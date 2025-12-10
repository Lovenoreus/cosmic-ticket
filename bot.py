import time
import json
import asyncio
import uuid
import re
from typing import TypedDict, List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
import os
from dotenv import load_dotenv
from utils import parse_json_from_response
import sys
from pathlib import Path

# Add tools directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "tools"))
from tools.vector_database_tools import cosmic_database_tool2
import config

load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(
    model=os.getenv("AGENT_MODEL_NAME", "gpt-4o-mini"),
    temperature=0
)

# Load known questions
KNOWN_QUESTIONS_PATH = Path(__file__).parent / "known_questions.json"
with open(KNOWN_QUESTIONS_PATH, "r", encoding="utf8") as f:
    KNOWN_QUESTIONS = json.load(f)


@dataclass
class Question:
    """Represents a question in the ticket creation process"""
    question: str
    index: int
    answered: bool
    answer: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "index": self.index,
            "answered": self.answered,
            "answer": self.answer
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Question':
        return cls(
            question=data.get("question", ""),
            index=data.get("index", 0),
            answered=data.get("answered", False),
            answer=data.get("answer", None)
        )


class AgentState(TypedDict):
    user_input: str
    intent: dict
    conversation_history: List[BaseMessage]
    last_cosmic_query: Optional[str]
    last_cosmic_query_response: Optional[str]
    bot_response: Optional[str]
    known_problem_identified: bool
    questioning: bool
    questions: List[Dict[str, Any]]  # List of Question dicts
    first_question_run_complete: bool


def detect_intent(state: AgentState) -> AgentState:
    """Detect user intent from input"""
    user_input = state["user_input"]
    conversation_history = state.get("conversation_history", [])
    questioning = state.get("questioning", False)
    questions = state.get("questions", [])
    
    # Check if all questions are answered
    all_answered = False
    if questions:
        all_answered = all(q.get("answered", False) for q in questions)
    
    first_question_run_complete = state.get("first_question_run_complete", False)
    
    # If all questions are answered AND questions have been asked at least once, route to create_ticket
    if questions and all_answered and first_question_run_complete:
        # Add user input to conversation history before routing
        updated_history = conversation_history + [HumanMessage(content=user_input)]
        return {"intent": {"mode": "create_ticket"}, "conversation_history": updated_history}
    
    # If questioning is active, route to questioner_agent
    if questioning:
        # Add user input to conversation history before routing
        updated_history = conversation_history + [HumanMessage(content=user_input)]
        return {"intent": {"mode": "questioning"}, "conversation_history": updated_history}
    
    system_prompt = """You are an intent detection agent. Analyze the user's message and determine their intent based on the conversation history.

Return ONLY a valid JSON object with one of these three structures:
- If the user is mentioning a problem or asking a question: {{"mode": "cosmic_search"}}
- If the user wants to START creating a support ticket (will ask questions first): {{"mode": "start_ticket"}}
- If the user wants to COMPLETE/FINISH creating the ticket NOW (skip questions, create immediately): {{"mode": "create_ticket"}}

CRITICAL DISTINCTION:
- "start_ticket" = User wants to BEGIN the ticket creation process. The system will ask questions to gather information.
- "create_ticket" = User wants to FINISH/COMPLETE the ticket creation RIGHT NOW, skipping any remaining questions.

Examples:
- "My printer is broken" -> {{"mode": "cosmic_search"}}
- "I need help with login" -> {{"mode": "cosmic_search"}}
- "How does <something> work?" -> {{"mode": "cosmic_search"}}
- "How do I do <something>?" -> {{"mode": "cosmic_search"}}
- "Why can't I<something>?" -> {{"mode": "cosmic_search"}}
- "I want to create a ticket" -> {{"mode": "start_ticket"}}
- "Can you help me file a support request?" -> {{"mode": "start_ticket"}}
- "Create a ticket for this problem" -> {{"mode": "start_ticket"}}
- "Create ticket from this problem" -> {{"mode": "start_ticket"}}
- "File a ticket" -> {{"mode": "start_ticket"}}
- "Start ticket creation" -> {{"mode": "start_ticket"}}
- "Proceed with ticket creation" -> {{"mode": "start_ticket"}}
- "Complete the ticket now" -> {{"mode": "create_ticket"}}
- "Finish the ticket" -> {{"mode": "create_ticket"}}
- "Create the ticket anyway" -> {{"mode": "create_ticket"}}
- "Skip remaining questions and create ticket" -> {{"mode": "create_ticket"}}
- "That's enough, create the ticket" -> {{"mode": "create_ticket"}}
- "Just create it" -> {{"mode": "create_ticket"}}
- "No more questions, create it now" -> {{"mode": "create_ticket"}}

Consider the conversation history when determining intent. If the user is continuing a previous conversation, use context to make the best decision.

Return ONLY the JSON, nothing else."""

    messages = [SystemMessage(content=system_prompt)]
    
    # Add conversation history
    messages.extend(conversation_history)
    
    # Add current user input
    messages.append(HumanMessage(content=user_input))
    
    response = llm.invoke(messages)
    response_text = response.content.strip()
    
    # Parse JSON from response using utility function
    intent = parse_json_from_response(response_text)
    
    # Update conversation history with new messages
    updated_history = conversation_history + [
        HumanMessage(content=user_input),
        AIMessage(content=json.dumps(intent))
    ]
    
    return {"intent": intent, "conversation_history": updated_history}


def cosmic_search_agent(state: AgentState) -> AgentState:
    """Cosmic search agent that handles cosmic_search mode"""
    # Extract the cosmic query from user input
    user_input = state.get("user_input", "")
    conversation_history = state.get("conversation_history", [])
    last_cosmic_query = user_input
    
    # Call the cosmic database tool (async function, run in sync context)
    # Use configuration values from config
    tool_result = asyncio.run(cosmic_database_tool2(
        query=last_cosmic_query,
        collection_name=config.COSMIC_DATABASE_COLLECTION_NAME,
        limit=config.QDRANT_RESULT_LIMIT,
        min_score=config.QDRANT_MIN_SCORE
    ))
    
    # Store the full tool result as JSON string in last_cosmic_query_response
    last_cosmic_query_response = json.dumps(tool_result, indent=2)
    
    # Extract the RAG answer from the tool result for bot response
    # The tool returns a dict with "message" field containing the RAG answer
    bot_response = tool_result.get("message", "")
    
    # If no message, use the full tool result as fallback
    if not bot_response:
        bot_response = 'I am sorry, I could not find any information on that topic.'
    
    # Update conversation history
    updated_history = list(conversation_history)
    # Check if user input is already in history (it should be from detect_intent)
    # If not, add it
    if not updated_history or (isinstance(updated_history[-1], AIMessage) and updated_history[-1].content != json.dumps({"mode": "cosmic_search"})):
        updated_history.append(HumanMessage(content=user_input))
    # Add the cosmic search response
    updated_history.append(AIMessage(content=bot_response))
    
    # Return the full modified state
    updated_state = dict(state)
    updated_state["last_cosmic_query"] = last_cosmic_query
    updated_state["last_cosmic_query_response"] = last_cosmic_query_response
    updated_state["bot_response"] = bot_response
    updated_state["conversation_history"] = updated_history
    
    return updated_state


def identify_known_question_agent(state: AgentState) -> AgentState:
    """Identify the most appropriate problem template from known_questions.json"""
    # Get the user problem from last cosmic query
    user_problem = state.get("last_cosmic_query", "")
    
    if not user_problem:
        # If no last cosmic query, use current user input
        user_problem = state.get("user_input", "")
    
    if not user_problem:
        # No problem to match, return unchanged
        return {"known_problem_identified": False}
    
    # Create a summary of all known question templates for the LLM
    templates_summary = []
    for idx, template in enumerate(KNOWN_QUESTIONS):
        template_info = {
            "index": idx,
            "issue_category": template.get("issue_category", ""),
            "description": template.get("description", ""),
            "keywords": template.get("keywords", []),
            "queue": template.get("queue", ""),
            "urgency_level": template.get("urgency_level", ""),
            "text": template.get("text", "")
        }
        templates_summary.append(template_info)
    
    # Create prompt for matching
    system_prompt = """You are a problem classification agent. Your task is to match a user's problem description to the most appropriate problem template from a list of known issue categories.

You will be given:
1. A user problem description
2. A list of problem templates, each with:
   - issue_category: The category name
   - description: Detailed description of what the category covers
   - keywords: Relevant keywords for this category
   - queue: Which department handles this
   - urgency_level: Typical urgency
   - text: Summary text

Your job is to:
1. Analyze the user's problem description
2. Compare it against all available templates
3. Consider keywords, descriptions, and context
4. Select the BEST matching template based on semantic similarity and relevance
5. Return ONLY a JSON object with the index of the matched template

Return format:
{
  "matched_index": <integer index of the best matching template>,
  "confidence": <float between 0.0 and 1.0 indicating match confidence>,
  "reasoning": "<brief explanation of why this template matches>"
}

If no template is a good match (confidence < 0.5), return:
{
  "matched_index": -1,
  "confidence": <low confidence value>,
  "reasoning": "<explanation of why no good match was found>"
}"""

    # Format templates for the prompt
    templates_text = json.dumps(templates_summary, indent=2)
    
    user_prompt = f"""User Problem Description:
{user_problem}

Available Problem Templates:
{templates_text}

Analyze the user's problem and identify the best matching template. Return ONLY the JSON response."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    response_text = response.content.strip()
    
    # Parse the response
    try:
        match_result = parse_json_from_response(response_text, default={"matched_index": -1, "confidence": 0.0, "reasoning": "Failed to parse response"})
        matched_index = match_result.get("matched_index", -1)
        confidence = match_result.get("confidence", 0.0)
        if config.DEBUG:
            print(f"Matched index: {matched_index}")
            print(f"Confidence: {confidence}")
            print(f"Reasoning: {match_result.get('reasoning', 'No reasoning provided')}")
        
        # If we have a valid match with reasonable confidence
        if matched_index >= 0 and matched_index < len(KNOWN_QUESTIONS) and confidence >= 0.5:
            matched_template = KNOWN_QUESTIONS[matched_index]
            # Initialize questions from the matched template
            questions_to_ask = matched_template.get("questions_to_ask", [])
            questions = []
            for idx, question_text in enumerate(questions_to_ask, start=1):
                questions.append(Question(
                    question=question_text,
                    index=idx,
                    answered=False,
                    answer=None
                ).to_dict())
            
            # Return the full modified state
            updated_state = dict(state)
            updated_state["known_problem_identified"] = True
            updated_state["matched_template"] = matched_template
            updated_state["match_confidence"] = confidence
            updated_state["questions"] = questions
            updated_state["questioning"] = True  # Start questioning process
            return updated_state
        else:
            # Low confidence or no match
            updated_state = dict(state)
            updated_state["known_problem_identified"] = False
            updated_state["match_confidence"] = confidence
            return updated_state
    except Exception as e:
        # Error in matching
        updated_state = dict(state)
        updated_state["known_problem_identified"] = False
        updated_state["match_error"] = str(e)
        return updated_state


def questioner_agent(state: AgentState) -> AgentState:
    """Questioner agent that gathers answers for ticket creation questions"""
    questions_data = state.get("questions", [])
    user_input = state.get("user_input", "")
    conversation_history = state.get("conversation_history", [])
    last_cosmic_query = state.get("last_cosmic_query", "")
    matched_template = state.get("matched_template", {})
    first_question_run_complete = state.get("first_question_run_complete", False)
    
    # If no questions initialized, return error state
    if not questions_data:
        updated_state = dict(state)
        updated_state["bot_response"] = "Error: No questions initialized. Please start ticket creation again."
        updated_state["questioning"] = False
        return updated_state
    
    # Convert question dicts to Question objects
    questions = [Question.from_dict(q) for q in questions_data]
    
    # Get user problem description
    user_problem = last_cosmic_query if last_cosmic_query else user_input
    
    # First run: scan conversation history and current input, return full formatted response
    if not first_question_run_complete:
        # First, scan conversation history for answers
        if not any(q.answered for q in questions):
            # Extract conversation text for analysis
            conversation_text = ""
            for msg in conversation_history:
                if isinstance(msg, HumanMessage):
                    conversation_text += f"User: {msg.content}\n"
                elif isinstance(msg, AIMessage):
                    conversation_text += f"Assistant: {msg.content}\n"
            
            # Use LLM to extract answers from conversation history
            if conversation_text:
                system_prompt = """You are an information extraction agent. Your task is to identify which questions have already been answered in the conversation history.

You will be given:
1. A list of questions that need answers
2. The full conversation history

For each question, determine if an answer can be found in the conversation history. If yes, extract the answer. If the user explicitly said they don't know or similar phrases, mark the answer as "unknown".

Return ONLY a JSON object with this structure:
{
  "answers": [
    {
      "question_index": <1-based index>,
      "answered": <true if answer found, false otherwise>,
      "answer": "<extracted answer or 'unknown' if user said they don't know, or null if not answered>"
    },
    ...
  ]
}"""

                questions_text = "\n".join([f"{q.index}. {q.question}" for q in questions])
                user_prompt = f"""Questions to check:
{questions_text}

Conversation History:
{conversation_text}

Analyze the conversation and identify which questions have been answered. Return ONLY the JSON response."""

                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                
                try:
                    response = llm.invoke(messages)
                    response_text = response.content.strip()
                    history_answers = parse_json_from_response(response_text, default={"answers": []})
                    
                    # Update questions with answers from history
                    for answer_info in history_answers.get("answers", []):
                        q_idx = answer_info.get("question_index", 0)
                        if 1 <= q_idx <= len(questions):
                            questions[q_idx - 1].answered = answer_info.get("answered", False)
                            if answer_info.get("answered"):
                                questions[q_idx - 1].answer = answer_info.get("answer", None)
                except Exception as e:
                    # If extraction fails, continue without history answers
                    pass
    
        # Now process current user input for answers (only if we have unanswered questions)
        unanswered_before = [q for q in questions if not q.answered]
        if user_input and unanswered_before:
            # Use LLM to extract answers from current user input
            unanswered_questions = [q for q in questions if not q.answered]
            if unanswered_questions:
                system_prompt = """You are an answer extraction agent. Your task is to identify which questions the user is answering in their current message.

The user may:
1. Answer one question
2. Answer multiple questions at once
3. Say they don't know (which is a valid answer - mark as "unknown")
4. Say they don't know the answers to remaining/other questions (mark ALL remaining unanswered questions as "unknown")
5. Provide partial information

IMPORTANT: If the user says phrases like:
- "I don't know the answers to the other questions"
- "I don't know the remaining questions"
- "I don't know answers to questions 4 and 5"
- "I don't know" (when referring to multiple/all remaining questions)

Then you MUST mark ALL remaining unanswered questions as answered with "unknown".

For each question, determine if the user's message contains an answer. If the user says "I don't know", "I'm not sure", "no idea", or similar phrases, that is a valid answer and should be marked as "unknown".

Return ONLY a JSON object with this structure:
{
  "answers": [
    {
      "question_index": <1-based index>,
      "answered": <true if answer found in this message, false otherwise>,
      "answer": "<extracted answer or 'unknown' if user said they don't know, or null if not answered>"
    },
    ...
  ]
}

CRITICAL: If the user indicates they don't know answers to remaining/other questions, you MUST include ALL unanswered questions in the response with answer="unknown"."""

                questions_text = "\n".join([f"{q.index}. {q.question}" for q in unanswered_questions])
                user_prompt = f"""Unanswered Questions:
{questions_text}

User's Current Message:
{user_input}

Analyze the user's message and identify which questions they are answering. Return ONLY the JSON response."""

                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                
                try:
                    response = llm.invoke(messages)
                    response_text = response.content.strip()
                    current_answers = parse_json_from_response(response_text, default={"answers": []})
                    
                    # Update questions with answers from current input
                    for answer_info in current_answers.get("answers", []):
                        q_idx = answer_info.get("question_index", 0)
                        if 1 <= q_idx <= len(questions):
                            questions[q_idx - 1].answered = answer_info.get("answered", False)
                            if answer_info.get("answered"):
                                questions[q_idx - 1].answer = answer_info.get("answer", None)
                except Exception as e:
                    # If extraction fails, continue
                    pass
        
        # Check if all questions are answered
        all_answered = all(q.answered for q in questions)
        
        # Build bot response - FULL FORMAT for first run
        answered_questions = [q for q in questions if q.answered]
        unanswered_questions = [q for q in questions if not q.answered]
        
        bot_response_parts = []
        
        # Opening statement
        bot_response_parts.append(f"I understand that you want to create a support ticket for {user_problem}.")
        
        if answered_questions:
            bot_response_parts.append("\nBut first I need to gather some information for the relevant department to better understand the problem.")
            bot_response_parts.append("You have already given me answers for the below questions:\n")
            for q in answered_questions:
                answer_text = q.answer if q.answer != "unknown" else "you mentioned that you don't know"
                bot_response_parts.append(f"{q.index}. {q.question}")
                bot_response_parts.append(f"   You have already mentioned that {answer_text}")
        
        if unanswered_questions:
            if not answered_questions:
                bot_response_parts.append("\nBut first I need to gather some information for the relevant department to better understand the problem.")
            bot_response_parts.append("\nPlease provide answers to the below questions:\n")
            for q in unanswered_questions:
                bot_response_parts.append(f"{q.index}. {q.question}")
        else:
            bot_response_parts.append("\nThank you! I have gathered all the necessary information.")
        
        bot_response = "\n".join(bot_response_parts)
        
        # Mark first run as complete
        first_question_run_complete = True
    
    else:
        # Subsequent runs: only process current user input, return simpler response
        unanswered_before = [q for q in questions if not q.answered]
        if user_input and unanswered_before:
            # Use LLM to extract answers from current user input
            unanswered_questions = [q for q in questions if not q.answered]
            if unanswered_questions:
                system_prompt = """You are an answer extraction agent. Your task is to identify which questions the user is answering in their current message.

The user may:
1. Answer one question
2. Answer multiple questions at once
3. Say they don't know (which is a valid answer - mark as "unknown")
4. Say they don't know the answers to remaining/other questions (mark ALL remaining unanswered questions as "unknown")
5. Provide partial information

IMPORTANT: If the user says phrases like:
- "I don't know the answers to the other questions"
- "I don't know the remaining questions"
- "I don't know answers to questions 4 and 5"
- "I don't know" (when referring to multiple/all remaining questions)

Then you MUST mark ALL remaining unanswered questions as answered with "unknown".

For each question, determine if the user's message contains an answer. If the user says "I don't know", "I'm not sure", "no idea", or similar phrases, that is a valid answer and should be marked as "unknown".

Return ONLY a JSON object with this structure:
{
  "answers": [
    {
      "question_index": <1-based index>,
      "answered": <true if answer found in this message, false otherwise>,
      "answer": "<extracted answer or 'unknown' if user said they don't know, or null if not answered>"
    },
    ...
  ]
}

CRITICAL: If the user indicates they don't know answers to remaining/other questions, you MUST include ALL unanswered questions in the response with answer="unknown"."""

                questions_text = "\n".join([f"{q.index}. {q.question}" for q in unanswered_questions])
                user_prompt = f"""Unanswered Questions:
{questions_text}

User's Current Message:
{user_input}

Analyze the user's message and identify which questions they are answering. Return ONLY the JSON response."""

                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                
                try:
                    response = llm.invoke(messages)
                    response_text = response.content.strip()
                    current_answers = parse_json_from_response(response_text, default={"answers": []})
                    
                    # Update questions with answers from current input
                    for answer_info in current_answers.get("answers", []):
                        q_idx = answer_info.get("question_index", 0)
                        if 1 <= q_idx <= len(questions):
                            questions[q_idx - 1].answered = answer_info.get("answered", False)
                            if answer_info.get("answered"):
                                questions[q_idx - 1].answer = answer_info.get("answer", None)
                except Exception as e:
                    # If extraction fails, continue
                    pass
        
        # Check if all questions are answered
        all_answered = all(q.answered for q in questions)
        
        # Build bot response - SIMPLER FORMAT for subsequent runs
        unanswered_questions = [q for q in questions if not q.answered]
        
        # Generate varied intro message
        import random
        intro_options = [
            "I see, what about",
            "Thank you for your last input. Please proceed to answer below questions as well",
            "Got it. Now I need a few more details",
            "Thanks for that information. Could you also provide answers to",
            "Understood. I still need answers to the following questions",
            "Good, thank you. Please also answer",
            "I appreciate that. Moving forward, I need responses to",
            "Noted. To complete the ticket, please answer",
            "Thank you. I still need information on",
            "Perfect. A couple more questions remain"
        ]
        intro = random.choice(intro_options)
        
        bot_response_parts = [intro]
        bot_response_parts.append("")
        
        # Add remaining questions with their indices
        for q in unanswered_questions:
            bot_response_parts.append(f"{q.index}. {q.question}")
        
        if all_answered:
            bot_response_parts.append("\nThank you! I have gathered all the necessary information.")
        
        bot_response = "\n".join(bot_response_parts)
    
    # Convert questions back to dicts
    questions_dicts = [q.to_dict() for q in questions]
    
    # Update state
    updated_state = dict(state)
    updated_state["questions"] = questions_dicts
    updated_state["questioning"] = not all_answered  # Set to False when all answered
    updated_state["bot_response"] = bot_response
    updated_state["first_question_run_complete"] = first_question_run_complete
    
    # Update conversation history
    # User input should already be added by detect_intent, but add it as a safety measure if missing
    updated_history = list(conversation_history)
    # Check if user input is already in history (should be the last message if added by detect_intent)
    if not updated_history or not (isinstance(updated_history[-1], HumanMessage) and updated_history[-1].content == user_input):
        # User input not in history yet, add it
        updated_history.append(HumanMessage(content=user_input))
    # Always add bot response
    updated_history.append(AIMessage(content=bot_response))
    updated_state["conversation_history"] = updated_history
    
    return updated_state


def create_ticket(state: AgentState) -> AgentState:
    """Create ticket agent that generates ticket title and saves ticket JSON file"""
    questions_data = state.get("questions", [])
    matched_template = state.get("matched_template", {})
    conversation_history = state.get("conversation_history", [])
    last_cosmic_query = state.get("last_cosmic_query", "")
    user_input = state.get("user_input", "")
    
    # Get user problem description
    user_problem = last_cosmic_query if last_cosmic_query else user_input
    
    # Convert questions to Question objects
    questions = [Question.from_dict(q) for q in questions_data] if questions_data else []
    
    # Generate ticket title using LLM
    system_prompt = """You are a ticket title generator. Your task is to create a concise, descriptive title for a support ticket.

The title should:
- Be between 3 to 10 words
- Be as short as possible without sacrificing meaning
- Accurately describe the problem
- Use lowercase letters and underscores instead of spaces
- Be suitable for use as a filename

Examples:
- "broken printer" -> "broken_printer"
- "HSAID configuration issue" -> "hsaid_configuration_issue"
- "ambulance station mapping problem" -> "ambulance_station_mapping"
- "patient registration error" -> "patient_registration_error"

Return ONLY the title text, nothing else. No quotes, no explanation, just the title."""
    
    # Build context for title generation
    context_parts = [f"Problem: {user_problem}"]
    
    if matched_template:
        context_parts.append(f"Issue Category: {matched_template.get('issue_category', '')}")
    
    # Add answered questions for context
    answered_questions = [q for q in questions if q.answered]
    if answered_questions:
        context_parts.append("\nKey Information:")
        for q in answered_questions[:3]:  # Use first 3 answered questions
            if q.answer and q.answer != "unknown":
                context_parts.append(f"- {q.answer}")
    
    context = "\n".join(context_parts)
    
    user_prompt = f"""Based on the following information, generate a concise ticket title:

{context}

Generate a short, descriptive title (3-10 words, use underscores instead of spaces)."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        ticket_title = response.content.strip()
        # Clean up title: remove quotes, convert to lowercase, replace spaces with underscores
        ticket_title = ticket_title.strip('"\'')
        ticket_title = ticket_title.lower().replace(" ", "_")
        # Remove any special characters except underscores and hyphens
        ticket_title = re.sub(r'[^a-z0-9_-]', '', ticket_title)
        # Ensure it's not empty
        if not ticket_title:
            ticket_title = "support_ticket"
    except Exception as e:
        # Fallback title
        if config.DEBUG:
            print(f"Title generation failed: {e}")
        ticket_title = "support_ticket"
    
    # Generate UUID
    ticket_uuid = str(uuid.uuid4())
    
    # Build ticket data
    ticket_data = {
        "ticket_title": ticket_title,
        "ticket_uuid": ticket_uuid,
        "description": user_problem,
        "category": matched_template.get("issue_category", ""),
        "assigned_queue": matched_template.get("queue", ""),
        "priority": matched_template.get("urgency_level", ""),
        "questions": [q.to_dict() for q in questions],
        "conversation_history": []
    }
    
    # Include entire conversation history in the ticket
    for msg in conversation_history:
        if isinstance(msg, HumanMessage):
            ticket_data["conversation_history"].append({
                "role": "user",
                "content": msg.content
            })
        elif isinstance(msg, AIMessage):
            ticket_data["conversation_history"].append({
                "role": "assistant",
                "content": msg.content
            })
    
    # Add current user input if not already in history
    if user_input and not any(msg.get("content") == user_input for msg in ticket_data["conversation_history"]):
        ticket_data["conversation_history"].append({
            "role": "user",
            "content": user_input
        })
    
    # Save ticket to file
    tickets_dir = Path(__file__).parent / "tickets"
    tickets_dir.mkdir(exist_ok=True)
    
    filename = f"{ticket_title}_{ticket_uuid}.json"
    filepath = tickets_dir / filename
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(ticket_data, f, indent=2, ensure_ascii=False)
        
        if config.DEBUG:
            print(f"Ticket saved to: {filepath}")
        
        # Generate bot response
        bot_response = f"Ticket created successfully! Ticket ID: {ticket_uuid}\nTitle: {ticket_title.replace('_', ' ').title()}\nSaved to: {filename}"
        
    except Exception as e:
        if config.DEBUG:
            print(f"Error saving ticket: {e}")
        bot_response = f"Error creating ticket: {str(e)}"
    
    # Update state
    updated_state = dict(state)
    updated_state["bot_response"] = bot_response
    updated_state["questioning"] = False  # Stop questioning
    
    # Update conversation history
    updated_history = list(conversation_history)
    if not updated_history or (isinstance(updated_history[-1], HumanMessage) and updated_history[-1].content != user_input):
        updated_history.append(HumanMessage(content=user_input))
    updated_history.append(AIMessage(content=bot_response))
    updated_state["conversation_history"] = updated_history

    print(conversation_history)
    
    return updated_state


def route_after_intent(state: AgentState) -> str:
    """Route to appropriate agent based on detected intent"""
    intent = state.get("intent", {})
    mode = intent.get("mode", "cosmic_search")
    questioning = state.get("questioning", False)
    first_question_run_complete = state.get("first_question_run_complete", False)
    
    # If create_ticket mode, only route to create_ticket if questions have been asked
    # Otherwise, route to start_ticket flow
    if mode == "create_ticket":
        if first_question_run_complete:
            return "create_ticket"
        else:
            # User wants to create ticket but questions haven't been asked yet
            # Route to start_ticket flow instead
            return "identify_known_question_agent"
    # If questioning is active, always route to questioner_agent
    elif questioning or mode == "questioning":
        return "questioner_agent"
    elif mode == "cosmic_search":
        return "cosmic_search_agent"
    elif mode == "start_ticket":
        return "identify_known_question_agent"
    else:
        return END


# Create the graph
workflow = StateGraph(AgentState)
workflow.add_node("detect_intent", detect_intent)
workflow.add_node("cosmic_search_agent", cosmic_search_agent)
workflow.add_node("identify_known_question_agent", identify_known_question_agent)
workflow.add_node("questioner_agent", questioner_agent)
workflow.add_node("create_ticket", create_ticket)
workflow.set_entry_point("detect_intent")
workflow.add_conditional_edges(
    "detect_intent",
    route_after_intent,
    {
        "cosmic_search_agent": "cosmic_search_agent",
        "identify_known_question_agent": "identify_known_question_agent",
        "questioner_agent": "questioner_agent",
        "create_ticket": "create_ticket",
        END: END
    }
)
workflow.add_edge("cosmic_search_agent", END)
workflow.add_conditional_edges(
    "identify_known_question_agent",
    lambda state: "questioner_agent" if state.get("questioning", False) else END,
    {
        "questioner_agent": "questioner_agent",
        END: END
    }
)
workflow.add_conditional_edges(
    "questioner_agent",
    lambda state: "create_ticket" if (
        state.get("questions") and 
        len(state.get("questions", [])) > 0 and
        all(q.get("answered", False) for q in state.get("questions", [])) and
        not state.get("questioning", True)  # questioning is False when all answered
    ) else END,
    {
        "create_ticket": "create_ticket",
        END: END
    }
)
workflow.add_edge("create_ticket", END)

# Compile the graph
app = workflow.compile()


def main():
    """Main loop for terminal interaction"""
    print("Intent Detection Bot - Type 'exit' to quit\n")
    
    # Initialize global state
    global_state = {
        "conversation_history": [],
        "last_cosmic_query": None,
        "last_cosmic_query_response": None,
        "bot_response": None,
        "known_problem_identified": False,
        "questioning": False,
        "questions": [],
        "first_question_run_complete": False
    }
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Start timing
            start_time = time.time()
            
            # Prepare state for this invocation (merge with global state)
            current_state = {
                "user_input": user_input,
                "conversation_history": global_state.get("conversation_history", []),
                "last_cosmic_query": global_state.get("last_cosmic_query"),
                "last_cosmic_query_response": global_state.get("last_cosmic_query_response"),
                "bot_response": None,
                "known_problem_identified": global_state.get("known_problem_identified", False),
                "questioning": global_state.get("questioning", False),
                "questions": global_state.get("questions", []),
                "first_question_run_complete": global_state.get("first_question_run_complete", False)
            }
            
            # Preserve matched_template if it exists
            if "matched_template" in global_state:
                current_state["matched_template"] = global_state["matched_template"]
            
            # Run the graph
            result = app.invoke(current_state)
            
            # Update global state with the result
            global_state.update({
                "conversation_history": result.get("conversation_history", []),
                "last_cosmic_query": result.get("last_cosmic_query"),
                "last_cosmic_query_response": result.get("last_cosmic_query_response"),
                "bot_response": result.get("bot_response"),
                "known_problem_identified": result.get("known_problem_identified", False),
                "questioning": result.get("questioning", False),
                "questions": result.get("questions", []),
                "first_question_run_complete": result.get("first_question_run_complete", False)
            })
            
            # Preserve matched_template if it exists
            if "matched_template" in result:
                global_state["matched_template"] = result["matched_template"]
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Display bot response (from cosmic_search_agent or default)
            bot_response = result.get("bot_response")
            if bot_response:
                print(f"\nBot: {bot_response} ({response_time:.3f}s)\n")
            else:
                # Fallback if no agent set a response (e.g., start_ticket mode)
                intent = result.get("intent", {})
                print(f"Bot: {json.dumps(intent)} ({response_time:.3f}s)\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")
        


if __name__ == "__main__":
    main()

