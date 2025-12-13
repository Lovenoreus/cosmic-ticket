# -------------------- Built-in Libraries --------------------
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
import logging

# Add tools directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "tools"))
from tools.vector_database_tools import cosmic_database_tool2
from tools.ask_qdrant import ask_question
import config
from jira_ticket_tool import create_jira_ticket

load_dotenv()

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)

logger = logging.getLogger(__name__)

# Set appropriate log levels for noisy libraries
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('langchain').setLevel(logging.WARNING)
logging.getLogger('langchain_core').setLevel(logging.WARNING)
logging.getLogger('langgraph').setLevel(logging.WARNING)

# ============================================================================
# LLM INITIALIZATION
# ============================================================================

# Initialize the LLM
llm = ChatOpenAI(
    model=os.getenv("AGENT_MODEL_NAME", "gpt-4o-mini"),
    temperature=0
)

logger.info(f"LLM initialized with model: {os.getenv('AGENT_MODEL_NAME', 'gpt-4o-mini')}")

# Known questions are now stored in Qdrant vector database
# Collection name: config.KNOWN_QUESTIONS_COLLECTION_NAME
logger.info(f"Known questions collection: {config.KNOWN_QUESTIONS_COLLECTION_NAME}")


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
    logger.info("=" * 80)
    logger.info("DETECT INTENT AGENT")
    logger.info("=" * 80)

    user_input = state["user_input"]
    conversation_history = state.get("conversation_history", [])
    questioning = state.get("questioning", False)
    questions = state.get("questions", [])

    logger.debug(f"User input: {user_input[:100]}...")
    logger.debug(f"Questioning mode: {questioning}")
    logger.debug(f"Number of questions: {len(questions)}")

    # Check if all questions are answered
    all_answered = False
    if questions:
        all_answered = all(q.get("answered", False) for q in questions)
        logger.debug(f"All questions answered: {all_answered}")

    first_question_run_complete = state.get("first_question_run_complete", False)
    logger.debug(f"First question run complete: {first_question_run_complete}")

    # If all questions are answered AND questions have been asked at least once, route to create_ticket
    if questions and all_answered and first_question_run_complete:
        logger.info("Routing to create_ticket - all questions answered")
        updated_history = conversation_history + [HumanMessage(content=user_input)]
        return {"intent": {"mode": "create_ticket"}, "conversation_history": updated_history}

    # If questioning is active, route to questioner_agent
    if questioning:
        logger.info("Routing to questioner_agent - questioning mode active")
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

    logger.debug("Invoking LLM for intent detection")
    response = llm.invoke(messages)
    response_text = response.content.strip()
    logger.debug(f"LLM response: {response_text[:200]}...")

    # Parse JSON from response using utility function
    intent = parse_json_from_response(response_text)

    logger.info(f"Detected intent: {intent}")

    # Update conversation history with new messages
    updated_history = conversation_history + [
        HumanMessage(content=user_input),
        AIMessage(content=json.dumps(intent))
    ]

    logger.info(f"Intent detection complete. Mode: {intent.get('mode', 'unknown')}")
    return {"intent": intent, "conversation_history": updated_history}


def cosmic_search_agent(state: AgentState) -> AgentState:
    """Cosmic search agent that handles cosmic_search mode"""
    logger.info("=" * 80)
    logger.info("COSMIC SEARCH AGENT")
    logger.info("=" * 80)

    # Extract the cosmic query from user input
    user_input = state.get("user_input", "")
    conversation_history = state.get("conversation_history", [])
    last_cosmic_query = user_input

    logger.info(f"Cosmic search query: {last_cosmic_query[:100]}...")

    # Call the cosmic database tool (async function, run in sync context)
    # Use configuration values from config
    logger.debug(f"Calling cosmic_database_tool2 with collection: {config.COSMIC_DATABASE_COLLECTION_NAME}")
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
        logger.warning("No message found in tool result, using fallback response")
    else:
        logger.info(f"Cosmic search response generated. Length: {len(bot_response)} characters")

    # Update conversation history
    updated_history = list(conversation_history)
    # Check if user input is already in history (it should be from detect_intent)
    # If not, add it
    if not updated_history or (isinstance(updated_history[-1], AIMessage) and updated_history[-1].content != json.dumps(
            {"mode": "cosmic_search"})):
        updated_history.append(HumanMessage(content=user_input))
    # Add the cosmic search response
    updated_history.append(AIMessage(content=bot_response))

    logger.debug(f"Conversation history updated. Total messages: {len(updated_history)}")

    # Return the full modified state
    updated_state = dict(state)
    updated_state["last_cosmic_query"] = last_cosmic_query
    updated_state["last_cosmic_query_response"] = last_cosmic_query_response
    updated_state["bot_response"] = bot_response
    updated_state["conversation_history"] = updated_history

    logger.info("Cosmic search agent completed successfully")
    return updated_state


def identify_known_question_agent(state: AgentState) -> AgentState:
    """Identify the most appropriate problem template from known_questions.json"""
    logger.info("=" * 80)
    logger.info("IDENTIFY KNOWN QUESTION AGENT")
    logger.info("=" * 80)

    # Get the user problem from last cosmic query
    user_problem = state.get("last_cosmic_query", "")

    if not user_problem:
        # If no last cosmic query, use current user input
        user_problem = state.get("user_input", "")

    logger.info(f"User problem: {user_problem[:100]}...")

    # Get conversation history to provide more context
    conversation_history = state.get("conversation_history", [])

    # Extract the latest 10 entries from conversation history
    recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history

    logger.debug(f"Using {len(recent_history)} recent conversation messages for context")

    # Format conversation history for context
    conversation_context = ""
    if recent_history:
        conversation_context = "\n\nRecent Conversation History:\n"
        for msg in recent_history:
            if isinstance(msg, HumanMessage):
                conversation_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                conversation_context += f"Assistant: {msg.content}\n"

    if not user_problem and not conversation_context:
        # No problem to match, return unchanged
        logger.warning("No problem description found")
        updated_state = dict(state)
        updated_state["known_problem_identified"] = False
        updated_state["bot_response"] = (
            "I understand you want to create a ticket, but I need more information about the problem. "
            "Could you please describe the issue you're experiencing?"
        )
        # Update conversation history
        user_input = state.get("user_input", "")
        updated_history = list(conversation_history)
        if not updated_history or not (
                isinstance(updated_history[-1], HumanMessage) and updated_history[-1].content == user_input):
            updated_history.append(HumanMessage(content=user_input))
        updated_history.append(AIMessage(content=updated_state["bot_response"]))
        updated_state["conversation_history"] = updated_history
        return updated_state

    # Build search query from user problem and conversation context
    search_query = user_problem
    if conversation_context:
        # Include conversation context in search query for better matching
        search_query = f"{user_problem}\n{conversation_context}"

    logger.debug(f"Searching known questions collection: {config.KNOWN_QUESTIONS_COLLECTION_NAME}")
    
    # Perform vector search in Qdrant
    try:
        # Use ask_question to search the known_questions collection
        # We only need the top result (limit=1)
        # ask_question is a synchronous function, so we call it directly
        search_result = ask_question(
            search_query,
            collection=config.KNOWN_QUESTIONS_COLLECTION_NAME,
            limit=1,
            qdrant_host=config.QDRANT_HOST,
            qdrant_port=config.QDRANT_PORT,
            openai_api_key=config.OPENAI_API_KEY if not config.USE_OLLAMA else None,
            ollama_model=config.EMBEDDINGS_MODEL_NAME if config.USE_OLLAMA else None,
            ollama_base_url=config.OLLAMA_BASE_URL if config.USE_OLLAMA else None
        )

        # Extract results from search
        results = search_result.get("results", [])
        
        if not results:
            logger.warning("No results found in known questions collection")
            updated_state = dict(state)
            updated_state["known_problem_identified"] = False
            updated_state["match_confidence"] = 0.0
            updated_state["bot_response"] = (
                "I understand you want to create a ticket, but I'm having trouble identifying "
                "the specific category for your issue. Could you please provide more details about "
                "the problem you're experiencing? For example:\n"
                "- Is it related to hardware (printer, computer, etc.)?\n"
            )
            # Update conversation history
            user_input = state.get("user_input", "")
            updated_history = list(conversation_history)
            if not updated_history or not (
                    isinstance(updated_history[-1], HumanMessage) and updated_history[-1].content == user_input):
                updated_history.append(HumanMessage(content=user_input))
            updated_history.append(AIMessage(content=updated_state["bot_response"]))
            updated_state["conversation_history"] = updated_history
            return updated_state

        # Get the top result (highest score)
        # results is a list of HybridResult objects
        top_result = results[0]
        confidence = float(top_result.score)
        payload = top_result.payload or {}
        
        logger.info(f"Top match - Score: {confidence}, Issue Category: {payload.get('issue_category', 'Unknown')}")

        # Use confidence threshold (0.5) to determine if match is good enough
        # Note: Qdrant cosine similarity scores are typically 0.0-1.0, where 1.0 is perfect match
        # We'll use 0.5 as the threshold (can be adjusted based on testing)
        min_confidence = 0.5
        
        if confidence >= min_confidence:
            # Extract template from payload (all template fields are stored in payload)
            matched_template = payload
            
            logger.info(f"Successfully matched template: {matched_template.get('issue_category', 'Unknown')}")

            # Initialize questions from the matched template
            questions_to_ask = matched_template.get("questions_to_ask", [])
            logger.debug(f"Initializing {len(questions_to_ask)} questions")

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

            logger.info("Questioning mode activated")
            return updated_state
        else:
            # Low confidence match
            logger.warning(f"Low confidence match (score: {confidence}, threshold: {min_confidence})")
            updated_state = dict(state)
            updated_state["known_problem_identified"] = False
            updated_state["match_confidence"] = confidence
            updated_state["bot_response"] = (
                "I understand you want to create a ticket, but I'm having trouble identifying "
                "the specific category for your issue. Could you please provide more details about "
                "the problem you're experiencing? For example:\n"
                "- Is it related to hardware (printer, computer, etc.)?\n"
                "- Is it a software or application issue?\n"
                "- Is it about network or connectivity?\n"
                "- Is it related to access or permissions?"
            )
            # Update conversation history
            user_input = state.get("user_input", "")
            updated_history = list(conversation_history)
            if not updated_history or not (
                    isinstance(updated_history[-1], HumanMessage) and updated_history[-1].content == user_input):
                updated_history.append(HumanMessage(content=user_input))
            updated_history.append(AIMessage(content=updated_state["bot_response"]))
            updated_state["conversation_history"] = updated_history
            return updated_state
    except Exception as e:
        # Error in matching
        logger.error(f"Error in template matching: {e}", exc_info=True)
        updated_state = dict(state)
        updated_state["known_problem_identified"] = False
        updated_state["match_error"] = str(e)
        updated_state["bot_response"] = (
            "I apologize, but I encountered an error while trying to identify your issue category. "
            "Could you please describe your problem in more detail so I can assist you better?"
        )
        # Update conversation history
        user_input = state.get("user_input", "")
        updated_history = list(conversation_history)
        if not updated_history or not (
                isinstance(updated_history[-1], HumanMessage) and updated_history[-1].content == user_input):
            updated_history.append(HumanMessage(content=user_input))
        updated_history.append(AIMessage(content=updated_state["bot_response"]))
        updated_state["conversation_history"] = updated_history
        return updated_state


def questioner_agent(state: AgentState) -> AgentState:
    """Questioner agent that gathers answers for ticket creation questions"""
    logger.info("=" * 80)
    logger.info("QUESTIONER AGENT")
    logger.info("=" * 80)

    questions_data = state.get("questions", [])
    user_input = state.get("user_input", "")
    conversation_history = state.get("conversation_history", [])
    last_cosmic_query = state.get("last_cosmic_query", "")
    matched_template = state.get("matched_template", {})
    first_question_run_complete = state.get("first_question_run_complete", False)

    logger.debug(f"Total questions: {len(questions_data)}")
    logger.debug(f"First run complete: {first_question_run_complete}")
    logger.debug(f"User input: {user_input[:100]}...")

    # If no questions initialized, return error state
    if not questions_data:
        logger.error("No questions initialized")
        updated_state = dict(state)
        updated_state["bot_response"] = "Error: No questions initialized. Please start ticket creation again."
        updated_state["questioning"] = False
        return updated_state

    # Convert question dicts to Question objects
    questions = [Question.from_dict(q) for q in questions_data]

    answered_count = sum(1 for q in questions if q.answered)
    logger.info(f"Questions status: {answered_count}/{len(questions)} answered")

    # Get user problem description
    user_problem = last_cosmic_query if last_cosmic_query else user_input

    # First run: scan conversation history and current input, return full formatted response
    if not first_question_run_complete:
        logger.info("First question run - scanning conversation history")

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
                logger.debug("Analyzing conversation history for pre-answered questions")
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
                    logger.debug("Invoking LLM for history analysis")
                    response = llm.invoke(messages)
                    response_text = response.content.strip()
                    history_answers = parse_json_from_response(response_text, default={"answers": []})

                    # Update questions with answers from history
                    history_answer_count = 0
                    for answer_info in history_answers.get("answers", []):
                        q_idx = answer_info.get("question_index", 0)
                        if 1 <= q_idx <= len(questions):
                            questions[q_idx - 1].answered = answer_info.get("answered", False)
                            if answer_info.get("answered"):
                                questions[q_idx - 1].answer = answer_info.get("answer", None)
                                history_answer_count += 1

                    logger.info(f"Extracted {history_answer_count} answers from conversation history")
                except Exception as e:
                    logger.warning(f"Failed to extract answers from history: {e}")
                    # If extraction fails, continue without history answers
                    pass

        # Now process current user input for answers (only if we have unanswered questions)
        unanswered_before = [q for q in questions if not q.answered]
        if user_input and unanswered_before:
            logger.debug(f"Processing current input for {len(unanswered_before)} unanswered questions")

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
                    logger.debug("Invoking LLM for current input analysis")
                    response = llm.invoke(messages)
                    response_text = response.content.strip()
                    current_answers = parse_json_from_response(response_text, default={"answers": []})

                    # Update questions with answers from current input
                    current_answer_count = 0
                    for answer_info in current_answers.get("answers", []):
                        q_idx = answer_info.get("question_index", 0)
                        if 1 <= q_idx <= len(questions):
                            questions[q_idx - 1].answered = answer_info.get("answered", False)
                            if answer_info.get("answered"):
                                questions[q_idx - 1].answer = answer_info.get("answer", None)
                                current_answer_count += 1

                    logger.info(f"Extracted {current_answer_count} answers from current input")
                except Exception as e:
                    logger.warning(f"Failed to extract answers from current input: {e}")
                    # If extraction fails, continue
                    pass

        # Check if all questions are answered
        all_answered = all(q.answered for q in questions)
        logger.info(f"All questions answered: {all_answered}")

        # Build bot response - FULL FORMAT for first run
        answered_questions = [q for q in questions if q.answered]
        unanswered_questions = [q for q in questions if not q.answered]

        logger.debug(f"Building response - {len(answered_questions)} answered, {len(unanswered_questions)} unanswered")

        bot_response_parts = []

        # Opening statement
        bot_response_parts.append(f"I understand that you want to create a support ticket for {user_problem}.")

        if answered_questions:
            bot_response_parts.append(
                "\nBut first I need to gather some information for the relevant department to better understand the problem.")
            bot_response_parts.append("You have already given me answers for the below questions:\n")
            for q in answered_questions:
                answer_text = q.answer if q.answer != "unknown" else "you mentioned that you don't know"
                bot_response_parts.append(f"{q.index}. {q.question}")
                bot_response_parts.append(f"   You have already mentioned that {answer_text}")

        if unanswered_questions:
            if not answered_questions:
                bot_response_parts.append(
                    "\nBut first I need to gather some information for the relevant department to better understand the problem.")
            bot_response_parts.append("\nPlease provide answers to the below questions:\n")
            for q in unanswered_questions:
                bot_response_parts.append(f"{q.index}. {q.question}")
        else:
            bot_response_parts.append("\nThank you! I have gathered all the necessary information.")

        bot_response = "\n".join(bot_response_parts)

        # Mark first run as complete
        first_question_run_complete = True
        logger.info("First question run completed")

    else:
        logger.info("Subsequent question run - processing user input only")

        # Subsequent runs: only process current user input, return simpler response
        unanswered_before = [q for q in questions if not q.answered]
        if user_input and unanswered_before:
            logger.debug(f"Processing {len(unanswered_before)} unanswered questions")

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
                    logger.debug("Invoking LLM for answer extraction")
                    response = llm.invoke(messages)
                    response_text = response.content.strip()
                    current_answers = parse_json_from_response(response_text, default={"answers": []})

                    # Update questions with answers from current input
                    answer_count = 0
                    for answer_info in current_answers.get("answers", []):
                        q_idx = answer_info.get("question_index", 0)
                        if 1 <= q_idx <= len(questions):
                            questions[q_idx - 1].answered = answer_info.get("answered", False)
                            if answer_info.get("answered"):
                                questions[q_idx - 1].answer = answer_info.get("answer", None)
                                answer_count += 1

                    logger.info(f"Extracted {answer_count} answers from user input")
                except Exception as e:
                    logger.warning(f"Failed to extract answers: {e}")
                    # If extraction fails, continue
                    pass

        # Check if all questions are answered
        all_answered = all(q.answered for q in questions)
        logger.info(f"All questions answered: {all_answered}")

        # Build bot response - SIMPLER FORMAT for subsequent runs
        unanswered_questions = [q for q in questions if not q.answered]

        logger.debug(f"Remaining unanswered questions: {len(unanswered_questions)}")

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

    logger.debug(f"Questioning mode: {updated_state['questioning']}")

    # Update conversation history
    # User input should already be added by detect_intent, but add it as a safety measure if missing
    updated_history = list(conversation_history)
    # Check if user input is already in history (should be the last message if added by detect_intent)
    if not updated_history or not (
            isinstance(updated_history[-1], HumanMessage) and updated_history[-1].content == user_input):
        # User input not in history yet, add it
        updated_history.append(HumanMessage(content=user_input))
    # Always add bot response
    updated_history.append(AIMessage(content=bot_response))
    updated_state["conversation_history"] = updated_history

    logger.info("Questioner agent completed")
    return updated_state


def create_ticket(state: AgentState) -> AgentState:
    """Create ticket agent that generates ticket title and saves ticket JSON file"""
    logger.info("=" * 80)
    logger.info("CREATE TICKET AGENT")
    logger.info("=" * 80)

    questions_data = state.get("questions", [])
    matched_template = state.get("matched_template", {})
    conversation_history = state.get("conversation_history", [])
    last_cosmic_query = state.get("last_cosmic_query", "")
    user_input = state.get("user_input", "")

    # Get user problem description
    user_problem = last_cosmic_query if last_cosmic_query else user_input
    logger.info(f"Creating ticket for problem: {user_problem[:100]}...")

    # Convert questions to Question objects
    questions = [Question.from_dict(q) for q in questions_data] if questions_data else []
    answered_count = sum(1 for q in questions if q.answered)
    logger.debug(f"Questions: {answered_count}/{len(questions)} answered")

    # Generate ticket title using LLM
    logger.debug("Generating ticket title")
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
        logger.info(f"Generated ticket title: {ticket_title}")
    except Exception as e:
        # Fallback title
        logger.error(f"Title generation failed: {e}", exc_info=True)
        ticket_title = "support_ticket"

    # Generate UUID
    ticket_uuid = str(uuid.uuid4())
    logger.info(f"Generated ticket UUID: {ticket_uuid}")

    # Format conversation history as strings
    conversation_history_str = []
    for msg in conversation_history:
        if isinstance(msg, HumanMessage):
            conversation_history_str.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            conversation_history_str.append(f"AI Assistant: {msg.content}")

    # Add current user input if not already in history
    if user_input:
        user_input_str = f"User: {user_input}"
        if not any(msg == user_input_str for msg in conversation_history_str):
            conversation_history_str.append(user_input_str)

    logger.debug(f"Conversation history: {len(conversation_history_str)} messages")

    # Generate summary using LLM
    logger.debug("Generating ticket summary")
    summary_prompt = """You are a ticket summary generator. Your task is to create a comprehensive summary of a support ticket based on:
1. The user's problem description
2. Questions asked and their answers
3. The conversation history

The summary should:
- Include all key information from the questions and answers
- Include any special information the user mentioned
- Be concise but comprehensive
- Be written in clear, professional language
- Focus on the problem and relevant details

Return ONLY the summary text, nothing else."""

    # Build context for summary generation
    summary_context_parts = [f"User Problem: {user_problem}"]

    if questions:
        summary_context_parts.append("\nQuestions and Answers:")
        for q in questions:
            if q.answered and q.answer:
                answer_text = q.answer if q.answer != "unknown" else "User indicated they do not know"
                summary_context_parts.append(f"Q: {q.question}")
                summary_context_parts.append(f"A: {answer_text}")

    summary_context_parts.append("\nConversation History:")
    summary_context_parts.append("\n".join(conversation_history_str))

    summary_context = "\n".join(summary_context_parts)

    summary_user_prompt = f"""Generate a comprehensive summary for this support ticket:

{summary_context}

Create a summary that includes all important details from the questions, answers, and conversation."""

    try:
        summary_messages = [
            SystemMessage(content=summary_prompt),
            HumanMessage(content=summary_user_prompt)
        ]
        summary_response = llm.invoke(summary_messages)
        summary = summary_response.content.strip()
        logger.info(f"Generated summary: {len(summary)} characters")
    except Exception as e:
        logger.error(f"Summary generation failed: {e}", exc_info=True)
        # Fallback summary
        summary = user_problem
        if questions:
            answered = [q for q in questions if q.answered and q.answer and q.answer != "unknown"]
            if answered:
                summary += "\n\nKey Information: " + "; ".join([q.answer for q in answered[:3]])

    # Get values from matched template
    assigned_queue = matched_template.get("queue", "Technical Support")
    category = matched_template.get("issue_category", "")
    priority = "High"  # Hard coded
    name = "Love Noreus"  # Hard coded

    logger.debug(f"Ticket details - Queue: {assigned_queue}, Category: {category}, Priority: {priority}")

    # Format conversation history for Jira description
    conversation_history_formatted = "\n".join(conversation_history_str)

    # Format description according to requirements
    ticket_title_display = ticket_title.replace("_", " ")
    description = f"""Summary:

{summary}

Assigned Queue:

{assigned_queue}

Priority:

{priority}

Name:

{name}

Category:

{category}

Conversation Topic:

{ticket_title_display}

Conversation History: 

{conversation_history_formatted}"""

    # Build ticket data
    ticket_data = {
        "ticket_title": ticket_title,
        "ticket_uuid": ticket_uuid,
        "description": user_problem,
        "category": category,
        "assigned_queue": assigned_queue,
        "priority": priority,
        "questions": [q.to_dict() for q in questions],
        "conversation_history": conversation_history_str
    }

    # Save ticket to file
    tickets_dir = Path(__file__).parent / "tickets"
    tickets_dir.mkdir(exist_ok=True)

    filename = f"{ticket_title}_{ticket_uuid}.json"
    filepath = tickets_dir / filename

    logger.info(f"Saving ticket to: {filepath}")

    # Create Jira ticket
    jira_result = None
    try:
        logger.info("Creating Jira ticket")
        jira_result = create_jira_ticket(
            conversation_topic=ticket_title_display,
            description=description,
            queue=assigned_queue,
            priority=priority,
            name=name,
            category=category
        )

        if jira_result.get("success"):
            logger.info(f"Jira ticket created successfully: {jira_result.get('key', 'Unknown')}")
        else:
            logger.error(f"Jira ticket creation failed: {jira_result.get('error', 'Unknown error')}")
    except Exception as e:
        logger.error(f"Error creating Jira ticket: {e}", exc_info=True)
        jira_result = {"success": False, "error": str(e)}

    # Save JSON ticket to file
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(ticket_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Ticket saved successfully to: {filepath}")

        # Generate bot response
        jira_info = ""
        if jira_result and jira_result.get("success"):
            jira_key = jira_result.get("key", "Unknown")
            jira_info = f"\nJira Ticket: {jira_key}"
        elif jira_result and not jira_result.get("success"):
            jira_info = f"\nJira Ticket: Failed to create ({jira_result.get('error', 'Unknown error')})"

        bot_response = f"Ticket created successfully! Ticket ID: {ticket_uuid}\nTitle: {ticket_title_display.title()}\nSaved to: {filename}{jira_info}"

    except Exception as e:
        logger.error(f"Error saving ticket: {e}", exc_info=True)
        bot_response = f"Error creating ticket: {str(e)}"

    # Reset state to initial state after ticket creation
    # The conversation history is already included in the ticket, so we can clear it completely
    logger.info("Resetting conversation state after ticket creation")
    updated_state = get_initial_state()
    updated_state["bot_response"] = bot_response

    logger.info("Ticket creation completed successfully")
    return updated_state


def create_ticket_without_known_question(state: AgentState) -> AgentState:
    """Create a ticket when no known question template is found"""
    logger.info("=" * 80)
    logger.info("CREATE TICKET WITHOUT KNOWN QUESTION")
    logger.info("=" * 80)

    # Get user problem and conversation history
    user_problem = state.get("last_cosmic_query", "")
    if not user_problem:
        user_problem = state.get("user_input", "")
    
    conversation_history = state.get("conversation_history", [])
    user_input = state.get("user_input", "")

    logger.info(f"Creating ticket for problem: {user_problem[:100]}...")

    # Format conversation history as strings
    conversation_history_str = []
    for msg in conversation_history:
        if isinstance(msg, HumanMessage):
            conversation_history_str.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            conversation_history_str.append(f"AI Assistant: {msg.content}")

    # Add current user input if not already in history
    if user_input:
        user_input_str = f"User: {user_input}"
        if not any(msg == user_input_str for msg in conversation_history_str):
            conversation_history_str.append(user_input_str)

    logger.debug(f"Conversation history: {len(conversation_history_str)} messages")

    # Generate ticket title and description using a single LLM call
    logger.debug("Generating ticket title and description")
    combined_prompt = """You are a ticket generator. Your task is to create both a title and description for a support ticket based on:
1. The user's problem description
2. The conversation history

For the title:
- Be 3-10 words long
- Be as short as possible without sacrificing meaning
- Accurately describe the problem
- Use lowercase letters and underscores instead of spaces
- Be suitable for use as a filename

Title examples:
- "broken printer" -> "broken_printer"
- "HSAID configuration issue" -> "hsaid_configuration_issue"
- "ambulance station mapping problem" -> "ambulance_station_mapping"
- "patient registration error" -> "patient_registration_error"

For the description:
- Include all key information from the conversation
- Be concise but comprehensive
- Be written in clear, professional language
- Focus on the problem and relevant details

Return ONLY a valid JSON object with this structure:
{
  "title": "ticket_title_here",
  "description": "comprehensive description here"
}"""

    # Build context for generation
    context_parts = [f"User Problem: {user_problem}"]
    context_parts.append("\nConversation History:")
    context_parts.append("\n".join(conversation_history_str))
    context = "\n".join(context_parts)

    user_prompt = f"""Generate both a title and description for this support ticket:

{context}

Return a JSON object with "title" and "description" fields."""

    try:
        messages = [
            SystemMessage(content=combined_prompt),
            HumanMessage(content=user_prompt)
        ]
        response = llm.invoke(messages)
        response_text = response.content.strip()
        
        # Parse JSON response
        result = parse_json_from_response(response_text)
        
        ticket_title = result.get("title", "").strip()
        summary = result.get("description", "").strip()
        
        # Clean up title: remove quotes, convert to lowercase, replace spaces with underscores
        ticket_title = ticket_title.strip('"\'')
        ticket_title = ticket_title.lower().replace(" ", "_")
        # Remove any special characters except underscores and hyphens
        ticket_title = re.sub(r'[^a-z0-9_-]', '', ticket_title)
        # Ensure it's not empty
        if not ticket_title:
            ticket_title = "support_ticket"
        
        # Ensure description is not empty
        if not summary:
            summary = user_problem
            if conversation_history_str:
                summary += "\n\nConversation History:\n" + "\n".join(conversation_history_str[:5])
        
        logger.info(f"Generated ticket title: {ticket_title}")
        logger.info(f"Generated description: {len(summary)} characters")
    except Exception as e:
        # Fallback values
        logger.error(f"Title and description generation failed: {e}", exc_info=True)
        ticket_title = "support_ticket"
        summary = user_problem
        if conversation_history_str:
            summary += "\n\nConversation History:\n" + "\n".join(conversation_history_str[:5])

    # Generate UUID
    ticket_uuid = str(uuid.uuid4())
    logger.info(f"Generated ticket UUID: {ticket_uuid}")

    # Default values for ticket creation
    assigned_queue = "Technical Support"
    category = "General Support"
    priority = "High"
    name = "Love Noreus"

    logger.debug(f"Ticket details - Queue: {assigned_queue}, Category: {category}, Priority: {priority}")

    # Format conversation history for Jira description
    conversation_history_formatted = "\n".join(conversation_history_str)

    # Format description according to requirements
    ticket_title_display = ticket_title.replace("_", " ")
    description = f"""Summary:

{summary}

Assigned Queue:

{assigned_queue}

Priority:

{priority}

Name:

{name}

Category:

{category}

Conversation Topic:

{ticket_title_display}

Conversation History: 

{conversation_history_formatted}"""

    # Build ticket data
    ticket_data = {
        "ticket_title": ticket_title,
        "ticket_uuid": ticket_uuid,
        "description": user_problem,
        "category": category,
        "assigned_queue": assigned_queue,
        "priority": priority,
        "conversation_history": conversation_history_str
    }

    # Save ticket to file
    tickets_dir = Path(__file__).parent / "tickets"
    tickets_dir.mkdir(exist_ok=True)

    filename = f"{ticket_title}_{ticket_uuid}.json"
    filepath = tickets_dir / filename

    logger.info(f"Saving ticket to: {filepath}")

    # Create Jira ticket
    jira_result = None
    try:
        logger.info("Creating Jira ticket")
        jira_result = create_jira_ticket(
            conversation_topic=ticket_title_display,
            description=description,
            queue=assigned_queue,
            priority=priority,
            name=name,
            category=category
        )

        if jira_result.get("success"):
            logger.info(f"Jira ticket created successfully: {jira_result.get('key', 'Unknown')}")
        else:
            logger.error(f"Jira ticket creation failed: {jira_result.get('error', 'Unknown error')}")
    except Exception as e:
        logger.error(f"Error creating Jira ticket: {e}", exc_info=True)
        jira_result = {"success": False, "error": str(e)}

    # Save JSON ticket to file
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(ticket_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Ticket saved successfully to: {filepath}")

        # Generate bot response
        jira_info = ""
        if jira_result and jira_result.get("success"):
            jira_key = jira_result.get("key", "Unknown")
            jira_info = f"\nJira Ticket: {jira_key}"
        elif jira_result and not jira_result.get("success"):
            jira_info = f"\nJira Ticket: Failed to create ({jira_result.get('error', 'Unknown error')})"

        bot_response = f"Ticket created successfully! Ticket ID: {ticket_uuid}\nTitle: {ticket_title_display.title()}\nSaved to: {filename}{jira_info}"

    except Exception as e:
        logger.error(f"Error saving ticket: {e}", exc_info=True)
        bot_response = f"Error creating ticket: {str(e)}"

    # Reset state to initial state after ticket creation
    logger.info("Resetting conversation state after ticket creation")
    updated_state = get_initial_state()
    updated_state["bot_response"] = bot_response

    logger.info("Ticket creation completed successfully")
    return updated_state


def route_after_intent(state: AgentState) -> str:
    """Route to appropriate agent based on detected intent"""
    intent = state.get("intent", {})
    mode = intent.get("mode", "cosmic_search")
    questioning = state.get("questioning", False)
    first_question_run_complete = state.get("first_question_run_complete", False)

    logger.debug(
        f"Routing - Mode: {mode}, Questioning: {questioning}, First run complete: {first_question_run_complete}")

    # If create_ticket mode, only route to create_ticket if questions have been asked
    # Otherwise, route to start_ticket flow
    if mode == "create_ticket":
        if first_question_run_complete:
            logger.info("Routing to: create_ticket")
            return "create_ticket"
        else:
            # User wants to create ticket but questions haven't been asked yet
            # Route to start_ticket flow instead
            logger.info(
                "Routing to: identify_known_question_agent (create_ticket requested but questions not initialized)")
            return "identify_known_question_agent"
    # If questioning is active, always route to questioner_agent
    elif questioning or mode == "questioning":
        logger.info("Routing to: questioner_agent")
        return "questioner_agent"
    elif mode == "cosmic_search":
        logger.info("Routing to: cosmic_search_agent")
        return "cosmic_search_agent"
    elif mode == "start_ticket":
        logger.info("Routing to: identify_known_question_agent")
        return "identify_known_question_agent"
    else:
        logger.info("Routing to: END")
        return END


# ============================================================================
# LANGGRAPH WORKFLOW SETUP
# ============================================================================

logger.info("=" * 80)
logger.info("BUILDING LANGGRAPH WORKFLOW")
logger.info("=" * 80)

# Create the graph
workflow = StateGraph(AgentState)
workflow.add_node("detect_intent", detect_intent)
workflow.add_node("cosmic_search_agent", cosmic_search_agent)
workflow.add_node("identify_known_question_agent", identify_known_question_agent)
workflow.add_node("questioner_agent", questioner_agent)
workflow.add_node("create_ticket", create_ticket)
workflow.add_node("create_ticket_without_known_question", create_ticket_without_known_question)
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
    lambda state: (
        "questioner_agent" if state.get("questioning", False) 
        else "create_ticket_without_known_question" if (
            not state.get("known_problem_identified", True) and 
            (state.get("last_cosmic_query") or state.get("user_input"))
        )
        else END
    ),
    {
        "questioner_agent": "questioner_agent",
        "create_ticket_without_known_question": "create_ticket_without_known_question",
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
workflow.add_edge("create_ticket_without_known_question", END)

# Compile the graph
app = workflow.compile()

logger.info("Workflow compiled successfully")
logger.info("Nodes: detect_intent, cosmic_search_agent, identify_known_question_agent, questioner_agent, create_ticket, create_ticket_without_known_question")
logger.info("=" * 80)

# Global state storage for multiple conversations (keyed by chat_id)
_conversation_states: Dict[str, dict] = {}


def get_initial_state() -> dict:
    """Return the initial state dictionary as if the application just started"""
    return {
        "conversation_history": [],
        "last_cosmic_query": None,
        "last_cosmic_query_response": None,
        "bot_response": None,
        "known_problem_identified": False,
        "questioning": False,
        "questions": [],
        "first_question_run_complete": False
    }


def process_message(chat_id: str, user_message: str) -> dict:
    """
    Process a single user message and return bot response.
    This function can be imported and used by external libraries (e.g., MCP server).

    Args:
        chat_id: Unique identifier for the conversation thread
        user_message: The user's message to process

    Returns:
        dict: Response containing:
            - success: bool
            - message: str (bot response)
            - ticket_mode: bool (whether in ticket creation mode)
            - ticket_ready: bool (whether ticket is ready to be created)
            - questions: list (current questions if in ticket mode)
            - answered_questions: list (answered questions if in ticket mode)
    """
    logger.info("=" * 80)
    logger.info(f"PROCESSING MESSAGE - Chat ID: {chat_id}")
    logger.info("=" * 80)

    # Initialize or get conversation state
    if chat_id not in _conversation_states:
        logger.info(f"Initializing new conversation state for chat_id: {chat_id}")
        _conversation_states[chat_id] = get_initial_state()
    else:
        logger.debug(f"Using existing conversation state for chat_id: {chat_id}")

    global_state = _conversation_states[chat_id]

    if not user_message or not user_message.strip():
        logger.warning("Empty user message received")
        return {
            "success": False,
            "message": "Please provide a message to process.",
            "ticket_mode": global_state.get("questioning", False),
            "ticket_ready": False
        }

    user_input = user_message.strip()
    logger.info(f"User message: {user_input[:100]}...")

    try:
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

        logger.debug(
            f"Current state - Questioning: {current_state['questioning']}, Questions: {len(current_state['questions'])}")

        # Run the graph
        logger.info("Invoking LangGraph workflow")
        start_time = time.time()
        result = app.invoke(current_state)
        elapsed_time = time.time() - start_time
        logger.info(f"Workflow completed in {elapsed_time:.2f}s")

        # Update global state with the result
        _conversation_states[chat_id].update({
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
            _conversation_states[chat_id]["matched_template"] = result["matched_template"]

        logger.debug("Global state updated successfully")

        # Get bot response
        bot_response = result.get("bot_response")
        if not bot_response:
            # Fallback if no agent set a response
            intent = result.get("intent", {})
            bot_response = json.dumps(intent)
            logger.warning("No bot_response found, using intent as fallback")

        # Determine ticket mode and status
        questioning = result.get("questioning", False)
        questions = result.get("questions", [])
        ticket_ready = False

        if questions:
            all_answered = all(q.get("answered", False) for q in questions)
            ticket_ready = all_answered and result.get("first_question_run_complete", False)
            logger.debug(f"Ticket status - All answered: {all_answered}, Ready: {ticket_ready}")

        # Build response
        response = {
            "success": True,
            "message": bot_response,
            "ticket_mode": questioning,
            "ticket_ready": ticket_ready
        }

        # Add question information if in ticket mode
        if questioning and questions:
            answered_questions = [q for q in questions if q.get("answered", False)]
            unanswered_questions = [q for q in questions if not q.get("answered", False)]

            response["questions"] = questions
            response["answered_questions"] = answered_questions
            response["unanswered_questions"] = unanswered_questions

            logger.debug(
                f"Questions - Total: {len(questions)}, Answered: {len(answered_questions)}, Unanswered: {len(unanswered_questions)}")

        logger.info(f"Message processed successfully - Ticket mode: {questioning}, Ticket ready: {ticket_ready}")
        return response

    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"An error occurred: {str(e)}",
            "ticket_mode": False,
            "ticket_ready": False,
            "error": str(e)
        }


def reset_conversation(chat_id: str) -> bool:
    """
    Reset a conversation state to initial state.

    Args:
        chat_id: Unique identifier for the conversation thread

    Returns:
        bool: True if conversation was reset, False if conversation didn't exist
    """
    logger.info(f"Resetting conversation for chat_id: {chat_id}")

    if chat_id in _conversation_states:
        _conversation_states[chat_id] = get_initial_state()
        logger.info(f"Conversation {chat_id} reset successfully")
        return True

    logger.warning(f"Conversation {chat_id} not found")
    return False


def main():
    """Main loop for terminal interaction"""
    logger.info("=" * 80)
    logger.info("STARTING TERMINAL INTERACTION MODE")
    logger.info("=" * 80)
    print("Intent Detection Bot - Type 'exit' to quit\n")

    # Use a default chat_id for terminal mode
    terminal_chat_id = "terminal_session"
    logger.info(f"Terminal chat_id: {terminal_chat_id}")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ['exit', 'quit', 'q']:
                logger.info("User requested exit")
                print("Goodbye!")
                break

            if not user_input:
                continue

            # Start timing
            start_time = time.time()

            # Process message using the process_message function
            result = process_message(terminal_chat_id, user_input)

            # Calculate response time
            response_time = time.time() - start_time

            # Display bot response
            if result.get("success"):
                bot_response = result.get("message", "")
                print(f"\nBot: {bot_response} ({response_time:.3f}s)\n")
                logger.debug(f"Response time: {response_time:.3f}s")
            else:
                error_msg = result.get("message", "Unknown error")
                print(f"\nError: {error_msg} ({response_time:.3f}s)\n")
                logger.error(f"Error displayed to user: {error_msg}")

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()