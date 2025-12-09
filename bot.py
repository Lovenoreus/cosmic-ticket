import time
import json
import asyncio
from typing import TypedDict, List, Optional
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


class AgentState(TypedDict):
    user_input: str
    intent: dict
    conversation_history: List[BaseMessage]
    last_cosmic_query: Optional[str]
    last_cosmic_query_response: Optional[str]
    bot_response: Optional[str]
    known_problem_identified: bool


def detect_intent(state: AgentState) -> AgentState:
    """Detect user intent from input"""
    user_input = state["user_input"]
    conversation_history = state.get("conversation_history", [])
    
    system_prompt = """You are an intent detection agent. Analyze the user's message and determine their intent based on the conversation history.

Return ONLY a valid JSON object with one of these two structures:
- If the user is mentioning a problem or asking a question: {"mode": "cosmic_search"}
- If the user is talking about creating a support ticket: {"mode": "ticket_creation"}


Examples:
- "My printer is broken" -> {"mode": "cosmic_search"}
- "I need help with login" -> {"mode": "cosmic_search"}
- "How does <something> work?" -> {"mode": "cosmic_search"}
- "How do I do <something>?" -> {"mode": "cosmic_search"}
- "Why can't I<something>?" -> {"mode": "cosmic_search"}
- "I want to create a ticket" -> {"mode": "ticket_creation"}
- "Can you help me file a support request?" -> {"mode": "ticket_creation"}

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
    
    # Return the full modified state
    updated_state = dict(state)
    updated_state["last_cosmic_query"] = last_cosmic_query
    updated_state["last_cosmic_query_response"] = last_cosmic_query_response
    updated_state["bot_response"] = bot_response
    
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

        print(f"Matched index: {matched_index}")
        print(f"Confidence: {confidence}")
        print(f"Reasoning: {match_result.get('reasoning', 'No reasoning provided')}")
        
        # If we have a valid match with reasonable confidence
        if matched_index >= 0 and matched_index < len(KNOWN_QUESTIONS) and confidence >= 0.5:
            matched_template = KNOWN_QUESTIONS[matched_index]
            # Return the full modified state
            updated_state = dict(state)
            updated_state["known_problem_identified"] = True
            updated_state["matched_template"] = matched_template
            updated_state["match_confidence"] = confidence
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


def route_after_intent(state: AgentState) -> str:
    """Route to appropriate agent based on detected intent"""
    intent = state.get("intent", {})
    mode = intent.get("mode", "cosmic_search")
    
    if mode == "cosmic_search":
        return "cosmic_search_agent"
    elif mode == "ticket_creation":
        return "identify_known_question_agent"
    else:
        return END


# Create the graph
workflow = StateGraph(AgentState)
workflow.add_node("detect_intent", detect_intent)
workflow.add_node("cosmic_search_agent", cosmic_search_agent)
workflow.add_node("identify_known_question_agent", identify_known_question_agent)
workflow.set_entry_point("detect_intent")
workflow.add_conditional_edges(
    "detect_intent",
    route_after_intent,
    {
        "cosmic_search_agent": "cosmic_search_agent",
        "identify_known_question_agent": "identify_known_question_agent",
        END: END
    }
)
workflow.add_edge("cosmic_search_agent", END)
workflow.add_edge("identify_known_question_agent", END)

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
        "known_problem_identified": False
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
                "known_problem_identified": global_state.get("known_problem_identified", False)
            }
            
            # Run the graph
            result = app.invoke(current_state)
            
            # Update global state with the result
            global_state.update({
                "conversation_history": result.get("conversation_history", []),
                "last_cosmic_query": result.get("last_cosmic_query"),
                "last_cosmic_query_response": result.get("last_cosmic_query_response"),
                "bot_response": result.get("bot_response"),
                "known_problem_identified": result.get("known_problem_identified", False)
            })
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Display bot response (from cosmic_search_agent or default)
            bot_response = result.get("bot_response")
            if bot_response:
                print(f"Bot: {bot_response} ({response_time:.3f}s)\n")
            else:
                # Fallback if no agent set a response (e.g., ticket_creation mode)
                intent = result.get("intent", {})
                print(f"Bot: {json.dumps(intent)} ({response_time:.3f}s)\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()

