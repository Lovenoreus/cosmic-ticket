import time
import json
from typing import TypedDict, List, Optional
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
import os
from dotenv import load_dotenv
from utils import parse_json_from_response

load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(
    model=os.getenv("AGENT_MODEL_NAME", "gpt-4o-mini"),
    temperature=0
)


class AgentState(TypedDict):
    user_input: str
    intent: dict
    conversation_history: List[BaseMessage]
    last_cosmic_query: Optional[str]
    last_cosmic_query_response: Optional[str]


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
    print("i am doing a search")
    
    # Extract the cosmic query from user input
    user_input = state.get("user_input", "")
    last_cosmic_query = user_input
    
    # TODO: Implement actual RAG search here
    # For now, return a placeholder response
    last_cosmic_query_response = "RAG response placeholder - to be implemented"
    
    # Return the full modified state
    updated_state = dict(state)
    updated_state["last_cosmic_query"] = last_cosmic_query
    updated_state["last_cosmic_query_response"] = last_cosmic_query_response
    
    return updated_state


def route_after_intent(state: AgentState) -> str:
    """Route to appropriate agent based on detected intent"""
    intent = state.get("intent", {})
    mode = intent.get("mode", "cosmic_search")
    
    if mode == "cosmic_search":
        return "cosmic_search_agent"
    else:
        return END


# Create the graph
workflow = StateGraph(AgentState)
workflow.add_node("detect_intent", detect_intent)
workflow.add_node("cosmic_search_agent", cosmic_search_agent)
workflow.set_entry_point("detect_intent")
workflow.add_conditional_edges(
    "detect_intent",
    route_after_intent,
    {
        "cosmic_search_agent": "cosmic_search_agent",
        END: END
    }
)
workflow.add_edge("cosmic_search_agent", END)

# Compile the graph
app = workflow.compile()


def main():
    """Main loop for terminal interaction"""
    print("Intent Detection Bot - Type 'exit' to quit\n")
    
    # Initialize global state
    global_state = {
        "conversation_history": [],
        "last_cosmic_query": None,
        "last_cosmic_query_response": None
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
                "last_cosmic_query_response": global_state.get("last_cosmic_query_response")
            }
            
            # Run the graph
            result = app.invoke(current_state)
            
            # Update global state with the result
            global_state.update({
                "conversation_history": result.get("conversation_history", []),
                "last_cosmic_query": result.get("last_cosmic_query"),
                "last_cosmic_query_response": result.get("last_cosmic_query_response")
            })
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Display result
            intent = result["intent"]
            print(f"Bot: {json.dumps(intent)} ({response_time:.3f}s)\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()

