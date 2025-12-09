"""
Minimal LangGraph Multi-Agent System
Requires: pip install langgraph langchain-openai python-dotenv
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY"))

# ===== STATE =====
class AgentState(TypedDict):
    messages: list[str]
    next_node: str
    user_input: str

# ===== NODES =====
def executor_node(state: AgentState) -> AgentState:
    """Routes to specialized nodes or handles general conversation"""
    user_input = state["user_input"]
    
    # Determine intent
    intent_prompt = f"""Analyze this user input and classify it as one of: MATH, MEDICAL, or GENERAL.
Reply with ONLY one word: MATH, MEDICAL, or GENERAL.

User input: {user_input}"""
    
    intent = llm.invoke(intent_prompt).content.strip().upper()
    
    if "MATH" in intent:
        state["next_node"] = "math"
    elif "MEDICAL" in intent:
        state["next_node"] = "medical"
    else:
        # Handle general conversation
        response = llm.invoke(f"You are a friendly assistant. Respond naturally to: {user_input}").content
        state["messages"].append(response)
        state["next_node"] = "end"
    
    return state

def math_node(state: AgentState) -> AgentState:
    """Handles mathematical queries"""
    user_input = state["user_input"]
    
    prompt = f"""You are a math expert. Solve this problem. no explanation needed. just the answer as brielfy as possible:

{user_input}"""
    
    response = llm.invoke(prompt).content
    state["messages"].append(response)
    state["next_node"] = "end"
    return state

def medical_node(state: AgentState) -> AgentState:
    """Handles medical queries"""
    user_input = state["user_input"]
    
    prompt = f"""You are a medical information assistant. Provide accurate information but remind users to consult healthcare professionals.

{user_input}"""
    
    response = llm.invoke(prompt).content
    state["messages"].append(response)
    state["next_node"] = "end"
    return state

# ===== ROUTING =====
def route_after_executor(state: AgentState) -> Literal["math", "medical", "end"]:
    """Route based on executor's decision"""
    return state["next_node"]

# ===== BUILD GRAPH =====
def build_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("executor", executor_node)
    workflow.add_node("math", math_node)
    workflow.add_node("medical", medical_node)
    
    # Set entry point
    workflow.set_entry_point("executor")
    
    # Add conditional routing from executor
    workflow.add_conditional_edges(
        "executor",
        route_after_executor,
        {
            "math": "math",
            "medical": "medical",
            "end": END
        }
    )
    
    # Both specialized nodes end after execution
    workflow.add_edge("math", END)
    workflow.add_edge("medical", END)
    
    return workflow.compile()

# ===== MAIN =====
def main():
    app = build_graph()
    
    print("Assistant: Hello! I can help with math problems, medical information, or general conversation.")
    print("Type 'quit' to exit.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nAssistant: Goodbye!")
                break
            
            # Run the graph
            state = {
                "messages": [],
                "next_node": "",
                "user_input": user_input
            }
            
            result = app.invoke(state)
            
            # Print response
            if result["messages"]:
                print(f"\nAssistant: {result['messages'][-1]}\n")
            
        except KeyboardInterrupt:
            print("\n\nAssistant: Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again.\n")

if __name__ == "__main__":
    main()