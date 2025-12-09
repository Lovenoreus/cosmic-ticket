"""
Hospital Support Ticket System with RAG
Requires: pip install langgraph langchain-openai python-dotenv
"""

import os
import json
import uuid
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional

load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY"))

# Known questions database
KNOWN_QUESTIONS = [
    {
        "issue_category": "Facilities or Fixtures malfunction",
        "queue": "Maintenance Department",
        "questions": [
            "Which specific room or area is affected by this issue?",
            "When did you first notice the problem?",
            "Is anyone's care or work being significantly affected by this issue?",
            "Have you noticed any changes in the issue either worsening or improving since it began?",
            "Are there any visible signs of damage, leaking, or unusual odors?"
        ]
    },
    {
        "issue_category": "Medical equipment malfunction",
        "queue": "Technical Support",
        "questions": [
            "Do you know the exact model and serial number of the equipment?",
            "What specific symptoms or errors is the equipment displaying?",
            "Is any patient care being delayed or compromised due to this issue?",
            "When was the equipment last functioning correctly?",
            "Has the equipment been moved recently or experienced any physical impact?"
        ]
    },
    {
        "issue_category": "IT or device malfunction",
        "queue": "IT Department",
        "questions": [
            "Do you know the device model and asset tag number?",
            "What specific error messages or behaviors are you experiencing?",
            "Were there any system updates, power outages, or changes before the issue started?",
            "Does the issue happen consistently or only under certain conditions?",
            "How is this affecting your ability to perform your duties or provide patient care?"
        ]
    }
]

# ===== STATE =====
class TicketState(TypedDict):
    user_input: str
    conversation_history: list
    intent: str
    rag_response: Optional[str]
    selected_category: Optional[dict]
    questions: list
    ticket_title: Optional[str]
    next_node: str

# ===== MOCK RAG TOOL =====
def qdrant_search_tool(query: str) -> str:
    """Mock RAG search - replace with actual Qdrant implementation"""
    return f"RAG Response for '{query}': The Cosmic Medical Journalling System allows you to create patient records, track medical history, and generate reports. For technical issues, please create a support ticket."

# ===== NODES =====
def intent_classifier(state: TicketState) -> TicketState:
    """Classify user intent: RAG search or ticket creation"""
    user_input = state["user_input"]
    
    prompt = f"""Classify user intent as either RAG_SEARCH or TICKET_CREATION.

RAG_SEARCH: User asks about how the system works, features, or general questions
TICKET_CREATION: User reports a problem, malfunction, or wants to create a support ticket

User input: {user_input}

Reply with ONLY: RAG_SEARCH or TICKET_CREATION"""
    
    intent = llm.invoke(prompt).content.strip().upper()
    
    state["intent"] = "rag_search" if "RAG_SEARCH" in intent else "ticket_creation"
    state["next_node"] = state["intent"]
    state["conversation_history"].append({"role": "user", "content": user_input})
    
    return state

def rag_search_node(state: TicketState) -> TicketState:
    """Execute RAG search and return response"""
    response = qdrant_search_tool(state["user_input"])
    state["rag_response"] = response
    state["conversation_history"].append({"role": "assistant", "content": response})
    state["next_node"] = "end"
    return state

def ticket_init_node(state: TicketState) -> TicketState:
    """Select category and initialize questions"""
    user_input = state["user_input"]
    
    # Build category list for LLM
    categories_text = "\n".join([f"{i+1}. {cat['issue_category']}" for i, cat in enumerate(KNOWN_QUESTIONS)])
    
    prompt = f"""Match the user's problem to the best category.

Categories:
{categories_text}

User problem: {user_input}

Reply with ONLY the category number (1, 2, or 3)"""
    
    category_num = int(llm.invoke(prompt).content.strip()) - 1
    selected_cat = KNOWN_QUESTIONS[category_num]
    
    # Initialize questions with original indexes
    questions = []
    for i, q in enumerate(selected_cat["questions"]):
        questions.append({
            "id": i + 1,
            "question": q,
            "answered": False,
            "answer": None
        })
    
    state["selected_category"] = selected_cat
    state["questions"] = questions
    
    # Present all questions
    questions_text = "\n".join([f"{q['id']}. {q['question']}" for q in questions])
    response = f"I'll help you create a support ticket for: {selected_cat['issue_category']}\n\nPlease answer these questions:\n{questions_text}"
    
    state["conversation_history"].append({"role": "assistant", "content": response})
    state["next_node"] = "question_collector"
    
    return state

def question_collector_node(state: TicketState) -> TicketState:
    """Collect answers, handle out-of-order responses"""
    user_input = state["user_input"]
    state["conversation_history"].append({"role": "user", "content": user_input})
    
    # Get unanswered questions
    unanswered = [q for q in state["questions"] if not q["answered"]]
    
    if not unanswered:
        state["next_node"] = "ticket_creator"
        return state
    
    # Build prompt for answer extraction
    questions_text = "\n".join([f"{q['id']}. {q['question']}" for q in unanswered])
    
    prompt = f"""Extract answers from user response. If user says "I don't know", "not sure", etc., that's a valid answer.

Questions asked:
{questions_text}

User response: {user_input}

Return JSON mapping question IDs to answers. Use null for unanswered questions.
Example: {{"1": "Room 302", "3": "I don't know", "5": null}}

JSON only:"""
    
    try:
        answers_json = llm.invoke(prompt).content.strip()
        # Remove markdown code blocks if present
        answers_json = answers_json.replace("```json", "").replace("```", "").strip()
        answers = json.loads(answers_json)
        
        # Update state with answers
        for q in state["questions"]:
            q_id_str = str(q["id"])
            if q_id_str in answers and answers[q_id_str] is not None:
                q["answered"] = True
                q["answer"] = answers[q_id_str]
    except:
        # If parsing fails, treat as no answers provided
        pass
    
    # Check if user wants to proceed
    proceed_prompt = f"""Does the user want to create the ticket now, even with incomplete information?
Look for phrases like "just create it", "that's all I know", "create the ticket", "I'm done", etc.

User message: {user_input}

Reply with ONLY: YES or NO"""
    
    wants_ticket = "YES" in llm.invoke(proceed_prompt).content.strip().upper()
    
    # Get remaining unanswered questions
    still_unanswered = [q for q in state["questions"] if not q["answered"]]
    
    if wants_ticket or not still_unanswered:
        state["next_node"] = "ticket_creator"
    else:
        # Ask remaining questions with original indexes preserved
        remaining_text = "\n".join([f"{q['id']}. {q['question']}" for q in still_unanswered])
        response = f"Thank you. Please answer the remaining questions:\n{remaining_text}"
        state["conversation_history"].append({"role": "assistant", "content": response})
        state["next_node"] = "question_collector"
    
    return state

def ticket_creator_node(state: TicketState) -> TicketState:
    """Generate ticket title and save to file"""
    
    # Generate ticket title
    answered_q = [q for q in state["questions"] if q["answered"]]
    answers_summary = "\n".join([f"{q['question']}: {q['answer']}" for q in answered_q])
    
    title_prompt = f"""Generate a short ticket title (max 8 words) based on this issue:

Category: {state['selected_category']['issue_category']}
Answers:
{answers_summary}

Title only:"""
    
    title = llm.invoke(title_prompt).content.strip().replace('"', '').replace("'", "")
    state["ticket_title"] = title
    
    # Create ticket data
    ticket_data = {
        "ticket_id": str(uuid.uuid4()),
        "title": title,
        "category": state["selected_category"]["issue_category"],
        "queue": state["selected_category"]["queue"],
        "created_at": datetime.now().isoformat(),
        "questions_and_answers": [
            {
                "question": q["question"],
                "answer": q["answer"] if q["answered"] else "Not answered"
            }
            for q in state["questions"]
        ],
        "conversation_history": state["conversation_history"]
    }
    
    # Save to file
    Path("tickets").mkdir(exist_ok=True)
    filename = f"tickets/{title.replace(' ', '_')}_{ticket_data['ticket_id'][:8]}.json"
    
    with open(filename, 'w') as f:
        json.dump(ticket_data, f, indent=2)
    
    response = f"✓ Ticket created successfully!\n\nTicket ID: {ticket_data['ticket_id']}\nTitle: {title}\nQueue: {state['selected_category']['queue']}\n\nSaved to: {filename}"
    state["conversation_history"].append({"role": "assistant", "content": response})
    state["next_node"] = "end"
    
    return state

# ===== ROUTING =====
def route_from_intent(state: TicketState) -> str:
    return state["next_node"]

def route_from_collector(state: TicketState) -> str:
    return state["next_node"]

# ===== BUILD GRAPH =====
def build_graph():
    workflow = StateGraph(TicketState)
    
    # Add nodes
    workflow.add_node("intent_classifier", intent_classifier)
    workflow.add_node("rag_search", rag_search_node)
    workflow.add_node("ticket_init", ticket_init_node)
    workflow.add_node("question_collector", question_collector_node)
    workflow.add_node("ticket_creator", ticket_creator_node)
    
    # Set entry point
    workflow.set_entry_point("intent_classifier")
    
    # Route from intent classifier
    workflow.add_conditional_edges(
        "intent_classifier",
        route_from_intent,
        {
            "rag_search": "rag_search",
            "ticket_creation": "ticket_init"
        }
    )
    
    # RAG ends
    workflow.add_edge("rag_search", END)
    
    # Ticket init goes to collector
    workflow.add_edge("ticket_init", "question_collector")
    
    # Collector loops or goes to creator
    workflow.add_conditional_edges(
        "question_collector",
        route_from_collector,
        {
            "question_collector": "question_collector",
            "ticket_creator": "ticket_creator"
        }
    )
    
    # Creator ends
    workflow.add_edge("ticket_creator", END)
    
    return workflow.compile()

# ===== MAIN =====
def main():
    app = build_graph()
    
    print("Hospital Support System")
    print("=" * 50)
    print("I can help you with:")
    print("1. Questions about the Cosmic Medical Journalling System")
    print("2. Creating support tickets for issues\n")
    
    # Conversation state persists across turns
    state = {
        "user_input": "",
        "conversation_history": [],
        "intent": "",
        "rag_response": None,
        "selected_category": None,
        "questions": [],
        "ticket_title": None,
        "next_node": ""
    }
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nAssistant: Goodbye!")
                break
            
            # Update state with new input
            state["user_input"] = user_input
            
            # Run graph
            result = app.invoke(state)
            
            # Update persistent state
            state = result
            
            # Print last assistant message
            if result["conversation_history"]:
                last_msg = result["conversation_history"][-1]
                if last_msg["role"] == "assistant":
                    print(f"\nAssistant: {last_msg['content']}")
            
            # Reset for new conversation after ticket creation or RAG response
            if result["next_node"] == "end":
                state = {
                    "user_input": "",
                    "conversation_history": [],
                    "intent": "",
                    "rag_response": None,
                    "selected_category": None,
                    "questions": [],
                    "ticket_title": None,
                    "next_node": ""
                }
            
        except KeyboardInterrupt:
            print("\n\nAssistant: Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    main()