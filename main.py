"""
Agent-based Ticket Creation Application
Uses specialized agents for ticket creation workflow
"""
import os
import json
import time
from smolagents.models import OpenAIServerModel
from dotenv import load_dotenv
from agents.ticket_creation_agents import (
    IdentifyKnownProblemAgent,
    UpdateTicketAgent,
    TicketExecutorAgent
)
from tools.ticket_creationtion_tools import create_ticket_tool

load_dotenv()

MODEL = "gpt-4o-mini"
DEPARTMENT = "General Hospital Operations"
NAME = "Love Noreus"
DEBUG = False

# Initialize the LLM model
llm_model = OpenAIServerModel(
    model_id=MODEL,
    api_key=os.environ.get("OPENAI_API_KEY")
)

# Load known questions
with open("data/known_questions.json", "r", encoding="utf8") as f:
    KNOWN_QUESTIONS = json.load(f)


def print_ticket_summary(state):
    """Print a formatted summary of the ticket information."""
    print("\n" + "="*60)
    print("TICKET SUMMARY")
    print("="*60)
    print(f"Category:           {state.get('category', 'N/A')}")
    print(f"Assigned Queue:     {state.get('assigned_queue', 'N/A')}")
    print(f"Priority:           {state.get('priority', 'N/A')}")
    print(f"Department:         {state.get('department', 'N/A')}")
    print(f"Name:               {state.get('name', 'N/A')}")
    print(f"Location:           {state.get('location', 'N/A')}")
    print(f"Description:        {state.get('description', 'N/A')}")
    print(f"Conversation Topic: {state.get('conversation_topic', 'N/A')}")
    print(f"Ticket Title:       {state.get('ticket_title', 'N/A')}")
    print("="*60 + "\n")


def run_bot():
    """Main bot loop using agent-based architecture"""
    ticket_mode = False
    issue = None
    questions = []
    answered = []
    
    # Initial empty ticket state
    ticket_state = {
        "description": "",
        "location": "",
        "assigned_queue": "",
        "priority": "",
        "department": DEPARTMENT,
        "name": NAME,
        "category": "",
        "conversation_topic": "",
        "ticket_title": ""
    }
    
    history = []
    
    # Initialize agents
    identify_agent = IdentifyKnownProblemAgent(llm_model)
    update_agent = UpdateTicketAgent(llm_model)
    executor_agent = TicketExecutorAgent(llm_model, KNOWN_QUESTIONS)
    create_ticket = create_ticket_tool()
    
    print("Hospital Support Assistant - Ready to help!")
    print("Type 'quit' or 'exit' to end the conversation.\n")
    
    while True:
        try:
            user_msg = input("> ").strip()
            
            if not user_msg:
                continue
            
            if user_msg.lower() in ['quit', 'exit', 'bye']:
                print("\nGoodbye! Have a great day!")
                break
            
            history.append({"role": "user", "content": user_msg})
            
            # ========================================
            # MODE 1: NORMAL CHAT (ticket_mode = false)
            # ========================================
            if not ticket_mode:
                # Check if user mentions a problem - use identify agent
                print("\n[Main] Normal chat mode - Checking if user mentioned a problem...")
                start_time = time.time()
                identify_result = identify_agent.identify(user_msg, KNOWN_QUESTIONS)
                elapsed_time = time.time() - start_time
                
                selected_category = identify_result.get("selected_category")
                
                if selected_category:
                    # Find the matching issue from known_questions.json
                    issue = None
                    for item in KNOWN_QUESTIONS:
                        if item.get("issue_category") == selected_category:
                            issue = item
                            break
                    
                    if issue:
                        # Switch to ticket mode
                        print(f"\n[Main] Problem identified: {selected_category}")
                        print(f"[Main] Switching to ticket mode...")
                        questions = issue["questions_to_ask"]
                        answered = [False] * len(questions)
                        print(f"[Main] Total questions to ask: {len(questions)}")
                        
                        # Fill in ticket base fields
                        ticket_state["assigned_queue"] = issue["queue"]
                        ticket_state["category"] = issue["issue_category"]
                        ticket_state["priority"] = issue["urgency_level"]
                        ticket_state["description"] = user_msg
                        ticket_state["conversation_topic"] = user_msg
                        
                        ticket_mode = True
                        
                        # Check if initial message already contains answers
                        unanswered_indices = [i for i in range(len(questions)) if not answered[i]]
                        if unanswered_indices:
                            initial_update = update_agent.update(
                                user_msg,
                                ticket_state,
                                questions,
                                unanswered_indices
                            )
                            
                            # Mark answered questions
                            initial_answered_indices = initial_update.get("answered_question_indices", [])
                            if isinstance(initial_answered_indices, list):
                                for idx in initial_answered_indices:
                                    if isinstance(idx, int) and 0 <= idx < len(questions):
                                        answered[idx] = True
                            
                            # Update state
                            initial_updates = initial_update.get("state_updates", {})
                            for k, v in initial_updates.items():
                                if isinstance(v, str):
                                    if v.strip():
                                        ticket_state[k] = v
                                elif v:
                                    ticket_state[k] = v
                        
                        # Ask remaining questions
                        unanswered_indices = [i for i in range(len(questions)) if not answered[i]]
                        if unanswered_indices:
                            ask_update = update_agent.update(
                                "Please ask all the required questions. Present them as a numbered list using the original question indices.",
                                ticket_state,
                                questions,
                                unanswered_indices
                            )
                            assistant_reply = ask_update.get("assistant_reply", "I'll help you create a support ticket.")
                        else:
                            assistant_reply = "I understand you're experiencing an issue. I'll help you create a support ticket."
                        
                        print(f"{assistant_reply} [{elapsed_time:.2f}s]\n")
                        history.append({"role": "assistant", "content": assistant_reply})
                        continue
                    else:
                        print(f"I couldn't find the selected category. Could you describe the problem again? [{elapsed_time:.2f}s]\n")
                        continue
                else:
                    # No problem identified - normal conversation
                    # Use executor for general conversation
                    print("\n[Main] No problem identified - Using executor for general conversation")
                    state_dict = {
                        "ticket_state": ticket_state,
                        "questions": questions,
                        "answered": answered,
                        "conversation_history": history
                    }
                    start_time = time.time()
                    result = executor_agent.process(user_msg, state_dict, ticket_mode=False)
                    elapsed_time = time.time() - start_time
                    
                    assistant_reply = result.get("assistant_reply", "I'm here to help. How can I assist you?")
                    print(f"\n{assistant_reply} [{elapsed_time:.2f}s]\n")
                    history.append({"role": "assistant", "content": assistant_reply})
                    continue
            
            # ========================================
            # MODE 2: TICKET MODE
            # ========================================
            else:
                # Check if user wants to create ticket early (before all questions answered)
                user_lower = user_msg.lower().strip()
                wants_to_create_early = (
                    "create ticket" in user_lower or 
                    "just create" in user_lower or 
                    "create it" in user_lower or
                    "proceed" in user_lower or 
                    "submit" in user_lower or
                    user_lower.startswith("create") or 
                    user_lower.startswith("just create") or
                    "go ahead" in user_lower
                )
                
                if wants_to_create_early:
                    print("[Main] User requested to create ticket early - Proceeding to confirmation...")
                    # Generate title if not already present
                    if not ticket_state.get("ticket_title") or ticket_state.get("ticket_title", "").strip() == "":
                        unanswered_indices = [i for i in range(len(questions)) if not answered[i]]
                        title_update = update_agent.update(
                            "Please generate a ticket title for this ticket.",
                            ticket_state,
                            questions,
                            unanswered_indices
                        )
                        title_updates = title_update.get("state_updates", {})
                        if "ticket_title" in title_updates:
                            ticket_state["ticket_title"] = title_updates["ticket_title"]
                    
                    # Show summary and ask for confirmation
                    print_ticket_summary(ticket_state)
                    print("Please review the ticket information above. Is this information accurate?")
                    print("You can make changes by describing what needs to be corrected, or confirm to proceed with ticket creation.\n")
                    
                    # Enter confirmation loop
                    while True:
                        user_msg = input("> ")
                        history.append({"role": "user", "content": user_msg})
                        
                        # Check if user wants to proceed
                        user_lower = user_msg.lower().strip()
                        proceed_keywords = ["yes", "correct", "accurate", "proceed", "create", "confirm", "ok", "okay", "go ahead", "create ticket", "submit", "good", "looks good", "sounds good", "that's good", "that looks good"]
                        is_confirmation = (
                            user_lower in proceed_keywords or
                            any(user_lower.startswith(kw + " ") or user_lower.startswith(kw + ".") or user_lower.startswith(kw + "!") or user_lower.startswith(kw + ",") for kw in proceed_keywords) or
                            "create ticket" in user_lower or
                            "go ahead" in user_lower or
                            "please create" in user_lower or
                            "looks good" in user_lower or
                            "sounds good" in user_lower or
                            user_lower == "ok" or user_lower == "okay" or user_lower == "good"
                        )
                        
                        if is_confirmation:
                            print("[Main] User confirmed - Creating ticket...")
                            ticket_state_json = json.dumps(ticket_state)
                            history_json = json.dumps(history)
                            result = create_ticket(ticket_state_json, history_json)
                            print(f"\n{result}\n")
                            break
                        
                        # User wants to make changes
                        print("[Main] User wants to make changes - Updating ticket...")
                        unanswered_indices = [i for i in range(len(questions)) if not answered[i]]
                        change_update = update_agent.update(
                            user_msg,
                            ticket_state,
                            questions,
                            unanswered_indices
                        )
                        
                        change_updates = change_update.get("state_updates", {})
                        for k, v in change_updates.items():
                            if k in ["description", "location", "conversation_topic", "ticket_title"]:
                                if isinstance(v, str):
                                    if v.strip():
                                        ticket_state[k] = v
                                elif v:
                                    ticket_state[k] = v
                        
                        assistant_reply = change_update.get("assistant_reply", "I've updated the ticket information.")
                        
                        # Filter out any questions from the reply since we're in confirmation mode
                        reply_lines = assistant_reply.split("\n")
                        filtered_lines = []
                        for line in reply_lines:
                            stripped = line.strip()
                            # Skip lines that look like questions (start with number.)
                            if stripped and stripped[0].isdigit() and "." in stripped:
                                continue
                            filtered_lines.append(line)
                        
                        assistant_reply = "\n".join(filtered_lines).strip()
                        if not assistant_reply:
                            assistant_reply = "I've updated the ticket information."
                        
                        print(f"{assistant_reply}\n")
                        history.append({"role": "assistant", "content": assistant_reply})
                        
                        print_ticket_summary(ticket_state)
                        print("Please review the updated ticket information. Is this information accurate?")
                        print("You can make additional changes or confirm to proceed with ticket creation.\n")
                    
                    break
                
                # Use update agent to process user response
                print(f"\n[Main] Ticket mode - Processing user response...")
                unanswered_indices = [i for i in range(len(questions)) if not answered[i]]
                print(f"[Main] Unanswered questions before processing: {unanswered_indices}")
                
                start_time = time.time()
                update_result = update_agent.update(
                    user_msg,
                    ticket_state,
                    questions,
                    unanswered_indices
                )
                elapsed_time = time.time() - start_time
                
                # Update ticket fields
                updates = update_result.get("state_updates", {})
                for k, v in updates.items():
                    # Only update authorized fields
                    if k in ["description", "location", "conversation_topic", "ticket_title"]:
                        if isinstance(v, str):
                            if v.strip():
                                ticket_state[k] = v
                        elif v:
                            ticket_state[k] = v
                
                # Mark answered questions
                answered_indices = update_result.get("answered_question_indices", [])
                print(f"[Main] Questions marked as answered: {answered_indices}")
                if isinstance(answered_indices, list):
                    for idx in answered_indices:
                        # Validate that the index is within the unanswered_indices that were passed
                        if isinstance(idx, int) and 0 <= idx < len(questions):
                            # Only mark as answered if it was in the unanswered list AND not already answered
                            if idx in unanswered_indices and not answered[idx]:
                                answered[idx] = True
                                print(f"[Main] Marked question {idx} as answered")
                            elif idx in unanswered_indices and answered[idx]:
                                print(f"[Main] Question {idx} was already answered - skipping duplicate mark")
                
                remaining_after = [i for i in range(len(questions)) if not answered[i]]
                print(f"[Main] Remaining unanswered questions after update: {remaining_after}")
                
                assistant_reply = update_result.get("assistant_reply", "Thank you for the information.")
                
                # If there are remaining questions, validate and fix the reply to ensure correct question text and indices
                if remaining_after:
                    # Build the correct question list with exact text from questions array
                    correct_questions_text = "\n".join([f"{idx}. {questions[idx]}" for idx in remaining_after])
                    
                    # Check if reply contains questions with correct indices
                    # Extract question lines from reply (lines starting with number.)
                    reply_lines = assistant_reply.split("\n")
                    question_lines_in_reply = [line.strip() for line in reply_lines if line.strip() and line.strip()[0].isdigit() and "." in line.strip()]
                    
                    # Check if all remaining questions are present with correct indices
                    all_correct = True
                    for idx in remaining_after:
                        expected_line = f"{idx}. {questions[idx]}"
                        # Check if this exact line exists in reply
                        if not any(expected_line in line or (line.startswith(f"{idx}.") and questions[idx] in line) for line in question_lines_in_reply):
                            all_correct = False
                            break
                    
                    # If questions are missing or incorrect, regenerate the questions section
                    if not all_correct or len(question_lines_in_reply) != len(remaining_after):
                        # Find where questions start in the reply
                        question_start_idx = None
                        for i, line in enumerate(reply_lines):
                            if line.strip() and line.strip()[0].isdigit() and "." in line.strip():
                                question_start_idx = i
                                break
                        
                        if question_start_idx is not None:
                            # Replace from question start to end
                            new_reply = "\n".join(reply_lines[:question_start_idx])
                            if new_reply.strip() and not new_reply.strip().endswith(":"):
                                new_reply += "\n\n"
                            new_reply += correct_questions_text
                            assistant_reply = new_reply
                        else:
                            # No questions found, add them
                            if assistant_reply.strip():
                                assistant_reply += "\n\n"
                            assistant_reply += f"Here are the remaining questions:\n\n{correct_questions_text}"
                
                print(f"\n{assistant_reply} [{elapsed_time:.2f}s]\n")
                history.append({"role": "assistant", "content": assistant_reply})
                
                # Check if all questions are answered
                if all(answered):
                    print("[Main] All questions answered - Ready for ticket creation")
                    # Generate title if not already present
                    if not ticket_state.get("ticket_title") or ticket_state.get("ticket_title", "").strip() == "":
                        unanswered_indices = []  # All answered
                        title_update = update_agent.update(
                            "Please generate a ticket title for this ticket.",
                            ticket_state,
                            questions,
                            unanswered_indices
                        )
                        title_updates = title_update.get("state_updates", {})
                        if "ticket_title" in title_updates:
                            ticket_state["ticket_title"] = title_updates["ticket_title"]
                    
                    # Show summary and ask for confirmation
                    print_ticket_summary(ticket_state)
                    print("Please review the ticket information above. Is this information accurate?")
                    print("You can make changes by describing what needs to be corrected, or confirm to proceed with ticket creation.\n")
                    
                    # Enter confirmation loop
                    while True:
                        user_msg = input("> ")
                        history.append({"role": "user", "content": user_msg})
                        
                        # Check if user wants to proceed
                        user_lower = user_msg.lower().strip()
                        proceed_keywords = ["yes", "correct", "accurate", "proceed", "create", "confirm", "ok", "okay", "go ahead", "create ticket", "submit", "good", "looks good", "sounds good", "that's good", "that looks good"]
                        is_confirmation = (
                            user_lower in proceed_keywords or
                            any(user_lower.startswith(kw + " ") or user_lower.startswith(kw + ".") or user_lower.startswith(kw + "!") or user_lower.startswith(kw + ",") for kw in proceed_keywords) or
                            "create ticket" in user_lower or
                            "go ahead" in user_lower or
                            "please create" in user_lower or
                            "looks good" in user_lower or
                            "sounds good" in user_lower or
                            user_lower == "ok" or user_lower == "okay" or user_lower == "good"
                        )
                        
                        if is_confirmation:
                            print("[Main] User confirmed - Creating ticket...")
                            # Create ticket
                            ticket_state_json = json.dumps(ticket_state)
                            history_json = json.dumps(history)
                            result = create_ticket(ticket_state_json, history_json)
                            print(f"\n{result}\n")
                            break
                        
                        # User wants to make changes
                        print("[Main] User wants to make changes - Updating ticket...")
                        unanswered_indices = []  # All answered, but user wants changes
                        change_update = update_agent.update(
                            user_msg,
                            ticket_state,
                            questions,
                            unanswered_indices
                        )
                        
                        change_updates = change_update.get("state_updates", {})
                        for k, v in change_updates.items():
                            if k in ["description", "location", "conversation_topic", "ticket_title"]:
                                if isinstance(v, str):
                                    if v.strip():
                                        ticket_state[k] = v
                                        print(f"[Main] Updated {k}: {v}")
                                elif v:
                                    ticket_state[k] = v
                                    print(f"[Main] Updated {k}: {v}")
                        
                        assistant_reply = change_update.get("assistant_reply", "I've updated the ticket information.")
                        
                        # Filter out any questions from the reply since we're in confirmation mode
                        # The user is making changes, not answering questions
                        reply_lines = assistant_reply.split("\n")
                        filtered_lines = []
                        skip_questions = False
                        for line in reply_lines:
                            stripped = line.strip()
                            # Skip lines that look like questions (start with number.)
                            if stripped and stripped[0].isdigit() and "." in stripped:
                                skip_questions = True
                                continue
                            if not skip_questions or (stripped and not (stripped[0].isdigit() and "." in stripped)):
                                filtered_lines.append(line)
                        
                        assistant_reply = "\n".join(filtered_lines).strip()
                        if not assistant_reply:
                            assistant_reply = "I've updated the ticket information."
                        
                        print(f"{assistant_reply}\n")
                        history.append({"role": "assistant", "content": assistant_reply})
                        
                        # Show updated summary
                        print_ticket_summary(ticket_state)
                        print("Please review the updated ticket information. Is this information accurate?")
                        print("You can make additional changes or confirm to proceed with ticket creation.\n")
                    
                    break
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            if DEBUG:
                import traceback
                traceback.print_exc()
            print("Please try again.\n")


if __name__ == "__main__":
    run_bot()
