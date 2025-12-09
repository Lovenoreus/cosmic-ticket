import os, json, uuid, time
from openai import OpenAI
from dotenv import load_dotenv
# from get_known_questions import search_issue  # COMMENTED OUT: Using known_questions.json instead
from pprint import pprint
load_dotenv()

MODEL = "gpt-4o-mini"
DEPARTMENT = "General Hospital Operations"
NAME = "Love Noreus"
DEBUG = False

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---- LOAD BASE PROMPT ----
with open("conversation_prompt.txt", "r", encoding="utf8") as f:
    BASE_PROMPT = f.read()

# ---- LOAD KNOWN QUESTIONS ----
with open("known_questions.json", "r", encoding="utf8") as f:
    KNOWN_QUESTIONS = json.load(f)


# ----------------------------------------
def call_llm(user_msg, ticket_mode, state, questions, answered, history, confirmation_mode=False):
    # normal chat vs ticket mode
    mode_block = {
        "ticket_mode": ticket_mode,
        "conversation_so_far": history,
        "ticket_state": state,
        "remaining_questions": questions,
        "confirmation_mode": confirmation_mode
    }
    
    # Add unanswered question indices for ticket mode
    if ticket_mode:
        unanswered_indices = [i for i in range(len(questions)) if not answered[i]]
        if DEBUG:
            print(f"UNANSWERED INDICES: {unanswered_indices}")
        mode_block["unanswered_question_indices"] = unanswered_indices
        # Add explicit instruction about preserving original indices with example
        mode_block["question_indexing_note"] = f"CRITICAL: When presenting remaining questions, use the EXACT indices from unanswered_question_indices = {unanswered_indices}. Do NOT renumber sequentially. Example: if unanswered_question_indices = [1, 2, 4], show as '1. [question 1]', '2. [question 2]', '4. [question 4]' - NOT as '1. ...', '2. ...', '3. ...'. The numbers MUST match the array exactly."
        # Also add a mapping for clarity
        if unanswered_indices:
            question_mapping = {idx: questions[idx] for idx in unanswered_indices}
            mode_block["unanswered_questions_mapping"] = question_mapping

    # Append known questions to the prompt
    known_questions_text = json.dumps(KNOWN_QUESTIONS, indent=2)
    full_prompt = BASE_PROMPT + "\n\n=== AVAILABLE ISSUE CATEGORIES ===\n" + known_questions_text

    messages = [
        {"role": "system", "content": full_prompt},
        {"role": "system", "content": json.dumps(mode_block)},
        {"role": "user", "content": user_msg}
    ]

    fmt = {"type": "json_object"}

    # Measure response time
    start_time = time.time()
    r = openai_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format=fmt
    )
    elapsed_time = time.time() - start_time
    
    result = json.loads(r.choices[0].message.content)
    result["_response_time"] = elapsed_time
    return result


# ----------------------------------------
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
    print("="*60 + "\n")


# ----------------------------------------
def save_conversation(history, ticket_uuid):
    """Save the entire conversation history to a JSON file."""
    os.makedirs("conversations", exist_ok=True)
    path = f"conversations/conversation_{ticket_uuid}.json"
    with open(path, "w", encoding="utf8") as f:
        json.dump(history, f, indent=2)


# ----------------------------------------
def save_ticket(state, history=None):
    """Save ticket and optionally save conversation history."""
    os.makedirs("tickets", exist_ok=True)
    tid = str(uuid.uuid4())
    # Use LLM-generated title, fallback to default if not provided
    title = state.get("ticket_title", "").strip()
    if not title or title == "N/A":
        title = "support_ticket"
    path = f"tickets/ticket_{title}_{tid}.json"
    with open(path, "w", encoding="utf8") as f:
        json.dump(state, f, indent=2)
    
    # Save conversation history if provided
    if history:
        save_conversation(history, tid)
    
    print(f"Ticket created: {path}\n")


# ----------------------------------------
def run_bot():
    ticket_mode = False
    issue = None
    questions = []
    answered = []

    # initial empty ticket state
    state = {
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

    while True:
        user_msg = input("> ")
        history.append({"role": "user", "content": user_msg})

        # -----------------------------------------
        # CALL LLM in the appropriate mode
        # -----------------------------------------
        resp = call_llm(user_msg, ticket_mode, state, questions, answered, history)
        if DEBUG: 
            print("\nRESP:")
            pprint(resp)

        # ========================================
        # MODE 1: NORMAL CHAT (ticket_mode = false)
        # ========================================
        if not ticket_mode:
            response_time = resp.get("_response_time", 0)
            assistant_reply = resp["assistant_reply"]
            print(f"{assistant_reply} [{response_time:.2f}s]\n")
            history.append({"role": "assistant", "content": assistant_reply})

            # Switch to ticket mode?
            if resp.get("switch_to_ticket_mode"):
                # LLM should have selected a category - extract it from response
                selected_category = resp.get("selected_category")
                if not selected_category:
                    print("I couldn't identify a technical issue category yet. Could you describe the problem again?")
                    continue

                # Find the matching issue from known_questions.json
                issue = None
                for item in KNOWN_QUESTIONS:
                    if item.get("issue_category") == selected_category:
                        issue = item
                        break

                if not issue:
                    print("I couldn't find the selected category. Could you describe the problem again?")
                    continue

                questions = issue["questions_to_ask"]
                answered = [False] * len(questions)

                # Fill in ticket base fields
                state["assigned_queue"] = issue["queue"]
                state["category"] = issue["issue_category"]
                state["priority"] = issue["urgency_level"]
                # Initialize description and conversation_topic with user's initial problem description
                state["description"] = user_msg
                state["conversation_topic"] = user_msg

                ticket_mode = True
                
                # Check if the initial user message already contains answers to any questions
                # Analyze the user's initial message to identify which questions were already answered
                initial_analysis_prompt = f"CRITICAL: Analyze the user's initial message that prompted ticket creation. The user said: '{user_msg}'. Review ALL questions in the remaining_questions array and identify which ones have already been answered in this message. Look for: time references (when did it start, how long), location mentions (room, area, floor), symptoms (heating, cooling, sounds, temperature), impact statements, damage descriptions, etc. IMPORTANT: If the user explicitly states that information is unknown, unavailable, not available, or will never be known (e.g., 'unknown', 'unavailable', 'forever unknown', 'cannot be determined'), mark that question as ANSWERED with 'unknown' as the answer. Be thorough - if the user mentioned ANY information that answers a question OR explicitly stated information is unknown, mark it as answered. Return the indices (0-based) of ALL questions that are already answered in answered_question_indices, and extract any relevant information into state_updates."
                initial_resp = call_llm(initial_analysis_prompt, ticket_mode, state, questions, answered, history)
                
                # Mark questions that were already answered in the initial message
                initial_answered_indices = initial_resp.get("answered_question_indices", [])
                if isinstance(initial_answered_indices, list):
                    for idx in initial_answered_indices:
                        if isinstance(idx, int) and 0 <= idx < len(questions):
                            answered[idx] = True
                
                # Also update state with any information extracted from initial message
                initial_updates = initial_resp.get("state_updates", {})
                for k, v in initial_updates.items():
                    if isinstance(v, str):
                        if v.strip():
                            state[k] = v
                    elif v:
                        state[k] = v
                
                # Immediately ask all remaining unanswered questions on entering ticket mode
                # Use a prompt to ask all questions
                resp = call_llm("Please ask all the required questions.", ticket_mode, state, questions, answered, history)
                response_time = resp.get("_response_time", 0)
                assistant_reply = resp["assistant_reply"]
                print(f"{assistant_reply} [{response_time:.2f}s]\n")
                history.append({"role": "assistant", "content": assistant_reply})
                continue
            
            # remain in natural conversation mode
            continue

        # ========================================
        # MODE 2: TICKET MODE
        # ========================================
        else:
            response_time = resp.get("_response_time", 0)
            assistant_reply = resp["assistant_reply"]
            print(f"{assistant_reply} [{response_time:.2f}s]\n")
            history.append({"role": "assistant", "content": assistant_reply})

        # update ticket fields
        updates = resp.get("state_updates", {})
        for k, v in updates.items():
            # Only update if value is provided and meaningful
            # For strings: must be non-empty after stripping
            # For other types: must be truthy
            if isinstance(v, str):
                if v.strip():  # Non-empty string
                    state[k] = v
            elif v:  # Non-string truthy value
                state[k] = v

        # mark answered questions (handle multiple answers at once)
        # Note: If user provides corrected info, they may answer the same question again
        answered_indices = resp.get("answered_question_indices", [])
        if isinstance(answered_indices, list):
            for idx in answered_indices:
                if isinstance(idx, int) and 0 <= idx < len(questions):
                    answered[idx] = True  # Mark as answered (even if previously answered, this updates it)
        # Backward compatibility: handle single answered_question if present
        elif resp.get("answered_question") is not None:
            idx = resp.get("answered_question")
            if isinstance(idx, int) and 0 <= idx < len(questions):
                answered[idx] = True

        # check if we are ready to show summary (all questions answered)
        if all(answered):
            # Generate title if not already present
            if not state.get("ticket_title") or state.get("ticket_title", "").strip() == "":
                resp = call_llm("Please generate a ticket title for this ticket.", ticket_mode, state, questions, answered, history)
                updates = resp.get("state_updates", {})
                if "ticket_title" in updates:
                    state["ticket_title"] = updates["ticket_title"]
            
            # Show summary and ask for confirmation
            print_ticket_summary(state)
            print("Please review the ticket information above. Is this information accurate?")
            print("You can make changes by describing what needs to be corrected, or confirm to proceed with ticket creation.\n")
            
            # Enter confirmation loop
            while True:
                user_msg = input("> ")
                history.append({"role": "user", "content": user_msg})
                
                # Check if user wants to proceed - use strict matching to avoid false positives
                user_lower = user_msg.lower().strip()
                proceed_keywords = ["yes", "correct", "accurate", "proceed", "create", "confirm", "ok", "okay", "go ahead", "create ticket", "submit"]
                # Only match if message is exactly a keyword, starts with a keyword followed by punctuation/space, or contains "create ticket"/"go ahead" as phrases
                is_confirmation = (
                    user_lower in proceed_keywords or
                    any(user_lower.startswith(kw + " ") or user_lower.startswith(kw + ".") or user_lower.startswith(kw + "!") for kw in proceed_keywords) or
                    "create ticket" in user_lower or
                    "go ahead" in user_lower or
                    user_lower == "ok" or user_lower == "okay"
                )
                if is_confirmation:
                    save_ticket(state, history)
                    break
                
                # User wants to make changes - process the update
                resp = call_llm(user_msg, ticket_mode, state, questions, answered, history, confirmation_mode=True)
                response_time = resp.get("_response_time", 0)
                assistant_reply = resp["assistant_reply"]
                print(f"{assistant_reply} [{response_time:.2f}s]\n")
                history.append({"role": "assistant", "content": assistant_reply})
                
                # Update state with any changes
                updates = resp.get("state_updates", {})
                for k, v in updates.items():
                    if isinstance(v, str):
                        if v.strip():
                            state[k] = v
                    elif v:
                        state[k] = v
                
                # ALWAYS show updated summary after any changes
                # Do NOT check for proceed keywords here - user must explicitly confirm in a separate message
                # This ensures corrections are processed and user sees updated summary before confirming
                print_ticket_summary(state)
                print("Please review the updated ticket information. Is this information accurate?")
                print("You can make additional changes or confirm to proceed with ticket creation.\n")
            
            break
        
        # If user explicitly requests to create ticket before all questions answered
        if resp.get("ready_to_create_ticket"):
            # Generate title if not already present
            if not state.get("ticket_title") or state.get("ticket_title", "").strip() == "":
                title_resp = call_llm("Please generate a ticket title for this ticket.", ticket_mode, state, questions, answered, history)
                title_updates = title_resp.get("state_updates", {})
                if "ticket_title" in title_updates:
                    state["ticket_title"] = title_updates["ticket_title"]
            
            print_ticket_summary(state)
            print("Please review the ticket information above. Is this information accurate?")
            print("You can make changes by describing what needs to be corrected, or confirm to proceed with ticket creation.\n")
            
            # Enter confirmation loop
            while True:
                user_msg = input("> ")
                history.append({"role": "user", "content": user_msg})
                
                # Check if user wants to proceed - use strict matching to avoid false positives
                user_lower = user_msg.lower().strip()
                proceed_keywords = ["yes", "correct", "accurate", "proceed", "create", "confirm", "ok", "okay", "go ahead", "create ticket", "submit"]
                # Only match if message is exactly a keyword, starts with a keyword followed by punctuation/space, or contains "create ticket"/"go ahead" as phrases
                is_confirmation = (
                    user_lower in proceed_keywords or
                    any(user_lower.startswith(kw + " ") or user_lower.startswith(kw + ".") or user_lower.startswith(kw + "!") for kw in proceed_keywords) or
                    "create ticket" in user_lower or
                    "go ahead" in user_lower or
                    user_lower == "ok" or user_lower == "okay"
                )
                if is_confirmation:
                    save_ticket(state, history)
                    break
                
                # User wants to make changes - process the update
                resp = call_llm(user_msg, ticket_mode, state, questions, answered, history, confirmation_mode=True)
                response_time = resp.get("_response_time", 0)
                assistant_reply = resp["assistant_reply"]
                print(f"{assistant_reply} [{response_time:.2f}s]\n")
                history.append({"role": "assistant", "content": assistant_reply})
                
                # Update state with any changes
                updates = resp.get("state_updates", {})
                for k, v in updates.items():
                    if isinstance(v, str):
                        if v.strip():
                            state[k] = v
                    elif v:
                        state[k] = v
                
                # ALWAYS show updated summary after any changes
                # Do NOT check for proceed keywords here - user must explicitly confirm in a separate message
                # This ensures corrections are processed and user sees updated summary before confirming
                print_ticket_summary(state)
                print("Please review the updated ticket information. Is this information accurate?")
                print("You can make additional changes or confirm to proceed with ticket creation.\n")
            
            break


# ----------------------------------------
if __name__ == "__main__":
    run_bot()
