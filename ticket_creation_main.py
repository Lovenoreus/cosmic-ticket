import os, json, uuid, time, asyncio, sys
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
# from get_known_questions import search_issue  # COMMENTED OUT: Using known_questions.json instead
from pprint import pprint

# Add tools directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

from tools.vector_database_tools import cosmic_database_tool2
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
def call_llm(user_msg, ticket_mode, cosmic_search_mode, state, questions, answered, history, confirmation_mode=False, last_cosmic_query=""):
    # normal chat vs ticket mode vs cosmic search mode
    mode_block = {
        "ticket_mode": ticket_mode,
        "cosmic_search_mode": cosmic_search_mode,
        "conversation_so_far": history,
        "ticket_state": state,
        "remaining_questions": questions,
        "confirmation_mode": confirmation_mode,
        "last_cosmic_query": last_cosmic_query
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
def format_cosmic_results(cosmic_result):
    """Format cosmic search results for display to user"""
    if cosmic_result.get("success"):
        print("go cosmic result successfully")
        message = cosmic_result.get("message", "Search completed.")
        sources = cosmic_result.get("sources", [])
        
        formatted = f"{message}\n"
        if sources:
            formatted += f"\nSources: {', '.join(sources)}"
        return formatted
    else:
        error = cosmic_result.get("error", "Unknown error occurred")
        return f"I couldn't find relevant information in the Cosmic database. {error}"


# ----------------------------------------
def run_bot():
    ticket_mode = False
    cosmic_search_mode = True  # Start in cosmic search mode
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
        "ticket_title": "",
        "last_cosmic_query": ""  # Track last cosmic query
    }

    history = []

    while True:
        user_msg = input("> ")
        history.append({"role": "user", "content": user_msg})

        # ========================================
        # MODE 0: COSMIC SEARCH MODE (cosmic_search_mode = true)
        # ========================================
        if cosmic_search_mode:
            # Quick check for explicit ticket creation language (before cosmic search)
            # Only skip cosmic search if user uses explicit ticket creation phrases
            user_lower = user_msg.lower().strip()
            explicit_ticket_keywords = [
                "create a ticket", "file a ticket", "open a ticket", "report this issue",
                "create ticket", "file ticket", "open ticket", "report issue",
                "i need to report", "i want to create a ticket", "create a support ticket",
                "file a support request", "report this problem", "create ticket for",
                "skapa en biljett", "skapa biljett", "rapportera detta problem"  # Swedish versions
            ]
            is_explicit_ticket_request = any(keyword in user_lower for keyword in explicit_ticket_keywords)
            
            # If NOT an explicit ticket request, always do cosmic search first
            if not is_explicit_ticket_request:
                # Perform cosmic search first
                print("\n[Searching Cosmic database...]")
                try:
                    cosmic_result = asyncio.run(cosmic_database_tool2(user_msg))
                    formatted_response = format_cosmic_results(cosmic_result)
                    print(f"\n{formatted_response}\n")
                    history.append({"role": "assistant", "content": formatted_response})
                    
                    # Save the last cosmic query
                    state["last_cosmic_query"] = user_msg
                    
                    # After showing cosmic search results, stay in cosmic search mode
                    # Only proceed to ticket creation if user explicitly requests it with keywords
                    # Check again for explicit ticket keywords in case user wants to create ticket now
                    if any(keyword in user_lower for keyword in explicit_ticket_keywords):
                        # User explicitly requested ticket creation after cosmic search
                        # Check with LLM to confirm
                        resp = call_llm(
                            user_msg, 
                            ticket_mode=False, 
                            cosmic_search_mode=True, 
                            state=state, 
                            questions=[], 
                            answered=[], 
                            history=history,
                            last_cosmic_query=state.get("last_cosmic_query", "")
                        )
                        if resp.get("switch_to_ticket_mode"):
                            # Fall through to ticket creation logic below - is_explicit_ticket_request is True
                            is_explicit_ticket_request = True
                        else:
                            # Even with keywords, LLM says no - continue in cosmic search mode
                            continue
                    else:
                        # No explicit ticket keywords - just continue in cosmic search mode
                        continue
                        
                except Exception as e:
                    print(f"\nError performing cosmic search: {str(e)}\n")
                    history.append({"role": "assistant", "content": f"Error performing cosmic search: {str(e)}"})
                    continue
            
            # Only reach here if is_explicit_ticket_request is True (initial explicit request)
            # Double-check: must have explicit keywords to proceed
            if not is_explicit_ticket_request:
                # This should not happen - if no explicit keywords, we should have continued above
                # Safety check: continue in cosmic search mode
                continue
            
            # Check ticket creation intent using LLM to confirm explicit request
            resp = call_llm(
                user_msg, 
                ticket_mode=False, 
                cosmic_search_mode=True, 
                state=state, 
                questions=[], 
                answered=[], 
                history=history,
                last_cosmic_query=state.get("last_cosmic_query", "")
            )
            
            # Only proceed if we have explicit ticket keywords AND LLM confirms
            if is_explicit_ticket_request and resp.get("switch_to_ticket_mode"):
                # Use last_cosmic_query if available, otherwise use current message
                problem_description = state.get("last_cosmic_query", user_msg)
                
                if not problem_description or not problem_description.strip():
                    # No previous cosmic query, use current message
                    problem_description = user_msg
                
                # Skip cosmic search and go directly to ticket creation flow
                cosmic_search_mode = False
                ticket_mode = False  # Will be set to True after identifying the problem
                
                # Call LLM to identify the category based on problem_description
                resp = call_llm(
                    f"I want to create a ticket for this issue: {problem_description}",
                    ticket_mode=False,
                    cosmic_search_mode=False,
                    state=state,
                    questions=[],
                    answered=[],
                    history=history,
                    last_cosmic_query=state.get("last_cosmic_query", "")
                )
                
                selected_category = resp.get("selected_category")
                if not selected_category:
                    print("I couldn't identify a technical issue category. Could you describe the problem again?")
                    cosmic_search_mode = True  # Return to cosmic search mode
                    continue
                
                # Find the matching issue from known_questions.json
                issue = None
                for item in KNOWN_QUESTIONS:
                    if item.get("issue_category") == selected_category:
                        issue = item
                        break
                
                if not issue:
                    print("I couldn't find the selected category. Could you describe the problem again?")
                    cosmic_search_mode = True  # Return to cosmic search mode
                    continue
                
                questions = issue["questions_to_ask"]
                answered = [False] * len(questions)
                
                # Fill in ticket base fields
                state["assigned_queue"] = issue["queue"]
                state["category"] = issue["issue_category"]
                state["priority"] = issue["urgency_level"]
                state["description"] = problem_description
                state["conversation_topic"] = problem_description
                
                ticket_mode = True
                
                # Check if the problem_description already contains answers to any questions
                initial_analysis_prompt = f"CRITICAL: Analyze the problem description that prompted ticket creation. The problem is: '{problem_description}'. Review ALL questions in the remaining_questions array and identify which ones have already been answered in this description. Look for: time references (when did it start, how long), location mentions (room, area, floor), symptoms (heating, cooling, sounds, temperature), impact statements, damage descriptions, etc. IMPORTANT: If the user explicitly states that information is unknown, unavailable, not available, or will never be known (e.g., 'unknown', 'unavailable', 'forever unknown', 'cannot be determined'), mark that question as ANSWERED with 'unknown' as the answer. Be thorough - if the user mentioned ANY information that answers a question OR explicitly stated information is unknown, mark it as answered. Return the indices (0-based) of ALL questions that are already answered in answered_question_indices, and extract any relevant information into state_updates."
                initial_resp = call_llm(initial_analysis_prompt, ticket_mode, cosmic_search_mode=False, state=state, questions=questions, answered=answered, history=history, last_cosmic_query=state.get("last_cosmic_query", ""))
                
                # Mark questions that were already answered
                initial_answered_indices = initial_resp.get("answered_question_indices", [])
                if isinstance(initial_answered_indices, list):
                    for idx in initial_answered_indices:
                        if isinstance(idx, int) and 0 <= idx < len(questions):
                            answered[idx] = True
                
                # Update state with any information extracted
                initial_updates = initial_resp.get("state_updates", {})
                for k, v in initial_updates.items():
                    if isinstance(v, str):
                        if v.strip():
                            state[k] = v
                    elif v:
                        state[k] = v
                
                # Immediately ask all remaining unanswered questions
                # Only ask if there are unanswered questions
                if not all(answered):
                    resp = call_llm("Please ask all the required questions.", ticket_mode, cosmic_search_mode=False, state=state, questions=questions, answered=answered, history=history, last_cosmic_query=state.get("last_cosmic_query", ""))
                    response_time = resp.get("_response_time", 0)
                    assistant_reply = resp["assistant_reply"]
                    print(f"{assistant_reply} [{response_time:.2f}s]\n")
                    history.append({"role": "assistant", "content": assistant_reply})
                    continue
                # If all questions are already answered (shouldn't happen, but handle it)
                continue

        # -----------------------------------------
        # CALL LLM in the appropriate mode (for ticket_mode or normal chat)
        # NOTE: This section is only reached if NOT in cosmic_search_mode
        # -----------------------------------------
        # Safety check: if we're in cosmic_search_mode, we should have already handled it above
        if cosmic_search_mode:
            # This should not happen - cosmic_search_mode should be handled in the block above
            # But if we somehow reach here, just continue
            continue
        
        # For normal chat mode: check for explicit ticket keywords BEFORE calling LLM
        # This prevents LLM from incorrectly detecting ticket intent
        user_lower = user_msg.lower().strip()
        explicit_ticket_keywords = [
            "create a ticket", "file a ticket", "open a ticket", "report this issue",
            "create ticket", "file ticket", "open ticket", "report issue",
            "i need to report", "i want to create a ticket", "create a support ticket",
            "file a support request", "report this problem", "create ticket for",
            "skapa en biljett", "skapa biljett", "rapportera detta problem"  # Swedish versions
        ]
        is_explicit_ticket_request = any(keyword in user_lower for keyword in explicit_ticket_keywords)
        
        # Only call LLM if NOT in ticket mode
        # If explicit ticket request, skip normal LLM call and go directly to ticket creation
        if not ticket_mode:
            if is_explicit_ticket_request:
                # User explicitly wants ticket - skip normal LLM call and go to ticket creation
                # Use last_cosmic_query if available, otherwise use current message
                problem_description = state.get("last_cosmic_query", user_msg)
                
                if not problem_description or not problem_description.strip():
                    problem_description = user_msg
                
                # Go directly to ticket creation flow
                resp = call_llm(
                    f"I want to create a ticket for this issue: {problem_description}",
                    ticket_mode=False,
                    cosmic_search_mode=False,
                    state=state,
                    questions=[],
                    answered=[],
                    history=history,
                    last_cosmic_query=state.get("last_cosmic_query", "")
                )
            else:
                # Normal conversation - call LLM for response
                resp = call_llm(user_msg, ticket_mode, cosmic_search_mode, state, questions, answered, history, last_cosmic_query=state.get("last_cosmic_query", ""))
        else:
            # Already in ticket mode - call LLM normally
            resp = call_llm(user_msg, ticket_mode, cosmic_search_mode, state, questions, answered, history, last_cosmic_query=state.get("last_cosmic_query", ""))
        
        if DEBUG: 
            print("\nRESP:")
            pprint(resp)

        # ========================================
        # MODE 1: NORMAL CHAT (ticket_mode = false, cosmic_search_mode = false)
        # ========================================
        # Safety check: must not be in cosmic_search_mode
        if cosmic_search_mode:
            continue
            
        if not ticket_mode and not cosmic_search_mode:
            response_time = resp.get("_response_time", 0)
            assistant_reply = resp["assistant_reply"]
            print(f"{assistant_reply} [{response_time:.2f}s]\n")
            history.append({"role": "assistant", "content": assistant_reply})

            # Switch to ticket mode? Only if explicit ticket request keywords are present
            if is_explicit_ticket_request:
                # Use last_cosmic_query if available, otherwise use current message
                problem_description = state.get("last_cosmic_query", user_msg)
                
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
                # Initialize description and conversation_topic with problem description
                state["description"] = problem_description
                state["conversation_topic"] = problem_description

                ticket_mode = True
                
                # Check if the problem description already contains answers to any questions
                # Analyze the problem description to identify which questions were already answered
                initial_analysis_prompt = f"CRITICAL: Analyze the problem description that prompted ticket creation. The problem is: '{problem_description}'. Review ALL questions in the remaining_questions array and identify which ones have already been answered in this description. Look for: time references (when did it start, how long), location mentions (room, area, floor), symptoms (heating, cooling, sounds, temperature), impact statements, damage descriptions, etc. IMPORTANT: If the user explicitly states that information is unknown, unavailable, not available, or will never be known (e.g., 'unknown', 'unavailable', 'forever unknown', 'cannot be determined'), mark that question as ANSWERED with 'unknown' as the answer. Be thorough - if the user mentioned ANY information that answers a question OR explicitly stated information is unknown, mark it as answered. Return the indices (0-based) of ALL questions that are already answered in answered_question_indices, and extract any relevant information into state_updates."
                initial_resp = call_llm(initial_analysis_prompt, ticket_mode, cosmic_search_mode=False, state=state, questions=questions, answered=answered, history=history, last_cosmic_query=state.get("last_cosmic_query", ""))
                
                # Mark questions that were already answered
                initial_answered_indices = initial_resp.get("answered_question_indices", [])
                if isinstance(initial_answered_indices, list):
                    for idx in initial_answered_indices:
                        if isinstance(idx, int) and 0 <= idx < len(questions):
                            answered[idx] = True
                
                # Also update state with any information extracted
                initial_updates = initial_resp.get("state_updates", {})
                for k, v in initial_updates.items():
                    if isinstance(v, str):
                        if v.strip():
                            state[k] = v
                    elif v:
                        state[k] = v
                
                # Immediately ask all remaining unanswered questions on entering ticket mode
                # Use a prompt to ask all questions
                resp = call_llm("Please ask all the required questions.", ticket_mode, cosmic_search_mode=False, state=state, questions=questions, answered=answered, history=history, last_cosmic_query=state.get("last_cosmic_query", ""))
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
        elif ticket_mode:
            # Only process if we have questions to ask (prevent showing summary immediately after entering ticket mode)
            if not questions or len(questions) == 0:
                print("Setting up ticket... Please wait.\n")
                continue
                
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
                resp = call_llm("Please generate a ticket title for this ticket.", ticket_mode, cosmic_search_mode=False, state=state, questions=questions, answered=answered, history=history, last_cosmic_query=state.get("last_cosmic_query", ""))
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
                resp = call_llm(user_msg, ticket_mode, cosmic_search_mode=False, state=state, questions=questions, answered=answered, history=history, confirmation_mode=True, last_cosmic_query=state.get("last_cosmic_query", ""))
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
                title_resp = call_llm("Please generate a ticket title for this ticket.", ticket_mode, cosmic_search_mode=False, state=state, questions=questions, answered=answered, history=history, last_cosmic_query=state.get("last_cosmic_query", ""))
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
                resp = call_llm(user_msg, ticket_mode, cosmic_search_mode=False, state=state, questions=questions, answered=answered, history=history, confirmation_mode=True, last_cosmic_query=state.get("last_cosmic_query", ""))
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
