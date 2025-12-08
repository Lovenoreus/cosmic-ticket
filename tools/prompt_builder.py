"""
Helper functions for building executor prompts
"""
import os
import json


def load_prompt(prompt_name: str) -> str:
    """Load a prompt from the prompts directory"""
    prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", f"{prompt_name}_prompt.txt")
    with open(prompt_path, "r", encoding="utf8") as f:
        return f.read()


def build_executor_prompt(state: dict, user_message: str, ticket_mode: bool = False) -> str:
    """
    Build the prompt for the executor agent.
    
    Args:
        state: Current state dictionary containing:
            - ticket_state: dict with ticket fields
            - questions: list of questions
            - answered: list of bools indicating which questions are answered
            - conversation_history: list of conversation messages
        user_message: The user's message
        ticket_mode: Whether we're in ticket mode
        
    Returns:
        Formatted prompt string
    """
    prompt_template = load_prompt("executor")
    
    # Build prompt components
    current_state_text = json.dumps(state.get("ticket_state", {}), indent=2)
    remaining_questions = state.get("questions", [])
    remaining_questions_text = json.dumps(remaining_questions, indent=2)
    
    # Calculate unanswered indices
    answered = state.get("answered", [])
    unanswered_indices = [i for i in range(len(remaining_questions)) if not answered[i]]
    unanswered_indices_text = json.dumps(unanswered_indices, indent=2)
    
    conversation_history = state.get("conversation_history", [])
    conversation_history_text = json.dumps(conversation_history[-10:], indent=2)  # Last 10 messages
    
    # Format the prompt template
    prompt = prompt_template.format(
        current_state=current_state_text,
        remaining_questions=remaining_questions_text,
        unanswered_indices=unanswered_indices_text,
        conversation_history=conversation_history_text
    )
    
    # Add mode information
    if ticket_mode:
        prompt += "\n\nMODE: TICKET MODE (ticket_mode = true)"
    else:
        prompt += "\n\nMODE: NORMAL CHAT MODE (ticket_mode = false)"
    
    # Add user message
    prompt += f"\n\nUser: {user_message}"
    
    return prompt

