"""
Tools for ticket creation functionality
"""
import os
import json
import uuid
from smolagents import tool


def save_conversation(history, ticket_uuid):
    """Save the entire conversation history to a JSON file."""
    os.makedirs("conversations", exist_ok=True)
    path = f"conversations/conversation_{ticket_uuid}.json"
    with open(path, "w", encoding="utf8") as f:
        json.dump(history, f, indent=2)


def create_ticket_tool():
    """
    Create a tool that saves the ticket to a file.
    
    Returns:
        A tool decorated function that can be used by the executor agent
    """
    @tool
    def create_ticket(ticket_state: str, conversation_history: str = None) -> str:
        """
        Call this tool when the user confirms they want to create the ticket.
        This tool saves the ticket information to a file.
        
        Args:
            ticket_state: JSON string containing the ticket state with fields:
                - description: Description of the issue
                - location: Location of the issue
                - assigned_queue: Queue to assign the ticket to
                - priority: Priority level
                - department: Department name
                - name: Requester name
                - category: Issue category
                - conversation_topic: Brief summary of the issue
                - ticket_title: Title for the ticket
            conversation_history: Optional JSON string containing conversation history
        """
        print("\n[Tool: create_ticket] Called - Saving ticket to file...")
        try:
            # Parse the ticket state
            if isinstance(ticket_state, str):
                state = json.loads(ticket_state)
            else:
                state = ticket_state
            
            # Generate ticket UUID
            tid = str(uuid.uuid4())
            
            # Use LLM-generated title, fallback to default if not provided
            title = state.get("ticket_title", "").strip()
            if not title or title == "N/A":
                title = "support_ticket"
            
            # Create tickets directory if it doesn't exist
            os.makedirs("tickets", exist_ok=True)
            
            # Save ticket
            path = f"tickets/ticket_{title}_{tid}.json"
            with open(path, "w", encoding="utf8") as f:
                json.dump(state, f, indent=2)
            
            # Save conversation history if provided
            if conversation_history:
                if isinstance(conversation_history, str):
                    history = json.loads(conversation_history)
                else:
                    history = conversation_history
                save_conversation(history, tid)
            
            print(f"[Tool: create_ticket] Ticket saved to: {path}")
            return f"Ticket created successfully: {path}"
            
        except Exception as e:
            print(f"[Tool: create_ticket] Error: {str(e)}")
            return f"Error creating ticket: {str(e)}"
    
    return create_ticket

