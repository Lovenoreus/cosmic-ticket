"""
Helper functions for creating executor tools
"""
import json
from smolagents import tool
from tools.ticket_creationtion_tools import create_ticket_tool


def create_executor_tools(identify_agent, update_agent, cosmic_agent, known_questions):
    """
    Create tools for the executor agent.
    
    Args:
        identify_agent: IdentifyKnownProblemAgent instance
        update_agent: UpdateTicketAgent instance
        cosmic_agent: AnswerCosmicQuestionsAgent instance
        known_questions: List of known question categories
        
    Returns:
        List of tool functions
    """
    create_ticket = create_ticket_tool()
    
    @tool
    def answer_cosmic_questions(user_question: str) -> str:
        """
        Call this tool to answer user questions using the cosmic database.
        Use this when the user asks a question or describes an issue that might be answered
        from the cosmic documentation database.
        
        Args:
            user_question: The user's question or issue description
        """
        print("\n[Tool: answer_cosmic_questions] Called by executor agent")
        result = cosmic_agent.answer(user_question)
        
        # Format the response for the executor
        if result.get("success"):
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            response_text = answer
            if sources:
                response_text += f"\n\nSources: {', '.join(sources)}"
            return response_text
        else:
            return result.get("answer", "I couldn't find relevant information.")
    
    @tool
    def identify_known_problem(user_message: str) -> str:
        """
        Call this tool to identify the problem category from the user's message.
        Use this when the user explicitly wants to create a ticket.
        
        IMPORTANT: If the current user message is just a ticket creation request (like "create a ticket for this problem"),
        you must pass the ACTUAL problem description from the conversation history, not the ticket creation request message.
        Look at previous user messages in the conversation history to find the original problem description.
        
        Args:
            user_message: The actual problem description from the user. If the user says "create a ticket for this problem",
                         extract the problem description from previous messages in the conversation history.
        """
        print("\n[Tool: identify_known_problem] Called by executor agent")
        result = identify_agent.identify(user_message, known_questions)
        return json.dumps(result)
    
    @tool
    def update_ticket(user_message: str, current_state: str, remaining_questions: str, 
                     unanswered_indices: str) -> str:
        """
        Call this tool to update ticket information from user response.
        Use this when the user provides information to fill out the ticket in ticket mode.
        
        Args:
            user_message: The user's message
            current_state: JSON string of current ticket state
            remaining_questions: JSON string of all questions to ask
            unanswered_indices: JSON string of indices of unanswered questions
        """
        print("\n[Tool: update_ticket] Called by executor agent")
        current_state_dict = json.loads(current_state) if isinstance(current_state, str) else current_state
        remaining_questions_list = json.loads(remaining_questions) if isinstance(remaining_questions, str) else remaining_questions
        unanswered_indices_list = json.loads(unanswered_indices) if isinstance(unanswered_indices, str) else unanswered_indices
        
        result = update_agent.update(
            user_message, 
            current_state_dict, 
            remaining_questions_list, 
            unanswered_indices_list
        )
        return json.dumps(result)
    
    return [answer_cosmic_questions, identify_known_problem, update_ticket, create_ticket]

