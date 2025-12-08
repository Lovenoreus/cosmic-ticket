"""
Agents for ticket creation functionality
"""
import os
import json
from smolagents import ToolCallingAgent
from smolagents.models import OpenAIServerModel
from openai import OpenAI


# Load prompts
def load_prompt(prompt_name: str) -> str:
    """Load a prompt from the prompts directory"""
    prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", f"{prompt_name}_prompt.txt")
    with open(prompt_path, "r", encoding="utf8") as f:
        return f.read()


class IdentifyKnownProblemAgent:
    """Agent specialized in identifying known problem categories"""
    
    def __init__(self, model):
        self.model = model
        self.prompt_template = load_prompt("identify_known_problem")
        # Get API key from model if available, or from environment
        try:
            if hasattr(model, 'api_key') and model.api_key:
                self.api_key = model.api_key
            else:
                self.api_key = os.environ.get("OPENAI_API_KEY")
        except:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        self.openai_client = OpenAI(api_key=self.api_key) if self.api_key else None
    
    def identify(self, user_message: str, known_questions: list) -> dict:
        """
        Identify the problem category from user message.
        
        Args:
            user_message: The user's message describing the problem
            known_questions: List of known question categories from known_questions.json
            
        Returns:
            Dictionary with selected_category, confidence, and reasoning
        """
        print("\n[IdentifyKnownProblemAgent] Analyzing user message to identify problem category...")
        known_questions_text = json.dumps(known_questions, indent=2)
        prompt = self.prompt_template.format(known_questions=known_questions_text)
        prompt += f"\n\nUser message: {user_message}"
        
        # Use OpenAI client directly for JSON response format
        if self.openai_client:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message}
            ]
            
            model_id = "gpt-4o-mini"
            if hasattr(self.model, 'model_id'):
                model_id = self.model.model_id
            elif hasattr(self.model, '_model_id'):
                model_id = self.model._model_id
            
            response = self.openai_client.chat.completions.create(
                model=model_id,
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            print(f"[IdentifyKnownProblemAgent] Result: {result.get('selected_category', 'None')} (confidence: {result.get('confidence', 'N/A')})")
        else:
            # Fallback to model agent
            agent = ToolCallingAgent(tools=[], model=self.model, verbosity_level=1)
            response = agent.run(prompt)
            response_str = str(response)
            try:
                if "{" in response_str and "}" in response_str:
                    start = response_str.find("{")
                    end = response_str.rfind("}") + 1
                    json_str = response_str[start:end]
                    result = json.loads(json_str)
                else:
                    result = json.loads(response_str)
            except:
                result = {
                    "selected_category": None,
                    "confidence": "low",
                    "reasoning": "Could not parse response"
                }
        
        return result


class UpdateTicketAgent:
    """Agent specialized in updating ticket information"""
    
    def __init__(self, model):
        self.model = model
        self.prompt_template = load_prompt("update_ticket")
        # Get API key from model if available, or from environment
        try:
            if hasattr(model, 'api_key') and model.api_key:
                self.api_key = model.api_key
            else:
                self.api_key = os.environ.get("OPENAI_API_KEY")
        except:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        self.openai_client = OpenAI(api_key=self.api_key) if self.api_key else None
    
    def update(self, user_message: str, current_state: dict, remaining_questions: list, 
               unanswered_indices: list) -> dict:
        """
        Update ticket information from user response.
        
        Args:
            user_message: The user's message
            current_state: Current ticket state
            remaining_questions: List of all questions to ask
            unanswered_indices: List of indices of unanswered questions
            
        Returns:
            Dictionary with state_updates, answered_question_indices, and assistant_reply
        """
        print(f"\n[UpdateTicketAgent] Processing user response...")
        print(f"[UpdateTicketAgent] Unanswered question indices: {unanswered_indices}")
        
        # Format the prompt with current state
        current_state_text = json.dumps(current_state, indent=2)
        remaining_questions_text = json.dumps(remaining_questions, indent=2)
        unanswered_indices_text = json.dumps(unanswered_indices, indent=2)
        
        prompt = self.prompt_template.format(
            current_state=current_state_text,
            remaining_questions=remaining_questions_text,
            unanswered_indices=unanswered_indices_text,
            user_message=user_message
        )
        
        # Use OpenAI client directly for JSON response format
        if self.openai_client:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message}
            ]
            
            model_id = "gpt-4o-mini"
            if hasattr(self.model, 'model_id'):
                model_id = self.model.model_id
            elif hasattr(self.model, '_model_id'):
                model_id = self.model._model_id
            
            response = self.openai_client.chat.completions.create(
                model=model_id,
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            answered = result.get("answered_question_indices", [])
            print(f"[UpdateTicketAgent] Questions answered in this interaction: {answered}")
            if answered:
                remaining = [i for i in unanswered_indices if i not in answered]
                print(f"[UpdateTicketAgent] Remaining unanswered questions: {remaining}")
        else:
            # Fallback to model agent
            agent = ToolCallingAgent(tools=[], model=self.model, verbosity_level=1)
            response = agent.run(prompt)
            response_str = str(response)
            try:
                if "{" in response_str and "}" in response_str:
                    start = response_str.find("{")
                    end = response_str.rfind("}") + 1
                    json_str = response_str[start:end]
                    result = json.loads(json_str)
                else:
                    result = json.loads(response_str)
            except Exception as e:
                result = {
                    "state_updates": {},
                    "answered_question_indices": [],
                    "assistant_reply": "I understand. Could you provide more details?"
                }
        
        return result


class TicketExecutorAgent:
    """Main executor agent that orchestrates ticket creation"""
    
    def __init__(self, model, known_questions: list):
        self.model = model
        self.known_questions = known_questions
        
        # Initialize specialized agents
        self.identify_agent = IdentifyKnownProblemAgent(model)
        self.update_agent = UpdateTicketAgent(model)
        
        # Create tools
        from tools.ticket_creationtion_tools import create_ticket_tool
        create_ticket = create_ticket_tool()
        
        # Create tool wrappers for agent calls
        from smolagents import tool
        
        # Store state reference for tools to access
        self._current_state = None
        
        @tool
        def identify_known_problem(user_message: str) -> str:
            """
            Call this tool to identify the problem category from the user's message.
            Use this when the user first mentions a problem or issue in normal chat mode.
            
            Args:
                user_message: The user's message describing the problem
            """
            print("\n[Tool: identify_known_problem] Called by executor agent")
            result = self.identify_agent.identify(user_message, self.known_questions)
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
            
            result = self.update_agent.update(
                user_message, 
                current_state_dict, 
                remaining_questions_list, 
                unanswered_indices_list
            )
            return json.dumps(result)
        
        # Store tools for access
        self.identify_tool = identify_known_problem
        self.update_tool = update_ticket
        self.create_ticket_tool = create_ticket
        
        # Initialize executor agent with tools
        self.executor = ToolCallingAgent(
            tools=[identify_known_problem, update_ticket, create_ticket],
            model=model,
            verbosity_level=1  # Enable verbose output
        )
        self.executor.verbose = 1
        self.prompt_template = load_prompt("executor")
    
    def process(self, user_message: str, state: dict, ticket_mode: bool = False) -> dict:
        """
        Process user input and return response with updated state.
        
        Args:
            user_message: The user's message
            state: Current state dictionary containing:
                - ticket_state: dict with ticket fields
                - questions: list of questions
                - answered: list of bools indicating which questions are answered
                - conversation_history: list of conversation messages
                - issue: dict with issue information (if in ticket mode)
            ticket_mode: Whether we're in ticket mode
            
        Returns:
            Dictionary with:
                - assistant_reply: str
                - updated_state: dict (updated state)
                - switch_to_ticket_mode: bool (if switching to ticket mode)
                - selected_category: str (if category was identified)
        """
        mode_str = "TICKET MODE" if ticket_mode else "NORMAL CHAT MODE"
        print(f"\n[TicketExecutorAgent] Processing user message in {mode_str}...")
        # Store state reference for tools
        self._current_state = state
        
        # Build prompt with current state
        current_state_text = json.dumps(state.get("ticket_state", {}), indent=2)
        remaining_questions = state.get("questions", [])
        remaining_questions_text = json.dumps(remaining_questions, indent=2)
        
        # Calculate unanswered indices
        answered = state.get("answered", [])
        unanswered_indices = [i for i in range(len(remaining_questions)) if not answered[i]]
        unanswered_indices_text = json.dumps(unanswered_indices, indent=2)
        
        conversation_history = state.get("conversation_history", [])
        conversation_history_text = json.dumps(conversation_history[-10:], indent=2)  # Last 10 messages
        
        prompt = self.prompt_template.format(
            current_state=current_state_text,
            remaining_questions=remaining_questions_text,
            unanswered_indices=unanswered_indices_text,
            conversation_history=conversation_history_text
        )
        
        if ticket_mode:
            prompt += "\n\nMODE: TICKET MODE (ticket_mode = true)"
        else:
            prompt += "\n\nMODE: NORMAL CHAT MODE (ticket_mode = false)"
        
        prompt += f"\n\nUser: {user_message}"
        
        # Run the executor
        response = self.executor.run(prompt)
        assistant_reply = str(response)
        
        # Initialize result with current state
        updated_state = state.copy()
        result = {
            "assistant_reply": assistant_reply,
            "updated_state": updated_state,
            "switch_to_ticket_mode": False,
            "selected_category": None
        }
        
        return result

