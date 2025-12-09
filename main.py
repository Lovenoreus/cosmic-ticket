"""
Simple Smolagents-based Agentic AI Application
Requires: pip install smolagents openai
"""

import os
from smolagents import ToolCallingAgent
from smolagents.models import OpenAIServerModel
from smolagents import tool
from dotenv import load_dotenv
# from tools import create_math_agent_tool, create_medical_agent_tool
# from agents.math_agent import MathAgent
# from agents.medical_agent import MedicalAgent
import time
load_dotenv()


# Initialize the LLM model (GPT-4o-mini for all agents)
llm_model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY")
)

# ===== AGENTS =====
class MedicalAgent:
    """Agent specialized in medical information"""
    
    def __init__(self, model):
        self.agent = ToolCallingAgent(
            tools=[],
            model=model
        )
        self.agent.verbose = 0
    
    def answer(self, query: str) -> str:
        """Answer medical information queries"""
        prompt = f"""You are a medical information assistant. Provide accurate medical 
information based on established medical knowledge. Always remind users to consult 
healthcare professionals for personal medical advice.

Query: {query}"""
        response = self.agent.run(prompt)
        return str(response)


def create_medical_agent_tool(medical_agent):
    """
    Create a tool that calls the medical agent to answer medical queries.
    
    Args:
        medical_agent: An instance of MedicalAgent
        
    Returns:
        A tool decorated function that can be used by the executor agent
    """
    @tool
    def call_medical_agent(query: str) -> str:
        """
        Call this tool when the user asks about medical information, health conditions,
        symptoms, treatments, or any medical-related query.
        
        Args:
            query: The medical information query
        """
        return medical_agent.answer(query)
    
    return call_medical_agent

class MathAgent:
    def __init__(self, model):
        self.model = model

    def solve(self, problem: str) -> str:
        prompt = f"""You are a mathematical expert. Give a short explanation and final answer.

Problem: {problem}"""
        
        # OpenAIServerModel expects messages in the correct format
        messages = [{"role": "user", "content": prompt}]
        response = self.model(messages)
        
        # Extract the text from the response
        return response


def create_math_agent_tool(math_agent):
    """
    Create a tool that calls the math agent to solve mathematical problems.
    
    Args:
        math_agent: An instance of MathAgent
        
    Returns:
        A tool decorated function that can be used by the executor agent
    """
    @tool
    def call_math_agent(problem: str) -> str:
        """
        Call this tool when the user asks to solve a mathematical problem, 
        calculation, equation, or any math-related query.
        
        Args:
            problem: The mathematical problem to solve
        """
        return math_agent.solve(problem)
    
    return call_math_agent




# ===== EXECUTOR NODE =====
class ExecutorNode:
    """Main executor that identifies user intent and routes to appropriate agents"""
    
    def __init__(self, model):
        self.model = model
        self.math_agent = MathAgent(model)
        self.medical_agent = MedicalAgent(model)
        
        # Create tools for the executor to call specialized agents
        call_math_agent = create_math_agent_tool(self.math_agent)
        call_medical_agent = create_medical_agent_tool(self.medical_agent)
        
        # Initialize executor agent with routing tools
        self.executor = ToolCallingAgent(
            tools=[call_math_agent, call_medical_agent],
            model=model
        )
        self.executor.verbose = 0
    
    def process(self, user_input: str) -> str:
        """Process user input and route to appropriate handler (agent or tool)"""

        prompt = f"""You are an intelligent assistant executor that routes user queries 
to specialized agents and tools.

- For MEDICAL queries: use call_medical_agent
- For MATH queries: use call_math_agent
- For SMALL TALK or general conversation: respond directly

Identify the user's intent and route accordingly.

User: {user_input}
"""
        response = self.executor.run(prompt)
        return str(response)


# ===== MAIN APPLICATION =====

def main():
    """Run the agentic AI application"""
    # Initialize the executor node
    executor = ExecutorNode(llm_model)
    
    # Main conversation loop
    while True:
        try:
            user_input = input("\nYou: ").strip()
            start_time=time.time()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nAssistant: Goodbye! Have a great day!")
                break
            
            # Process the user input through executor
            response = executor.process(user_input)
            elapsed_time = time.time() - start_time
            print(f"\nAssistant: {response}({elapsed_time})")
            
        except KeyboardInterrupt:
            print("\n\nAssistant: Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again.")


if __name__ == "__main__":
    main()