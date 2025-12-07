"""
Simple Smolagents-based Agentic AI Application
Requires: pip install smolagents openai
"""

import os
from smolagents import ToolCallingAgent
from smolagents.models import OpenAIServerModel
from dotenv import load_dotenv
from tools import create_math_agent_tool, create_medical_agent_tool

load_dotenv()


# Initialize the LLM model (GPT-4o-mini for all agents)
llm_model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY")
)


# ===== SPECIALIZED AGENT NODES =====

class MathAgent:
    """Agent specialized in solving mathematical problems"""
    
    def __init__(self, model):
        self.agent = ToolCallingAgent(
            tools=[],
            model=model, 
            verbosity_level=0
        )
        self.agent.verbose = 0
    
    def solve(self, problem: str) -> str:
        """Solve a mathematical problem"""
        prompt = f"""You are a mathematical expert. Solve this math problem step by step.
Show your work clearly and provide the final answer.

Problem: {problem}"""
        response = self.agent.run(prompt)
        return str(response)


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
        """Process user input and route to appropriate handler"""
        prompt = f"""You are an intelligent assistant executor that routes user queries 
to specialized agents.

- For MATH problems (calculations, equations, word problems): Use call_math_agent
- For MEDICAL queries (health info, conditions, symptoms): Use call_medical_agent
- For SMALL TALK or general conversation: Respond directly in a friendly manner

Identify the user's intent and either handle it yourself (for small talk) or 
route to the appropriate specialized agent.

User: {user_input}"""
        response = self.executor.run(prompt)
        return str(response)


# ===== MAIN APPLICATION =====

def main():
    """Run the agentic AI application"""
    
    print("=" * 60)
    print("Smolagents Agentic AI Application")
    print("=" * 60)
    print("I can help you with:")
    print("  • Mathematical problems")
    print("  • Medical information")
    print("  • General conversation")
    print("\nType 'quit' or 'exit' to end the conversation.")
    print("=" * 60)
    
    # Initialize the executor node
    executor = ExecutorNode(llm_model)
    
    # Main conversation loop
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nAssistant: Goodbye! Have a great day!")
                break
            
            # Process the user input through executor
            response = executor.process(user_input)
            print(f"\nAssistant: {response}")
            
        except KeyboardInterrupt:
            print("\n\nAssistant: Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again.")


if __name__ == "__main__":
    main()