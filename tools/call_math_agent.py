"""
Tool for calling the math agent to solve mathematical problems
"""
from smolagents import tool


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

