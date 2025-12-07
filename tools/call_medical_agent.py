"""
Tool for calling the medical agent to answer medical information queries
"""
from smolagents import tool


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

