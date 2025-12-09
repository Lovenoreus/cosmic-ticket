"""
Tools package for the agentic AI application
"""
from .call_math_agent import create_math_agent_tool
from .call_medical_agent import create_medical_agent_tool

__all__ = ['create_math_agent_tool', 'create_medical_agent_tool']

