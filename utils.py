import json
import re


def parse_json_from_response(response_text: str, default: dict = None) -> dict:
    """
    Parse JSON from LLM response, handling markdown code blocks and fallback extraction.
    
    Args:
        response_text: The raw response text from the LLM
        default: Default dictionary to return if parsing fails (default: {"mode": "cosmic_search"})
    
    Returns:
        Parsed JSON as a dictionary
    """
    if default is None:
        default = {"mode": "cosmic_search"}
    
    response_text = response_text.strip()
    
    # Try to parse JSON from response
    try:
        # Remove markdown code blocks if present
        if "```" in response_text:
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        response_text = response_text.strip()
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Fallback: try to extract JSON from response
        json_match = re.search(r'\{[^}]+\}', response_text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                return default
        else:
            # Default fallback
            return default

