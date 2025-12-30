"""
Calculator tool for mathematical operations.
"""
from typing import Annotated
from langchain_core.tools import tool
import math
import re


@tool
def calculator(
    expression: Annotated[str, "A mathematical expression to evaluate, e.g., '2 + 2' or 'sqrt(16) * 3'."]
) -> str:
    """
    Evaluate a mathematical expression.
    Supports basic arithmetic (+, -, *, /, **, %), functions (sqrt, sin, cos, tan, log, exp, abs),
    and constants (pi, e). Use this for any calculations.
    """
    try:
        # Allowed names for safe evaluation
        allowed_names = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            # Math functions
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "log2": math.log2,
            "exp": math.exp,
            "floor": math.floor,
            "ceil": math.ceil,
            "factorial": math.factorial,
            # Constants
            "pi": math.pi,
            "e": math.e,
        }
        
        # Sanitize: allow basic math characters and comparison operators
        sanitized = re.sub(r'[^0-9+\-*/.()%,\s\w><=!]', '', expression)
        
        # Evaluate safely
        result = eval(sanitized, {"__builtins__": {}}, allowed_names)
        
        # Format result nicely
        if isinstance(result, float):
            # Avoid floating point display issues
            if result == int(result):
                return str(int(result))
            return f"{result:.10g}"
        
        return str(result)
    
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Calculation error: {str(e)}"
