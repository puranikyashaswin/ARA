"""
Code execution tool using E2B sandbox.
"""
from typing import Annotated
from langchain_core.tools import tool
from e2b_code_interpreter import Sandbox


@tool
def execute_python(
    code: Annotated[str, "The Python code to execute in a sandboxed environment."]
) -> str:
    """
    Execute Python code in a secure E2B sandbox.
    Use this for calculations, data analysis, plotting, or any computation.
    Returns stdout, stderr, and any generated artifacts (like plots).
    """
    try:
        sandbox = Sandbox.create()
        try:
            execution = sandbox.run_code(code)
            
            results = []
            
            # Collect stdout (may be a list)
            if execution.logs.stdout:
                stdout = ''.join(execution.logs.stdout) if isinstance(execution.logs.stdout, list) else execution.logs.stdout
                results.append(f"Output:\n{stdout}")
            
            # Collect stderr (may be a list)
            if execution.logs.stderr:
                stderr = ''.join(execution.logs.stderr) if isinstance(execution.logs.stderr, list) else execution.logs.stderr
                results.append(f"Stderr:\n{stderr}")
            
            # Collect results (return values, plots, etc.)
            if execution.results:
                for result in execution.results:
                    if result.is_main_result:
                        if hasattr(result, 'text') and result.text:
                            results.append(f"Result: {result.text}")
                        if hasattr(result, 'png') and result.png:
                            results.append("[Generated a plot/image]")
            
            # Handle errors
            if execution.error:
                results.append(f"Error: {execution.error.name}: {execution.error.value}")
            
            return "\n".join(results) if results else "Code executed successfully with no output."
        finally:
            sandbox.kill()
    
    except Exception as e:
        return f"Execution failed: {str(e)}"
