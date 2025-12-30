"""
Quick test script to verify NVIDIA NIM connection and basic agent functionality.
"""
import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

def test_nvidia_connection():
    """Test basic NVIDIA NIM API connection."""
    print("🔌 Testing NVIDIA NIM Connection...")
    
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("❌ NVIDIA_API_KEY not found in environment!")
        print("   → Create a .env file with your NVIDIA_API_KEY")
        return False
    
    print(f"✅ API Key found: {api_key[:20]}...")
    
    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        
        llm = ChatNVIDIA(
            model="meta/llama-3.1-70b-instruct",
            nvidia_api_key=api_key,
            temperature=0,
            max_tokens=50,
        )
        
        response = llm.invoke("Say 'Hello, ARA!' and nothing else.")
        print(f"✅ Model Response: {response.content}")
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


def test_calculator():
    """Test the calculator tool."""
    print("\n🧮 Testing Calculator Tool...")
    
    try:
        from src.tools.calculator import calculator
        
        result = calculator.invoke("sqrt(144) + 5")
        print(f"✅ sqrt(144) + 5 = {result}")
        return True
    except Exception as e:
        print(f"❌ Calculator test failed: {e}")
        return False


def test_e2b_connection():
    """Test E2B sandbox connection."""
    print("\n🐍 Testing E2B Sandbox...")
    
    api_key = os.getenv("E2B_API_KEY")
    if not api_key:
        print("⚠️  E2B_API_KEY not found - skipping E2B test")
        return None
    
    try:
        from e2b_code_interpreter import Sandbox
        
        sandbox = Sandbox.create()
        try:
            result = sandbox.run_code("print('Hello from E2B sandbox!')")
            if result.logs.stdout:
                # stdout is a list of strings
                output = ''.join(result.logs.stdout) if isinstance(result.logs.stdout, list) else result.logs.stdout
                print(f"✅ E2B Response: {output.strip()}")
                return True
        finally:
            sandbox.kill()
        return False
    except Exception as e:
        print(f"❌ E2B test failed: {e}")
        return False


def main():
    print("=" * 60)
    print("🧠 ARA - System Verification")
    print("=" * 60)
    
    results = {
        "nvidia": test_nvidia_connection(),
        "calculator": test_calculator(),
        "e2b": test_e2b_connection(),
    }
    
    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)
    
    for component, status in results.items():
        if status is True:
            print(f"  ✅ {component.upper()}: OK")
        elif status is False:
            print(f"  ❌ {component.upper()}: FAILED")
        else:
            print(f"  ⚠️  {component.upper()}: SKIPPED")
    
    if all(v is not False for v in results.values()):
        print("\n🎉 All systems operational!")
    else:
        print("\n⚠️  Some components need attention.")


if __name__ == "__main__":
    main()
