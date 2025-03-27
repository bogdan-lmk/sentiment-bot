import os
from dotenv import load_dotenv
from src.reporting.llm_providers.deepseek_provider import DeepSeekProvider

load_dotenv()

def test_deepseek_provider():
    """Test DeepSeek LLM provider functionality"""
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("DEEPSEEK_API_KEY not found in .env")
        return False

    try:
        provider = DeepSeekProvider(api_key, "deepseek-reasoner")
        
        # Simple test prompt
        test_prompt = "Translate this to English: Привіт, як справи?"
        response = provider.generate(test_prompt)
        
        if not response:
            print("DeepSeek provider returned empty response")
            return False
            
        print("DeepSeek test successful. Response:")
        print(response)
        return True
        
    except Exception as e:
        print(f"DeepSeek test failed: {str(e)}")
        return False

if __name__ == "__main__":
    if test_deepseek_provider():
        print("✅ DeepSeek provider works correctly")
    else:
        print("❌ DeepSeek provider test failed")
