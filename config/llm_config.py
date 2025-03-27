from dotenv import load_dotenv
import os

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')

# DeepSeek Configuration 
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-reasoner')

# Validate required configurations
if not OPENAI_API_KEY and not DEEPSEEK_API_KEY:
    raise ValueError("At least one LLM API key (OPENAI or DEEPSEEK) must be configured in .env")
