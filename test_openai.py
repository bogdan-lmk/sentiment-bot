# Must precede any llm module imports


import os
from dotenv import load_dotenv
import openai

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("OPENAI_API_KEY not found in .env")
    exit(1)

openai.api_key = api_key

try:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Say 'test successful'"}],
        max_tokens=10
    )
    print("API test successful:", response.choices[0].message.content)
except Exception as e:
    print("API test failed:", str(e))
