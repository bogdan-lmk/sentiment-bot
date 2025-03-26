import openai
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = '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'

try:
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt="Test",
        max_tokens=5
    )
    print("API key is valid!")
    print(response)
except Exception as e:
    print(f"API key error: {e}")
