import openai
from typing import Optional
from .base_provider import BaseLLMProvider
from src.utils.logger import logger

class OpenAIProvider(BaseLLMProvider):
    """OpenAI API implementation for LLM generation."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        openai.api_key = api_key

    def generate(self, prompt: str, system_message: Optional[str] = None, **kwargs) -> Optional[str]:
        """Generate text using OpenAI API."""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return None
