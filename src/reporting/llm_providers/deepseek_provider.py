import requests
from typing import Optional
from .base_provider import BaseLLMProvider
from src.utils.logger import logger

class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek API implementation for LLM generation."""
    
    def __init__(self, api_key: str, model: str = "deepseek-reasoner"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.deepseek.com/v1"

    def generate(self, prompt: str, system_message: Optional[str] = None, **kwargs) -> Optional[str]:
        """Generate text using DeepSeek API."""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                **kwargs
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            return data['choices'][0]['message']['content']
            
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            return None
