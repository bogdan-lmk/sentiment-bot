from abc import ABC, abstractmethod
from typing import Optional

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, system_message: Optional[str] = None, **kwargs) -> Optional[str]:
        """Generate text from the LLM.
        
        Args:
            prompt: The user prompt to generate from
            system_message: Optional system message to guide the LLM
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated text or None if error occurred
        """
        pass
