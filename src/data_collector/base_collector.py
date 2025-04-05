import abc
from abc import ABC

class BaseCollector(ABC):
    """Abstract base class for data collectors"""
    
    @abc.abstractmethod
    async def run(self) -> bool:
        """Main entry point for data collection
        Returns:
            bool: True if collection succeeded, False otherwise
        """
        pass
