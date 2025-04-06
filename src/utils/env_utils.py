import os
from typing import Optional

def get_required_env(name: str) -> str:
    """Get required environment variable or raise error"""
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value

def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Get optional environment variable with default"""
    return os.getenv(name) or default
