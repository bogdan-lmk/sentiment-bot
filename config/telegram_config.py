import os
import sys
from dotenv import load_dotenv

# Load environment variables first
load_dotenv(override=True)

def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        print(f"Error: Required environment variable {name} is not set")
        sys.exit(1)
    return value

def get_optional_env(name: str, default: str = "") -> str:
    return os.getenv(name, default)

# Для парсера (ваши личные данные)
try:
    API_ID = int(os.getenv("TELEGRAM_API_ID", "0")) or None  # 0 will be treated as None
except ValueError:
    print("Error: TELEGRAM_API_ID must be a valid integer if provided")
    API_ID = None

API_HASH = get_optional_env("TELEGRAM_API_HASH")
CHAT_LINK = get_optional_env("CHAT_LINK")  # Keep as string to support both IDs and usernames

# Для бота
TELEGRAM_BOT_TOKEN = get_required_env("TELEGRAM_BOT_TOKEN")
TELEGRAM_REPORT_CHAT_ID = get_required_env("TELEGRAM_REPORT_CHAT_ID")  # Куда отправлять отчёты
