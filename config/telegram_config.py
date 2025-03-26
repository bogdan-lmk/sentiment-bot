from dotenv import load_dotenv
import os

load_dotenv()

# Для парсера (ваши личные данные)
API_ID = int(os.getenv("TELEGRAM_API_ID", ""))
API_HASH = os.getenv("TELEGRAM_API_HASH", "")
CHAT_LINK = os.getenv("CHAT_LINK", "")  # Keep as string to support both IDs and usernames

# Для бота
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_REPORT_CHAT_ID = os.getenv("TELEGRAM_REPORT_CHAT_ID")  # Куда отправлять отчёты