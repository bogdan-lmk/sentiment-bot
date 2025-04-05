from pathlib import Path
from dotenv import load_dotenv
from src.utils.env_utils import get_required_env

# Load environment variables
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(env_path, override=True)

# Telegram configuration
TELEGRAM_BOT_TOKEN = get_required_env("TELEGRAM_BOT_TOKEN")
TELEGRAM_REPORT_CHAT_ID = get_required_env("TELEGRAM_REPORT_CHAT_ID")
API_ID = get_required_env("TELEGRAM_API_ID")
API_HASH = get_required_env("TELEGRAM_API_HASH")

# Гео-группы для анализа
GEO_GROUPS = {
    "DEU": {
        "chat_ids": [-1002158812012,-1001783625336],  # Converted to integers
        "keywords": ["germany", "deutschland", "берлин"],
        "timezone": "Europe/Berlin"
    },
    "ESP": {
        "chat_ids": [-1001727866141,-1001713113247],
        "keywords": ["spain", "españa", "мадрид"],
        "timezone": "Europe/Madrid"
    },
    "PRT": {
        "chat_ids": [-1001342547202,-1002239405289,-1001590941393],
        "keywords": ["portugal", "португалия", "лиссабон"],
        "timezone": "Europe/Lisbon"
    }
}
