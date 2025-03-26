import logging
from typing import List, Dict, Optional
from telethon.sync import TelegramClient
import pandas as pd
import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

class TelegramParser:
    def __init__(self):
        """Initialize Telegram parser with credentials from .env"""
        self.api_id = int(os.getenv("TELEGRAM_API_ID"))
        self.api_hash = os.getenv("TELEGRAM_API_HASH")
        self.phone = os.getenv("TELEGRAM_PHONE")
        self.chat_id = os.getenv("CHAT_LINK")

    async def _fetch_messages(self) -> Optional[List[Dict]]:
        """Fetch messages from Telegram chat"""
        client = TelegramClient('sessions/telegram_session', self.api_id, self.api_hash)
        
        try:
            # Try connecting with existing session first
            await client.connect()
            if not await client.is_user_authorized():
                raise ConnectionError("Session invalid")
        except Exception as e:
            logger.warning(f"Session invalid: {e}")
            # Fall back to phone login if session is invalid
            if not self.phone:
                raise ValueError("No phone number provided and no valid session exists")
            await client.start(self.phone)
        
        try:
            # Handle both numeric IDs and usernames
            chat_id = int(self.chat_id) if self.chat_id.lstrip('-').isdigit() else self.chat_id
            entity = await client.get_entity(chat_id)
            logger.info(f"Successfully accessed chat: {entity.title if hasattr(entity, 'title') else entity.id}")
            
            messages = []
            async for message in client.iter_messages(entity, limit=100):
                if message.text:
                    messages.append({
                        "date": message.date.strftime("%Y-%m-%d %H:%M:%S"),
                        "author": message.sender_id,
                        "text": message.text
                    })
            return messages
        except Exception as e:
            logger.error(f"Failed to fetch messages: {e}")
            logger.error("Please ensure:")
            logger.error("1. You're signed in as the correct user")
            logger.error("2. You've joined the channel")
            logger.error(f"3. Chat ID {self.chat_id} is correct")
            return None

    async def parse_messages(self, output_path: str = "data/raw/messages.csv") -> bool:
        """Parse messages from Telegram group and save to CSV"""
        logger.info("Fetching messages from Telegram...")
        messages = await self._fetch_messages()
        
        if not messages:
            logger.warning("No messages found in the chat")
            return False
            
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            pd.DataFrame(messages).to_csv(output_path, index=False)
            logger.info(f"Saved {len(messages)} messages to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save messages: {e}")
            return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = TelegramParser()
    asyncio.run(parser.parse_messages())
