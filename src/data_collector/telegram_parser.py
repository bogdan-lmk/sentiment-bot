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
        self.chat_id = '-1002239405289,-1001590941393,-1001200251912,-1001342547202'

    async def _fetch_messages(self, output_path: str = "data/raw/messages.csv") -> Optional[List[Dict]]:
        """Fetch messages from Telegram chat
        
        Args:
            output_path: Path to existing messages CSV file to check for last date
        """
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
        
        messages = []
        chat_ids = [cid.strip() for cid in self.chat_id.split(',')]
        
        for chat_id in chat_ids:
            try:
                # Handle numeric IDs, usernames, and channel IDs with suffixes
                if '_' in chat_id:
                    # Keep channel IDs with suffixes as strings
                    entity = await client.get_entity(chat_id)
                else:
                    # Convert regular numeric IDs to integers
                    chat_id = int(chat_id) if chat_id.lstrip('-').isdigit() else chat_id
                    entity = await client.get_entity(chat_id)
                logger.info(f"Successfully accessed chat: {entity.title if hasattr(entity, 'title') else entity.id}")
                
                # Get last message date from existing file if it exists
                last_date = None
                if os.path.exists(output_path):
                    try:
                        existing_df = pd.read_csv(output_path)
                        if not existing_df.empty:
                            last_date = pd.to_datetime(existing_df['date']).max()
                    except Exception as e:
                        logger.warning(f"Couldn't read existing messages: {e}")
                
                # Process messages in batches to avoid rate limits
                batch_size = 500
                total_messages = 0
                async for message in client.iter_messages(entity, limit=1000, offset_date=last_date):
                    if message.text:
                        messages.append({
                            "date": message.date.strftime("%Y-%m-%d %H:%M:%S"),
                            "author": message.sender_id,
                            "text": message.text,
                            "chat_id": chat_id
                        })
                        total_messages += 1
                        
                        # Stop if we've reached 1000 messages for this chat
                        if total_messages >= 500:
                            logger.info(f"Reached 500 message limit for chat {chat_id}")
                            break
                            
                        # Save batch periodically and sleep to avoid rate limits
                        if len(messages) % batch_size == 0:
                            logger.info(f"Processed {total_messages} messages from chat {chat_id}")
                            await asyncio.sleep(1)  # Brief pause between batches
            except Exception as e:
                logger.error(f"Failed to fetch messages from chat {chat_id}: {e}")
                continue
                
        return messages if messages else None

    async def parse_messages(self, output_path: str = "data/raw/messages.csv") -> bool:
        """Parse messages from Telegram group and save to CSV"""
        logger.info("Fetching messages from Telegram...")
        messages = await self._fetch_messages(output_path)
        
        if not messages:
            logger.warning("No messages found in any chat")
            return False
            
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df = pd.DataFrame(messages)
            
            # Append to existing file if it exists
            if os.path.exists(output_path):
                existing_df = pd.read_csv(output_path)
                df = pd.concat([existing_df, df]).drop_duplicates(subset=['date', 'author', 'text', 'chat_id'])
                
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(messages)} messages from {len(set(m['chat_id'] for m in messages))} chats to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save messages: {e}")
            return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = TelegramParser()
    asyncio.run(parser.parse_messages())
