import sys
import logging
from typing import List, Dict, Tuple
import os

# Add project root directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from telethon.sync import TelegramClient
from src.data_collector.base_collector import BaseCollector
from contextlib import asynccontextmanager
from typing import AsyncIterator

class ClientSessionManager:
    """Manages pool of Telegram client connections with rate limiting"""
    def __init__(self, client: TelegramClient, pool_size: int = 5):
        self.client = client
        self.semaphore = asyncio.Semaphore(pool_size)

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[TelegramClient]:
        async with self.semaphore:
            yield self.client

    async def __aenter__(self):
        await self.client.connect()
        if not await self.client.is_user_authorized():
            if not self.phone:
                raise ValueError("No phone number provided and no valid session exists")
            await self.client.start(self.phone)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.disconnect()
import pandas as pd
import os
import asyncio
from config.telegram_config import GEO_GROUPS

logger = logging.getLogger(__name__)

class TelegramParser(BaseCollector):
    def __init__(self, geo_group: str = None):
        """Initialize Telegram parser with credentials and geo-groups
        Args:
            geo_group: Specific geo group to process (None for all)
        """
        api_id = os.getenv("TELEGRAM_API_ID")
        if not api_id:
            raise ValueError("TELEGRAM_API_ID environment variable not set")
        self.api_id = int(api_id)
        self.api_hash = os.getenv("TELEGRAM_API_HASH")
        self.phone = os.getenv("TELEGRAM_PHONE")
        self.geo_groups = GEO_GROUPS
        self.target_geo = geo_group
        self.semaphore = asyncio.Semaphore(5)  # Limit concurrent connections

    async def _process_geo_group(self, client: TelegramClient, geo_code: str, group_config: dict) -> List[Dict]:
        """Process messages for a specific geo group"""
        messages = []
        try:
            for chat_id in group_config['chat_ids']:
                entity = None
                try:
                    entity = await client.get_input_entity(chat_id)
                    if not entity:
                        raise ValueError(f"Entity not found for chat_id {chat_id}")
                except Exception as e:
                    logger.error(f"Error resolving entity {chat_id}: {e}")
                    continue
                try:
                    # Handle different entity types safely
                    channel_id = getattr(entity, 'channel_id', getattr(entity, 'id', 'unknown'))
                    title = getattr(entity, 'title', f'Channel#{channel_id}')
                    # Get entity ID based on type
                    entity_id = (
                        entity.channel_id if hasattr(entity, 'channel_id') else
                        entity.chat_id if hasattr(entity, 'chat_id') else
                        entity.user_id if hasattr(entity, 'user_id') else
                        'unknown'
                    )
                    logger.info(f"Processing {geo_code} chat: {title} (Type: {type(entity).__name__}, ID: {entity_id})")
                except Exception as e:
                    logger.error(f"Error logging channel info: {e}")
                    continue
                
                try:
                    async for message in client.iter_messages(entity, limit=500):
                        if message.text and any(kw.lower() in message.text.lower() for kw in group_config['keywords']):
                            messages.append({
                                "date": message.date.strftime("%Y-%m-%d %H:%M:%S"),
                                "author": message.sender_id,
                                "text": message.text,
                                "geo_code": geo_code,
                                "chat_id": chat_id
                            })
                except Exception as e:
                    logger.error(f"Error processing messages in {geo_code} chat {chat_id}: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing {geo_code} group: {str(e)}")
            logger.debug(f"Entity type: {type(entity).__name__} | Entity attributes: {vars(entity)}")
        return messages

    async def _fetch_messages(self) -> Tuple[List[Dict], List[str]]:
        """Fetch messages from geo groups with parallel processing"""
        client = TelegramClient('sessions/telegram_session', self.api_id, self.api_hash)
        messages = []
        processed_geo = []
        failed_geo = []
        
        try:
            await client.connect()
            if not await client.is_user_authorized():
                if not self.phone:
                    raise ValueError("No phone number provided and no valid session exists")
                await client.start(self.phone)

            # Filter geo groups if target specified
            target_geos = [self.target_geo] if self.target_geo else self.geo_groups.keys()

            # Process geo groups in parallel with connection pool
            async with ClientSessionManager(client, pool_size=5) as session:
                tasks = [
                    self._process_geo_group(session.client, geo_code, group_config)
                    for geo_code in target_geos
                    if (group_config := self.geo_groups.get(geo_code))
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for geo_code, result in zip(target_geos, results):
                    if isinstance(result, Exception):
                        logger.error(f"Failed processing {geo_code}: {str(result)}")
                        failed_geo.append(geo_code)
                    elif result:
                        messages.extend(result)
                        processed_geo.append(geo_code)
                if failed_geo:
                    logger.warning(f"Failed to process {len(failed_geo)} geo groups: {', '.join(failed_geo)}")
                
        finally:
            await client.disconnect()
            
        return messages, processed_geo
    
    async def run_parser(self) -> bool:
        """Alias for run() method to maintain compatibility"""
        return await self.run()

    async def run(self) -> bool:
        """Main entry point for parsing messages from geo groups"""
        logger.info("Starting geo-group message parsing...")
        messages, processed_geo = await self._fetch_messages()
        
        if not messages:
            logger.warning("No messages found in any geo group")
            return False
            
        try:
            # Group messages by geo code and save to separate files
            df = pd.DataFrame(messages)
            geo_codes = df['geo_code'].unique()
            
            for geo_code in geo_codes:
                geo_df = df[df['geo_code'] == geo_code]
                output_dir = f"data/raw/{geo_code}"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"messages_{geo_code}.csv")
                
                # Append to existing file if it exists
                if os.path.exists(output_path):
                    existing_df = pd.read_csv(output_path)
                    geo_df = pd.concat([existing_df, geo_df]).drop_duplicates(
                        subset=['date', 'author', 'text', 'chat_id']
                    )
                
                geo_df.to_csv(output_path, index=False)
                logger.info(f"Saved {len(geo_df)} messages for {geo_code} to {output_path}")
            
            logger.info(f"Successfully processed {len(processed_geo)} geo groups")
            return True
        except Exception as e:
            logger.error(f"Failed to save messages: {e}")
            return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = TelegramParser()
    asyncio.run(parser.run())
