from telethon import TelegramClient
import asyncio
import os
import pandas as pd
from config.telegram_config import API_ID, API_HASH, CHAT_LINK
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

SESSION_NAME = "telegram_session"

class TelegramParser:
    def __init__(self):
        self.client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
        self.analyzer = SentimentIntensityAnalyzer()

    async def connect(self):
        await self.client.start()
        print("Client connected successfully.")

    async def fetch_messages(self, chat_name, limit=100):
        """Fetch messages from a given Telegram chat and analyze sentiment."""
        messages = []
        async for message in self.client.iter_messages(chat_name, limit=limit):
            sentiment_score = self.analyzer.polarity_scores(message.text)["compound"] if message.text else 0
            messages.append({
                "date": message.date,
                "author": message.sender_id,
                "text": message.text,
                "source": chat_name,
                "sentiment": sentiment_score
            })
        return messages

    async def save_messages_to_csv(self, chat_name, limit=100, output_path="data/raw/messages.csv"):
        """Fetch messages, analyze sentiment, and save them to a CSV file."""
        messages = await self.fetch_messages(chat_name, limit)
        df = pd.DataFrame(messages)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Saved {len(messages)} messages to {output_path}")

    async def close(self):
        await self.client.disconnect()
        print("Client disconnected.")

if __name__ == "__main__":
    parser = TelegramParser()
    limit = 500
    
    asyncio.run(parser.connect())
    asyncio.run(parser.save_messages_to_csv(CHAT_LINK, limit))
    asyncio.run(parser.close())
