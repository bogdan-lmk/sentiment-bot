import os
import asyncio
from telethon import TelegramClient
from config.telegram_config import TELEGRAM_BOT_TOKEN, TELEGRAM_REPORT_CHAT_ID as REPORT_CHAT_ID, API_ID, API_HASH

class TelegramReporter:
    """
    Класс для отправки отчетов в Telegram.
    """
    
    def __init__(self):
        self.client = TelegramClient('reporter_session', API_ID, API_HASH)
    
    async def send_message(self, message, max_retries=3):
        """Отправка текстового сообщения в Telegram."""
        if not REPORT_CHAT_ID:
            print("Ошибка: Не указан ID чата для отчетов")
            return False
            
        for attempt in range(max_retries):
            try:
                await self.client.start(bot_token=TELEGRAM_BOT_TOKEN)
                await self.client.send_message(REPORT_CHAT_ID, message)
                print("Сообщение отправлено в Telegram")
                return True
            except Exception as e:
                print(f"Ошибка отправки сообщения (попытка {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return False
                await asyncio.sleep(2)
        return False
    
    async def send_file(self, file_path, max_retries=3):
        """Отправка файла в Telegram."""
        if not REPORT_CHAT_ID:
            print("Ошибка: Не указан ID чата для отчетов")
            return False
            
        if not os.path.exists(file_path):
            print("Файл не найден.")
            return False
            
        for attempt in range(max_retries):
            try:
                await self.client.start(bot_token=TELEGRAM_BOT_TOKEN)
                await self.client.send_file(REPORT_CHAT_ID, file_path)
                print(f"Файл {file_path} отправлен в Telegram")
                return True
            except Exception as e:
                print(f"Ошибка отправки файла (попытка {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return False
                await asyncio.sleep(2)
        return False
    
    def send_report(self, report_path):
        """Отправляет отчет в виде файла в Telegram."""
        try:
            success = asyncio.run(self.send_file(report_path))
            if not success:
                print("Не удалось отправить отчет в Telegram")
            return success
        finally:
            asyncio.run(self.client.disconnect())
    
    def send_text_report(self, report_text):
        """Отправляет текстовый отчет в Telegram."""
        try:
            success = asyncio.run(self.send_message(report_text))
            if not success:
                print("Не удалось отправить текстовый отчет в Telegram")
            return success
        finally:
            asyncio.run(self.client.disconnect())
    
if __name__ == "__main__":
    reporter = TelegramReporter()
    reporter.send_report("reports/llm_report.pdf")
