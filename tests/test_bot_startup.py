import pytest
from src.bot.bot import TelegramBot
from aiogram import Dispatcher

def test_bot_initialization():
    """Test basic bot initialization"""
    bot = TelegramBot()
    assert bot.bot is not None, "Bot instance should be created"
    assert isinstance(bot.dp, Dispatcher), "Dispatcher should be initialized"

if __name__ == "__main__":
    pytest.main(["-s", __file__])
