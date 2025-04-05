import pytest
import os
from unittest.mock import AsyncMock, MagicMock, patch
from src.data_collector.telegram_parser import TelegramParser
import pandas as pd
import asyncio

@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_API_ID", "12345")
    monkeypatch.setenv("TELEGRAM_API_HASH", "test_hash")
    monkeypatch.setenv("TELEGRAM_PHONE", "+123456789")

@pytest.fixture
def geo_groups_config():
    return {
        "DEU": {
            "chat_ids": [-10012345678],
            "keywords": ["test", "pytest"]
        }
    }

@pytest.mark.asyncio
async def test_telegram_parser_initialization(mock_env):
    parser = TelegramParser()
    assert parser.api_id == 12345
    assert parser.api_hash == "test_hash"
    assert parser.semaphore._value == 5

@pytest.mark.asyncio
async def test_successful_message_fetching(mock_env, geo_groups_config):
    with patch('telethon.sync.TelegramClient') as mock_client:
        mock_client_instance = AsyncMock()
        mock_client.return_value = mock_client_instance
        
        # Mock messages
        mock_message = MagicMock()
        mock_message.text = "Test message with keyword"
        mock_message.date = MagicMock()
        mock_message.date.strftime.return_value = "2024-01-01 00:00:00"
        mock_message.sender_id = 123456
        
        # Mock iterator
        mock_client_instance.iter_messages = AsyncMock(return_value=[mock_message])
        
        parser = TelegramParser(geo_group="DEU")
        messages, processed_geo = await parser._fetch_messages()
        
        assert len(messages) > 0
        assert "DEU" in processed_geo
        assert messages[0]["text"] == "Test message with keyword"

@pytest.mark.asyncio
async def test_failed_geo_processing(mock_env):
    parser = TelegramParser(geo_group="INVALID_GEO")
    messages, processed_geo = await parser._fetch_messages()
    
    assert len(messages) == 0
    assert len(processed_geo) == 0

@pytest.mark.asyncio
async def test_message_saving(tmp_path, mock_env):
    test_messages = [{
        "date": "2024-01-01 00:00:00",
        "author": 123456,
        "text": "Test message",
        "geo_code": "DEU",
        "chat_id": -10012345678
    }]
    
    with patch.object(TelegramParser, '_fetch_messages', AsyncMock(return_value=(test_messages, ["DEU"]))), \
         patch('os.makedirs') as mock_makedirs:
        
        parser = TelegramParser()
        result = await parser.run_parser()
        
        assert result is True
        mock_makedirs.assert_called_once_with("data/raw/DEU", exist_ok=True)

def test_geo_group_configuration(mock_env, geo_groups_config):
    parser = TelegramParser()
    assert "DEU" in parser.geo_groups
    assert parser.geo_groups["DEU"]["chat_ids"] == [-10012345678]
