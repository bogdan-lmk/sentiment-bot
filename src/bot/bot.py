import sys
import os
import logging
import asyncio
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from aiogram import Bot, Dispatcher, F, types
from config.telegram_config import TELEGRAM_BOT_TOKEN
from aiogram.types import (
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    CallbackQuery,
    Message,
    ReplyKeyboardMarkup,
    KeyboardButton
)
from aiogram.filters import CommandStart, Command
from aiogram.fsm.context import FSMContext
from aiogram.types.input_file import FSInputFile
from config.telegram_config import TELEGRAM_BOT_TOKEN, TELEGRAM_REPORT_CHAT_ID, GEO_GROUPS

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TelegramBot:
    def __init__(self, token: str = TELEGRAM_BOT_TOKEN, chat_id: str = TELEGRAM_REPORT_CHAT_ID):
        self.bot = Bot(token=token)
        self.report_chat_id = chat_id
        self.dp = Dispatcher()
        self.active_geo = None
        self.running = False
        self._register_handlers()
        self.setup_handlers()

    async def start(self):
        """Start the Telegram bot with async polling"""
        self.running = True
        await self.dp.start_polling(self.bot)
        logger.info("Telegram bot started successfully")

    async def stop(self):
        """Stop the Telegram bot gracefully"""
        self.running = False
        await self.dp.storage.close()
        await self.dp.storage.wait_closed()
        await self.bot.session.close()
        logger.info("Telegram bot stopped")

    def setup_handlers(self):
        """Setup bot command handlers"""
        self.dp.message.register(self.send_welcome, CommandStart())
        self.dp.message.register(self.prompt_geo_selection, F.text == "Выбрать гео")
        self.dp.message.register(self.handle_report_request, F.text == "Сформировать отчет")
        self.dp.message.register(self.handle_chart_request, F.text == "Показать графики")

    def _register_handlers(self):
        """Регистрация обработчиков команд и callback запросов."""
        self.dp.message(CommandStart())(self.send_welcome)
        self.dp.callback_query(F.data == "get_pdf_report")(self.send_pdf_report)
        self.dp.callback_query(F.data == "get_text_report")(self.send_text_report)
        self.dp.callback_query(F.data == "get_short_pdf_report")(self.send_short_pdf_report)
        self.dp.callback_query(F.data == "get_short_text_report")(self.send_short_text_report)
        self.dp.callback_query(F.data == "chart_selection")(self.send_chart)
        self.dp.callback_query(F.data == "back_to_reports")(self.back_to_reports)
        self.dp.callback_query(F.data == "select_geo")(self.send_geo_selection)
        self.dp.callback_query(F.data.startswith("geo_"))(self.handle_geo_selection)
        self.dp.callback_query(F.data.startswith("get_"))(self.handle_chart_selection)

    async def send_welcome(self, message: types.Message):
        """Обработчик команды /start с минимальной клавиатурой"""
        keyboard = ReplyKeyboardMarkup(
            resize_keyboard=True,
            keyboard=[[KeyboardButton(text="Выбрать гео")]]
        )

        await message.answer(
            "🖥️ *Добро пожаловать в TG-AI-REPORTER!* 🖥️\n\n"
            "Для начала работы выберите географический регион:",
            reply_markup=keyboard,
            parse_mode="Markdown"
        )

    def get_report_buttons(self):
        """Создание инлайн-клавиатуры с кнопками отчетов."""
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="📄 Полный PDF отчёт", callback_data="get_pdf_report"),
             InlineKeyboardButton(text="📄 Краткий PDF отчёт", callback_data="get_short_pdf_report")],
            [InlineKeyboardButton(text="📜 Полный текстовый отчёт", callback_data="get_text_report"),
             InlineKeyboardButton(text="📜 Краткий текстовый отчёт", callback_data="get_short_text_report")],
            [InlineKeyboardButton(text="📊 Графики", callback_data="chart_selection"),
             InlineKeyboardButton(text="🌍 Выбрать регион", callback_data="select_geo")]
        ])
        return keyboard

    def get_chart_buttons(self):
        """Клавиатура для выбора графиков."""
        buttons = [
            [InlineKeyboardButton(text="📈 Тематическое распределение", callback_data="get_theme_distribution"),
             InlineKeyboardButton(text="📊 Топ ключевых слов", callback_data="get_top_keywords")],
            [InlineKeyboardButton(text="📉 Тренды сообщений", callback_data="get_message_trends"),
             InlineKeyboardButton(text="🧩 Распределение потребностей", callback_data="get_needs_distribution")],
            [InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_reports")]
        ]
        return InlineKeyboardMarkup(inline_keyboard=buttons)

    def get_geo_buttons(self):
        """Клавиатура для выбора гео-региона."""
        buttons = [
            [InlineKeyboardButton(text=f"🇩🇪 Германия", callback_data="geo_DEU"),
             InlineKeyboardButton(text=f"🇪🇸 Испания", callback_data="geo_ESP")],
            [InlineKeyboardButton(text=f"🇵🇹 Португалия", callback_data="geo_PRT")]
        ]
        return InlineKeyboardMarkup(inline_keyboard=buttons)

    async def send_geo_selection(self, callback: CallbackQuery):
        """Обработчик выбора гео-региона."""
        await callback.message.answer(
            "Выберите регион для анализа:",
            reply_markup=self.get_geo_buttons()
        )
        await callback.answer()

    async def prompt_geo_selection(self, message: types.Message):
        """Handle text command to show geo selection"""
        await message.answer(
            "Выберите регион для анализа:",
            reply_markup=self.get_geo_buttons()
        )

    async def handle_geo_selection(self, callback: CallbackQuery):
        """Handle geo selection from inline buttons"""
        geo_code = callback.data.replace("geo_", "")
        self.active_geo = geo_code
        await callback.message.answer(f"✅ Выбран регион: {geo_code}")
        
        try:
            await callback.message.answer(f"⏳ Начинаем парсинг для региона {geo_code}...")
            from src.data_collector.telegram_parser import TelegramParser
            parser = TelegramParser(geo_group=geo_code)
            await parser.run()
            await asyncio.to_thread(os.system, f"python generate_visualizations.py --geo {geo_code}")
            await self.generate_reports(geo_code)
            await callback.message.answer(
                f"✅ Анализ для региона {geo_code} завершен!",
                reply_markup=self.get_report_buttons()
            )
        except Exception as e:
            logger.error(f"Ошибка обработки гео-региона: {e}")
            await callback.message.answer(f"❌ Ошибка: {str(e)}")
        finally:
            await callback.answer()

    async def generate_reports(self, geo_code: str):
        """Асинхронная генерация отчетов для конкретного региона."""
        from src.reporting.pdf_reporter import PDFReporter
        pdf_reporter = PDFReporter(input_data_path=f"data/processed/{geo_code}")
        await asyncio.to_thread(pdf_reporter.generate_report)

    async def handle_report_request(self, message: types.Message):
        """Обработчик запроса отчетов"""
        await message.answer(
            "Выберите тип отчета:",
            reply_markup=self.get_report_buttons()
        )

    async def handle_chart_request(self, message: types.Message):
        """Обработчик запроса графиков"""
        await message.answer(
            "Выберите тип графика:",
            reply_markup=self.get_chart_buttons()
        )
        
        if self.active_geo:
            from src.reporting.llm_reporter import LLMReporter
            llm_reporter = LLMReporter(active_geo=self.active_geo)
            report_content = await asyncio.to_thread(llm_reporter.generate_report)
            if report_content:
                await message.answer("✅ LLM-отчет успешно сгенерирован")

    async def handle_chart_selection(self, callback: CallbackQuery):
        """Обработчик выбора графиков с автоматическим парсингом при отсутствии данных"""
        try:
            if not self.active_geo:
                logger.warning("Chart selection attempted without active geo (User: %s)", callback.from_user.id)
                await callback.answer("❌ Сначала выберите регион!")
                return

            chart_type = callback.data.replace("get_", "")
            await callback.answer(f"⏳ Проверяем данные для {chart_type.replace('_', ' ')}...")
            
            data_files = {
                "theme_distribution": "sentiment_analysis.csv",
                "top_keywords": "keywords.csv", 
                "message_trends": "message_trends.csv",
                "message_clusters": "message_clusters.csv",
                "needs_distribution": "needs_analysis.csv"
            }
            
            geo_path = f"data/processed/{self.active_geo}"
            file_path = os.path.join(geo_path, data_files[chart_type])
            
            if not os.path.exists(file_path):
                await callback.message.answer(f"⚠️ Данных нет, запускаем парсинг для {self.active_geo}...")
                from src.data_collector.telegram_parser import TelegramParser
                parser = TelegramParser(geo_group=self.active_geo)
                await parser.run()
                await asyncio.to_thread(os.system, f"python generate_visualizations.py --geo {self.active_geo}")
                
            if not os.path.exists(file_path):
                await callback.answer("❌ Данные недоступны после парсинга")
                return

            await callback.message.answer(f"📊 Генерируем {chart_type.replace('_', ' ')}...")
            from src.visualization.charts import DataVisualizer
            df = pd.read_csv(file_path)
            fig = DataVisualizer(df).plot_clusters(df) if chart_type == "message_clusters" else \
                  DataVisualizer(df).plot_sentiment_distribution() if chart_type == "theme_distribution" else \
                  DataVisualizer(df).plot_top_keywords(df) if chart_type == "top_keywords" else \
                  DataVisualizer(df).plot_trends(df) if chart_type == "message_trends" else \
                  DataVisualizer(df).plot_needs_distribution(df)
            
            if fig:
                chart_path = f"reports/visualizations/{chart_type}_{self.active_geo}.png"
                os.makedirs(os.path.dirname(chart_path), exist_ok=True)
                if fig:
                    fig.savefig(chart_path, bbox_inches='tight')
                    plt.close(fig)
                
                await self.bot.send_photo(
                    chat_id=callback.message.chat.id,
                    photo=FSInputFile(chart_path),
                    caption=f"📊 {chart_type.replace('_', ' ').title()} - {self.active_geo}"
                )
            
        except FileNotFoundError as e:
            error_msg = f"DATA_NOT_FOUND: Missing data file for {self.active_geo} ({str(e)})"
            logger.error(error_msg)
            await callback.answer("❌ Отсутствуют необходимые данные")
        except pd.errors.EmptyDataError as e:
            error_msg = f"EMPTY_DATA: Corrupted data file for {self.active_geo} ({str(e)})"
            logger.error(error_msg)
            await callback.answer("❌ Ошибка в данных анализа")
        except Exception as e:
            error_code = "INTERNAL_ERROR"
            error_msg = f"{error_code}: Unexpected error generating chart ({str(e)})"
            logger.error(error_msg, exc_info=True)
            await callback.answer(f"❌ Системная ошибка ({error_code})")

    # Остальные методы остаются без изменений
    async def send_pdf_report(self, callback: CallbackQuery):
        try:
            await callback.answer("⏳ Формируем полный PDF отчет...")
            
            if not self.active_geo:
                await callback.message.answer("❌ Сначала выберите регион!")
                return

            from src.reporting.pdf_reporter import PDFReporter
            pdf_reporter = PDFReporter(
                input_data_path=f"data/processed/{self.active_geo}",
                output_dir=f"reports/{self.active_geo}"
            )
            
            report_path = pdf_reporter.generate_report()
            
            if report_path and os.path.exists(report_path):
                await callback.message.answer_document(
                    FSInputFile(report_path),
                    caption=f"📊 Полный отчет для региона {self.active_geo}"
                )
            else:
                await callback.message.answer("❌ Не удалось сформировать отчет")
        except Exception as e:
            logger.error(f"Ошибка отправки отчета: {e}")
            await callback.message.answer(f"❌ Ошибка формирования отчета: {str(e)}")

    async def send_text_report(self, callback: CallbackQuery):
        await callback.answer("⏳ Формируем полный текстовый отчет...")

    async def send_short_pdf_report(self, callback: CallbackQuery):
        await callback.answer("⏳ Формируем краткий PDF отчет...")

    async def send_short_text_report(self, callback: CallbackQuery):
        await callback.answer("⏳ Формируем краткий текстовый отчет...")

    async def send_chart(self, callback: CallbackQuery):
        await callback.answer("📊 Загрузка доступных графиков...")

    async def back_to_reports(self, callback: CallbackQuery):
        await callback.message.answer(
            "Выберите тип отчета:",
            reply_markup=self.get_report_buttons()
        )
        await callback.answer()
