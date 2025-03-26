import logging
import os
import asyncio
from aiogram import Bot, Dispatcher, F
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery, Message
from aiogram.filters import CommandStart, Command
from aiogram.fsm.context import FSMContext
from aiogram.types.input_file import FSInputFile

from config.telegram_config import TELEGRAM_BOT_TOKEN, TELEGRAM_REPORT_CHAT_ID

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
        self._register_handlers()

    def _register_handlers(self):
        """Регистрация обработчиков команд и callback запросов."""
        self.dp.message(CommandStart())(self.send_welcome)
        self.dp.callback_query(F.data == "get_pdf_report")(self.send_pdf_report)
        self.dp.callback_query(F.data == "get_text_report")(self.send_text_report)
        self.dp.callback_query(F.data == "get_chart")(self.send_chart)

    def get_report_buttons(self):
        """Создание инлайн-клавиатуры с кнопками отчетов."""
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="📄 PDF отчёт", callback_data="get_pdf_report")],
            [InlineKeyboardButton(text="📜 Текстовый отчёт", callback_data="get_text_report")],
            [InlineKeyboardButton(text="📊 График", callback_data="get_chart")]
        ])
        return keyboard

    async def send_welcome(self, message: Message):
        """Обработчик команды /start."""
        try:
            await message.answer(
                "Привет! Я бот для анализа сообщений 📊\nВыберите нужный отчёт:",
                reply_markup=self.get_report_buttons()
            )
        except Exception as e:
            logger.error(f"Ошибка при отправке приветственного сообщения: {e}")

    async def send_pdf_report(self, callback: CallbackQuery):
        """Отправка PDF отчёта."""
        pdf_path = "reports/llm_report.pdf"
        try:
            if not os.path.exists(pdf_path):
                await callback.message.answer("⌛ PDF отчёт генерируется...")
                # Try generating report if not exists
                from src.reporting.pdf_reporter import PDFReporter
                reporter = PDFReporter()
                reporter.generate_report()
                
            if os.path.exists(pdf_path):
                await callback.message.answer_document(FSInputFile(pdf_path))
            else:
                await callback.message.answer("❌ Не удалось сгенерировать PDF отчёт.")
            # Отправляем клавиатуру с выбором отчётов после отправки
            await callback.message.answer(
                "Выберите другой отчёт:",
                reply_markup=self.get_report_buttons()
            )
            await callback.answer()
        except Exception as e:
            logger.error(f"Ошибка при отправке PDF отчёта: {e}")
            await callback.message.answer("❌ Ошибка: " + str(e))
            await callback.answer()

    async def send_text_report(self, callback: CallbackQuery):
        """Отправка текстового отчёта."""
        text_report_path = "reports/ai_report.txt"
        try:
            if not os.path.exists(text_report_path):
                await callback.message.answer("⌛ Текстовый отчёт генерируется...")
                # Try generating report if not exists
                from src.reporting.llm_reporter import LLMReporter
                reporter = LLMReporter()
                reporter.generate_report()
                
            if os.path.exists(text_report_path):
                with open(text_report_path, 'r') as f:
                    report_text = f.read()
                await callback.message.answer(report_text)
            else:
                await callback.message.answer("❌ Не удалось сгенерировать текстовый отчёт.")
            # Отправляем клавиатуру с выбором отчётов после отправки
            await callback.message.answer(
                "Выберите другой отчёт:",
                reply_markup=self.get_report_buttons()
            )
            await callback.answer()
        except Exception as e:
            logger.error(f"Ошибка при отправке текстового отчёта: {e}")
            await callback.message.answer("❌ Ошибка: " + str(e))
            await callback.answer()

    async def send_chart(self, callback: CallbackQuery):
        """Отправка диаграммы."""
        chart_path = "data/processed/chart.png"
        try:
            if os.path.exists(chart_path):
                await callback.message.answer_photo(FSInputFile(chart_path))
                await callback.answer()
            else:
                await callback.message.answer("❌ Диаграмма не найдена.")
                await callback.answer()
            # Отправляем клавиатуру с выбором отчётов после отправки
            await callback.message.answer(
                "Выберите другой отчёт:",
                reply_markup=self.get_report_buttons()
            )
        except Exception as e:
            logger.error(f"Ошибка при отправке диаграммы: {e}")
            await callback.message.answer("Произошла ошибка при отправке диаграммы.")

    async def start(self):
        """Запуск бота."""
        try:
            logger.info("Запуск Telegram бота...")
            await self.dp.start_polling(self.bot)
        except Exception as e:
            logger.error(f"Ошибка при запуске бота: {e}")

async def create_and_run_bot():
    """Создание и запуск бота."""
    try:
        bot_instance = TelegramBot()
        await bot_instance.start()
    except Exception as e:
        logger.error(f"Fatal error in bot: {e}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(create_and_run_bot())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
