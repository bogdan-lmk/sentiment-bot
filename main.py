import os
import logging
import asyncio
from config.telegram_config import TELEGRAM_BOT_TOKEN, TELEGRAM_REPORT_CHAT_ID
from src.bot.bot import TelegramBot
from src.analyzer.keyword_extractor import AdvancedNLPAnalyzer
from src.reporting.llm_reporter import LLMReporter
from src.reporting.pdf_reporter import PDFReporter
from src.reporting.csv_reporter import CSVReporter
from src.visualization.dashboard import create_dashboard
import threading
import signal

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def run_telegram_bot():
    """Initialize and run the Telegram bot."""
    try:
        logger.info("Starting the Telegram bot...")
        bot = TelegramBot(TELEGRAM_BOT_TOKEN, TELEGRAM_REPORT_CHAT_ID)
        await bot.start()
    except asyncio.CancelledError:
        logger.info("Telegram bot stopped")
    except Exception as e:
        logger.error(f"Error in Telegram bot: {e}")
        raise

async def parse_telegram_messages():
    """Parse messages from Telegram group."""
    try:
        logger.info("Parsing Telegram messages...")
        from src.data_collector.telegram_parser import TelegramParser
        parser = TelegramParser()
        success = await parser.parse_messages(output_path="data/raw/messages.csv")
        if not success:
            logger.error("Failed to parse messages - no messages found")
            return False
        if not os.path.exists("data/raw/messages.csv"):
            logger.error("Messages file not created")
            return False
        logger.info("Message parsing completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Error parsing messages: {e}")
        return False

def start_nlp_analysis():
    """Initialize and start the NLP analysis."""
    try:
        logger.info("Starting the NLP analysis...")
        analyzer = AdvancedNLPAnalyzer(language='russian')
        analyzer.comprehensive_analysis(input_path="data/raw/messages.csv", output_dir="data/processed")
        logger.info("NLP analysis completed successfully.")
    except Exception as e:
        logger.error(f"Error starting the NLP analysis: {e}")

def start_reporting():
    """Generate reports and send them to the Telegram bot."""
    try:
        logger.info("Starting report generation...")
        
        # Wait for NLP analysis to complete by checking for any output files
        processed_files = [
            "keywords.csv",
            "message_clusters.csv", 
            "sentiment_analysis.csv",
            "theme_classification.csv"
        ]
        if not any(os.path.exists(f"data/processed/{f}") for f in processed_files):
            logger.error("Processed data files not found in data/processed/")
            return

        # Generate LLM report and get the text
        llm_report = LLMReporter(input_data_path="data/processed", provider="deepseek")
        report_text = llm_report.generate_report()
        
        if report_text:
            # Generate PDF report using the same text
            pdf_report = PDFReporter(input_data_path="data/processed")
            pdf_report.generate_report(report_text)

            # Generate CSV report
            csv_report = CSVReporter(input_data_path="data/processed")
            csv_report.generate_report()
            
            logger.info("Reports generated successfully.")
        else:
            logger.error("Failed to generate LLM report text")
    except Exception as e:
        logger.error(f"Error generating reports: {e}")

def start_dashboard():
    """Start the Streamlit dashboard in a separate thread."""
    try:
        logger.info("Starting the dashboard...")
        dashboard_thread = threading.Thread(target=create_dashboard)
        dashboard_thread.daemon = True
        dashboard_thread.start()
        return dashboard_thread
    except Exception as e:
        logger.error(f"Error starting the dashboard: {e}")
        return None

async def main():
    """Main function to start the services."""
    logger.info("Starting the main application...")
    dashboard_thread = None

    try:
        # First parse messages from Telegram
        success = await parse_telegram_messages()
        if not success:
            logger.error("Cannot proceed - message parsing failed")
            return

        # Then run NLP analysis on parsed messages
        start_nlp_analysis()

        # Generate reports from analysis
        start_reporting()

        # Start dashboard if data exists
        if os.path.exists('data/processed/sentiment_analysis.csv'):
            dashboard_thread = start_dashboard()
        else:
            logger.error("Cannot start dashboard - sentiment analysis data not found")

        # Run Telegram bot (this will block until stopped)
        await run_telegram_bot()

    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Fatal error in application: {e}")
        raise
    finally:
        if dashboard_thread and dashboard_thread.is_alive():
            logger.info("Stopping dashboard...")
            os.kill(os.getpid(), signal.SIGTERM)

if __name__ == '__main__':
    asyncio.run(main())
