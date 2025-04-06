# Must precede any llm module imports
import os
import logging
import asyncio
import matplotlib.pyplot as plt
import schedule
import time
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
        success = await parser.run()
        if not success:
            logger.error("Failed to parse messages - no messages found")
            return False
        # Check if any geo message files were created
        geo_dirs = [d for d in os.listdir("data/raw") if os.path.isdir(os.path.join("data/raw", d))]
        if not any(os.path.exists(f"data/raw/{geo}/messages_{geo}.csv") for geo in geo_dirs):
            logger.error("No message files found in data/raw/* directories")
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
        from config.analyzer_config import ANALYZER_CONFIG
        
        # Get geo codes from raw data directories
        geo_dirs = [d for d in os.listdir("data/raw") 
                   if os.path.isdir(os.path.join("data/raw", d))]
        
        if not geo_dirs:
            logger.error("No geo directories found in data/raw/")
            return

        analyzer = AdvancedNLPAnalyzer(language=ANALYZER_CONFIG["default_language"])
        
        # Process each geo region
        for geo_code in geo_dirs:
            logger.info(f"Processing geo region: {geo_code}")
            analyzer.comprehensive_analysis(
                geo_code=geo_code,
                input_dir=ANALYZER_CONFIG["input_dir"],
                output_dir=ANALYZER_CONFIG["output_dir"]
            )
        
        logger.info("NLP analysis completed successfully.")
    except Exception as e:
        logger.error(f"Error starting the NLP analysis: {e}")

def start_reporting():
    """Generate reports and send them to the Telegram bot."""
    try:
        logger.info("Starting report generation...")
        
        # Check for processed files in geo directories
        geo_dirs = [d for d in os.listdir("data/processed") if os.path.isdir(os.path.join("data/processed", d))]
        if not geo_dirs:
            logger.error("No processed geo directories found in data/processed/")
            return

        # Wait for NLP analysis to complete by checking for any output files
        processed_files = [
            "keywords.csv",
            "message_clusters.csv", 
            "sentiment_analysis.csv",
            "theme_classification.csv",
            "top_phrases.csv"
        ]
        if not any(os.path.exists(f"data/processed/{f}") for f in processed_files):
            logger.error("Processed data files not found in data/processed/")
            return

        # Generate visualizations
        from src.visualization.charts import DataVisualizer
        import pandas as pd
        
        # Process visualizations per geo
        geo_dirs = [d for d in os.listdir("data/processed") if os.path.isdir(os.path.join("data/processed", d))]
        
        for geo in geo_dirs:
            geo_path = os.path.join("data/processed", geo)
            vis_path = os.path.join("reports", geo, "visualizations")
            os.makedirs(vis_path, exist_ok=True)

            try:
                # Load geo-specific data
                sentiment_df = pd.read_csv(os.path.join(geo_path, "sentiment_analysis.csv"))
                keywords_df = pd.read_csv(os.path.join(geo_path, "keywords.csv"))
                trends_df = pd.read_csv(os.path.join(geo_path, "message_trends.csv"))
                clusters_df = pd.read_csv(os.path.join(geo_path, "message_clusters.csv"))
                phrases_df = pd.read_csv(os.path.join(geo_path, "top_phrases.csv"))

                visualizer = DataVisualizer(sentiment_df)
                
                # Generate all visualizations
                visualizations = [
                    ("sentiment_distribution.png", visualizer.plot_sentiment_distribution()),
                    ("top_keywords.png", visualizer.plot_top_keywords(keywords_df)),
                    ("message_trends.png", visualizer.plot_trends(trends_df)),
                    ("message_clusters.png", visualizer.plot_clusters(clusters_df)),
                    ("top_phrases.png", visualizer.plot_top_phrases(phrases_df))
                ]

                for filename, fig in visualizations:
                    if fig is not None:
                        fig.savefig(os.path.join(vis_path, filename))
                        plt.close(fig)

            except Exception as e:
                logger.error(f"Error processing visualizations for {geo}: {e}")

        # Generate reports per geo
        for geo in geo_dirs:
            geo_path = os.path.join("data/processed", geo)
            
            # Initialize reporters with geo-specific data
            llm_report = LLMReporter(input_data_path=geo_path, provider="deepseek")
            pdf_report = PDFReporter(input_data_path=geo_path)
            csv_report = CSVReporter(input_data_path=geo_path)

            # Generate full reports
            report_text = llm_report.generate_report()
            if report_text:
                pdf_report.generate_report(report_text)
                csv_report.generate_report()
                
                # Generate short reports
                short_report_text = llm_report.generate_short_report()
                if short_report_text:
                    pdf_report.generate_short_report(short_report_text)
            
            logger.info("All reports and visualizations generated successfully.")
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

async def process_data():
    """Process data pipeline - parse messages, analyze, and generate reports."""
    try:
        logger.info("Starting scheduled data processing...")
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
        if any(f.endswith('sentiment_analysis.csv') for f in os.listdir('data/processed')):
            start_dashboard()
        else:
            logger.error("Cannot start dashboard - sentiment analysis data not found")
    except Exception as e:
        logger.error(f"Error in data processing: {e}")

async def main():
    """Main function to start the services."""
    logger.info("Starting the main application...")

    # Schedule data processing every 6 hours
    schedule.every(6).hours.do(lambda: asyncio.run(process_data()))

    # Run initial processing
    await process_data()

    # Start scheduler in background
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(1)

    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()

    try:
        # Run Telegram bot (this will block until stopped)
        await run_telegram_bot()
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Fatal error in application: {e}")
        raise

if __name__ == '__main__':
    asyncio.run(main())
