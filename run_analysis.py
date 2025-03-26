import logging
from main import start_nlp_analysis, start_reporting

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting analysis pipeline...")
    start_nlp_analysis()
    start_reporting()
    logger.info("Analysis pipeline completed")

if __name__ == '__main__':
    main()
