import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logger(name, log_file='app.log', level=logging.INFO):
    """Setup a logger with rotating file handler"""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    handler = RotatingFileHandler(
        os.path.join('logs', log_file),
        maxBytes=1024*1024,  # 1MB
        backupCount=5
    )
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger

# Create default logger instance
logger = setup_logger('ai-santiment')
