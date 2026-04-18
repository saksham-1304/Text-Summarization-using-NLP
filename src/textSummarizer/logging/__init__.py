"""Logging configuration for Text Summarization pipeline.

Creates a logger that writes to both console and a rotating log file.
Log format: [timestamp: level: module: message]
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler

LOG_FORMAT = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "running_logs.log")
LOGGER_NAME = "textSummarizer"

os.makedirs(LOG_DIR, exist_ok=True)

# Rotating file handler: max 5MB per file, keep 5 backups
file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# Configure the project logger exactly once.
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)