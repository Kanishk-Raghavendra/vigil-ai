"""Logger utility for VIGIL system."""

import logging
import sys
from pathlib import Path
from typing import Optional


class Logger:
    """Simple logger wrapper for VIGIL system."""
    
    _loggers = {}
    _log_file = None
    
    def __init__(self, name: str):
        """
        Initialize logger with given name.
        
        Args:
            name: Logger name (typically __name__)
        """
        if name not in Logger._loggers:
            logger = logging.getLogger(name)
            logger.setLevel(logging.DEBUG)
            
            # Console handler (INFO level)
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                '[%(name)s] %(levelname)s: %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            Logger._loggers[name] = logger
        
        self.logger = Logger._loggers[name]
    
    @staticmethod
    def set_log_file(log_path: Path):
        """
        Set log file for file-based logging.
        
        Args:
            log_path: Path to log file
        """
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        Logger._log_file = log_path
        
        # Add file handler to all loggers
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        for logger in Logger._loggers.values():
            logger.addHandler(file_handler)
    
    def info(self, msg: str):
        """Log info message."""
        self.logger.info(msg)
    
    def debug(self, msg: str):
        """Log debug message."""
        self.logger.debug(msg)
    
    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)
    
    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)
