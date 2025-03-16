import os
import logging
from pathlib import Path
from typing import Optional, Union

# Global dictionary to track configured loggers
# This prevents adding duplicate handlers
CONFIGURED_LOGGERS = set()

def setup_logger(
    name: str = "af_claseq",
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    propagate: bool = True,
    add_console_handler: bool = True,
) -> logging.Logger:
    """
    Set up and return a logger with the given name.
    
    Args:
        name: Logger name, ideally hierarchical (e.g., 'af_claseq.module_name')
        log_file: Path to log file (if None, no file handler will be added)
        level: Logging level
        propagate: Whether to propagate logs to parent loggers
        add_console_handler: Whether to add a console handler
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Only configure each logger once to avoid duplicate handlers
    if name in CONFIGURED_LOGGERS:
        return logger
    
    logger.setLevel(level)
    logger.propagate = propagate
    
    # Create a consistent formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add file handler if log_file is provided
    if log_file:
        # Ensure the log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if requested
    if add_console_handler:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Mark this logger as configured
    CONFIGURED_LOGGERS.add(name)
    
    return logger


def get_logger(module_name: str) -> logging.Logger:
    """
    Get a logger for a specific module that inherits from the main af_claseq logger.
    This should be used in module code instead of creating new loggers.
    
    Args:
        module_name: Name of the module (e.g., 'm_fold_sampling')
        
    Returns:
        A logger with appropriate name
    """
    logger_name = f"af_claseq.{module_name}"
    return logging.getLogger(logger_name)