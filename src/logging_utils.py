import logging
import sys

def get_logger(name: str = "geospatial_pipeline", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a logger with standard formatting."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Output to stdout
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger
