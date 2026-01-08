import os
import logging
import csv

def setup_logging(log_dir="log", log_level=logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level (default: INFO)
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove existing handlers
    logger.handlers = []
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = os.path.join(log_dir, "training.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


class CSVLogger:
    """Logger for saving training metrics to CSV file."""
    def __init__(self, log_dir="log", filename="training_metrics.csv"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, filename)
        self.fieldnames = ['step', 'type', 'value', 'lr', 'norm', 'dt_ms', 'tokens_per_sec']
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Initialize CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
    
    def log(self, step, metric_type, value, lr=None, norm=None, dt_ms=None, tokens_per_sec=None):
        """Log a training metric to CSV"""
        row = {
            'step': step,
            'type': metric_type,
            'value': value,
            'lr': lr if lr is not None else '',
            'norm': norm if norm is not None else '',
            'dt_ms': dt_ms if dt_ms is not None else '',
            'tokens_per_sec': tokens_per_sec if tokens_per_sec is not None else ''
        }
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)