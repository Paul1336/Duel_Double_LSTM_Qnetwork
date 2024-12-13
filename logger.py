import logging
import os
from logging.handlers import TimedRotatingFileHandler

LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOG_FORMAT = '%(asctime)s [%(levelname)s] [%(name)s]: %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
time_formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
log_path = os.path.join(LOG_DIR, "main.log")

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

time_handler = TimedRotatingFileHandler(
    filename=log_path,
    when='midnight',
    interval=1,
    backupCount=31,
    encoding='utf-8',
    utc=False
)
time_handler.setFormatter(time_formatter)
time_handler.setLevel(logging.DEBUG)
root_logger.addHandler(time_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(time_formatter)
root_logger.addHandler(console_handler)

def get_logger(name: str = None, log_file:str = None):#with given path, the logger works independent
    return logging.getLogger(name) 
    #if log_file == None:
    #    return logging.getLogger(name) 
    #else:
    #    logger = logging.getLogger(name)
    #    logger.propagate = False
    #    handler = logging.FileHandler(
    #        filename=os.path.join(LOG_DIR, log_file),
    #        mode='a',  # Append mode
    #        encoding='utf-8'
    #    )
    #    handler.setFormatter(time_formatter)
    #    handler.setLevel(logging.DEBUG)
    #    logger.addHandler(handler)
    #    return logger