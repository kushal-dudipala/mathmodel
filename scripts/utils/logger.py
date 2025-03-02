import logging
import os

log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]',
    datefmt='%d-%b-%y %H:%M:%S',
    filename=os.path.join(log_dir, 'pipeline.log'),
    filemode='w'
    )

logger = logging.getLogger(__name__)

def get_logger():
    return logger