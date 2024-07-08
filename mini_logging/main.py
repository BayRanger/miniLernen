# myapp.py
import logging
import mylib
from datetime import datetime

logger = logging.getLogger(__name__)

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Set the log file name with the timestamp
    log_filename = f'./server_app_{timestamp}.log'

    logging.basicConfig(filename=log_filename,  level=logging.DEBUG,\
    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    logger.info('Started')
    logger.debug("cliche")
    mylib.do_something()
    logger.info('Finished')

if __name__ == '__main__':
    main()
