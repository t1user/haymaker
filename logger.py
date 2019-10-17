import sys
from datetime import datetime
from logbook import Logger, StreamHandler, FileHandler, set_datetime_format


def logger(name):
    set_datetime_format('local')
    StreamHandler(sys.stdout, bubble=True).push_application()
    filename = __file__.split('/')[-1][:-3]
    FileHandler(
        f'logs/{name}_{datetime.today().strftime("%Y-%m-%d_%H-%M")}.log',
        bubble=True, delay=True).push_application()
    return Logger(name)
