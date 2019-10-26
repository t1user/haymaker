import sys
from datetime import datetime
from logbook import Logger, StreamHandler, FileHandler, set_datetime_format, DEBUG


def logger(name, stream_level=DEBUG, file_level=DEBUG):
    set_datetime_format('local')
    StreamHandler(sys.stdout, level=stream_level,
                  bubble=True).push_application()
    filename = __file__.split('/')[-1][:-3]
    FileHandler(
        f'logs/{name}_{datetime.today().strftime("%Y-%m-%d_%H-%M")}.log',
        bubble=True, level=file_level, delay=True).push_application()
    return Logger(name)
