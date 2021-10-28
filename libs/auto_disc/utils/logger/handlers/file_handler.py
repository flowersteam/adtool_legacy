import logging
from logging import FileHandler

class SetFileHandler(FileHandler):

    def __init__(self, file_log_path, formatter):
        FileHandler.__init__(self, file_log_path)
        self.setLevel(logging.NOTSET)
        self.setFormatter(formatter)