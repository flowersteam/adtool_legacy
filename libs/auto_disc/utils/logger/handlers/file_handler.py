import logging
from logging import FileHandler

class SetFileHandler(FileHandler):

    def __init__(self, folder_log_path, experiment_id):
        FileHandler.__init__(self, "{}exp_{}.log".format(folder_log_path, experiment_id))
        self.setLevel(logging.NOTSET)
        self.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
        self.experiment_id = experiment_id