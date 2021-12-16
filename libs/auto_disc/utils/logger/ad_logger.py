import logging
from random import seed


class AutoDiscLogger(logging.Logger):

    def __init__(self, experiment_id, seed, specific_handler):
        self.__experiment_id = experiment_id
        self._seed = seed
        self._shared_logger = logging.getLogger("ad_tool_logger")
        # create handler
        # add handler
        if not any(handler.experiment_id == self.__experiment_id for handler in self._shared_logger.handlers):
            stream_h = logging.StreamHandler()
            stream_h.setLevel(logging.NOTSET)
            logging.getLogger().setLevel(logging.NOTSET)
            formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            stream_h.setFormatter(formatter)
            stream_h.experiment_id = experiment_id
            self._shared_logger.addHandler(stream_h)
            if isinstance(specific_handler, list):
                for handler in specific_handler:
                    self._shared_logger.addHandler(handler)
            else:
                self._shared_logger.addHandler(specific_handler)

    def debug(self, *args):
        self._shared_logger.debug(*args, {"experiment_id": self.__experiment_id,"seed": self._seed})

    def info(self, *args):
        self._shared_logger.info(*args, {"experiment_id": self.__experiment_id,"seed": self._seed})
    
    def warning(self, *args):
        self._shared_logger.warning(*args, {"experiment_id": self.__experiment_id,"seed": self._seed})

    def error(self, *args):
        self._shared_logger.error(*args, {"experiment_id": self.__experiment_id,"seed": self._seed})

    def critical(self, *args):
        self._shared_logger.critical(*args, {"experiment_id": self.__experiment_id,"seed": self._seed})