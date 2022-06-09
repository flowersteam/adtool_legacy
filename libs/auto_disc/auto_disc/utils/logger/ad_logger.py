import logging
from random import seed


class AutoDiscLogger(logging.Logger):

    def __init__(self, experiment_id, seed, handlers):
        self.__experiment_id = experiment_id
        self._seed = seed
        self._shared_logger = logging.getLogger("ad_tool_logger")
        self.__index = 0
        # create handler
        # add handler
        if not any(handler.experiment_id == self.__experiment_id for handler in self._shared_logger.handlers):
            stream_h = logging.StreamHandler()
            stream_h.setLevel(logging.NOTSET)
            logging.getLogger().setLevel(logging.NOTSET)
            formatter = logging.Formatter('%(name)s - %(levelname)s - SEED %(seed)s - LOG_ID %(id)s - %(message)s')
            stream_h.setFormatter(formatter)
            stream_h.experiment_id = experiment_id
            self._shared_logger.addHandler(stream_h)
            for handler in handlers:
                self._shared_logger.addHandler(handler)
            self._shared_logger.addFilter(ContextFilter())

    def debug(self, *args):
        self.__index += 1
        self._shared_logger.debug(*args, {"experiment_id": self.__experiment_id,"seed": self._seed, "id":"{}_{}_{}".format(self.__experiment_id, self._seed, self.__index)})

    def info(self, *args):
        self.__index += 1
        self._shared_logger.info(*args, {"experiment_id": self.__experiment_id,"seed": self._seed, "id":"{}_{}_{}".format(self.__experiment_id, self._seed, self.__index)})
    
    def warning(self, *args):
        self.__index += 1
        self._shared_logger.warning(*args, {"experiment_id": self.__experiment_id,"seed": self._seed, "id":"{}_{}_{}".format(self.__experiment_id, self._seed, self.__index)})

    def error(self, *args):
        self.__index += 1
        self._shared_logger.error(*args, {"experiment_id": self.__experiment_id,"seed": self._seed, "id":"{}_{}_{}".format(self.__experiment_id, self._seed, self.__index)})

    def critical(self, *args):
        self.__index += 1
        self._shared_logger.critical(*args, {"experiment_id": self.__experiment_id,"seed": self._seed, "id":"{}_{}_{}".format(self.__experiment_id, self._seed, self.__index)})

class ContextFilter(logging.Filter):
    """
    This is a filter which injects contextual information into the log.
    """
    def filter(self, record):
        record.seed = record.args["seed"]
        record.id = record.args["id"]
        return True
