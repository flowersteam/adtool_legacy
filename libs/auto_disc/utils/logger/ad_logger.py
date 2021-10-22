import logging
from random import seed


class AutoDiscLogg(logging.Logger):

    def __init__(self, experiment_id, seed, checkpoint_id, specific_handler):
        self._experiment_id = experiment_id
        self._seed = seed
        self._checkpoint_id = checkpoint_id
        if "ad_tool_logger" not in logging.root.manager.loggerDict:
            self._shared_logger = logging.getLogger("ad_tool_logger")
            # create handler
            # add handler
            stream_h = logging.StreamHandler()
            stream_h.setLevel(logging.NOTSET)
            logging.getLogger().setLevel(logging.NOTSET)
            formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            stream_h.setFormatter(formatter)
            self._shared_logger.addHandler(stream_h)
            if isinstance(specific_handler, list):
                for handler in specific_handler:
                    self._shared_logger.addHandler(handler)
            else:
                self._shared_logger.addHandler(specific_handler)
        else:
            self._shared_logger = logging.getLogger("ad_tool_logger")
    
    @property 
    def seed(self):
        return self._seed
    
    @seed.setter
    def seed(self, seed):
        self._seed = seed

    @property 
    def checkpoint_id(self):
        return self._checkpoint_id
    
    @checkpoint_id.setter 
    def checkpoint_id(self, checkpoint_id ):
        self._checkpoint_id = checkpoint_id

    def debug(self, *args):
        self._shared_logger.debug(*args, {"experiment_id": self.experiment_id, "checkpoint_id": self._checkpoint_id,"seed": self._seed})

    def info(self, *args):
        self._shared_logger.info(*args, {"experiment_id": self.experiment_id, "checkpoint_id": self._checkpoint_id,"seed": self._seed})
    
    def warning(self, *args):
        self._shared_logger.warning(*args, {"experiment_id": self.experiment_id, "checkpoint_id": self._checkpoint_id,"seed": self._seed})

    def error(self, *args):
        self._shared_logger.error(*args, {"experiment_id": self.experiment_id, "checkpoint_id": self._checkpoint_id,"seed": self._seed})

    def critical(self, *args):
        self._shared_logger.critical(*args, {"experiment_id": self.experiment_id, "checkpoint_id": self._checkpoint_id,"seed": self._seed})