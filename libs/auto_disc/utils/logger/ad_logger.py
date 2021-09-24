import logging
import enum

class ADToolLogger():

    def __init__(self, specific_handler):
        # set handler to log in console
        self.logger = logging.getLogger("adtool_logger")
        stream_h = logging.StreamHandler()
        stream_h.setLevel(logging.NOTSET)
        logging.getLogger().setLevel(logging.NOTSET)
        self.formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        stream_h.setFormatter(self.formatter)
        self.logger.addHandler(stream_h)

        #dict of all level match with enum of log levels
        self.levels={
            "debug" : self.logger.debug,
            "info" : self.logger.info,
            "warning" : self.logger.warning,
            "error" : self.logger.error,
            "critical" : self.logger.critical
        }

        # set others handler
        if isinstance(specific_handler, list):
            for handler in specific_handler:
                self.logger.addHandler(handler)
        else:
            self.logger.addHandler(specific_handler)

    def __call__(self, level, msg, context):
        self.levels[level.value](msg, context)
    
    def __call__(self, level, msg):
        if isinstance(level, enum.Enum):
            self.levels[level.value](msg, {"experiment_id" : self.experiment_id,"checkpoint_id": self.checkpoint_id,"seed": self.seed})
        elif isinstance(level, str):
            self.levels[level](msg, {"experiment_id" : self.experiment_id,"checkpoint_id": self.checkpoint_id,"seed": self.seed})
        else:
            raise NotImplementedError()