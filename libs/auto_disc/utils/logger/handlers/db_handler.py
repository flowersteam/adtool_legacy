from logging import StreamHandler
import logging
import requests
import re

class SetDBHandler(StreamHandler):
    
    def __init__(self, formatter, base_url):
        StreamHandler.__init__(self)
        self.log_levels_id ={
            'NOTSET' : 1,
            'DEBUG' : 2,
            'INFO' : 3,
            'WARNING' : 4,
            'ERROR' : 5,
            'CRITICAL' : 6
        }
        self.base_url = base_url # http://127.0.0.1:3000
        self.setLevel(logging.NOTSET)
        self.setFormatter(formatter)


    def emit(self, record):
        if(len(re.sub("[^0-9]", "", record.levelname)) >0):
            levelname = int(re.sub("[^0-9]", "", record.levelname))
        else:
            levelname = self.log_levels_id[record.levelname]
        requests.post(self.base_url+"/logs", 
                                    json={
                                            "experiment_id":record.args["experiment_id"],
                                            "checkpoint_id":record.args["checkpoint_id"],
                                            "seed":record.args["seed"],
                                            "log_level_id":levelname,
                                            "error_message":"{} - {} - {} \n".format(record.name, record.levelname, record.msg)
                                        }
                                )