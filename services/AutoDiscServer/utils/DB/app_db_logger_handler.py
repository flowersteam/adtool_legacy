from logging import StreamHandler
import logging
import requests
import re

class AppDBLoggerHandler(StreamHandler):
    
    def __init__(self, base_url, experiment_id, get_checkpoint_id_from_seed_fn):
        StreamHandler.__init__(self)
        self.log_levels_id ={
            'NOTSET' : 1,
            'DEBUG' : 2,
            'INFO' : 3,
            'WARNING' : 4,
            'ERROR' : 5,
            'CRITICAL' : 6
        }
        self.experiment_id = experiment_id
        self.base_url = base_url # http://127.0.0.1:3000
        self.setLevel(logging.NOTSET)
        self.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
        self.get_checkpoint_id_from_seed_fn = get_checkpoint_id_from_seed_fn


    def emit(self, record):
        if self.experiment_id == record.args["experiment_id"]:
            if(len(re.sub("[^0-9]", "", record.levelname)) >0):
                levelname = int(re.sub("[^0-9]", "", record.levelname))
            else:
                levelname = self.log_levels_id[record.levelname]
            checkpoint_id = self.get_checkpoint_id_from_seed_fn(record.args["seed"])
            self.save(checkpoint_id, record.args["seed"], levelname, record.args["id"], record.msg)
            
    
    def save(self, checkpoint_id, seed, level_name, log_id, message):
        response = requests.post(self.base_url+"/logs", 
            json={
                    "experiment_id":self.experiment_id,
                    "checkpoint_id":checkpoint_id,
                    "name":log_id,
                    "seed":seed,
                    "log_level_id":level_name,
                    "error_message":"{} - {} \n".format(list(self.log_levels_id.keys())[list(self.log_levels_id.values()).index(level_name)], message)
                }
        )
