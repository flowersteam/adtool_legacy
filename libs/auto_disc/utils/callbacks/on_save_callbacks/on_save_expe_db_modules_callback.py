from auto_disc.utils.callbacks.on_save_callbacks import BaseOnSaveCallback
import requests

import pickle
import matplotlib.pyplot as plt
from torch import Tensor
import json
import pickle

class OnSaveExpeDBModulesCallback(BaseOnSaveCallback):
    def __init__(self, base_url):
        """
        brief:  init the callback 
        param:  base_url: string, url to save discoveries on database
        param:  to_save_outputs: string list, key of "SAVABLE_OUTPUTS" (parent's attribute) to select the outpouts who we want to save
        """
        self.base_url = base_url
                        
    def __call__(self, **kwargs):
        #TODO convert to_save_modules --> self.to_save_modules (like on_discovery_*_callback)
        to_save_modules = ["systems","explorers","input_wrappers","output_representations","db"]

        
        for save_module in to_save_modules:
            module_to_save = pickle.dumps(kwargs[save_module])
            save_to_file = {save_module+"_file": module_to_save}           
            response = requests.post(self.base_url + "/"+save_module, 
                                    json={
                                        "checkpoint_id": kwargs["checkpoint_id"],
                                        "run_idx": kwargs["run_idx"],
                                        "seed": kwargs["seed"]
                                    }
                                )
            json_response = json.loads(response.text)
            module_id = json_response["ID"]

            requests.post(self.base_url + "/" + save_module + "/" + module_id + "/files", 
                        files=save_to_file)