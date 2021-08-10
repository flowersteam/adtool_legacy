from auto_disc.utils.callbacks.on_save_callbacks import BaseOnSaveCallback
import requests

import pickle
import json

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
        to_save_modules = ["system","explorer","input_wrappers","output_representations","in_memory_db"]

        files_to_save={}
        for module in to_save_modules:
            if isinstance(kwargs[module], list):
                to_pickle = []
                for element in kwargs[module]:
                    to_pickle.append(element.save())
            else:
                to_pickle = kwargs[module].save()

            module_to_save = pickle.dumps(to_pickle)
            files_to_save[module] = module_to_save
        
        response = requests.post(self.base_url + "/checkpoint_saves", 
                                    json={
                                        "checkpoint_id": kwargs["checkpoint_id"],
                                        "run_idx": kwargs["run_idx"],
                                        "seed": kwargs["seed"]
                                    }
                                )
        json_response = json.loads(response.text)
        module_id = json_response["ID"]

        requests.post(self.base_url + "/checkpoint_saves/" + module_id + "/files", 
                    files=files_to_save)