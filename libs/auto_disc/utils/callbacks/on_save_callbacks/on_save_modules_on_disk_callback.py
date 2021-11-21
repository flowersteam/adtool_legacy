from auto_disc.utils.callbacks.on_save_callbacks import BaseOnSaveCallback
import requests

import pickle
import matplotlib.pyplot as plt
from torch import Tensor
import json
import os

class OnSaveModulesOnDiskCallback(BaseOnSaveCallback):
    def __init__(self, folder_path):
        """
        brief:  init the callback 
        param:  base_url: string, url to save discoveries on database
        param:  to_save_outputs: string list, key of "SAVABLE_OUTPUTS" (parent's attribute) to select the outpouts who we want to save
        """
        self.folder_path = folder_path
                        
    def __call__(self, **kwargs):
        #TODO convert to_save_modules --> self.to_save_modules (like on_discovery_*_callback)
        to_save_modules = ["system","explorer","input_wrappers","output_representations","in_memory_db"]

        
        for save_module in to_save_modules:
            if isinstance(kwargs[save_module], list):
                to_pickle = []
                for element in kwargs[save_module]:
                    to_pickle.append(element.save())
            else:
                to_pickle = kwargs[save_module].save()

            folder = "{}{}/{}/{}".format(self.folder_path, kwargs["experiment_id"], kwargs["seed"], save_module)
            filename = "{}/idx_{}.pickle".format(folder, kwargs["run_idx"])

            if not os.path.isdir(folder):
                print(folder)
                os.makedirs(folder)
            with open(filename, 'wb') as out_file:
                pickle.dump(to_pickle, out_file)
            
        folder = "{}{}/{}/".format(self.folder_path, kwargs["experiment_id"], kwargs["seed"])
        self.logger.info("New modules saved : {} : {} :{}".format(folder, to_save_modules, kwargs["run_idx"]))