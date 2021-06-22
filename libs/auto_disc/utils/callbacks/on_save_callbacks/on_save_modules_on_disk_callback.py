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
        to_save_modules = ["systems","explorers","input_wrappers","output_representations"]

        
        for save_module in to_save_modules:
            folder = self.folder_path+save_module
            filename = "{}/exp_{}_idx_{}.pickle".format(folder, kwargs["experiment_id"], kwargs["run_idx"])

            if not os.path.isdir(folder):
                print(folder)
                os.makedirs(folder)
            with open(filename, 'wb') as out_file:
                pickle.dump(kwargs[save_module], out_file)