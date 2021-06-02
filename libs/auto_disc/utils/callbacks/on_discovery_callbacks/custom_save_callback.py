from auto_disc.utils.callbacks import BaseCallback

import pickle
import matplotlib.pyplot as plt
import os

class CustomSaveCallback(BaseCallback):
    def __init__(self, folder_path):
        """
        init the callback with a path to a folder to save discoveries
        folder_path: string, path to folder
        """
        super().__init__()
        self.folder_path = folder_path

    def __call__(self, pipeline, run_idx, raw_run_parameters, run_parameters, raw_output, output, rendered_output, step_observations, **kwargs):
        """
        callback saves the 'rendered_output' (the discoveries) as a jpg and the input parameters 'run_parameters' in a pickle in folder
        define by 'self.folder_path'
        """
        if not os.path.isdir(self.folder_path+"output/"):
                print(self.folder_path)
                os.makedirs(self.folder_path+"output/")
        with open(self.folder_path+"output/"+str(run_idx)+"."+rendered_output[1], 'wb') as out:
            out.write(rendered_output[0].getbuffer())