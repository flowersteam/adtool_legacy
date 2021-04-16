from auto_disc.utils.callbacks import BaseCallback

import pickle
import matplotlib.pyplot as plt

class CustomSaveCallback(BaseCallback):
    def __init__(self, folder_path):
        """
        init the callback with a path to a folder to save discoveries
        folder_path: string, path to folder
        """
        super().__init__()
        self.folder_path = folder_path

    def __call__(self, pipeline, run_idx, raw_run_parameters, run_parameters, raw_output, output, rendered_output, step_observations):
        """
        callback saves the 'rendered_output' (the discoveries) as a jpg and the input parameters 'run_parameters' in a pickle in folder
        define by 'self.folder_path'
        """   
        if isinstance(rendered_output, list):
            rendered_output[0].save(self.folder_path+"output/"+str(run_idx)+'.gif', save_all=True, append_images=rendered_output[1:], duration=100, loop=0)
        else:
            plt.imsave(self.folder_path+"output/"+str(run_idx)+".jpg", rendered_output, format='jpg')
        pickle_file=self.folder_path+"run_parameters/"+str(run_idx)

        

        with open(pickle_file, 'wb') as f1:
            pickle.dump(run_parameters, f1)