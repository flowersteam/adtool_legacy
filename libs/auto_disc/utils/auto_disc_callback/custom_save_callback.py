from libs.auto_disc.utils import BaseAutoDiscCallback

import pickle
import matplotlib.pyplot as plt

class CustomSaveCallback(BaseAutoDiscCallback):
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        #self._output_representations = output_representations

    def __call__(self, pipeline, run_idx, raw_run_parameters, run_parameters, raw_output, output, rendered_output, step_observations):
        # save_folder="/home/mperie/project/save_callback/"
        plt.imsave(self.folder_path+"output/"+str(run_idx)+".jpg", rendered_output, format='jpg')
        pickle_file=self.folder_path+"run_parameters/"+str(run_idx)
        with open(pickle_file, 'wb') as f1:
            pickle.dump(rendered_output, f1)

        with open(pickle_file, 'wb') as f1:
            pickle.dump(run_parameters, f1)