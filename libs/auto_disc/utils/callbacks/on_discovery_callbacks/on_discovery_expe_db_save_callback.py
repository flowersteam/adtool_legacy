from auto_disc.utils.callbacks import BaseCallback
import requests

import pickle
import matplotlib.pyplot as plt

import os
from torch import Tensor
import json
import pickle

class OnDiscoveryExpeDBSaveCallback(BaseCallback):
    def __init__(self, base_url, experiment_id):
        """
        init the callback with a path to a folder to save discoveries
        """
        super().__init__()
        self.base_url = base_url
        self.experiment_id = experiment_id

    def _serialize_autodisc_space(self, space):
        serialized_space = {}
        for key in space:
            if isinstance(space[key], Tensor):
                serialized_space[key] = space[key].tolist()
            else:
                serialized_space[key] = space[key]
        return serialized_space

    def __call__(self, pipeline, run_idx, raw_run_parameters, run_parameters, raw_output, output, rendered_output, step_observations):
        """
        callback saves the 'rendered_output' (the discoveries) as a jpg and the input parameters 'run_parameters' in a pickle in folder
        define by 'self.folder_path'
        """
        folder = "./tmp_output_results"
        if not os.path.exists(folder):
            os.makedirs(folder)

        filename = "{}/exp_{}_idx_{}".format(folder, self.experiment_id, run_idx)

        if isinstance(rendered_output, list):
            filename += ".gif"
            rendered_output[0].save(filename, save_all=True, append_images=rendered_output[1:], duration=100, loop=0)
        else:
            filename += ".jpg"
            plt.imsave(filename, rendered_output, format='jpg')

        response = requests.post(self.base_url + "/discoveries", 
                                    json={
                                        "checkpoint_id": "0",
                                        "parameters":{
                                            "raw": self._serialize_autodisc_space(raw_run_parameters),
                                            "wrapped": self._serialize_autodisc_space(run_parameters)
                                        },
                                        "output":{
                                            "representation": output.tolist()
                                        }
                                    })
        json_response = json.loads(response.text)
        discovery_id = json_response["ID"]

        with open(filename, 'rb') as rendered_output_file:
            requests.post(self.base_url + "/discoveries/" + discovery_id + "/files", 
                        files={
                            "raw_output": ('raw_output_{}_{}'.format(self.experiment_id, run_idx), pickle.dumps(raw_output), 'application/json'),
                            "rendered_output": rendered_output_file
                        })
        os.remove(filename) # Remove temporary stored file