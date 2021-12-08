from auto_disc.utils.callbacks.on_discovery_callbacks import BaseOnDiscoveryCallback
import requests

import pickle
import matplotlib.pyplot as plt
from torch import Tensor
import json
import pickle

class OnDiscoveryExpeDBSaveCallback(BaseOnDiscoveryCallback):
    def __init__(self, base_url, to_save_outputs):
        """
        brief:  init the callback 
        param:  base_url: string, url to save discoveries on database
        param:  to_save_outputs: string list, key of "SAVABLE_OUTPUTS" (parent's attribute) to select the outpouts who we want to save
        """
        super().__init__(to_save_outputs)
        self.to_save_outputs.extend(["run_idx", "experiment_id", "seed"])
        self.base_url = base_url

    def _serialize_autodisc_space(self, space):
        """
        brief:  transform space into serializable object for json (tensor to list)
        param:  space: one of possible outputs we want save
        """
        serialized_space = {}
        if isinstance(space, Tensor):
            serialized_space = space.tolist()
        elif isinstance(space, list):
            for i in range(len(space)):
                space[i] = self._serialize_autodisc_space(space[i])
            serialized_space =space
        elif isinstance(space, dict):
            for key in space:
                if isinstance(space[key], Tensor):
                    serialized_space[key] = space[key].tolist()
                else:
                    serialized_space[key] = space[key]
        else:
            serialized_space = space
        return serialized_space

    def __call__(self, **kwargs):
        """
        brief:      callback saves the discoveries outputs we want to save on database.
        comment:    always saved : run_idx(json), experiment_id(json)
                    saved if key in self.to_save_outputs: raw_run_parameters(json)
                                                        run_parameters,(json)
                                                        raw_output(file),
                                                        output(json),
                                                        rendered_output(file),
                                                        step_observations(file)
        """
        saves={}
        save_to_file={}

        for save_item in self.to_save_outputs:
            if save_item == "step_observations":
                kwargs[save_item] = self._serialize_autodisc_space(kwargs[save_item])

            if save_item == "raw_output" or save_item == "step_observations":
                save_to_file[save_item] = ('{}_{}_{}'.format(save_item, kwargs["experiment_id"], kwargs["run_idx"]), pickle.dumps(kwargs[save_item]), 'application/json')
            elif save_item == "rendered_output":
                filename = "exp_{}_idx_{}".format(kwargs["experiment_id"], kwargs["run_idx"])
                filename=filename+"."+kwargs["rendered_output"][1]
                save_to_file["rendered_output"] = (filename, kwargs["rendered_output"][0].getbuffer())
            else:
                saves[save_item] = self._serialize_autodisc_space(kwargs[save_item])
        
        response = requests.post(self.base_url + "/discoveries", 
                                    json=saves
                                )
        json_response = json.loads(response.text)
        discovery_id = json_response["ID"]

        requests.post(self.base_url + "/discoveries/" + discovery_id + "/files", 
                    files=save_to_file)