from auto_disc.utils.callbacks.on_discovery_callbacks import BaseOnDiscoveryCallback

import pickle
import os

class OnDiscoverySaveCallbackOnDisk(BaseOnDiscoveryCallback):
    def __init__(self, folder_path, to_save_outputs):
        """
        brief:  init the callback with a path to a folder to save discoveries
        param:  folder_path: string, path to folder
        param:  to_save_outputs: string list, key of "SAVABLE_OUTPUTS" (parent's attribute) to select the outpouts who we want to save
        """
        super().__init__(to_save_outputs)
        self.folder_path = folder_path

    def __call__(self, **kwargs):
        """
        brief:  callback saves the 'rendered_output' (the discoveries) as a jpg and the input parameters 'run_parameters' in a pickle in folder
                define by 'self.folder_path'

        comment:callback save the discoveries outputs we want to save on disk.
                always saved : run_idx(pickle), checkpoint_id(pickle)
                saved if key in self.to_save_outputs: raw_run_parameters(pickle)
                                                    run_parameters,(pickle)
                                                    raw_output(pickle),
                                                    output(pickle),
                                                    rendered_output(changes according to the render function of the current system),
                                                    step_observations(pickle)
        """
        for save_item in self.to_save_outputs:
            folder = self.folder_path+save_item
            if save_item != "rendered_output":
                filename = "{}/exp_{}_idx_{}.pickle".format(folder, kwargs["experiment_id"], kwargs["run_idx"])
            else:
                filename = "{}/exp_{}_idx_{}.{}".format(folder, kwargs["experiment_id"], kwargs["run_idx"], kwargs["rendered_output"][1])
            
            if not os.path.isdir(folder):
                print(folder)
                os.makedirs(folder)
            with open(filename, 'wb') as out_file:
                if save_item != "rendered_output":
                    pickle.dump(kwargs[save_item], out_file)
                else:
                    out_file.write(kwargs["rendered_output"][0].getbuffer())