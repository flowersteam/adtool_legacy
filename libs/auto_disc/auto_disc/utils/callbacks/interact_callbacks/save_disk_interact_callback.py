import os
import pickle

from auto_disc.utils.callbacks.interact_callbacks import BaseInteractCallback

class SaveDiskInteractCallback(BaseInteractCallback):
    '''
    Base class for cancelled callbacks used by the experiment pipelines when the experiment was cancelled.
    '''

    def __init__(self, **kwargs) -> None:
        """
            initialize attributes common to all cancelled callbacks

            Args:
                kwargs: some usefull args (e.g. experiment_id...)
        """
        super().__init__(**kwargs)
        self.folder_path = kwargs["folder_path"]

    def __call__(self, data, config, dict_info=None, **kwargs) -> None:
        """
            The function to call to effectively raise cancelled callback.
            Inform the user that the experience is in canceled status
            Args:
                experiment_id: current experiment id
                seed: current seed number
                kwargs: somme usefull parameters
        """
        data["custom dict"] = "new beautifull dict"
        kwargs["data"] = data
        kwargs["dict_info"] = dict_info

        item_to_save = ["data", "dict_info"]
        for save_item in item_to_save:
            folder = "{}{}/{}/{}".format(self.folder_path, config["experiment_id"], config["seed"], save_item)
            filename = "{}/idx_{}.pickle".format(folder, config["idx"])
            
            if not os.path.isdir(folder):
                os.makedirs(folder)
            with open(filename, 'wb') as out_file:
                pickle.dump(kwargs[save_item], out_file)
        print("Saved in '{}' discovery {} for experiment {}".format(self.folder_path, config["idx"], config["experiment_id"]))
        folder = "{}{}/{}/".format(self.folder_path, config["experiment_id"], config["seed"])
        self.logger.info("New data saved : {} : {} :{}".format(folder, item_to_save, config["idx"]))
        config["idx"] += 1
        # self.interactMethod(data, self.seed, dict_info)