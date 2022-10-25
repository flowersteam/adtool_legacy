from auto_disc.utils.callbacks.interact_callbacks import BaseInteractCallback

class SaveExpeDBInteractCallback(BaseInteractCallback):
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
        if dict_info != None:
            dict_info.update({"idx": config["idx"]})
        self.interactMethod(data, config["seed"], dict_info)
        config["idx"] += 1