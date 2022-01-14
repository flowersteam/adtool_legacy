from auto_disc.utils.callbacks import BaseCallback
class BaseOnDiscoveryCallback(BaseCallback):

    SAVABLE_OUTPUTS = ["raw_run_parameters",
                        "run_parameters", 
                        "raw_output", 
                        "output",
                        "rendered_output",
                        "step_observations"]

    def __init__(self, to_save_outputs, **kwargs):
        super().__init__(**kwargs)
        self.to_save_outputs = to_save_outputs


    def __call__(self, experiment_id, seed, **kwargs):
        print("New discovery for experiment {} with seed {}".format(experiment_id, seed))