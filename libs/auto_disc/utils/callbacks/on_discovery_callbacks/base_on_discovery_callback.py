from auto_disc.utils.callbacks import BaseCallback
class BaseOnDiscoveryCallback(BaseCallback):

    SAVABLE_OUTPUTS = {
        "Parameters sent by the explorer before input wrappers": "raw_run_parameters",
        "Parameters sent by the explorer after input wrappers": "run_parameters",
        "Raw system output": "raw_output",
        "Representation of system output": "output",
        "Rendered system output": "rendered_output",
        "System's step observations": "step_observations"
    }

    def __init__(self, to_save_outputs):
        self.to_save_outputs = [v for k, v in self.SAVABLE_OUTPUTS.items() if k in to_save_outputs]
        self.to_save_outputs.extend(["run_idx", "checkpoint_id", "seed"])


    def __call__(self, experiment_id, seed, **kwargs):
        print("New discovery for experiment {} with seed {}".format(experiment_id, seed))