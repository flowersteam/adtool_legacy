from auto_disc.utils.callbacks import BaseCallback
class BaseOnFinishedCallback(BaseCallback):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __call__(self, experiment_id, seed, **kwargs):
        print("Experiment {} with seed {} finished".format(experiment_id, seed))