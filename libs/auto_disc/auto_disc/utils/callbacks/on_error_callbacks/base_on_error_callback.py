from auto_disc.utils.callbacks import BaseCallback
class BaseOnErrorCallback(BaseCallback):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __call__(self, experiment_id, seed, **kwargs):
        print("Error for experiment {} with seed {}".format(experiment_id, seed))