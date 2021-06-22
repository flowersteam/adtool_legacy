from auto_disc.utils.callbacks import BaseCallback
class BaseOnSaveFinishedCallback(BaseCallback):

    def __call__(self, experiment_id, seed, **kwargs):
        print("Experiment {} with seed {} saved".format(experiment_id, seed))