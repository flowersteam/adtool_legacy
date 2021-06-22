from auto_disc.utils.callbacks import BaseCallback
class BaseOnFinishedCallback(BaseCallback):

    def __call__(self, experiment_id, seed, **kwargs):
        print("Experiment {} with seed {} finished".format(experiment_id, seed))