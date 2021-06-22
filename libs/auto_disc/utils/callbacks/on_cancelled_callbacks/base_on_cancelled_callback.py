from auto_disc.utils.callbacks import BaseCallback
class BaseOnCancelledCallback(BaseCallback):

    def __call__(self, experiment_id, seed, **kwargs):
        print("Experiment {} with seed {} cancelled".format(experiment_id, seed))