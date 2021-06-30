from auto_disc.utils.callbacks import BaseCallback
class BaseOnSaveCallback(BaseCallback):

    def __call__(self, experiment_id, seed, **kwargs):
        print("Saving experiment {} with seed {}".format(experiment_id, seed))