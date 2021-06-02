class BaseOnSavedCallback():

    def __call__(self, experiment_id, seed, **kwargs):
        print("Saving experiment {} with seed {}".format(experiment_id, seed))