class BaseOnErrorCallback():

    def __call__(self, experiment_id, seed, **kwargs):
        print("Error for experiment {} with seed {}".format(experiment_id, seed))