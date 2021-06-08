class BaseOnCancelledCallback():

    def __call__(self, experiment_id, seed, **kwargs):
        print("Experiment {} with seed {} cancelled".format(experiment_id, seed))