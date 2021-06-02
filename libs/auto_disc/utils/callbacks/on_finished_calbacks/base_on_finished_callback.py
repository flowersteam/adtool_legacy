class BaseOnFinishedCallback():

    def __call__(self, experiment_id, seed, **kwargs):
        print("Experiment {} with seed {} finished".format(experiment_id, seed))