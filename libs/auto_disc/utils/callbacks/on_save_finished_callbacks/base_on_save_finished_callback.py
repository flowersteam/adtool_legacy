class BaseOnSaveFinishedCallback():

    def __call__(self, experiment_id, seed, **kwargs):
        print("Experiment {} with seed {} saved".format(experiment_id, seed))