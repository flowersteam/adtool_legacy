class BaseCallback():
    """
    Base class for callbacks used by the experiment pipelines when progress is made (e.g. new dicovery, explorer's optimization).
    """

    def __init__(self, logger=None, **kwargs) -> None:
        """
            Initialize attributes common to all auto_disc callbacks

            #### Args:
            - logger: the logger which will make it possible to keep information on the progress of an experiment in the database or on files
        """
        self.logger = logger

    def __call__(self, experiment_id: int, seed: int, **kwargs) -> None:
        """
            The call which happens when callback is raised

            #### Args:
            - experiment_id: current experiment id
            - seed: current seed number
            - kwargs: somme useful parameters
        """
        raise NotImplementedError
