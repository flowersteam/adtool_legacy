class BaseCallback():
    '''
    Base class for callbacks used by the experiment pipelines when progress is made (e.g. new dicovery, explorer's optimization).
    '''
    def __init__(self, logger=None, **kwargs) -> None:
        self.logger = logger
        
    def __call__(self, experiment_id, seed, **kwargs):
        raise NotImplementedError