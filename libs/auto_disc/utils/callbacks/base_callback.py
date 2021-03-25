class BaseCallback():
    '''
    Base class for callbacks used by the experiment pipelines when progress is made (e.g. new dicovery, explorer's optimization).
    '''
    def __call__(self, pipeline, run_idx, raw_run_parameters, run_parameters, raw_output, output, rendered_output, step_observations):
        raise NotImplementedError