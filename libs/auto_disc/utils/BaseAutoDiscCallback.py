class BaseAutoDiscCallback():
    def __call__(self, pipeline, run_idx, raw_run_parameters, run_parameters, raw_output, output, rendered_output, step_observations):
        raise NotImplementedError