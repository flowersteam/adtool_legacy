class ExperimentPipeline():
    def __init__(self, system, explorer, input_wrappers=None, output_representations=None, action_policy=None, 
                 on_exploration_callbacks=None):
        self._system = system
        self._explorer = explorer
        self._input_wrappers = input_wrappers
        self._output_representations = output_representations
        self._action_policy = action_policy
        self._on_exploration_callbacks = on_exploration_callbacks

    def _process_observations(self, observations):
        for _output_representation in self._output_representations:
            observations = _output_representation.map(observations)
        return observations

    def _process_run_parameters(self, run_parameters):
        for input_wrapper in self._input_wrappers:
            run_parameters = input_wrapper.map(run_parameters)
        return run_parameters

    def run(self, n_exploration_runs):
        run_idx = 0
        system_steps = [0]

        while run_idx < n_exploration_runs:
            raw_run_parameters = self._explorer.emit()
            run_parameters = self._process_run_parameters(raw_run_parameters)

            o, r, i, d = self._system.reset(run_parameters), 0, None, False
            raw_observations = [o]

            while not d:
                if self._action_policy is not None:
                    a = self._action_policy(o, r)
                else:
                    a = None

                o, r, i, d = self._system.step(a)
                raw_observations.append(o)
                system_steps[-1] += 1

            observations = self._process_observations(raw_observations)
                
            self._explorer.archive(raw_run_parameters, observations)

            for callback in self._on_exploration_callbacks:
                callback(run_parameters, raw_observations)
            run_idx += 1

            self._explorer.optimize() # TODO callbacks

        self._system.close()
                
 