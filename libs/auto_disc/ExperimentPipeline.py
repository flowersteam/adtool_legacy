class ExperimentPipeline():
    def __init__(self, system, explorer, input_wrappers=None, output_representations=None, action_policy=None, 
                 on_exploration_callbacks=None):
        self._system = system
        self._input_wrappers = input_wrappers
        self._output_representations = output_representations
        self._explorer = explorer
        self._explorer.initialize(self._input_wrappers[-1], self._output_representations[-1])
        self._action_policy = action_policy
        self._on_exploration_callbacks = on_exploration_callbacks

    def _process_output(self, output):
        for _output_representation in self._output_representations:
            output = _output_representation.map(output)
        return output

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

            o, r, d, i = self._system.reset(run_parameters), 0, None, False
            step_observations = [o]

            while not d:
                if self._action_policy is not None:
                    a = self._action_policy(o, r)
                else:
                    a = None

                o, r, d, i = self._system.step(a)
                step_observations.append(o)
                system_steps[-1] += 1

            raw_output = self._system.observe()
            output = self._process_output(raw_output)
            rendered_output = self._system.render()
                
            self._explorer.archive(raw_run_parameters, output)

            for callback in self._on_exploration_callbacks:
                callback(
                    pipeline=self,
                    run_idx=run_idx,
                    raw_run_parameters=raw_run_parameters,
                    run_parameters=run_parameters,
                    raw_output=raw_output,
                    output=output,
                    rendered_output=rendered_output,
                    step_observations=step_observations
                )

            run_idx += 1

            self._explorer.optimize() # TODO callbacks

        self._system.close()
 