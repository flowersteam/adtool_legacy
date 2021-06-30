from AutoDiscServer import experiments
from auto_disc.output_representations.generic import DummyOutputRepresentation
from auto_disc.input_wrappers.generic import DummyInputWrapper
import torch
import traceback

class CancellationToken:
    def __init__(self):
        self._token = False
    def get(self):
        return self._token
    def trigger(self):
        self._token = True

class ExperimentPipeline():
    '''
    Pipeline of an automated discovery experiment.
    An experiment is at least constitued of a system and an explorer. Additionally, input wrappers and output representations can be added (multiple can be stacked).
    When the system requires an action at each timestep, an `action_policy` must be provided.
    In order to monitor the experiment, you must provide `on_exploration_classbacks`, which will be called every time a discovery has been made. Please provide callbacks overriding the `libs.auto_disc.utils.BaseAutoDiscCallback`.
    '''
    def __init__(self, experiment_id, seed, checkpoint_id, system, explorer, input_wrappers=None, output_representations=None, action_policy=None, 
                 save_frequency=100, on_discovery_callbacks=[], on_save_finished_callbacks=[], on_finished_callbacks=[], on_cancelled_callbacks=[],
                  on_save_callbacks=[], on_error_callbacks=[]):
        self.experiment_id = experiment_id
        self.seed = seed
        self.checkpoint_id = checkpoint_id
        self.save_frequency = save_frequency

        ### SYSTEM ###
        self._system = system
        
        ### OUTPUT REPRESENTATIONS ###
        if output_representations is not None and len(output_representations) > 0:
            self._output_representations = output_representations
        else:
            self._output_representations = [DummyOutputRepresentation()]

        for i in range(len(self._output_representations)):
            if i == 0:
                self._output_representations[i].initialize(input_space=self._system.output_space)
            else:
                self._output_representations[i].initialize(input_space=self._output_representations[i-1].output_space)

        ### INPUT WRAPPERS ###
        if input_wrappers is not None and len(input_wrappers) > 0:
            self._input_wrappers = input_wrappers
        else:
            self._input_wrappers = [DummyInputWrapper()]
            
        for i in reversed(range(len(self._input_wrappers))):
            if i == len(self._input_wrappers) - 1:
                self._input_wrappers[i].initialize(output_space=self._system.input_space)
            else:
                self._input_wrappers[i].initialize(output_space=self._input_wrappers[i+1].input_space)

        ### EXPLORER ###
        self._explorer = explorer
        self._explorer.initialize(input_space=self._output_representations[-1].output_space,
                                  output_space=self._input_wrappers[0].input_space, 
                                  input_distance_fn=self._output_representations[-1].calc_distance)
        
        self._action_policy = action_policy
        self._on_discovery_callbacks = on_discovery_callbacks
        self._on_save_finished_callbacks = on_save_finished_callbacks
        self._on_finished_callbacks = on_finished_callbacks
        self._on_cancelled_callbacks = on_cancelled_callbacks
        self._on_error_callbacks = on_error_callbacks
        self._on_save_callbacks = on_save_callbacks
        self.cancellation_token = CancellationToken()

    def _process_output(self, output):
        for output_representation in self._output_representations:
            output = output_representation.map(output)
        return output

    def _process_run_parameters(self, run_parameters):
        for input_wrapper in self._input_wrappers:
            run_parameters = input_wrapper.map(run_parameters)
        return run_parameters

    def _raise_callbacks(self, callbacks, **kwargs):
        callbacks_res = {}
        for callback in callbacks:
            callback_res = callback(
                                    pipeline=self,
                                    **kwargs
                                    )
            if callback_res != None:
                callbacks_res.update(callback_res)
        return callbacks_res

    def run(self, n_exploration_runs):
        '''
        Launches the experiment for `n_exploration_runs` explorations.
        '''
        run_idx = 0
        system_steps = [0]
        try:
            while run_idx < n_exploration_runs:
                if self.cancellation_token.get():
                    break

                raw_run_parameters = self._explorer.emit()
                with torch.no_grad():
                    run_parameters = self._process_run_parameters(raw_run_parameters)

                o, r, d, i = self._system.reset(run_parameters), 0, None, False
                step_observations = [o]

                while not d:
                    if self._action_policy is not None:
                        a = self._action_policy(o, r)
                    else:
                        a = None

                    with torch.no_grad():
                        o, r, d, i = self._system.step(a)
                    step_observations.append(o)
                    system_steps[-1] += 1
                    
                with torch.no_grad():
                    raw_output = self._system.observe()
                    output = self._process_output(raw_output)
                    rendered_output = self._system.render()
                        
                self._explorer.archive(raw_run_parameters, output)

                self._raise_callbacks(
                    self._on_discovery_callbacks,
                    run_idx=run_idx,
                    seed=self.seed,
                    raw_run_parameters=raw_run_parameters,
                    run_parameters=run_parameters,
                    raw_output=raw_output,
                    output=output,
                    rendered_output=rendered_output,
                    step_observations=step_observations,
                    experiment_id=self.experiment_id,
                    checkpoint_id = self.checkpoint_id
                )

                run_idx += 1

                self._explorer.optimize() # TODO callbacks

                if run_idx % self.save_frequency == 0:
                    self._raise_callbacks(
                        self._on_save_callbacks,
                        run_idx=run_idx,
                        seed=self.seed,
                        experiment_id=self.experiment_id,
                        checkpoint_id = self.checkpoint_id,
                        systems=self._system,
                        explorers=self._explorer,
                        input_wrappers=self._input_wrappers,
                        output_representations=self._output_representations
                    )
                    callbacks_res = self._raise_callbacks(
                        self._on_save_finished_callbacks,# TODO
                        run_idx=run_idx,
                        seed=self.seed,
                        experiment_id=self.experiment_id,
                        checkpoint_id = self.checkpoint_id
                    )
                    self.checkpoint_id = callbacks_res["checkpoint_id"]
                
        except Exception as ex:
            self._raise_callbacks(
                self._on_error_callbacks,# TODO
                run_idx=run_idx,
                seed=self.seed,
                experiment_id=self.experiment_id,
                checkpoint_id = self.checkpoint_id,
                message = "error exp_{}_check_{}_run_{}_seed_{} = {}".format(self.experiment_id, self.checkpoint_id, run_idx, self.seed, traceback.format_exc())
            )
        self._system.close()
        self._raise_callbacks(
            self._on_cancelled_callbacks if self.cancellation_token.get() else self._on_finished_callbacks,
            run_idx=run_idx,
            seed=self.seed,
            experiment_id=self.experiment_id,
            checkpoint_id = self.checkpoint_id
        )