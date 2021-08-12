from copy import copy
import traceback

import torch

from auto_disc import BaseAutoDiscModule
from auto_disc.output_representations.generic import DummyOutputRepresentation
from auto_disc.input_wrappers.generic import DummyInputWrapper
from auto_disc.utils.misc import DB


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

        self.db = DB()

        ### SYSTEM ###
        self._system = system
        self._system.set_call_output_history_update_fn(self._update_outputs_history)
        # self._system.set_call_run_parameters_history_update_fn(self._update_run_parameters_history)
        self._system.set_history_access_fn(lambda: self.db.to_autodisc_history(self.db.all(), 
                                                                          ['idx', 'run_parameters', 'raw_output'], 
                                                                          ['idx', 'input', 'output']))
        
        ### OUTPUT REPRESENTATIONS ###
        if output_representations is not None and len(output_representations) > 0:
            self._output_representations = output_representations
        else:
            self._output_representations = [DummyOutputRepresentation()]

        for i in range(len(self._output_representations)):
            self._output_representations[i].set_call_output_history_update_fn(lambda: self._update_outputs_history(i))
            if i == 0:
                self._output_representations[i].initialize(input_space=self._system.output_space)
                input_key = 'raw_output'
            else:
                input_key = f'output_{i-1}'
                self._output_representations[i].initialize(input_space=self._output_representations[i-1].output_space)

            if i == len(self._output_representations) - 1:
                output_key = f'output'
            else:
                output_key = f'output_{i}'

            self._output_representations[i].set_history_access_fn(lambda i=input_key, o=output_key: self.db.to_autodisc_history(self.db.all(), 
                                                                                                    ['idx', i, o], 
                                                                                                    ['idx', 'input', 'output']))
                

        ### INPUT WRAPPERS ###
        if input_wrappers is not None and len(input_wrappers) > 0:
            self._input_wrappers = input_wrappers
        else:
            self._input_wrappers = [DummyInputWrapper()]
            
        for i in reversed(range(len(self._input_wrappers))):
            # self._input_wrappers[i].set_call_run_parameters_history_update_fn(self._update_run_parameters_history)
            if i == len(self._input_wrappers) - 1:
                self._input_wrappers[i].initialize(output_space=self._system.input_space)
                output_key = f'run_parameters'
            else:
                self._input_wrappers[i].initialize(output_space=self._input_wrappers[i+1].input_space)
                output_key = f'run_parameters_{i}'

            if i == 0:
                input_key = 'raw_run_parameters'
            else:
                input_key = f'run_parameters_{i-1}'

            self._input_wrappers[i].set_history_access_fn(lambda i=input_key, o=output_key: self.db.to_autodisc_history(self.db.all(), 
                                                                                            ['idx', i, o], 
                                                                                            ['idx', 'input', 'output']))

        ### EXPLORER ###
        self._explorer = explorer
        self._explorer.set_call_output_history_update_fn(self._update_outputs_history)
        # self._explorer.set_call_run_parameters_history_update_fn(self._update_run_parameters_history)
        self._explorer.initialize(input_space=self._output_representations[-1].output_space,
                                  output_space=self._input_wrappers[0].input_space, 
                                  input_distance_fn=self._output_representations[-1].calc_distance)
        self._explorer.set_history_access_fn(lambda: self.db.to_autodisc_history(self.db.all(), 
                                                                            ['idx', 'output', 'raw_run_parameters'], 
                                                                            ['idx', 'input', 'output']))
        
        self._action_policy = action_policy
        self._on_discovery_callbacks = on_discovery_callbacks
        self._on_save_finished_callbacks = on_save_finished_callbacks
        self._on_finished_callbacks = on_finished_callbacks
        self._on_cancelled_callbacks = on_cancelled_callbacks
        self._on_error_callbacks = on_error_callbacks
        self._on_save_callbacks = on_save_callbacks
        self.cancellation_token = CancellationToken()

    def _process_output(self, output, document_id, starting_index=0, is_output_new_discovery=True):
        for i, output_representation in enumerate(self._output_representations[starting_index:]):
            output = output_representation.map(output, is_output_new_discovery)
            if i == len(self._output_representations) - 1:
                self.db.update({'output': copy(output)}, doc_ids=[document_id])
            else:
                self.db.update({f'output_{i}': copy(output)}, doc_ids=[document_id])
        return output

    def _update_outputs_history(self, output_representation_idx):
        '''
            Iterate over history and update values of outputs produced after `output_representation_idx`.
        '''
        for document in self.db.all():
            if output_representation_idx == 0:
                output =  document['raw_output'] # starting from first output => raw_output
            else:
                output = document[f'output_{output_representation_idx-1}']
            self._process_output(output, document.doc_id, starting_index=output_representation_idx, is_output_new_discovery=False)

    def _process_run_parameters(self, run_parameters, document_id, starting_index=0, is_input_new_discovery=True):
        for i, input_wrapper in enumerate(self._input_wrappers[starting_index:]):
            run_parameters = input_wrapper.map(run_parameters, is_input_new_discovery)
            if i == len(self._input_wrappers) - 1:
                self.db.update({'run_parameters': copy(run_parameters)}, doc_ids=[document_id])
            else:
                self.db.update({f'run_parameters_{i}': copy(run_parameters)}, doc_ids=[document_id])
        return run_parameters
    
    # def _update_run_parameters_history(self, run_parameters_idx):
    #     '''
    #         Iterate over history and update values of run_parameters produced after `run_parameters_idx`.
    #     '''
    #     for document in self.db.all():
    #         if run_parameters_idx == 0:
    #             run_parameters =  document['raw_run_parameters'] # starting from first run_parameters => raw_run_parameters
    #         else:
    #             run_parameters = document[f'run_parameters_{run_parameters_idx-1}']
    #         self._process_run_parameters(run_parameters, document.doc_id, starting_index=run_parameters_idx, is_input_new_discovery=False)

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
        BaseAutoDiscModule.CURRENT_RUN_INDEX = 0
        system_steps = [0]
        try:
            while run_idx < n_exploration_runs:
                if self.cancellation_token.get():
                    break

                raw_run_parameters = self._explorer.emit()
                document_id = self.db.insert({'idx': run_idx, 'raw_run_parameters': copy(raw_run_parameters)})
                with torch.no_grad():
                    run_parameters = self._process_run_parameters(raw_run_parameters, document_id)

                o, r, d, i = self._system.reset(run_parameters), 0, None, False
                step_observations = [o]

                while not d:
                    if self._action_policy is not None:
                        a = self._action_policy(o, r)
                    else:
                        a = None

                    with torch.no_grad():
                        o, r, d, i = self._system.step(a)
                    # step_observations.append(o) # TODO: Remove step observations ?
                    system_steps[-1] += 1
                    
                with torch.no_grad():
                    raw_output = self._system.observe()
                    self.db.update({'raw_output': copy(raw_output)}, doc_ids=[document_id])
                    output = self._process_output(raw_output, document_id)
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

                self._explorer.optimize() # TODO callbacks

                if (run_idx+1) % self.save_frequency == 0:
                    self._raise_callbacks(
                        self._on_save_callbacks,
                        run_idx=run_idx,
                        seed=self.seed,
                        experiment_id=self.experiment_id,
                        checkpoint_id = self.checkpoint_id,
                        system=self._system,
                        explorer=self._explorer,
                        input_wrappers=self._input_wrappers,
                        output_representations=self._output_representations,
                        in_memory_db=self.db
                    )
                    callbacks_res = self._raise_callbacks(
                        self._on_save_finished_callbacks,# TODO
                        run_idx=run_idx,
                        seed=self.seed,
                        experiment_id=self.experiment_id,
                        checkpoint_id = self.checkpoint_id
                    )
                    self.checkpoint_id = callbacks_res["checkpoint_id"]

                run_idx += 1
                BaseAutoDiscModule.CURRENT_RUN_INDEX += 1
                
        except Exception as ex:
            self._raise_callbacks(
                self._on_error_callbacks,
                run_idx=run_idx,
                seed=self.seed,
                experiment_id=self.experiment_id,
                checkpoint_id = self.checkpoint_id,
                message = "error exp_{}_check_{}_run_{}_seed_{} = {}".format(self.experiment_id, self.checkpoint_id, run_idx, self.seed, traceback.format_exc())
            )
            self.db.close()
            raise

        self._system.close()
        self._raise_callbacks(
            self._on_cancelled_callbacks if self.cancellation_token.get() else self._on_finished_callbacks,
            run_idx=run_idx,
            seed=self.seed,
            experiment_id=self.experiment_id,
            checkpoint_id = self.checkpoint_id
        )
        self.db.close()