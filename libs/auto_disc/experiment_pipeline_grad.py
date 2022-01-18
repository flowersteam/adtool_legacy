from auto_disc import ExperimentPipeline
from auto_disc import BaseAutoDiscModule
from auto_disc.utils.callbacks import BaseCallback
from copy import deepcopy
import traceback

class ExperimentPipelineGecko(ExperimentPipeline):

    def _process_output(self, output, document_id, starting_index=0, is_output_new_discovery=True):
        for i, output_representation in enumerate(self._output_representations[starting_index:]):
            output = output_representation.map(output, is_output_new_discovery)
            # TODO - quick fix
            output_copy = {}
            for k, v in output.items():
                output_copy[k] = deepcopy(v.detach())
            if i == len(self._output_representations) - 1:
                self.db.update({'output': output_copy}, doc_ids=[document_id])
            else:
                self.db.update({f'output_{i}': output_copy}, doc_ids=[document_id])
        return output

    def _process_run_parameters(self, run_parameters, document_id, starting_index=0, is_input_new_discovery=True):
        for i, input_wrapper in enumerate(self._input_wrappers[starting_index:]):
            run_parameters = input_wrapper.map(run_parameters, is_input_new_discovery)
            # TODO - quick fix
            run_parameters_copy = {}
            for k, v in run_parameters.items():
                run_parameters_copy[k] = deepcopy(v.detach())
            if i == len(self._input_wrappers) - 1:
                self.db.update({'run_parameters': run_parameters_copy}, doc_ids=[document_id])
            else:
                self.db.update({f'run_parameters_{i}': run_parameters_copy}, doc_ids=[document_id])
        return run_parameters


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
                # TODO - quick fix
                raw_run_parameters_copy = {}
                for k, v in raw_run_parameters.items():
                    raw_run_parameters_copy[k] = deepcopy(v.detach())
                document_id = self.db.insert({'idx': run_idx, 'raw_run_parameters': raw_run_parameters_copy})
                #with torch.no_grad():
                run_parameters = self._process_run_parameters(raw_run_parameters, document_id)

                o, r, d, i = self._system.reset(run_parameters), 0, None, False
                step_observations = [o]

                while not d:
                    if self._action_policy is not None:
                        a = self._action_policy(o, r)
                    else:
                        a = None

                    #with torch.no_grad():
                    o, r, d, i = self._system.step(a)
                    # step_observations.append(o) # TODO: Remove step observations ?
                    system_steps[-1] += 1
                    
                #with torch.no_grad():
                raw_output = self._system.observe()
                # TODO - quick fix
                raw_output_copy = {}
                for k,v in raw_output.items():
                    raw_output_copy[k] = deepcopy(v.detach())
                self.db.update({'raw_output': raw_output_copy}, doc_ids=[document_id])
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
                BaseAutoDiscModule.logger.info("New discovery from experiment {} with seed {}".format(self.experiment_id, self.seed))
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
                    BaseAutoDiscModule.logger.info("Experiment {} with seed {} and checkpoint_id {} saved".format(self.experiment_id, self.seed, self.checkpoint_id))
                    if("checkpoint_id" in callbacks_res):
                        self.checkpoint_id = callbacks_res["checkpoint_id"]
                    BaseAutoDiscModule.logger.checkpoint_id = self.checkpoint_id
                    BaseCallback.logger.checkpoint_id = self.checkpoint_id

                run_idx += 1
                BaseAutoDiscModule.CURRENT_RUN_INDEX += 1
                
        except Exception as ex:
            message = "error exp_{}_check_{}_run_{}_seed_{} = {}".format(self.experiment_id, self.checkpoint_id, run_idx, self.seed, traceback.format_exc())
            if len(message) > 8000: # Cut message to match varchar length of AppDB
                message = message[:7997] + '...'
            BaseAutoDiscModule.logger.error(message)
            self._raise_callbacks(
                self._on_error_callbacks,
                run_idx=run_idx,
                seed=self.seed,
                experiment_id=self.experiment_id,
                checkpoint_id = self.checkpoint_id,
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
        BaseAutoDiscModule.logger.info("Experiment {} with seed {} finished".format(self.experiment_id, self.seed))
        self.db.close()