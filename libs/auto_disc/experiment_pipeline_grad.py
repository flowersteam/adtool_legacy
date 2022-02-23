import torch
from auto_disc import ExperimentPipeline
from auto_disc import BaseAutoDiscModule
from auto_disc.utils.misc.dict_utils import map_nested_dicts
from copy import deepcopy
import traceback

from tinydb import Query 

class ExperimentPipelineGrad(ExperimentPipeline):

    def _process_output(self, output, document_id, starting_index=0, is_output_new_discovery=True):
        for i, output_representation in enumerate(self._output_representations[starting_index:]):
            output = output_representation.map(output, is_output_new_discovery)
            # TODO - quick fix
            output_copy = {}
            for k, v in output.items():
                if isinstance(v, (dict, torch.Tensor)):
                    v = map_nested_dicts(v, lambda x: x.detach().cpu())
                output_copy[k] = deepcopy(v)
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
                if isinstance(v, (dict, torch.Tensor)):
                    v = map_nested_dicts(v, lambda x: x.detach().cpu())
                run_parameters_copy[k] = deepcopy(v)
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
                    if isinstance(v, (dict, torch.Tensor)):
                        v = map_nested_dicts(v, lambda x: x.detach().cpu())
                    raw_run_parameters_copy[k] = deepcopy(v)
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
                    if isinstance(v, (dict, torch.Tensor)):
                        v = map_nested_dicts(v, lambda x: x.detach().cpu())
                    raw_output_copy[k] = deepcopy(v)
                self.db.update({'raw_output': raw_output_copy}, doc_ids=[document_id])
                output = self._process_output(raw_output, document_id)
                rendered_output = self._system.render()

                self._explorer.archive(raw_run_parameters, output)

                # quick fix to save detached/cpu tensors
                Data = Query()
                cur_data = self.db.search(Data.idx==run_idx)[0]
                self._raise_callbacks(
                    self._on_discovery_callbacks,
                    run_idx=run_idx,
                    seed=self.seed,
                    raw_run_parameters=cur_data["raw_run_parameters"],
                    run_parameters=cur_data["run_parameters"],
                    raw_output=cur_data["raw_output"],
                    output=cur_data["output"],
                    rendered_output=rendered_output,
                    step_observations=step_observations, 
                    experiment_id=self.experiment_id
                )
                self._system.logger.info(
                    "[DISCOVERY] - New discovery from experiment {} with seed {}".format(self.experiment_id, self.seed))
                self._explorer.optimize()  # TODO callbacks

                if (run_idx + 1) % self.save_frequency == 0:
                    self._raise_callbacks(
                        self._on_save_callbacks,
                        run_idx=run_idx,
                        seed=self.seed,
                        experiment_id=self.experiment_id,
                        system=self._system,
                        explorer=self._explorer,
                        input_wrappers=self._input_wrappers,
                        output_representations=self._output_representations,
                        in_memory_db=self.db
                    )
                    self._raise_callbacks(
                        self._on_save_finished_callbacks,
                        run_idx=run_idx,
                        seed=self.seed,
                        experiment_id=self.experiment_id
                    )
                    self._system.logger.info(
                        "[SAVED] - experiment {} with seed {} saved".format(self.experiment_id, self.seed))

                run_idx += 1
                BaseAutoDiscModule.CURRENT_RUN_INDEX += 1

        except Exception as ex:
            message = "error in experiment {} run_idx {} seed {} = {}".format(self.experiment_id, run_idx, self.seed,
                                                                              traceback.format_exc())
            if len(message) > 8000:  # Cut message to match varchar length of AppDB
                message = message[:7997] + '...'
            self._system.logger.error("[ERROR] - " + message)
            self._raise_callbacks(
                self._on_error_callbacks,
                run_idx=run_idx,
                seed=self.seed,
                experiment_id=self.experiment_id
            )
            self.db.close()
            raise

        self._system.close()
        if self.cancellation_token.get():
            self._system.logger.info(
                "[CANCELLED] - experiment {} with seed {} cancelled".format(self.experiment_id, self.seed))
        else:
            self._system.logger.info(
                "[FINISHED] - experiment {} with seed {} finished".format(self.experiment_id, self.seed))
        self._raise_callbacks(
            self._on_cancelled_callbacks if self.cancellation_token.get() else self._on_finished_callbacks,
            run_idx=run_idx,
            seed=self.seed,
            experiment_id=self.experiment_id
        )
        self.db.close()