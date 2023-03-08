from copy import copy
import traceback
import torch
from typing import Dict, Callable, List, Any, Tuple

from auto_disc.input_wrappers import BaseInputWrapper
from auto_disc.output_representations import BaseOutputRepresentation
from auto_disc.utils.callbacks.interact_callbacks import Interact
from auto_disc.systems.base_system import BaseSystem
from auto_disc.explorers.base_explorer import BaseExplorer

from auto_disc.newarch.wrappers.IdentityWrapper import IdentityWrapper


class CancellationToken:
    """
        Manages the cancellation token which allows you to stop an experiment in progress
    """

    def __init__(self) -> None:
        """
            Init the cancellation token to false
        """
        self._token = False

    def get(self) -> bool:
        """
            Give access to the cancellation token

            #### Returns:
            - **token**: a boolean indicating if the current experiment must be cancelled
        """
        return self._token

    def trigger(self) -> None:
        """
            Sets the cancellation token to true (the experiment must be cancelled)
        """
        self._token = True


class ExperimentPipeline():
    '''
    Pipeline of an automated discovery experiment.

    An experiment is at least constituted of a system and an explorer. 

    Additionally, input wrappers and output representations can be added and composed.

    When the system requires an action at each timestep, an `action_policy` must be provided.

    In order to monitor the experiment, you must provide **callbacks**, which will be called every time a discovery has been made. 
    Please see: `auto_disc.utils.callbacks.base_callback.BaseCallback`.
    '''

    def __init__(self, experiment_id: int, seed: int,
                 system: BaseSystem,
                 explorer: BaseExplorer,
                 input_wrappers: List[BaseInputWrapper] = None,
                 output_representations: List[BaseOutputRepresentation] = None,
                 action_policy=None,
                 save_frequency: int = 100,
                 on_discovery_callbacks: List[Callable] = [],
                 on_save_finished_callbacks: List[Callable] = [],
                 on_finished_callbacks: List[Callable] = [],
                 on_cancelled_callbacks: List[Callable] = [],
                 on_save_callbacks: List[Callable] = [],
                 on_error_callbacks: List[Callable] = [],
                 interact_callbacks: List[Callable] = []) -> None:
        """
            Initializes state of experiment pipeline, setting all necessary attributes given by the following arguments.

            #### Args:
            - **experiment_id**: ID of current experiment
            - **seed**: Current seed for random number generation
            - **system**: System to be explored
            - **explorer**: Explorer used
            - **input_wrappers**: List of wrappers which transform the input space
            - **output_representations**: List of wrappers which transform the output space
            - **action_policy**: User-defined method ran on each loop, based on policy
            - **save_frequency**: Frequency to save state of the experiment
            - **on_discovery_callbacks**: Callbacks raised when a discovery is made
            - **on_save_finished_callbacks**: Callbacks raised when the experiment save is complete
            - **on_finished_callbacks**: Callbacks raised when the experiment is finished
            - **on_cancelled_callbacks**: Callbacks raised when the experiment is cancelled
            - **on_save_callbacks**: Callbacks raised when a experiment save is made
            - **on_error_callbacks**: Callbacks raised when an error is raised
        """
        self.run_idx = 0
        self.experiment_id = experiment_id
        self.seed = seed
        self.save_frequency = save_frequency

        ### SYSTEM ###
        self._system = system

        ### OUTPUT REPRESENTATIONS ###
        if output_representations is not None and \
                len(output_representations) > 0:
            self._output_representations = output_representations
        else:
            self._output_representations = [IdentityWrapper()]

        for i in range(len(self._output_representations)):
            input_key, output_key = self.compose_outputs(i)

        ### INPUT WRAPPERS ###
        if input_wrappers is not None and \
                len(input_wrappers) > 0:
            self._input_wrappers = input_wrappers
        else:
            self._input_wrappers = [IdentityWrapper()]

        for i in reversed(range(len(self._input_wrappers))):
            input_key, output_key = self.compose_wrappers(i)

        ### EXPLORER ###
        self._explorer = explorer

        self._action_policy = action_policy
        self._on_discovery_callbacks = on_discovery_callbacks
        self._on_save_finished_callbacks = on_save_finished_callbacks
        self._on_finished_callbacks = on_finished_callbacks
        self._on_cancelled_callbacks = on_cancelled_callbacks
        self._on_error_callbacks = on_error_callbacks
        self._on_save_callbacks = on_save_callbacks
        self.interact_callbacks = interact_callbacks
        self.cancellation_token = CancellationToken()

    def compose_outputs(self, i):
        if i == 0:
            self._output_representations[i].initialize(
                input_space=self._system.output_space)
            input_key = 'raw_output'
        else:
            input_key = f'output_{i-1}'
            self._output_representations[i].initialize(
                input_space=self._output_representations[i-1].output_space)

        if i == len(self._output_representations) - 1:
            output_key = f'output'
        else:
            output_key = f'output_{i}'

        return input_key, output_key

    def compose_wrappers(self, i) -> Tuple[str, str]:
        if i == len(self._input_wrappers) - 1:
            output_key = 'run_parameters'
        else:
            output_key = f'run_parameters_{i}'

        if i == 0:
            input_key = 'raw_run_parameters'
        else:
            input_key = f'run_parameters_{i-1}'

        return (input_key, output_key)

    def _raise_callbacks(self, callbacks: List[Callable], **kwargs) -> None:
        """
            Raise all callbacks linked to the current event (new discovery, a save, the end of the experiment...)

            Args:
                callbacks: list of all callbacks must be raise
                kwargs: some usefull parameters like self.run_idx, seed, experiment_id, some modules...
        """
        for callback in callbacks:
            callback(
                pipeline=self,
                **kwargs
            )

    def run(self, n_exploration_runs: int) -> None:
        '''
        Launches the experiment for `n_exploration_runs` number of explorations.

        #### Args:
        - **n_exploration_runs**: number of explorations
        '''
        system_steps = [0]
        Interact.init_seed(self.interact_callbacks, {
                           "experiment_id": self.experiment_id, "seed": self.seed, "idx": 0})
        try:
            while self.run_idx < n_exploration_runs:

                # check for termination
                if self.cancellation_token.get():
                    break

                # sample trial set of parameters
                raw_run_parameters = self._explorer.sample()

                # wraps run_parameters and stores in DB
                with torch.no_grad():
                    run_parameters = self._process_run_parameters(
                        raw_run_parameters)

                # initialize system state
                observation, reward, is_done, info = self._system.reset(
                    copy(run_parameters)
                ), 0, None, False

                while not is_done:
                    # set exploration policy
                    if self._action_policy is not None:
                        a = self._action_policy(observation, reward)

                    # step through dynamics of system
                    with torch.no_grad():
                        observation, reward, is_done, info = self._system.step(
                            a)
                    system_steps[-1] += 1

                with torch.no_grad():

                    # get system output
                    raw_output = self._system.observe()
                    # transform based on output_wrapper
                    output = self._process_output(raw_output)

                    rendered_output = self._system.render()

                # pass results of system pipeline to explorer
                self._explorer.observe(copy(raw_run_parameters), copy(output))

                self._raise_callbacks(
                    self._on_discovery_callbacks,
                    run_idx=self.run_idx,
                    seed=self.seed,
                    raw_run_parameters=raw_run_parameters,
                    run_parameters=run_parameters,
                    raw_output=raw_output,
                    output=output,
                    rendered_output=rendered_output,
                    experiment_id=self.experiment_id
                )

                self._system.logger.info(
                    "[DISCOVERY] - New discovery from experiment {} with seed {}"
                    .format(self.experiment_id, self.seed)
                )

                self._explorer.optimize()  # TODO callbacks

                self.save(n_exploration_runs)

                self.run_idx += 1

        except Exception as ex:
            message = "error in experiment {} self.run_idx {} seed {} = {}".format(
                self.experiment_id, self.run_idx, self.seed, traceback.format_exc())
            if len(message) > 8000:  # Cut message to match varchar length of AppDB
                message = message[:7997] + '...'
            self._system.logger.error("[ERROR] - " + message)
            self._raise_callbacks(
                self._on_error_callbacks,
                run_idx=self.run_idx,
                seed=self.seed,
                experiment_id=self.experiment_id
            )
            raise

        # cleanup any system state
        self._system.close()

        # log termination of experiment
        if self.cancellation_token.get():
            self._system.logger.info(
                "[CANCELLED] - experiment {} with seed {} cancelled"
                .format(self.experiment_id, self.seed)
            )

            self._raise_callbacks(
                self._on_cancelled_callbacks,
                run_idx=self.run_idx,
                seed=self.seed,
                experiment_id=self.experiment_id
            )
        else:
            self._system.logger.info(
                "[FINISHED] - experiment {} with seed {} finished"
                .format(self.experiment_id, self.seed)
            )

            self._raise_callbacks(
                self._on_finished_callbacks,
                run_idx=self.run_idx,
                seed=self.seed,
                experiment_id=self.experiment_id
            )

    def save(self, n_exploration_runs):

        if (self.run_idx + 1) % self.save_frequency == 0 or self.run_idx + 1 == n_exploration_runs:
            self._raise_callbacks(
                self._on_save_callbacks,
                run_idx=self.run_idx,
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
                run_idx=self.run_idx,
                seed=self.seed,
                experiment_id=self.experiment_id
            )
            self._system.logger.info(
                "[SAVED] - experiment {} with seed {} saved"
                .format(self.experiment_id, self.seed)
            )
