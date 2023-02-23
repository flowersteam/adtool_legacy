from copy import copy
import traceback
import torch
from typing import Dict, Callable, List, Any, Tuple

from leaf.leaf import Leaf
from leaf.locators import FileLocator
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


class ExperimentPipeline(Leaf):
    """
    Pipeline of an automated discovery experiment.

    An experiment is at least constituted of a system and an explorer. 

    Additionally, input wrappers and output representations can be added and composed.

    When the system requires an action at each timestep, an `action_policy` must be provided.

    In order to monitor the experiment, you must provide **callbacks**, which will be called every time a discovery has been made. 
    Please see: `auto_disc.utils.callbacks.base_callback.BaseCallback`.
    """

    def __init__(self, experiment_id: int = 0, seed: int = 0,
                 system=None,
                 explorer=None,
                 input_pipeline=None,
                 output_pipeline=None,
                 action_policy=None,
                 save_frequency: int = 100,
                 on_discovery_callbacks: List[Callable] = [],
                 on_save_finished_callbacks: List[Callable] = [],
                 on_finished_callbacks: List[Callable] = [],
                 on_cancelled_callbacks: List[Callable] = [],
                 on_save_callbacks: List[Callable] = [],
                 on_error_callbacks: List[Callable] = [],
                 logger=None,
                 resource_uri: str = ""
                 ) -> None:
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
        super().__init__()
        self.locator = FileLocator()
        self.run_idx = 0
        self.experiment_id = experiment_id
        self.seed = seed
        self.save_frequency = save_frequency
        self.logger = logger
        self.resource_uri = resource_uri

        ### SYSTEM ###
        self._system = system

        ### OUTPUT PIPELINE ###
        self._output_pipeline = output_pipeline

        ### INPUT PIPELINE ###
        self._input_pipeline = input_pipeline

        ### EXPLORER ###
        self._explorer = explorer

        self._action_policy = action_policy
        self._on_discovery_callbacks = on_discovery_callbacks
        self._on_save_finished_callbacks = on_save_finished_callbacks
        self._on_finished_callbacks = on_finished_callbacks
        self._on_cancelled_callbacks = on_cancelled_callbacks
        self._on_error_callbacks = on_error_callbacks
        self._on_save_callbacks = on_save_callbacks
        self.cancellation_token = CancellationToken()

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

    def run(self, n_exploration_runs: int) -> str:
        '''
        Launches the experiment for `n_exploration_runs` number of explorations.

        `n_exploration_runs` is specified so more optimized looping routines
        can be chosen in the inner loop if desired. Interfacing with the
        objects in the inner loop should be done via the callback interface.

        #### Args:
        - **n_exploration_runs**: number of explorations

        #### Returns:
        - **LeafUID**: returns the UID associated to the experiment
        '''
        # initialize in case exception is thrown
        uid = ""
        try:
            data_dict = self._explorer.bootstrap()

            while self.run_idx < n_exploration_runs:
                # check for termination
                if self.cancellation_token.get():
                    break

                # pass trial parameters through system
                data_dict = self._system.map(data_dict)

                # render system output
                rendered_output = self._system.render(data_dict)

                # exploration phase : emits new trial parameters for next loop
                data_dict = self._explorer.map(data_dict)

                discovery = self._explorer.read_last_discovery()

                self._raise_callbacks(
                    self._on_discovery_callbacks,
                    run_idx=self.run_idx,
                    seed=self.seed,
                    run_parameters=discovery[self._explorer.postmap_key],
                    output=discovery[self._explorer.premap_key],
                    raw_output=discovery["raw_" + self._explorer.premap_key],
                    rendered_output=rendered_output,
                    experiment_id=self.experiment_id
                )

                self.logger.info(
                    "[DISCOVERY] - New discovery from experiment {} with seed {}"
                    .format(self.experiment_id, self.seed)
                )

                self.run_idx += 1

                # avoids divide by zero
                run_idx_start_from_one = self.run_idx + 1

                if (run_idx_start_from_one % self.save_frequency == 0
                        or run_idx_start_from_one == n_exploration_runs):
                    uid = self.save(resource_uri=self.resource_uri)

        except Exception as _:
            message = "error in experiment {} self.run_idx {} seed {} = {}".format(
                self.experiment_id, self.run_idx, self.seed, traceback.format_exc())

            # TODO: do this in appdb side
            if len(message) > 8000:  # Cut message to match varchar length of AppDB
                message = message[:7997] + '...'

            self.logger.error("[ERROR] - " + message)
            self._raise_callbacks(
                self._on_error_callbacks,
                run_idx=self.run_idx,
                seed=self.seed,
                experiment_id=self.experiment_id
            )
            raise

        ## CLEANUP ##

        # log termination of experiment
        if self.cancellation_token.get():
            self.logger.info(
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
            self.logger.info(
                "[FINISHED] - experiment {} with seed {} finished"
                .format(self.experiment_id, self.seed)
            )

            self._raise_callbacks(
                self._on_finished_callbacks,
                run_idx=self.run_idx,
                seed=self.seed,
                experiment_id=self.experiment_id
            )
        return uid

    def save(self, resource_uri: str) -> str:
        self._raise_callbacks(
            self._on_save_callbacks,
            run_idx=self.run_idx,
            seed=self.seed,
            experiment_id=self.experiment_id,
            system=self._system,
            explorer=self._explorer,
        )
        uid = self.save_leaf(resource_uri=resource_uri)
        self._raise_callbacks(
            self._on_save_finished_callbacks,
            uid=uid
        )
        self.logger.info(
            "[SAVED] - experiment {} with seed {} saved"
            .format(self.experiment_id, self.seed)
        )
        return uid
