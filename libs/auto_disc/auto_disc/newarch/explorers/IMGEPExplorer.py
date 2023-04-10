from leaf.Leaf import Leaf
from leaf.locators.locators import BlobLocator, LinearLocator
from auto_disc.newarch.wrappers.IdentityWrapper import IdentityWrapper
from auto_disc.newarch.wrappers.SaveWrapper import SaveWrapper
from auto_disc.newarch.maps.MeanBehaviorMap import MeanBehaviorMap
from auto_disc.newarch.maps.UniformParameterMap import UniformParameterMap
from auto_disc.newarch.wrappers.mutators import add_gaussian_noise
from auto_disc.utils.config_parameters import (
    DecimalConfigParameter,
    IntegerConfigParameter,
    StringConfigParameter,
    DictConfigParameter
)
from typing import Dict, Tuple, Callable, List
import torch
from copy import deepcopy
from functools import partial


@IntegerConfigParameter("equil_time", default=1, min=1)
@IntegerConfigParameter("param_dim", default=1, min=1)
@DecimalConfigParameter("param_init_low", default=0.)
@DecimalConfigParameter("param_init_high", default=0.)
@DecimalConfigParameter("param_bound_low", default=float('-inf'))
@DecimalConfigParameter("param_bound_high", default=float('inf'))
@IntegerConfigParameter("system_output_dim", default=1, min=1)
@DecimalConfigParameter("mutation_noise_std", default=0., min=0.)
@StringConfigParameter("behavior_map",
                       possible_values=["mean"],
                       default="mean")
@DictConfigParameter("behavior_map_config", default={})
@StringConfigParameter("parameter_map",
                       possible_values=["uniform"],
                       default="uniform")
class IMGEPFactory:
    """
    Factory class providing interface with config parameters and therefore the
    frontend
    """
    CONFIG_DEFINITION = {}

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self):
        behavior_map = self.make_behavior_map()
        param_map = self.make_parameter_map()
        mutator = self.make_mutator()
        equil_time = self.config["equil_time"]
        explorer = IMGEPExplorer(parameter_map=param_map,
                                 behavior_map=behavior_map,
                                 equil_time=equil_time,
                                 mutator=mutator)

        return explorer

    def make_behavior_map(self):
        if self.config["behavior_map"] == "mean":
            kwargs = self.config["behavior_map_config"]
            behavior_map = MeanBehaviorMap(**kwargs)
        else:
            # this branch should be unreachable,
            # because the ConfigParameter decorator checks
            raise Exception("unreachable")

        return behavior_map

    def make_parameter_map(self):
        if self.config["parameter_map"] == "uniform":
            param_size = torch.Size([self.config["param_dim"]])
            init_low = self.config["param_init_low"]
            init_high = self.config["param_init_high"]
            tensor_low = torch.full(size=param_size, fill_value=init_low)
            tensor_high = torch.full(size=param_size, fill_value=init_high)
            float_bound_low = self.config["param_bound_low"]
            float_bound_high = self.config["param_bound_high"]

            param_map = UniformParameterMap(
                tensor_low=tensor_low,
                tensor_high=tensor_high,
                float_bound_low=float_bound_low,
                float_bound_high=float_bound_high
            )
        else:
            # this branch should be unreachable,
            # because the ConfigParameter decorator checks
            raise Exception("unreachable")

        return param_map

    def make_mutator(self):
        if self.config["mutation_noise_std"] > 0:
            mutator = partial(add_gaussian_noise,
                              std=self.config["mutation_noise_std"])
        else:
            mutator = torch.nn.Identity()

        return mutator


class IMGEPExplorer(Leaf):
    def __init__(self,
                 premap_key: str = "output",
                 postmap_key: str = "params",
                 parameter_map: Leaf = IdentityWrapper(),
                 behavior_map: Leaf = IdentityWrapper(),
                 mutator: Leaf = torch.nn.Identity(),
                 equil_time: int = 0) -> None:
        super().__init__()
        self.locator = BlobLocator()
        self.premap_key = premap_key
        self.postmap_key = postmap_key
        self.parameter_map = parameter_map
        self.behavior_map = behavior_map
        self.equil_time = equil_time
        self.timestep = 0

        self.mutator = mutator

        self._history_saver = SaveWrapper()

    def bootstrap(self) -> Dict:
        """
        need special initialization logic for the t=0 run of IMGEP
        """
        data_dict = {}
        # initialize sample
        data_shape = self.parameter_map.output_shape
        params_init = self.parameter_map.sample(data_shape)
        data_dict[self.postmap_key] = params_init

        # first timestep recorded
        # NOTE: therefore, regardless of self.equil_time, 1 equil step
        # will always happen
        data_dict["equil"] = 1
        self.timestep += 1

        return data_dict

    def map(self, system_output: Dict) -> Dict:
        """
        The "map" when the `Explorer` is viewed as a function, which takes the
        feature vector in behavior space and maps it to a (randomly chosen)
        subsequent parameter configuration to try.
        """
        # either do nothing, or update dict by changing "output" -> "raw_output"
        # and adding new "output" key which is the result of the behavior map
        new_trial_data = self.observe_results(system_output)

        # save results
        trial_data_reset = self._history_saver.map(new_trial_data)

        # TODO: check gradients here
        if self.timestep < self.equil_time:
            # sets "params" key
            trial_data_reset = self.parameter_map.map(trial_data_reset)

            # label which trials were from random initialization
            trial_data_reset["equil"] = 1
        else:
            # suggest_trial reads history
            params_trial = self.suggest_trial()

            # assemble dict and update parameter_map state
            trial_data_reset[self.postmap_key] = params_trial

            # label that trials are now from the usual IMGEP procedure
            trial_data_reset["equil"] = 0

        self.timestep += 1

        return trial_data_reset

    def suggest_trial(self, lookback_length: int = -1) -> torch.Tensor:
        """
        Samples according to the policy a new trial of parameters for the
        system.

        The `lookback_length` parameter is the number of previous trials to
        consider when choosing the next trial, i.e., it is a batch size
        based on the save frequency. 

        Note that the default `lookback_length = -1` will retrieve the entire 
        history.
        """
        goal = self.behavior_map.sample()

        source_policy = self._vector_search_for_goal(lookback_length, goal)

        params_trial = self.mutator(source_policy)

        return params_trial

    def observe_results(self, system_output: Dict) -> Dict:
        """
        Reads the behavior discovered and processes it.
        """
        # check we are not in the initialization case
        if system_output.get(self.premap_key, None) is not None:
            # recall that behavior_maps will remove the dict entry of
            # self.premap_key
            system_output = self.behavior_map.map(system_output)
        else:
            pass

        return system_output

    def read_last_discovery(self) -> Dict:
        return self._history_saver.buffer[-1]

    def optimize(self):
        """
        Optimization step for online learning of the `Explorer` policy.
        """
        pass

    def _extract_tensor_history(self,
                                dict_history: List[Dict],
                                key: str) -> torch.Tensor:
        """
        Extracts tensor history from an array of dicts with labelled data,
        with the tensor being labelled by key 
        """
        # append history of tensors along a new dimension at index 0
        tensor_history = \
            dict_history[0][key].unsqueeze(0)
        for dict in dict_history[1:]:
            tensor_history = torch.cat(
                (tensor_history, dict[key].unsqueeze(0)), dim=0)

        return tensor_history

    def _find_closest(self, goal: torch.Tensor, goal_history: torch.Tensor):
        # TODO: simple L2 distance right now
        return torch.argmin((goal_history-goal).pow(2).sum(-1))

    def _vector_search_for_goal(self, goal: torch.Tensor,
                                lookback_length: int) -> torch.Tensor:
        history_buffer = self._history_saver.get_history(
            lookback_length=lookback_length)
        goal_history = self._extract_tensor_history(history_buffer,
                                                    self.premap_key)
        param_history = self._extract_tensor_history(history_buffer,
                                                     self.postmap_key)
        source_policy_idx = self._find_closest(goal, goal_history)
        source_policy = param_history[source_policy_idx]

        return source_policy
