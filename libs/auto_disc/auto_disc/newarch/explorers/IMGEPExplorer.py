from leaf.leaf import Leaf
from leaf.locators import FileLocator
from auto_disc.newarch.wrappers.IdentityWrapper import IdentityWrapper
from typing import Dict, Tuple, Callable
import torch
from copy import deepcopy


class IMGEPExplorer(Leaf):
    def __init__(self,
                 premap_key: str = "output",
                 postmap_key: str = "params",
                 parameter_map: Leaf = IdentityWrapper(),
                 behavior_map: Leaf = IdentityWrapper(),
                 mutator: Leaf = torch.nn.Identity(),
                 equil_time: int = 0) -> None:
        super().__init__()
        self.locator = FileLocator()
        self.premap_key = premap_key
        self.postmap_key = postmap_key
        self.parameter_map = parameter_map
        self.behavior_map = behavior_map
        self.equil_time = equil_time
        self.timestep = 0

        self.mutator = mutator

    def map(self, system_output: Dict) -> Dict:
        """
        The "map" when the `Explorer` is viewed as a function, which takes the
        feature vector in behavior space and maps it to a (randomly chosen)
        subsequent parameter configuration to try.
        """
        behavior_tensor, orig_data = self.observe_results(
            system_output)

        data_shape = behavior_tensor.size()

        # note that suggest_trial() increments the timestep
        params_trial = self.suggest_trial(data_shape)

        new_trial_data = deepcopy(orig_data)
        del new_trial_data[self.premap_key]
        new_trial_data[self.postmap_key] = params_trial

        return new_trial_data

    def suggest_trial(self, data_shape: Tuple) -> torch.Tensor:
        """
        Samples according to the policy a new trial of parameters for the
        system, generating sample according to the data_shape
        """
        if self.timestep < self.equil_time:
            params_trial = self.parameter_map.sample(data_shape)
        else:
            goal_history = self.behavior_map.get_tensor_history()
            param_history = self.parameter_map.get_tensor_history()

            goal = self.behavior_map.sample()
            source_policy_idx = self._find_closest(goal, goal_history)
            source_policy = param_history[source_policy_idx]

            params_trial = self.mutator(source_policy)

        self.timestep += 1

        return params_trial

    def observe_results(self, system_output: Dict) -> Tuple[
        torch.Tensor, Callable[[], torch.Tensor], Dict
    ]:
        """
        Reads the behavior discovered and processes it.
        """
        output_dict = self.behavior_map.map(system_output)

        # for convenience, return a pointer to the feature tensor and
        behavior_tensor = output_dict[self.premap_key]

        return behavior_tensor, output_dict

    def optimize():
        """
        Optimization step for online learning of the `Explorer` policy.
        """
        pass

    def _find_closest(self, goal: torch.Tensor, goal_history: torch.Tensor):
        # TODO: simple L2 distance right now
        return torch.argmin((goal_history-goal).pow(2).sum(-1))
