from addict import Dict
from auto_disc.explorers import BaseExplorer
from auto_disc.utils.config_parameters import StringConfigParameter, IntegerConfigParameter, DictConfigParameter
from auto_disc.utils.misc.dict_utils import map_nested_dicts
import random
import torch

@StringConfigParameter(name="tensors_device", default="cpu", possible_values=["cuda", "cpu", ])


#TODO: think what is best to implement the goal_sampling / candidate_selection / candidate_optimization strategies: in the input/output spaces or here?
@StringConfigParameter(name="goal_selection_type", possible_values=["random"], default="random")
@StringConfigParameter(name="source_policy_selection_type", possible_values=["optimal", "random"], default="optimal")
@StringConfigParameter(name="policy_optimization_type", possible_values=["random", "none"], default="random")
@IntegerConfigParameter(name="num_of_random_initialization", default=10, min=1)

class IMGEPExplorer(BaseExplorer):
    """
    Basic explorer that samples goals in a goalspace and uses a policy library to generate parameters to reach the goal.
    """
    CONFIG_DEFINITION = {}
    

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def initialize(self, input_space, output_space):
        super().initialize(input_space, output_space)
        if len(self._input_space) > 1:
            raise NotImplementedError("Only 1 vector can be accepted as input space")
        self._outter_input_space_key = list(self._input_space.spaces.keys())[0] # select first key in DictSpace


    def _get_next_goal(self):
        """ Defines the goal sampling policy. """

        if self.config.goal_selection_type == 'random':
            target_goal = self._input_space[self._outter_input_space_key].sample()

        else:
            raise ValueError(
                'Unknown goal generation type {!r} in the configuration!'.format(self.config.goal_selection_type))

        return target_goal


    def _get_source_policy_idx(self, target_goal):
        history = self._access_history()
        goal_library = [h[self._outter_input_space_key] for h in history['input']] # get goal history as tensor

        if self.config.source_policy_selection_type == 'optimal':
            # get distance to other goals
            goal_distances = self._input_space[self._outter_input_space_key].calc_distance(target_goal, goal_library)

            # select goal with minimal distance
            source_policy_idx = torch.argmin(goal_distances)

        elif self.config.source_policy_selection_type == 'random':
            source_policy_idx = random.randint(0, self.CURRENT_RUN_INDEX)

        else:
            raise ValueError('Unknown source policy selection type {!r} in the configuration!'.format(
                self.config.source_policy_selection_type))

        return source_policy_idx

    def _optimize_policy(self, target_goal, source_policy):
        if self.config.policy_optimization_type == 'random':
            policy_parameters = self._output_space.mutate(source_policy)

        elif self.config.policy_optimization_type == 'none':
            policy_parameters = source_policy

        else:
            raise ValueError(
                'Unknown policy optimization type {!r} in the configuration!'.format(self.config.policy_optimization))

        return policy_parameters


    def emit(self):

        # random sampling if not enough in library
        if self.CURRENT_RUN_INDEX < self.config.num_of_random_initialization:
            with torch.no_grad():
                policy_parameters = self._output_space.sample()
                policy_parameters = map_nested_dicts(policy_parameters, lambda x: x.to(self.config.tensors_device) if torch.is_tensor(x) else x)

        else:
            # sample a goal space from the goal space
            target_goal = self._get_next_goal()

            # get source policy which should be mutated
            history = self._access_history()
            source_policy_idx = self._get_source_policy_idx(target_goal)
            source_policy = history[int(source_policy_idx)]['output']

            with torch.no_grad():
                policy_parameters = self._optimize_policy(target_goal, source_policy)

        return policy_parameters


    def archive(self, parameters, observations):
        pass

    def optimize(self):
        pass

    def save(self):
        return {'input_space': self._input_space}

    def load(self, saved_dict):
        self._input_space = saved_dict['input_space']

