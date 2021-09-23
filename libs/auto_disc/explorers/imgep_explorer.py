from addict import Dict
from auto_disc.explorers import BaseExplorer

from auto_disc.utils.config_parameters import StringConfigParameter, DecimalConfigParameter, IntegerConfigParameter, BooleanConfigParameter

import random
import torch

from copy import deepcopy

@StringConfigParameter(name="source_policy_selection_type", possible_values=["optimal", "random"], default="optimal")
@StringConfigParameter(name="goal_selection_type", possible_values=["random", "specific", "function", None], default="random")
@IntegerConfigParameter(name="num_of_random_initialization", default=10, min=1)
@BooleanConfigParameter(name="use_exandable_goal_space", default=True)

class IMGEPExplorer(BaseExplorer):
    """
    Basic explorer that samples goals in a goalspace and uses a policy library to generate parameters to reach the goal.
    """
    CONFIG_DEFINITION = {}
    config = Dict()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def initialize(self, input_space, output_space):
        super().initialize(input_space, output_space)
        if len(self._input_space) > 1:
            raise NotImplementedError("Only 1 vector can be accepted as input space")
        self._outter_input_space_key = list(self._input_space.spaces.keys())[0] # select first key in DictSpace

    # def expand_box_goal_space(self, space, observations):
    #     observations = observations.type(space.dtype)
    #     is_nan_mask = torch.isnan(observations)
    #     if is_nan_mask.sum() > 0:
    #         observations[is_nan_mask] = space.low[is_nan_mask]
    #         observations[is_nan_mask] = space.high[is_nan_mask]
    #     space.low = torch.min(space.low, observations)
    #     space.high = torch.max(space.high, observations)

    def _get_next_goal(self):
        """ Defines the next goal of the exploration. """

        if self.config.goal_selection_type == 'random':
            target_goal = self._input_space.sample()
        else:
            raise ValueError(
                'Unknown goal generation type {!r} in the configuration!'.format(self.config.goal_selection_type))

        return target_goal[self._outter_input_space_key]


    def _get_source_policy_idx(self, target_goal):
        history = self._access_history()
        goal_library = [h[self._outter_input_space_key] for h in history['input']] # get goal history as tensor

        if self.config.source_policy_selection_type == 'optimal':
            # get distance to other goals
            goal_distances = self._input_space[self._outter_input_space_key].calc_distance(target_goal, goal_library)

            # select goal with minimal distance
            source_policy_idx = torch.argmin(goal_distances)

        elif self.config.source_policy_selection_type == 'random':
            source_policy_idx = random.randint(0, len(goal_library)-1)

        else:
            raise ValueError('Unknown source policy selection type {!r} in the configuration!'.format(
                self.config.source_policy_selection_type))

        return source_policy_idx

    def emit(self):
        target_goal = None
        source_policy_idx = None
        policy_parameters = Dict()  # policy parameters (output of IMGEP policy)

        # random sampling if not enough in library
        if self.CURRENT_RUN_INDEX < self.config.num_of_random_initialization:
            # initialize the parameters
            policy_parameters = self._output_space.sample()
            # for parameter_key, parameter_space in self._output_space.items():
            #     policy_parameters[parameter_key] = sample_value(parameter_space)

        else:
            # sample a goal space from the goal space
            target_goal = self._get_next_goal()

            # get source policy which should be mutated
            history = self._access_history()
            source_policy_idx = self._get_source_policy_idx(target_goal)
            source_policy = history[int(source_policy_idx)]['output']

            policy_parameters = self._output_space.mutate(source_policy)

        # TODO: Target goal
        # run with parameters
        run_parameters = deepcopy(policy_parameters) #self._convert_policy_to_run_parameters(policy_parameters)

        return run_parameters


    def archive(self, parameters, observations):
        pass
        #if self.config.use_exandable_goal_space:
            #self._input_space[self._outter_input_space_key].expand(observations[self._outter_input_space_key])
            #self.expand_box_goal_space(self._input_space[self._outter_input_space_key], observations[self._outter_input_space_key])

    def optimize(self):
        pass

    def save(self):
        return {'input_space': self._input_space}

    def load(self, saved_dict):
        self._input_space = saved_dict['input_space']

