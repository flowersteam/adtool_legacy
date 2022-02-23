from addict import Dict
from auto_disc.explorers import IMGEPExplorer
from auto_disc.utils.config_parameters import StringConfigParameter
import random
import torch

@StringConfigParameter(name="goal_space_selection_type", possible_values=["random"], default="random")

class IMGEPHOLMESExplorer(IMGEPExplorer):
    CONFIG_DEFINITION = {}
    config = Dict()

    def _get_next_goal_space(self):
        """ Defines the goal space sampling policy. """

        if self.config.goal_space_selection_type == 'random':
            goal_space_key = random.choice(list(self._input_space[self._outter_input_space_key].spaces.keys()))

        else:
            raise ValueError(
                'Unknown goal space generation type {!r} in the configuration!'.format(self.config.goal_space_selection_type))

        return goal_space_key


    def _get_next_goal(self):
        """ Defines the goal sampling policy. """

        goal_space_key = self._get_next_goal_space()

        if self.config.goal_selection_type == 'random':
            target_goal = self._input_space[self._outter_input_space_key][goal_space_key].sample()

        else:
            raise ValueError(
                'Unknown goal generation type {!r} in the configuration!'.format(self.config.goal_selection_type))

        return (goal_space_key, target_goal)


    def _get_source_policy_idx(self, target_goal):
        goal_space_key, target_goal = target_goal
        history = self._access_history()
        goal_library = [h[self._outter_input_space_key][goal_space_key] for h in history['input']] # get goal history as tensor

        if self.config.source_policy_selection_type == 'optimal':
            # get distance to other goals
            goal_distances = self._input_space[self._outter_input_space_key][goal_space_key].calc_distance(target_goal, goal_library)

            # select goal with minimal distance
            source_policy_idx = torch.argmin(goal_distances)

        elif self.config.source_policy_selection_type == 'random':
            source_policy_idx = random.randint(0, self.CURRENT_RUN_INDEX)


        else:
            raise ValueError('Unknown source policy selection type {!r} in the configuration!'.format(
                self.config.source_policy_selection_type))

        return source_policy_idx

