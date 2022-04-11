import auto_disc
from auto_disc.explorers import BaseExplorer
from auto_disc.utils.config_parameters import BooleanConfigParameter, IntegerConfigParameter, DictConfigParameter, StringConfigParameter
from auto_disc.utils.misc.dict_utils import map_nested_dicts

from addict import Dict
from copy import deepcopy
import random
import torch

@StringConfigParameter(name="tensors_device", default="cpu", possible_values=["cuda", "cpu", ])
@BooleanConfigParameter(name="tensors_require_grad", default=False)

@IntegerConfigParameter(name="num_of_random_initialization", default=10, min=1)

@StringConfigParameter(name="goal_achievement_type", possible_values=["input_space_calc_distance", "mse", "custom"],  default="input_space_calc_distance")
@DictConfigParameter(name="goal_achievement_parameters", default={})

@StringConfigParameter(name="goal_sampling_type", possible_values=["input_space_sample", "specific", "custom"], default="input_space_sample")
@DictConfigParameter(name="goal_sampling_parameters", default={})

@StringConfigParameter(name="candidate_selection_type", possible_values=["optimal", "random", "previous", "custom"], default="optimal")
@DictConfigParameter(name="candidate_selection_parameters", default={})

@StringConfigParameter(name="candidate_optimization_type", possible_values=["output_space_mutate", "Adam", "none", "custom"], default="output_space_mutate")
@DictConfigParameter(name="candidate_optimization_parameters", default={})
#TODO: callbacks instead of "custom"?: e.g to reset optimizer when loss running average < epsilon, when human say to reset

@StringConfigParameter(name="outter_input_space_key", default="")
class IMGEPExplorer(BaseExplorer):
    """
    Basic explorer that samples goals in a goalspace and uses a policy library to generate parameters to reach the goal.
    """
    CONFIG_DEFINITION = {}
    

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize(self, input_space, output_space):
        super().initialize(input_space, output_space)
        if len(self._input_space) == 1 and self._outter_input_space_key is None:
            self._outter_input_space_key = list(self._input_space.spaces.keys())[0]
        assert self._outter_input_space_key is not None, self.logger.error("_outter_input_space_key must be specifed when len(input_space)>1")

        self.set_goal_achievement_loss()

        self.set_goal_sampler()

        self.set_candidate_selector()

        self.set_candidate_optimizer()


    def set_goal_achievement_loss(self):

        if self.config.goal_achievement_type == 'input_space_calc_distance':
            goal_achievement_loss = self._input_space[self._outter_input_space_key].calc_distance

        elif hasattr(torch.nn.functional, self.config.goal_achievement_type + "_loss"):
            goal_achievement_loss = eval(f"torch.nn.functional.{self.config.goal_achievement_type}_loss")

        elif self.config.goal_achievement_type == 'custom':
            gd = {'self': self, 'torch': torch}  # /!\ exec is risky so we only give access to some packages?
            ld = {}
            exec(self.config.goal_achievement_parameters["func_def"], gd, ld)
            goal_achievement_loss = ld['goal_achievement_loss']

        else:
            raise ValueError(
                'Unknown goal achievement type {!r} in the configuration!'.format(self.config.goal_achievement_type))

        self._goal_achievement_loss = lambda input, target: goal_achievement_loss(input, target,
                                                                                       **self.config.goal_achievement_parameters)

    def set_goal_sampler(self):
        if self.config.goal_sampling_type == 'input_space_sample':
            def goal_sampler():
                return self._input_space[self._outter_input_space_key].sample().to(self.config.tensors_device)

        elif self.config.goal_sampling_type == 'specific':
            # correct config is e.g. self.config.goal_selection_parameters = {"range(0,10)": "goal_1", "range(10,100)"": "goal_2", ...}
            def goal_sampler():
                for k,v in self.config.goal_sampling_parameters.items():
                    if self.CURRENT_RUN_INDEX in eval(k):
                       return eval(v).to(self.config.tensors_device)
                raise ValueError(f'Run idx {self.CURRENT_RUN_INDEX} misses target goal in the goal sampling parameters configuration!')

        elif self.config.goal_sampling_type == 'custom':
            gd = {'self': self, 'torch': torch}  # /!\ exec is risky so we only give access to some packages?
            ld = {}
            exec(self.config.goal_sampling_parameters["func_def"], gd, ld)
            goal_sampler = ld['goal_sampler']

        else:
            raise ValueError(
                'Unknown goal sampling type {!r} in the configuration!'.format(self.config.goal_sampling_type))

        self._goal_sampler = lambda: goal_sampler()

    def set_candidate_selector(self):
        if self.config.candidate_selection_type == 'optimal':
            def candidate_selector(target_goal):
                history = self._access_history()
                goal_library = torch.Tensor([h[self._outter_input_space_key] for h in history['input']])
                goal_distances = self._goal_achievement_loss(target_goal, goal_library)
                return torch.argmin(goal_distances)

        elif self.config.candidate_selection_type == 'random':
            def candidate_selector(target_goal):
                return random.randint(0, self.CURRENT_RUN_INDEX)

        elif self.config.candidate_selection_type == 'previous':
            def candidate_selector(target_goal):
                return self.CURRENT_RUN_INDEX - 1

        elif self.config.candidate_selection_type == 'custom':
            gd = {'self': self, 'torch': torch} #/!\ exec is risky so we only give access to some packages?
            ld = {}
            exec(self.config.candidate_selection_parameters["func_def"], gd, ld)
            candidate_selector = ld['candidate_selector']

        else:
            raise ValueError('Unknown source policy selection type {!r} in the configuration!'.format(
                self.config.candidate_selection_type))

        self._candidate_selector = lambda target_goal: candidate_selector(target_goal)

    def set_candidate_optimizer(self):

        if self.config.candidate_optimization_type == 'output_space_mutate':
            def candidate_optimizer(target_goal, candidate):
                return self._output_space.mutate(candidate)

        elif hasattr(torch.optim, self.config.candidate_optimization_type):
            self.set_sgd_optimizer()

            # The candidate optimizer gets back the values from optimized parameters
            def candidate_optimizer(target_goal, candidate):
                for k,v in self.optimized_parameters.items():
                    candidate[k] = v
                return candidate

        elif self.config.candidate_optimization_type == 'none':
            def candidate_optimizer(target_goal, candidate):
                return candidate

        elif self.config.candidate_optimization_type == 'custom':
            gd = {'self': self, 'torch': torch}  # /!\ exec is risky so we only give access to some packages?
            ld = {}
            exec(self.config.candidate_optimization_parameters["func_def"], gd, ld)
            candidate_optimizer = ld['candidate_optimizer']

        else:
            raise ValueError(
                'Unknown policy optimization type {!r} in the configuration!'.format(self.config.policy_optimization))

        self._candidate_optimizer = lambda target_goal, candidate: candidate_optimizer(target_goal, candidate,
                                                                          **self.config.candidate_optimizer_parameters)


    def set_sgd_optimizer(self, optimized_parameters_init={}):
        self.optimized_parameters = torch.nn.ParameterDict()
        optimizer_class = eval(f"torch.optim.{self.config.candidate_optimization_type}")

        for param_group_idx, (param_group_list, param_group_lr) in \
                enumerate(self.config.candidate_optimization_parameters.items()):

            param_group_list = param_group_list.split(",")

            for param_key in param_group_list:
                if param_key not in optimized_parameters_init.keys():
                    optimized_parameters_init[param_key] = self._output_space[param_key].sample()
                self.optimized_parameters[param_key] = torch.nn.Parameter(optimized_parameters_init[param_key])

            if param_group_idx == 0:
                self.sgd_optimizer = optimizer_class(
                    [self.optimized_parameters[param_key] for param_key in param_group_list],
                    lr=param_group_lr)
            else:
                self.sgd_optimizer.add_param_group({
                    'params': [self.optimized_parameters[param_key] for param_key in param_group_list],
                    'lr': param_group_lr
                })


    def emit(self):

        torch.set_grad_enabled(self.config.tensors_require_grad)

        # random sampling if not enough in library
        if self.CURRENT_RUN_INDEX < self.config.num_of_random_initialization:
            with torch.no_grad():
                policy_parameters = self._output_space.sample()

        else:
            # sample a goal space from the goal space
            self.target_goal = self._goal_sampler()

            # get candidate policy
            history = self._access_history()
            candidate_idx = self._candidate_selector(self.target_goal)
            candidate = deepcopy(history[int(candidate_idx)]['output'])

            # optimize candidate policy
            policy_parameters = self._candidate_optimizer(self.target_goal, candidate)

        policy_parameters = map_nested_dicts(policy_parameters,
                                             lambda x: x.to(self.config.tensors_device) if torch.is_tensor(x) else x)

        history = self._access_history()

        return Dict(policy_parameters)


    def archive(self, parameters, observations):
        if (self.CURRENT_RUN_INDEX >= self.config.num_of_random_initialization) \
            and hasattr(self, "sgd_optimizer"):

                goal_achievement_loss = self._goal_achievement_loss(self.target_goal,
                                                                    observations[self._outter_input_space_key].unsqueeze(0)
                                                                    )
                print(f"LOSS: {goal_achievement_loss.item()}")
                goal_achievement_loss.backward()
                # for k,v in self.optimized_parameters.items():
                #     assert v.grad is not None
                self.sgd_optimizer.step()
                self.sgd_optimizer.zero_grad()


        torch.cuda.empty_cache()



    def optimize(self):
        pass

    def save(self):
        return {'input_space': self._input_space}

    def load(self, saved_dict):
        self._input_space = saved_dict['input_space']

