from addict import Dict
from auto_disc.explorers import BaseExplorer
from auto_disc.utils.sampling import sample_value
from auto_disc.utils.sampling import mutate_value

from auto_disc.utils.auto_disc_parameters import ParameterTypesEnum, AutoDiscParameter

from tqdm import tqdm
import torch
from torch import nn

import numpy as np

class IMGEPExplorer(BaseExplorer):
    """
    Basic explorer that samples goals in a goalspace and uses a policy library to generate parameters to reach the goal.
    """

    CONFIG_DEFINITION = [
        AutoDiscParameter(
                    name="num_of_random_initialization", 
                    type=ParameterTypesEnum.get('INTEGER'), 
                    values_range=[1, np.inf], 
                    default=10),
        AutoDiscParameter(
                    name="source_policy_selection_type", 
                    type=ParameterTypesEnum.get('STRING'), 
                    values_range=["optimal", "random"], 
                    default="optimal"),
        AutoDiscParameter(
                    name="goal_selection_type", 
                    type=ParameterTypesEnum.get('STRING'), 
                    values_range=["random", "specific", "function", None], 
                    default="random"),
    ]

    # @staticmethod
    # def default_config():
    #     default_config = AttrDict()
    #     # base config
    #     default_config.num_of_random_initialization = 10  # number of random runs at the beginning of exploration to populate the IMGEP memory

    #     # Pi: policy parameters config
    #     default_config.source_policy_selection = AttrDict()
    #     default_config.source_policy_selection.type = 'optimal'  # either: 'optimal', 'random'
    #     default_config.policy_parameters = []  # config to init and mutate the run parameters

    #     # R: goal space representation config
    #     default_config.goal_space_representation = AttrDict()

    #     # G: goal selection config
    #     default_config.goal_selection = AttrDict()
    #     default_config.goal_selection.type = None  # either: 'random', 'function'

    #     # Optimizer to reach goal
    #     default_config.reach_goal_optimizer = AttrDict()
    #     default_config.reach_goal_optimizer.optim_steps = 10
    #     default_config.reach_goal_optimizer.name = "SGD"
    #     default_config.reach_goal_optimizer.parameters = AttrDict()
    #     default_config.reach_goal_optimizer.parameters.lr = 0.1
    #     default_config.reach_goal_optimizer.is_scheduler = False
    #     # default_config.reach_goal_optimizer.scheduler = AttrDict()
    #     # default_config.reach_goal_optimizer.scheduler.name = "ExponentialLR"
    #     # default_config.reach_goal_optimizer.scheduler.parameters = {"gamma": 0.9999}

    #     return default_config

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # initialize policy library
        self.policy_library = []
    
    def initialize(self, input_space, output_space, input_distance_fn):
        super().initialize(input_space, output_space, input_distance_fn)
        if len(self._input_space) > 1:
            raise NotImplementedError("Only 1 vector can be accepted as input space")
        self._input_space = self._input_space[next(iter(self._input_space))]
        # initialize goal library
        self.goal_library = torch.empty((0, self._input_space.dims[0]))

    def _get_next_goal(self):
        """ Defines the next goal of the exploration. """

        if self.config.goal_selection_type == 'random':
            target_goal = sample_value(self._input_space)
        else:
            raise ValueError(
                'Unknown goal generation type {!r} in the configuration!'.format(self.config.goal_selection_type))

        return target_goal


    def _get_source_policy_idx(self, target_goal):

        if self.config.source_policy_selection_type == 'optimal':
            # get distance to other goals
            goal_distances = self._input_distance_fn(target_goal, self.goal_library)

            # select goal with minimal distance
            source_policy_idx = torch.argmin(goal_distances)
        elif self.config.source_policy_selection_type == 'random':
            source_policy_idx = sample_value(('discrete', 0, len(self.goal_library)-1))
        else:
            raise ValueError('Unknown source policy selection type {!r} in the configuration!'.format(
                self.config.source_policy_selection_type))

        return source_policy_idx


    # def _convert_policy_to_run_parameters(self, policy_parameters):
    #     run_parameters = AttrDict()
    #     for policy_param_conf in self.config.policy_parameters:
    #         run_parameters[policy_param_conf.name] = policy_parameters[policy_param_conf.name].clamp(policy_param_conf.mutate['min'], policy_param_conf.mutate['max'])

    #     run_parameters['init_state'] = torch.zeros((self.system.config.SX, self.system.config.SY))
    #     run_parameters['init_state'][self.system.config.SX//2, self.system.config.SY//2] = 1.0
    #     #run_parameters['init_state'] = self.config.goal_selection.config.lenia_animals[20]

    #     run_parameters['R'] = 13
    #     run_parameters['T'] = 10
    #     run_parameters['b'] = torch.tensor([1])
    #     run_parameters['kn'] = 1
    #     run_parameters['gn'] = 1
    #     return run_parameters

    def emit(self):
        target_goal = None
        source_policy_idx = None
        policy_parameters = Dict()  # policy parameters (output of IMGEP policy)

        # random sampling if not enough in library
        if len(self.policy_library) < self.config.num_of_random_initialization:
            # initialize the parameters
            for parameter_key, parameter_space in self._output_space.items():
                policy_parameters[parameter_key] = sample_value(parameter_space)

        else:
            # sample a goal space from the goal space
            target_goal = self._get_next_goal()

            # get source policy which should be mutated
            source_policy_idx = self._get_source_policy_idx(target_goal)
            source_policy = self.policy_library[source_policy_idx]

            for parameter_key, parameter_space in self._output_space.items():
                
                if parameter_space.mutation != None:
                    policy_parameter = mutate_value.mutate_value(source_policy[parameter_key], parameter_space)
                else:
                    raise ValueError(
                        'Unknown run_parameter type {!r} in configuration.'.format(parameter_space.mutation.distribution))

                policy_parameters[parameter_key] = source_policy[parameter_key]

        # TODO: Target goal
        # run with parameters
        run_parameters = policy_parameters #self._convert_policy_to_run_parameters(policy_parameters)
        self.policy_library.append(policy_parameters)

        return run_parameters


    def archive(self, parameters, observations):
        self.goal_library = torch.cat([self.goal_library, observations.reshape(1,-1)]) 

    def optimize(self):
        pass


    # def run(self, n_exploration_runs):

    #     self.policy_library = []
    #     self.goal_library = torch.empty((0, self.goal_space_representation.n_latents))

    #     print('Exploration: ')
    #     progress_bar = tqdm(total=n_exploration_runs)
    #     run_idx = 0
    #     while run_idx < n_exploration_runs:

    #         target_goal = None
    #         source_policy_idx = None
    #         policy_parameters = AttrDict()  # policy parameters (output of IMGEP policy)

    #         # random sampling if not enough in library
    #         if len(self.policy_library) < self.config.num_of_random_initialization:
    #             # initialize the parameters
    #             for parameter_config in self.config.policy_parameters:
    #                 if parameter_config.type == "sampling":
    #                     policy_parameters[parameter_config['name']] = sampling.sample_value(parameter_config['init'])
    #                 else:
    #                     raise ValueError('Unknown run_parameter type {!r} in configuration.'.format(parameter_config.type))

    #         else:

    #             # sample a goal space from the goal space
    #             target_goal = self.get_next_goal()

    #             # get source policy which should be mutated
    #             source_policy_idx = self.get_source_policy_idx(target_goal)
    #             source_policy = self.policy_library[source_policy_idx]

    #             for parameter_config in self.config.policy_parameters:

    #                 if parameter_config.type == 'sampling':
    #                     policy_parameter = sampling.mutate_value(val=source_policy[parameter_config['name']], config=parameter_config['mutate'])
    #                 else:
    #                     raise ValueError(
    #                         'Unknown run_parameter type {!r} in configuration.'.format(parameter_config.type))

    #                 policy_parameters[parameter_config['name']] = policy_parameter

    #         if self.config.reach_goal_optimizer.optim_steps > 0 and target_goal is not None:
    #             # make policy parameters as nn.Parameters
    #             for param_k, param_v in policy_parameters.items():
    #                 policy_parameters[param_k] = nn.Parameter(param_v)
    #             optimizer_class = eval(f'torch.optim.{self.config.reach_goal_optimizer.name}')
    #             self.optimizer = optimizer_class(list(policy_parameters.values()), **self.config.reach_goal_optimizer.parameters)
    #             if self.config.reach_goal_optimizer.is_scheduler:
    #                 self.scheduler = eval(f'torch.optim.lr_scheduler.{self.config.reach_goal_optimizer.scheduler.name}')(self.optimizer, **self.config.reach_goal_optimizer.scheduler.parameters)

    #             print(f'Run {run_idx}, optimisation toward goal: ')
    #             for optim_step_idx in tqdm(range(1, self.config.reach_goal_optimizer.optim_steps)):

    #                 # run with parameters
    #                 run_parameters = self.convert_policy_to_run_parameters(policy_parameters)
    #                 observations = self.system.run(run_parameters=run_parameters)
    #                 reached_goal = self.goal_space_representation.calc(observations)

    #                 # compute error between reached_goal and target_goal
    #                 loss = self.goal_space_representation.calc_distance(target_goal, reached_goal)
    #                 print(f'step {optim_step_idx}: distance to target={loss.item():0.2f}')

    #                 # optimisation step
    #                 self.optimizer.zero_grad()
    #                 loss.backward()
    #                 self.optimizer.step()
    #                 if self.config.reach_goal_optimizer.is_scheduler:
    #                     self.scheduler.step()

    #                 if optim_step_idx>5 and abs(old_loss-loss.item()) < 1e-4:
    #                     break;
    #                 old_loss = loss.item()

    #             # remove grad
    #             for param_k, param_v in policy_parameters.items():
    #                 policy_parameters[param_k] = param_v.data

    #             dist_to_target = loss.item()


    #         else:
    #             # run with parameters
    #             run_parameters = self.convert_policy_to_run_parameters(policy_parameters)
    #             observations = self.system.run(run_parameters=run_parameters)
    #             reached_goal = self.goal_space_representation.calc(observations)
    #             optim_step_idx = 0
    #             dist_to_target = None


    #         # save results
    #         self.db.add_run_data(id=run_idx,
    #                                run_parameters=run_parameters,
    #                                observations=observations,
    #                                source_policy_idx=source_policy_idx,
    #                                target_goal=target_goal,
    #                                reached_goal=reached_goal,
    #                                n_optim_steps_to_reach_goal=optim_step_idx,
    #                                dist_to_target=dist_to_target)

    #         # add policy and reached goal into the libraries
    #         # do it after the run data is saved to not save them if there is an error during the saving
    #         self.policy_library.append(policy_parameters)
    #         self.goal_library = torch.cat([self.goal_library, reached_goal.reshape(1,-1)])

    #         # increment run_idx
    #         run_idx += 1
    #         progress_bar.update(1)

