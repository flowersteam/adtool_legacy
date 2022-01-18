from auto_disc.explorers import IMGEPExplorer
from auto_disc.utils.config_parameters import StringConfigParameter, IntegerConfigParameter, DictConfigParameter, BooleanConfigParameter
from auto_disc.utils.misc.tensorboard_utils import logger_add_image_list

from copy import copy
import io
import numpy as np
import os
from PIL import Image
import sys
import requests
from tensorboardX import SummaryWriter
import time
import torch

@StringConfigParameter(name="tensors_device", default="cpu", possible_values=["cuda", "cpu", ])

@BooleanConfigParameter(name="use_tensorboard", default=True)
@StringConfigParameter(name="tb_folder", default="./tensorboard_imgep/")
@IntegerConfigParameter(name="tb_record_loss_frequency", default=1, min=1) # TODO: replace tensorboard frequency with callbacks
@IntegerConfigParameter(name="tb_record_images_frequency", default=10, min=1)
class IMGEPExplorerGecko(IMGEPExplorer):
    def initialize(self, input_space, output_space):
        super().initialize(input_space,output_space)

        # Custom for GECKO
        self.load_target_img()
        self.set_optimizer()
        self.policy_parameters = {}

        self.set_tensorboard()


    def load_target_img(self):
        url = "https://raw.githubusercontent.com/googlefonts/noto-emoji/main/png/128/emoji_u1f98e.png"
        r = requests.get(url)
        img = Image.open(io.BytesIO(r.content))
        img = img.resize((256,256), Image.ANTIALIAS)
        with torch.no_grad():
            img = torch.as_tensor(np.float64(img)) / 255.0
            # premultiply RGB by Alpha
            img[..., :3] *= img[..., 3:]
            rgb, a = img[..., :3], img[..., 3:4].clamp(0.0, 1.0)
            img = 1.0 - a + rgb
            gray_target_img = img.matmul(torch.DoubleTensor([[0.2989, 0.5870, 0.1140]]).t()).squeeze()
            self.target_img  = (1.0 - gray_target_img).flatten().unsqueeze(0).to(self.config.tensors_device)

    def set_optimizer(self):
        self.optimized_parameters = torch.nn.ParameterDict({
                                    'init_state': torch.nn.Parameter(self._output_space['init_state'].sample()),
                                    'r': torch.nn.Parameter(self._output_space['r'].sample()),
                                    'rk': torch.nn.Parameter(self._output_space['rk'].sample()),
                                    'b': torch.nn.Parameter(self._output_space['b'].sample()),
                                    'w': torch.nn.Parameter(self._output_space['w'].sample()),
                                    'h': torch.nn.Parameter(self._output_space['h'].sample()),
                                    'm': torch.nn.Parameter(self._output_space['m'].sample()),
                                    's': torch.nn.Parameter(self._output_space['s'].sample())
                                    }).to(self.config.tensors_device)
        #TODO: differentiable and not differentiable
        self.optimizer = torch.optim.Adam([{'params': self.optimized_parameters.init_state, 'lr': 0.5e-1},
                                          {'params': self.optimized_parameters.r},
                                          {'params': self.optimized_parameters.rk},
                                          {'params': self.optimized_parameters.b},
                                          {'params': self.optimized_parameters.w},
                                          {'params': self.optimized_parameters.h},
                                          {'params': self.optimized_parameters.m},
                                          {'params': self.optimized_parameters.s},
                                           ], lr=1e-3)

        self.loss_buffer = []
        self.loss_buffer_size = 10

    def set_tensorboard(self):
        if not os.path.exists(self.config.tb_folder):
            os.makedirs(self.config.tb_folder)
        self.tensorboard_logger = SummaryWriter(self.config.tb_folder)


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
        self.start_run_time = time.time()

        # random sampling if not enough in library
        if self.CURRENT_RUN_INDEX < self.config.num_of_random_initialization:
            self.policy_parameters = self._output_space.sample()
            with torch.no_grad():
                for k,v in self.policy_parameters.items():
                    self.policy_parameters[k] = v.to(self.config.tensors_device)
        else:
            if (self.CURRENT_RUN_INDEX == self.config.num_of_random_initialization) or (len(self.loss_buffer)==self.loss_buffer_size and self.loss_running_average < 0.1):
                # get source policy which should be mutated
                history = self._access_history()
                source_policy_idx = self._get_source_policy_idx(self.target_img)
                source_policy = history[int(source_policy_idx)]['output']


                for k,v in source_policy.items():
                    if k in self.optimized_parameters.keys():
                        self.optimized_parameters[k].data = v.to(self.config.tensors_device)

            for k,v in self.optimized_parameters.items():
                self.policy_parameters[k] = v


        return self.policy_parameters


    def archive(self, parameters, observations):
        if self.CURRENT_RUN_INDEX < self.config.num_of_random_initialization:
            torch.cuda.empty_cache()
        else:
            #loss = (0.9*self.target_img - observations[self._outter_input_space_key]).pow(2).sum().sqrt()
            loss = self._input_space[self._outter_input_space_key].calc_distance(self.target_img, [observations[self._outter_input_space_key]])
            loss.backward()
            # for k,v in self.optimized_parameters.items():
            #     assert v.grad is not None
            self.optimizer.step()
            self.optimizer.zero_grad()
            torch.cuda.empty_cache()


            # update loss running average
            self.loss_buffer.append(loss.item())
            self.loss_buffer = self.loss_buffer[-self.loss_buffer_size:]
            loss_speed = torch.FloatTensor(self.loss_buffer[1:]) - torch.FloatTensor(self.loss_buffer[:-1])
            self.loss_running_average = loss_speed.mean().abs()

            self.end_run_time = time.time()

            # Tensorboard log loss
            if self.config.use_tensorboard and (self.CURRENT_RUN_INDEX % self.config.tb_record_loss_frequency == 0):
                self.tensorboard_logger.add_scalar('loss/', loss.item(), self.CURRENT_RUN_INDEX)
                self.tensorboard_logger.add_scalar('loss_running_average/', self.loss_running_average.item(), self.CURRENT_RUN_INDEX)
                self.tensorboard_logger.add_text('time/',
                                                 f'Run {self.CURRENT_RUN_INDEX}: {self.end_run_time - self.start_run_time} secs')

            # Tensorboard log reconstruction accuracy
            logger_add_image_list(self.tensorboard_logger,
                                  [self.target_img.view(1,256,256).cpu(), observations[self._outter_input_space_key].view(1,256,256).cpu()],
                                  "reconstructions", global_step=self.CURRENT_RUN_INDEX)


    def optimize(self):
        pass

    def save(self):
        return {'input_space': self._input_space}

    def load(self, saved_dict):
        self._input_space = saved_dict['input_space']
