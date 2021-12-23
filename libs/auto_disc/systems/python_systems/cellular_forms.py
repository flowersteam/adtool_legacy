from auto_disc.systems.python_systems import BasePythonSystem
from auto_disc.utils.config_parameters import StringConfigParameter, DecimalConfigParameter, IntegerConfigParameter
from auto_disc.utils.spaces.utils import ConfigParameterBinding
from auto_disc.utils.spaces import DictSpace, BoxSpace, DiscreteSpace
from auto_disc.utils.mutators import GaussianMutator

from addict import Dict
from auto_disc.utils.misc.torch_utils import SphericPad, roll_n, complex_mult_torch

import torch
import matplotlib.pyplot as plt
import numpy as np

import math
import time
import subprocess

from pathlib import Path
from PIL import Image

from matplotlib.animation import FuncAnimation
import io

""" =============================================================================================
Utils
============================================================================================= """

GEN_PATH = Path('/home/laetitia/Documents/workspace/work/AutomatedDiscoveryTool/experiment_results/gens')
EXEC_PATH = Path('bin/cellularForms14_flowers_1.0')
CWD = Path('home/laetitia/Documents/workspace/work/AutomatedDiscoveryTool')
# EXEC_PATH = 'echo'

class LogBoxSpace(BoxSpace):

    def sample(self):
        # assert type(self.high) == torch.float64

        bounded = self.bounded_below & self.bounded_above

        sample = (self.low.log() - self.high.log()) \
                  * torch.rand(self.high.shape, dtype=torch.float64) + \
                  self.high[bounded].log()

        return sample.exp()

def four_digit_str(n):
    if n > 9999:
        return '9999'
    zstr = '0' * (4 - len(str(n)))
    return zstr + str(n)

# TODO adapt gaussian mutator

def read_image(idx, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        # try to get the image file until timeout
        imgpath = GEN_PATH / f'form_ambocc.{four_digit_str(idx)}.png'
        try:
            img = Image.open(str(imgpath)).convert('L')
            img = torch.tensor(np.array(img))
            return img
        except FileNotFoundError:
            time.sleep(0.5)
    return None


""" =============================================================================================
System definition
============================================================================================= """
@StringConfigParameter(name="version", default="14_flowers_1.0", possible_values=["14_flowers_1.0"])
@IntegerConfigParameter(name="final_step", default=200, min=1, max=1000)
@IntegerConfigParameter(name="num_particles", default=2000)
@IntegerConfigParameter(name="resolution", default=1024)
@IntegerConfigParameter(name="frame_step", default=10)
@IntegerConfigParameter(name="step_timeout", default=10)
class CellularForm(BasePythonSystem):
    CONFIG_DEFINITION = {}
    config = Dict()

    input_space = DictSpace(
        food_inc_min = LogBoxSpace(low=0.01, high=10.0, shape=()),
        food_inc_rand = LogBoxSpace(low=0.01, high=10.0, shape=()),
        planar_factor = LogBoxSpace(low=0.01, high=10.0, shape=()),
        spring_factor = LogBoxSpace(low=0.01, high=10.0, shape=()),
        bulge_factor = LogBoxSpace(low=0.01, high=10.0, shape=()),
        short_split_factor = LogBoxSpace(low=0.01, high=10.0, shape=()),
        bend_split_factor = LogBoxSpace(low=0.01, high=10.0, shape=()),
        colinear_split_factor = LogBoxSpace(low=0.01, high=10.0, shape=()),
        opposite_edge_factor = LogBoxSpace(low=0.01, high=10.0, shape=()),
        random_split_factor = LogBoxSpace(low=0.01, high=10.0, shape=()),
        collision_radius = BoxSpace(low=0.0, high=5.0, shape=()),
        collision_strength = BoxSpace(low=0.0, high=5.0, shape=())
    )

    output_space = DictSpace(
        states = BoxSpace(low=0, high=1,
                          shape=(ConfigParameterBinding("final_step"),
                                 ConfigParameterBinding("resolution"),
                                 ConfigParameterBinding("resolution")))
    )

    step_output_space = DictSpace(
        states = BoxSpace(low=0, high=1,
                          shape=(ConfigParameterBinding("resolution"),
                                 ConfigParameterBinding("resolution")))
    )


    def reset(self, run_parameters):

        if not GEN_PATH.exists():
            GEN_PATH.mkdir(parents=True)

        param_list = [str(v.item()) for v in list(run_parameters.values())]
        additional_params = [
            str(self.config.num_particles),
            str(self.config.resolution),
            str(self.config.frame_step),
            str(GEN_PATH / 'form'),
        ]
        bash_command = ' '.join([str(EXEC_PATH)] + param_list + additional_params).split()

        # TODO terminate the process if it takes too much time
        process = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
        output, error = process.communicate()

        self.step_idx = 0
        self._observations = Dict()
        self._observations.states = torch.empty(self.config.final_step,
                                                self.config.resolution,
                                                self.config.resolution)

        output = torch.zeros(
            self.config.resolution,
            self.config.resolution,
        )
        img = read_image(self.step_idx, self.config.step_timeout)
        if img is not None:
            output = img

        self._observations.states[0] = output

        return output

    def step(self, action=None):

        if self.step_idx >= self.config.final_step:
            raise Exception("Final step already reached, please reset the system.")

        self.step_idx += 1

        timeout = self.config.step_timeout
        output = torch.zeros(
            self.config.resolution,
            self.config.resolution,
        )

        img = read_image(self.step_idx, timeout)
        if img is not None:
            output = img

        current_observation = Dict()
        current_observation.state = output

        self._observations.states[self.step_idx] = current_observation.state

        return current_observation, 0, self.step_idx >= self.config.final_step - 1, None

    def observe(self):
        return self._observations

    def render(self):
        im_array = []
        for img in self._observations.states:
            im_array.append(Image.fromarray(img.cpu().detach().numpy()))

        byte_img = io.BytesIO()
        # TODO gif from np.array
        im_array[0].save(byte_img, format='GIF', save_all=True, append_images=im_array[1:], duration=100, loop=0)
        # return (Image.fromarray(im_array[0]), 'png')
        return (byte_img, 'gif')

    def close(self):
        pass