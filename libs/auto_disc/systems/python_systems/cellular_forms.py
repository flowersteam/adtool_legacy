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

GEN_PATH = Path('~/Documents/workspace/work/AutoDiscTool/experiment_results/gens')
EXEC_PATH = Path('~/Documents/workspace/work/AutoDiscTool/bin/cellularForms14_flowers_1.0')

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

""" =============================================================================================
System definition
============================================================================================= """
@StringConfigParameter(name="version", default="14_flowers_1.0", possible_values=["14_flowers_1.0"])
@IntegerConfigParameter(name="final_step", default=200, min=1, max=1000)
@IntegerConfigParameter(name="num_particles", default=200000)
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

        param_list = [str(v.item()) for v in list(run_parameters.values())]
        additional_params = [
            str(self.config.num_particles),
            str(self.config.resolution),
            str(self.config.frame_step),
            str(GEN_PATH / 'form'),
        ]
        bash_command = ' '.join([str(EXEC_PATH)] + param_list + additional_params)

        cwd = Path(__file__).absolute().parent
        process = subprocess.Popen(bash_command, stdout=subprocess.PIPE, cwd=str(cwd))
        # output, error = process.communicate()

        return torch.zeros(
            ConfigParameterBinding("resolution"),
            ConfigParameterBinding("resolution")
        )

    def step(self, action=None):

        if self.step_idx >= self.config.final_step:
            raise Exception("Final step already reached, please reset the system.")

        self.step_idx += 1

        timeout = ConfigParameterBinding('step_timeout')
        t0 = time.time()
        output = None

        while time.time() - t0 < timeout:
            # try to get the image file until timeout
            imgpath = GEN_PATH / f'form_ambocc.{four_digit_str(self.step_idx)}.png'
            try:
                img = Image.open(str(imgpath))
                output = torch.tensor(np.array(img))
                break
            except FileNotFoundError:
                time.sleep(0.5)

        current_observation = Dict()
        current_observation.state = output

        return current_observation

    def observe(self):
        return self._observations

    def close(self):
        pass
