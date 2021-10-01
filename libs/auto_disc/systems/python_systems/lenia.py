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

from matplotlib.animation import FuncAnimation
import io
import imageio

""" =============================================================================================
System definition
============================================================================================= """
@StringConfigParameter(name="version", possible_values=["pytorch_fft", "pytorch_conv2d"], default="pytorch_fft")
@IntegerConfigParameter(name="SX", default=256, min=1)
@IntegerConfigParameter(name="SY", default=256, min=1)
@IntegerConfigParameter(name="final_step", default=200, min=1, max=1000)
@IntegerConfigParameter(name="scale_init_state", default=1, min=1)
class Lenia(BasePythonSystem):
    CONFIG_DEFINITION = {}
    config = Dict()
    
    input_space = DictSpace(
        init_state = BoxSpace(low=0.0, high=1.0, mutator=GaussianMutator(mean=0.0, std=0.5), 
                              shape=(ConfigParameterBinding("SX") // ConfigParameterBinding("scale_init_state"), 
                                     ConfigParameterBinding("SY") // ConfigParameterBinding("scale_init_state"))),
        R = DiscreteSpace(n=20, mutator=GaussianMutator(mean=0.0, std=0.5), indpb=1.0),
        T = BoxSpace(low=1.0, high=20.0, mutator=GaussianMutator(mean=0.0, std=0.5), shape=(), indpb=1.0, dtype=torch.float32),
        b = BoxSpace(low=0.0, high=1.0, mutator=GaussianMutator(mean=0.0, std=0.1), shape=(4,), indpb=1.0, dtype=torch.float32),
        m = BoxSpace(low=0.0, high=1.0, mutator=GaussianMutator(mean=0.0, std=0.1), shape=(), indpb=1.0, dtype=torch.float32),
        s = BoxSpace(low=0.001, high=0.3, mutator=GaussianMutator(mean=0.0, std=0.05), shape=(), indpb=1.0, dtype=torch.float32),
        # kn = DiscreteSpace(n=4, mutator=GaussianMutator(mean=0.0, std=0.1), indpb=1.0),
        # gn = DiscreteSpace(n=3, mutator=GaussianMutator(mean=0.0, std=0.1), indpb=1.0),
    )

    output_space = DictSpace(
        states = BoxSpace(low=0, high=1, shape=(ConfigParameterBinding("final_step"), ConfigParameterBinding("SX"), ConfigParameterBinding("SY")))
    )

    step_output_space = DictSpace(
        state = BoxSpace(low=0, high=1, shape=(ConfigParameterBinding("SX"), ConfigParameterBinding("SY")))
    )

    def reset(self, run_parameters):
        run_parameters.kn = 0
        run_parameters.gn = 1
        init_state = torch.zeros(1,1, self.config.SY, self.config.SX, dtype=torch.float64)
        init_state[
            0,
            0, 
            self.config.SY // 2 - math.ceil(self.input_space['init_state'].shape[0] / 2):self.config.SY // 2 + self.input_space['init_state'].shape[0] // 2,
            self.config.SX // 2 - math.ceil(self.input_space['init_state'].shape[0] / 2):self.config.SX // 2 + self.input_space['init_state'].shape[0] // 2
        ] = run_parameters.init_state
        # self.state = init_state.to(self.device)
        self.state = init_state
        del run_parameters.init_state

        if self.config.version.lower() == 'pytorch_fft':
            self.automaton = LeniaStepFFT(SX=self.config.SX, SY=self.config.SY, **run_parameters)
        elif self.config.version.lower() == 'pytorch_conv2d':
            self.automaton = LeniaStepConv2d(**run_parameters)
        else:
            raise ValueError('Unknown lenia version (config.version = {!r})'.format(self.config.version))

        self._observations = Dict()
        # self._observations.timepoints = list(range(self.config.final_step))
        self._observations.states = torch.empty((self.config.final_step, 1, self.config.SX, self.config.SY))
        self._observations.states[0] = self.state[0]

        self.step_idx = 0

        current_observation = Dict()
        current_observation.state = self._observations.states[0]

        return current_observation

    def step(self, action=None):
        if self.step_idx >= self.config.final_step:
            raise Exception("Final step already reached, please reset the system.")

        self.step_idx += 1
        self.state = self.automaton(self.state)

        self._observations.states[self.step_idx] = self.state[0]

        current_observation = Dict()
        current_observation.state = self._observations.states[self.step_idx]

        return current_observation, 0, self.step_idx >= self.config.final_step - 1, None

    def observe(self):
        return self._observations

    def render(self, mode="PIL_image"):

        colormap = create_colormap(np.array(
            [[255, 255, 255], [119, 255, 255], [23, 223, 252], [0, 190, 250], [0, 158, 249], [0, 142, 249],
             [81, 125, 248], [150, 109, 248], [192, 77, 247], [232, 47, 247], [255, 9, 247], [200, 0, 84]]) / 255 * 8)
        im_array = []
        for img in self._observations.states:
            im = im_from_array_with_colormap(img.squeeze().cpu().detach().numpy(), colormap)
            im_array.append(im.convert('RGB'))
        

        if mode == "human":
            fig = plt.figure(figsize=(4, 4))
            animation = FuncAnimation(fig, lambda frame: plt.imshow(frame), frames=im_array)
            plt.axis('off')
            plt.tight_layout()
            return plt.show()
        elif mode == "PIL_image":
            byte_img = io.BytesIO()
            imageio.mimwrite(byte_img, im_array, 'mp4',  fps=30, output_params=["-f", "mp4"])
            return (byte_img, "mp4")
        else:
            raise NotImplementedError

    def close(self):
        pass

""" =============================================================================================
Lenia Main
============================================================================================= """

from PIL import Image

def create_colormap(colors, is_marker_w=True):
    MARKER_COLORS_W = [0x5F, 0x5F, 0x5F, 0x7F, 0x7F, 0x7F, 0xFF, 0xFF, 0xFF]
    MARKER_COLORS_B = [0x9F, 0x9F, 0x9F, 0x7F, 0x7F, 0x7F, 0x0F, 0x0F, 0x0F]
    nval = 253
    ncol = colors.shape[0]
    colors = np.vstack((colors, np.array([[0, 0, 0]])))
    v = np.repeat(range(nval), 3)  # [0 0 0 1 1 1 ... 252 252 252]
    i = np.array(list(range(3)) * nval)  # [0 1 2 0 1 2 ... 0 1 2]
    k = v / (nval - 1) * (ncol - 1)  # interpolate between 0 .. ncol-1
    k1 = k.astype(int)
    c1, c2 = colors[k1, i], colors[k1 + 1, i]
    c = (k - k1) * (c2 - c1) + c1  # interpolate between c1 .. c2
    return np.rint(c / 8 * 255).astype(int).tolist() + (MARKER_COLORS_W if is_marker_w else MARKER_COLORS_B)


def im_from_array_with_colormap(np_array, colormap):
    '''
    Function that transforms the color palette of a PIL image

    input:
        - image: the PIL image to transform
        - colormap: the desired colormap
    output: the transformed PIL image
    '''
    image_array = np.uint8(np_array.astype(float) * 252.0)
    transformed_image = Image.fromarray(image_array)
    transformed_image.putpalette(colormap)

    return transformed_image


""" =============================================================================================
Lenia Main
============================================================================================= """

# Lenia family of functions for the kernel K and for the growth mapping g
kernel_core = {
    0: lambda r: (4 * r * (1 - r)) ** 4,  # polynomial (quad4)
    1: lambda r: torch.exp(4 - 1 / (r * (1 - r))),  # exponential / gaussian bump (bump4)
    2: lambda r, q=1 / 4: (r >= q).double() * (r <= 1 - q).double(),  # step (stpz1/4)
    3: lambda r, q=1 / 4: (r >= q).double() * (r <= 1 - q).double() + (r < q).double() * 0.5  # staircase (life)
}
field_func = {
    0: lambda n, m, s: torch.max(torch.zeros_like(n), 1 - (n - m) ** 2 / (9 * s ** 2)) ** 4 * 2 - 1,
    # polynomial (quad4)
    1: lambda n, m, s: torch.exp(- (n - m) ** 2 / (2 * s ** 2)) * 2 - 1,  # exponential / gaussian (gaus)
    2: lambda n, m, s: (torch.abs(n - m) <= s).double() * 2 - 1  # step (stpz)
}


# Lenia Step FFT version (faster)
class LeniaStepFFT(torch.nn.Module):
    """ Module pytorch that computes one Lenia Step with the fft version"""

    def __init__(self, R, T, b, m, s, kn, gn, is_soft_clip=False, SX=256, SY=256, device='cpu'):
        torch.nn.Module.__init__(self)

        self.register_buffer('R', R+2)
        self.register_parameter('T', torch.nn.Parameter(T))
        self.register_buffer('b', b)
        self.register_parameter('m', torch.nn.Parameter(m))
        self.register_parameter('s', torch.nn.Parameter(s))

        self.kn = 0
        self.gn = 1

        self.SX = SX
        self.SY = SY
        self.spheric_pad = SphericPad(self.R)
        self.is_soft_clip = is_soft_clip

        self.device = device

        self.compute_kernel()

    def compute_kernel(self):

        # implementation of meshgrid in torch
        x = torch.arange(self.SX)
        y = torch.arange(self.SY)
        xx = x.repeat(self.SY, 1)
        yy = y.view(-1, 1).repeat(1, self.SX)
        X = (xx - int(self.SX / 2)).double() / float(self.R)
        Y = (yy - int(self.SY / 2)).double() / float(self.R)

        # distance to center in normalized space
        D = torch.sqrt(X ** 2 + Y ** 2)

        # kernel
        k = len(self.b)  # modification to allow b always of length 4
        kr = k * D
        b = self.b[torch.min(torch.floor(kr).long(), (k - 1) * torch.ones_like(kr).long())]
        kfunc = kernel_core[self.kn]
        kernel = (D < 1).double() * kfunc(torch.min(kr % 1, torch.ones_like(kr))) * b
        kernel_sum = torch.sum(kernel)
        # normalization of the kernel
        self.kernel_norm = (kernel / kernel_sum).unsqueeze(0).unsqueeze(0)
        # fft of the kernel
        if torch.__version__ >= "1.7.1":
            self.kernel_FFT = torch.fft.rfft2(self.kernel_norm).to(self.device)
        else:
            self.kernel_FFT = torch.rfft(self.kernel_norm, signal_ndim=2, onesided=False).to(self.device)


    def forward(self, input):

        if torch.__version__ >= "1.7.1":
            world_FFT = torch.fft.rfft2(input)
            potential_FFT = self.kernel_FFT * world_FFT  # complex mutliplication
            potential = torch.fft.irfft2(potential_FFT)
        else:
            world_FFT = torch.rfft(input, signal_ndim=2, onesided=False)
            potential_FFT = complex_mult_torch(self.kernel_FFT, world_FFT)  # complex mutliplication
            potential = torch.irfft(potential_FFT, signal_ndim=2, onesided=False)

        potential = roll_n(potential, 3, potential.detach().size(3) // 2)
        potential = roll_n(potential, 2, potential.detach().size(2) // 2)

        gfunc = field_func[self.gn]
        field = gfunc(potential, self.m, self.s)

        if not self.is_soft_clip:
            output_img = torch.clamp(input + (1.0 / self.T) * field, min=0., max=1.)
        else:
            output_img = soft_clip(input + (1.0 / self.T) * field, 0, 1, self.T)

        if torch.any(torch.isnan(potential)):
            raise AssertionError(f"lenia paremeters {self.T, self.R, self.b, self.m, self.s} are causing NaN values")

        return output_img


# Lenia Step Conv2D version
class LeniaStepConv2d(torch.nn.Module):
    """ Module pytorch that computes one Lenia Step with the conv2d version"""

    def __init__(self, R, T, b, m, s, kn, gn, is_soft_clip=False, device='cpu'):
        torch.nn.Module.__init__(self)


        self.register_buffer('R', R+2)
        self.register_parameter('T', torch.nn.Parameter(T))
        self.register_buffer('b', b)
        self.register_parameter('m', torch.nn.Parameter(m))
        self.register_parameter('s', torch.nn.Parameter(s))


        self.kn = 0
        self.gn = 1

        self.spheric_pad = SphericPad(self.R)
        self.is_soft_clip = is_soft_clip

        self.device = device

        self.compute_kernel()


    def compute_kernel(self):
        SY = 2 * self.R + 1
        SX = 2 * self.R + 1

        # implementation of meshgrid in torch
        x = torch.arange(SX)
        y = torch.arange(SY)
        xx = x.repeat(SY, 1)
        yy = y.view(-1, 1).repeat(1, SX)
        X = (xx - int(SX / 2)).double() / float(self.R)
        Y = (yy - int(SY / 2)).double() / float(self.R)

        # distance to center in normalized space
        D = torch.sqrt(X ** 2 + Y ** 2)

        # kernel
        k = len(self.b)
        kr = k * D
        b = self.b[torch.min(torch.floor(kr).long(), (k - 1) * torch.ones_like(kr).long())]
        kfunc = kernel_core[self.kn]
        kernel = (D < 1).double() * kfunc(torch.min(kr % 1, torch.ones_like(kr))) * b
        kernel_sum = torch.sum(kernel)
        # normalization of the kernel
        self.kernel_norm = (kernel / kernel_sum).unsqueeze(0).unsqueeze(0).to(self.device)


    def forward(self, input):
        potential = torch.nn.functional.conv2d(self.spheric_pad(input), weight=self.kernel_norm)
        gfunc = field_func[self.gn]
        field = gfunc(potential, self.m, self.s)

        if not self.is_soft_clip:
            output_img = torch.clamp(input + (1.0 / self.T) * field, min=0., max=1.)
        else:
            output_img = soft_clip(input + (1.0 / self.T) * field, 0, 1, self.T)

        return output_img