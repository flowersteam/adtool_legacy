from auto_disc.systems.python_systems import BasePythonSystem
from auto_disc.utils.config_parameters import StringConfigParameter, IntegerConfigParameter, BooleanConfigParameter
from auto_disc.utils.spaces.utils import ConfigParameterBinding
from auto_disc.utils.spaces import DictSpace, BoxSpace, DiscreteSpace, MultiDiscreteSpace
from auto_disc.utils.mutators import GaussianMutator

from addict import Dict
from auto_disc.utils.misc.torch_utils import roll_n, complex_mult_torch

import torch
import matplotlib.pyplot as plt
import numpy as np

import math

from matplotlib.animation import FuncAnimation
from matplotlib import colors
import io
import imageio

from PIL import Image

""" =============================================================================================
System definition
============================================================================================= """
@StringConfigParameter(name="version", possible_values=["pytorch_fft", "pytorch_conv2d"], default="pytorch_fft")
@StringConfigParameter(name="size", default="(256,256)")
@IntegerConfigParameter(name="final_step", default=200, min=1, max=1000)
@IntegerConfigParameter(name="scale_init_state", default=1, min=1)
@IntegerConfigParameter(name="nb_k", default=10, min=1)
@IntegerConfigParameter(name="nb_c", default=1, min=1)
@BooleanConfigParameter(name="wall_c", default=True)

#TODO: other env kernels
class LeniaExpandedDiff(BasePythonSystem):
    CONFIG_DEFINITION = {}
    config = Dict()
    
    input_spaces = dict(
        #init_state=BoxSpace(low=0.0, high=1.0, mutator=GaussianMutator(mean=0.0, std=0.01), indpb=0.0, dtype=torch.float32, shape=()),
        R=DiscreteSpace(n=25, mutator=GaussianMutator(mean=0.0, std=0.01), indpb=0.01),
        T=BoxSpace(low=1.0, high=10.0, mutator=GaussianMutator(mean=0.0, std=0.1), shape=(), indpb=0.01, dtype=torch.float32),
        #c0=MultiDiscreteSpace(nvec=[1] * 10, mutator=GaussianMutator(mean=0.0, std=0.1), indpb=0.1),
        #c1=MultiDiscreteSpace(nvec=[1] * 10, mutator=GaussianMutator(mean=0.0, std=0.1), indpb=0.1),
        rk=BoxSpace(low=0, high=1, shape=(ConfigParameterBinding("nb_k"), 3), mutator=GaussianMutator(mean=0.0, std=0.2), indpb=0.25, dtype=torch.float32),
        b=BoxSpace(low=0.0, high=1.0, shape=(ConfigParameterBinding("nb_k"), 3), mutator=GaussianMutator(mean=0.0, std=0.2), indpb=0.25, dtype=torch.float32),
        w=BoxSpace(low=0.01, high=0.5, shape=(ConfigParameterBinding("nb_k"), 3), mutator=GaussianMutator(mean=0.0, std=0.2), indpb=0.25, dtype=torch.float32),
        m=BoxSpace(low=0.05, high=0.5, shape=(ConfigParameterBinding("nb_k"), ), mutator=GaussianMutator(mean=0.0, std=0.2), indpb=0.25, dtype=torch.float32),
        s=BoxSpace(low=0.001, high=0.18, shape=(ConfigParameterBinding("nb_k"), ), mutator=GaussianMutator(mean=0.0, std=0.01), indpb=0.25, dtype=torch.float32),
        h=BoxSpace(low=0, high=1.0, shape=(ConfigParameterBinding("nb_k"), ), mutator=GaussianMutator(mean=0.0, std=0.2), indpb=0.25, dtype=torch.float32),
        r=BoxSpace(low=0.2, high=1.0, shape=(ConfigParameterBinding("nb_k"), ), mutator=GaussianMutator(mean=0.0, std=0.2), indpb=0.25, dtype=torch.float32)
        # kn = DiscreteSpace(n=4, mutator=GaussianMutator(mean=0.0, std=0.1), indpb=1.0),
        # gn = DiscreteSpace(n=3, mutator=GaussianMutator(mean=0.0, std=0.1), indpb=1.0),
    )

    input_space = DictSpace(spaces=input_spaces)

    output_space = DictSpace(
        #states=BoxSpace(low=0, high=1, shape=(ConfigParameterBinding("final_step"), ConfigParameterBinding("nb_c") + ConfigParameterBinding("wall_c"), 256, 256))
    )

    step_output_space = DictSpace(
        #state=BoxSpace(low=0, high=1, shape=(ConfigParameterBinding("nb_c") + ConfigParameterBinding("wall_c"), *eval(ConfigParameterBinding("size"))))
    )

    def __init__(self):

        #quick fix
        self.config.size = eval(self.config.size) #convert string to tuple
        self.input_space.spaces["init_state"] = BoxSpace(low=0.0, high=1.0,
                                                         mutator=GaussianMutator(mean=0.0, std=0.01), indpb=0.0, dtype=torch.float32,
                                                         shape=(self.config.nb_c, *tuple([s//self.config.scale_init_state for s in self.config.size])))
        self.input_space.spaces["c0"] = MultiDiscreteSpace(nvec=[self.config.nb_c] * self.config.nb_k, mutator = GaussianMutator(mean=0.0, std=0.1), indpb = 0.1)
        self.input_space.spaces["c1"] = MultiDiscreteSpace(nvec=[self.config.nb_c] * self.config.nb_k, mutator = GaussianMutator(mean=0.0, std=0.1), indpb = 0.1)
        if self.config.wall_c:
            self.input_space.spaces["init_wall"] = BoxSpace(low=0.0, high=1.0, mutator=GaussianMutator(mean=0.0, std=0.01),
                                                 shape=(1, *self.config.size), indpb=0.0, dtype=torch.float32)

        self.output_space.spaces["states"] = BoxSpace(low=0, high=1, shape=(self.config.final_step, self.config.nb_c + int(self.config.wall_c), *self.config.size))

        self.step_output_space.spaces["state"] = BoxSpace(low=0, high=1, shape=(self.config.nb_c + int(self.config.wall_c), *self.config.size))

        super().__init__() #after quick fix because need to call initialize()

    def reset(self, run_parameters):
        #TODO: clamp parameters if not contained in space definition

        init_state = torch.zeros(1, self.config.nb_c, *self.config.size, dtype=torch.float64)
        init_mask = [0, slice(0, self.config.nb_c)] + [slice(s//2 - math.ceil(run_parameters.init_state.shape[i+1]/2), s//2+run_parameters.init_state.shape[i+1]//2) for i,s in enumerate(self.config.size)]
        init_state[init_mask] = run_parameters.init_state

        # wall init
        if self.config.wall_c:
            init_wall = run_parameters.init_wall.unsqueeze(0)
            init_wall[init_wall<0.7] = 0.0

        self.state = torch.cat([init_state, init_wall], 1)
        del run_parameters.init_state
        del run_parameters.init_wall

        self.lenia_step = LeniaStepFFT(nb_c=self.config.nb_c, wall_c=self.config.wall_c, size=self.config.size, **run_parameters)

        self._observations = Dict()
        self._observations.states = torch.empty((self.config.final_step, self.config.nb_c+int(self.config.wall_c), *self.config.size))
        self._observations.states[0] = self.state[0]

        self.step_idx = 0

        current_observation = Dict()
        current_observation.state = self._observations.states[0]

        return current_observation

    def step(self, action=None):
        if self.step_idx >= self.config.final_step:
            raise Exception("Final step already reached, please reset the system.")

        self.state = self.lenia_step(self.state)
        self.step_idx += 1

        self._observations.states[self.step_idx] = self.state[0]

        current_observation = Dict()
        current_observation.state = self._observations.states[self.step_idx]

        return current_observation, 0, self.step_idx >= self.config.final_step - 1, None

    def observe(self):
        return self._observations

    def render(self, mode="PIL_image"):

        channel_colors = [colors.to_rgb(color) for color in colors.TABLEAU_COLORS.values()][:self.config.nb_c+int(self.config.wall_c)]
        channel_colors = np.array(channel_colors).transpose()

        im_array = []
        for img in self._observations.states:
            slices = [slice(0, self.config.nb_c+int(self.config.wall_c))]
            for i, s in enumerate(self.config.size):
                if i < 2:
                    slices.append(slice(0,s))
                else:
                    slices.append(s//2)
            im = Image.fromarray(np.uint8(255*channel_colors@img[slices].squeeze().cpu().detach().numpy().reshape(self.config.nb_c+int(self.config.wall_c), -1)).reshape(3, self.config.size[0], self.config.size[1]).transpose(1,2,0), "RGB")
            im_array.append(im)
        

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
Lenia Step
============================================================================================= """

# Lenia family of functions for the kernel K and for the growth mapping g
kernel_core = {
    0: lambda x,r,w,b : (b*torch.exp(-((x.unsqueeze(-1)-r)/w)**2 / 2)).sum(-1)
}

field_func = {
    0: lambda n, m, s: torch.max(torch.zeros_like(n), 1 - (n - m) ** 2 / (9 * s ** 2)) ** 4 * 2 - 1, # polynomial (quad4)
    1: lambda n, m, s: torch.exp(- (n - m) ** 2 / (2 * s ** 2)-1e-3) * 2 - 1,  # exponential / gaussian (gaus)
    2: lambda n, m, s: (torch.abs(n - m) <= s).float() * 2 - 1 , # step (stpz)
    3: lambda n, m, s: - torch.clamp(n-m,0,1)*s #food eating kernl
}

# TODO: .to(self.device)
class LeniaStepFFT(torch.nn.Module):
    """ Module pytorch that computes one Lenia Step with the fft version"""

    def __init__(self, nb_c, R, T, c0, c1, r, rk, b, w, h, m, s, kn=0, gn=1, wall_c=False, is_soft_clip=False, size=(256, 256)):
        torch.nn.Module.__init__(self)

        self.register_buffer('R', R)
        self.register_buffer('T', T)
        self.register_buffer('c0', c0)
        self.register_buffer('c1', c1)
        self.register_parameter('r', torch.nn.Parameter(r))
        self.register_parameter('rk', torch.nn.Parameter(rk))
        self.register_parameter('b', torch.nn.Parameter(b))
        self.register_parameter('w', torch.nn.Parameter(w))
        self.register_parameter('h', torch.nn.Parameter(h))
        self.register_parameter('m', torch.nn.Parameter(m))
        self.register_parameter('s', torch.nn.Parameter(s))

        self.kn = kn
        self.gn = gn
        self.nb_c = nb_c
        self.nb_k = c0.shape[0]
        self.wall_c = wall_c

        self.size = size
        self.is_soft_clip = is_soft_clip

        self.compute_kernel()
        if self.wall_c:
            self.compute_kernel_env()

    def compute_kernel(self):

        if torch.__version__ >= "1.7.1":
            self.kernels = torch.zeros((self.nb_k, *[s for s in self.size[:-1]], self.size[-1]//2+1))
        else:
            self.kernels = torch.zeros((self.nb_k, *self.size, 2))

        dims = [slice(0, s) for s in self.size]
        I = list(reversed(np.mgrid[list(reversed(dims))]))  # I, J, K, L
        MID = [int(s / 2) for s in self.size]
        X = [torch.from_numpy(i - mid) for i, mid in zip(I, MID)]  # X, Y, Z, S #TODO: removed division by self.R

        for k in range(self.nb_k):
            # distance to center in normalized space
            D = torch.sqrt(torch.stack([x ** 2 for x in X]).sum(axis=0)) / ((self.R+15) * self.r[k]) #TODO: why +15

            # kernel
            kfunc = kernel_core[self.kn]
            kernel = torch.sigmoid(-(D - 1) * 10) * kfunc(D, self.rk[k], self.w[k], self.b[k]) #TODO: why -(D-1)*10
            kernel_sum = torch.sum(kernel)
            kernel_norm = (kernel / kernel_sum).unsqueeze(0).unsqueeze(0)
            # fft of the kernel
            if torch.__version__ >= "1.7.1":
                kernel_FFT = torch.fft.rfftn(kernel_norm, dim=tuple(range(-len(self.size),0)))
            else:
                kernel_FFT = torch.rfft(kernel_norm, signal_ndim=len(self.size), onesided=False)
            self.kernels[k] = kernel_FFT


    def compute_kernel_env(self):

        dims = [slice(0, s) for s in self.size]
        I = list(reversed(np.mgrid[list(reversed(dims))]))  # I, J, K, L
        MID = [int(s / 2) for s in self.size]
        X = [torch.from_numpy(i - mid) for i, mid in zip(I, MID)]  # X, Y, Z, S

        D = torch.sqrt(torch.stack([x ** 2 for x in X]).sum(axis=0)) / 4.0 #TODO: why 4

        kfunc = kernel_core[self.kn]
        kernel = torch.sigmoid(-(D - 1) * 10) * kfunc(D, torch.tensor([0, 0, 0]),
                                                      torch.tensor([0.5, 0.1, 0.1]),
                                                      torch.tensor([1, 0, 0]))
        kernel_sum = torch.sum(kernel)
        kernel_norm = (kernel / kernel_sum).unsqueeze(0)
        if torch.__version__ >= "1.7.1":
            kernel_FFT = torch.fft.rfftn(kernel_norm, dim=tuple(range(-len(self.size), 0)))
        else:
            kernel_FFT = torch.rfft(kernel_norm, signal_ndim=len(self.size), onesided=False)
        self.kernel_wall = kernel_FFT


    def forward(self, input):

        # intra-creature updates
        self.D = torch.zeros(input.shape)

        if torch.__version__ >= "1.7.1":
            world_FFT = [torch.fft.rfftn(input[:, i, :], dim=tuple(range(-len(self.size), 0))) for i in range(self.nb_c)]
            potential_FFT = [self.kernels[k].unsqueeze(0) * world_FFT[self.c0[k]] for k in range(self.nb_k)]
            potentials = [torch.fft.irfftn(potential_FFT[k], dim=tuple(range(-len(self.size), 0))) for k in range(self.nb_k)]
        else:
            world_FFT = [torch.rfft(input[:, i, :], signal_ndim=len(self.size), onesided=False) for i in range(self.nb_c)]
            potential_FFT = [complex_mult_torch(self.kernels[k].unsqueeze(0), world_FFT[self.c0[k]]) for k in range(self.nb_k)]
            potentials = [torch.irfft(potential_FFT[k], signal_ndim=len(self.size), onesided=False) for k in range(self.nb_k)]

        gfunc = field_func[self.gn]
        potentials = [roll_n(potential, 2, potential.size(2) // 2) for potential in potentials]
        potentials = [roll_n(potential, 1, potential.size(1) // 2)for potential in potentials]
        fields = [gfunc(potentials[k], self.m[k], self.s[k]) for k in range(self.nb_k)]

        for k in range(self.nb_k):
            self.D[:, self.c1[k], :] = self.D[:, self.c1[k], :] + self.h[k] * fields[k]

        # wall updates
        if self.wall_c:
            if torch.__version__ >= "1.7.1":
                wall_FFT = torch.fft.rfftn(input[:, -1, :], dim=tuple(range(-len(self.size), 0)))
                potential_FFT = self.kernel_wall * wall_FFT
                potential = torch.fft.irfftn(potential_FFT, dim=tuple(range(-len(self.size), 0)))
            else:
                wall_FFT = torch.rfft(input[:, -1, :], signal_ndim=len(self.size), onesided=False)
                potential_FFT = complex_mult_torch(self.kernel_wall, wall_FFT)
                potential = torch.irfft(potential_FFT, signal_ndim=len(self.size), onesided=False)
            potential = roll_n(potential, 2, potential.size(2) // 2)
            potential = roll_n(potential, 1, potential.size(1) // 2)

            gfunc = field_func[3]
            field = gfunc(potential, 1e-8, 10)
            for i in range(self.nb_c):
                self.D[:, i, :] = self.D[:, i, :] + 1 * field

        if not self.is_soft_clip:
            output_img = torch.clamp(input + (1.0 / self.T) * self.D, min=0., max=1.)
        else:
            output_img = torch.clamp(input + (1.0 / self.T) * self.D, min=0., max=1.)


        if torch.any(torch.isnan(self.D)):
            raise AssertionError(f"lenia paremeters "
                                 f"nb_c={self.nb_c}, R={self.R}, T={self.T}, c0={self.c0}, c1={self.c1}, "
                                 f"r={self.r}, rk={self.rk}, b={self.b}, w={self.w}, h={self.h}, m={self.m}, s={self.s}, "
                                 f"kn={self.kn}, gn={self.gn}, is_soft_clip={self.is_soft_clip}"
                                 f"are causing NaN values")

        return output_img