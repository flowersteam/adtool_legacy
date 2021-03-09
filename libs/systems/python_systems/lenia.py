from libs.systems.python_systems import BasePythonSystem

from libs.utils.auto_disc_parameters import AutoDiscParameter, ConfigParameterBinding, ParameterTypesEnum, AutoDiscSpaceDefinition

from libs.utils import AttrDict
from libs.utils.torch_utils import SphericPad, roll_n, complex_mult_torch

import torch
import matplotlib.pyplot as plt
import numpy as np

class LeniaStepFFT(torch.nn.Module):
    """ Module pytorch that computes one Lenia Step with the fft version"""

    def __init__(self, run_parameters, use_gpu, SX, SY):
        super(LeniaStepFFT, self).__init__()
        self.run_parameters = run_parameters
        # for k, v in self.run_parameters.items():
        #     self.run_parameters[k] = torch.nn.Parameter(v)

        self.spheric_pad = SphericPad(int(self.run_parameters.R))
        self.is_soft_clip = True  # do only soft clip for differentiable lenia
        self.use_gpu = use_gpu
        self.SX = SX
        self.SY = SY

        self.compute_kernel()

    def compute_kernel(self):

        # implementation of meshgrid in torch
        x = torch.arange(self.SX)
        y = torch.arange(self.SY)
        xx = x.repeat(self.SY, 1)
        yy = y.view(-1, 1).repeat(1, self.SX)
        X = (xx - int(self.SX / 2)).float() / float(self.run_parameters.R)
        Y = (yy - int(self.SY / 2)).float() / float(self.run_parameters.R)

        # distance to center in normalized space
        D = torch.sqrt(X ** 2 + Y ** 2)

        # kernel
        k = len(self.run_parameters.b)
        kr = k * D
        bs = torch.tensor([float(f) for f in self.run_parameters.b])
        b = bs[torch.min(torch.floor(kr).long(), (k - 1) * torch.ones_like(kr).long())]
        kfunc = AutomatonPytorch.kernel_core[self.run_parameters.kn - 1]
        kernel = (D < 1).float() * kfunc(torch.min(kr % 1, torch.ones_like(kr))) * b
        kernel_sum = torch.sum(kernel)
        # normalization of the kernel
        self.kernel_norm = (kernel / kernel_sum).unsqueeze(0).unsqueeze(0)
        # fft of the kernel
        self.kernel_FFT = torch.rfft(self.kernel_norm, signal_ndim=2, onesided=False)

        self.kernel_updated = False

    def forward(self, input):
        if self.use_gpu:
            input = input.cuda()
            self.kernel_FFT = self.kernel_FFT.cuda()

        self.world_FFT = torch.rfft(input, signal_ndim=2, onesided=False)
        self.potential_FFT = complex_mult_torch(self.kernel_FFT, self.world_FFT)
        self.potential = torch.irfft(self.potential_FFT, signal_ndim=2, onesided=False)
        self.potential = roll_n(self.potential, 3, self.potential.size(3) // 2)
        self.potential = roll_n(self.potential, 2, self.potential.size(2) // 2)

        gfunc = AutomatonPytorch.field_func[min(self.run_parameters.gn, 2)]
        self.field = gfunc(self.potential, self.run_parameters.m, self.run_parameters.s)

        if not self.is_soft_clip:
            output_img = torch.clamp(input + (1.0 / self.run_parameters.T) * self.field, min=0., max=1.)
        else:
            output_img = AutomatonPytorch.soft_clip(input + (1.0 / self.run_parameters.T) * self.field, 0, 1,
                                                    self.run_parameters.T)

        return output_img


class LeniaStepConv2d(torch.nn.Module):
    """ Module pytorch that computes one Lenia Step with the conv2d version"""

    def __init__(self, run_parameters, use_gpu):
        super(LeniaStepConv2d, self).__init__()
        self.run_parameters = run_parameters
        # for k, v in self.run_parameters.items():
        #     self.run_parameters[k] = torch.nn.Parameter(v)
        self.spheric_pad = SphericPad(int(self.run_parameters.R))
        self.is_soft_clip = True
        self.use_gpu = use_gpu

        self.compute_kernel()

    def compute_kernel(self):
        SY = 2 * self.run_parameters.R + 1
        SX = 2 * self.run_parameters.R + 1

        # implementation of meshgrid in torch
        x = torch.arange(SX)
        y = torch.arange(SY)
        xx = x.repeat(SY, 1)
        yy = y.view(-1, 1).repeat(1, SX)
        X = (xx - int(SX / 2)).float() / float(self.run_parameters.R)
        Y = (yy - int(SY / 2)).float() / float(self.run_parameters.R)

        # distance to center in normalized space
        D = torch.sqrt(X ** 2 + Y ** 2)

        # kernel
        k = len(self.run_parameters.b)
        kr = k * D
        bs = torch.tensor([float(f) for f in self.run_parameters.b])
        b = bs[torch.min(torch.floor(kr).long(), (k - 1) * torch.ones_like(kr).long())]
        kfunc = AutomatonPytorch.kernel_core[self.run_parameters.kn - 1]
        kernel = (D < 1).float() * kfunc(torch.min(kr % 1, torch.ones_like(kr))) * b
        kernel_sum = torch.sum(kernel)
        # normalization of the kernel
        self.kernel_norm = (kernel / kernel_sum).unsqueeze(0).unsqueeze(0)

        self.kernel_updated = False

    def forward(self, input):
        if self.use_gpu:
            input = input.cuda()
            self.kernel_norm = self.kernel_norm.cuda()

        self.potential = torch.nn.functional.conv2d(self.spheric_pad(input), weight=self.kernel_norm)
        gfunc = AutomatonPytorch.field_func[self.run_parameters.gn]
        self.field = gfunc(self.potential, self.run_parameters.m, self.run_parameters.s)

        if not self.is_soft_clip:
            output_img = torch.clamp(input + (1.0 / self.run_parameters.T) * self.field, 0,
                                     1)  # A_new = A + dt * torch.clamp(D, -A/dt, (1-A)/dt)
        else:
            output_img = AutomatonPytorch.soft_clip(input + (1.0 / self.run_parameters.T) * self.field, 0, 1,
                                                    self.run_parameters.T)  # A_new = A + dt * Automaton.soft_clip(D, -A/dt, (1-A)/dt, 1)

        return output_img


class AutomatonPytorch:
    kernel_core = {
        0: lambda r: (4 * r * (1 - r)) ** 4,  # polynomial (quad4)
        1: lambda r: torch.exp(4 - 1 / (r * (1 - r))),  # exponential / gaussian bump (bump4)
        2: lambda r, q=1 / 4: (r >= q).float() * (r <= 1 - q).float(),  # step (stpz1/4)
        3: lambda r, q=1 / 4: (r >= q).float() * (r <= 1 - q).float() + (r < q).float() * 0.5  # staircase (life)
    }
    field_func = {
        0: lambda n, m, s: torch.max(torch.zeros_like(n), 1 - (n - m) ** 2 / (9 * s ** 2)) ** 4 * 2 - 1,
        # polynomial (quad4)
        1: lambda n, m, s: torch.exp(- (n - m) ** 2 / (2 * s ** 2)) * 2 - 1,  # exponential / gaussian (gaus)
        2: lambda n, m, s: (torch.abs(n - m) <= s).float() * 2 - 1  # step (stpz)
    }

    @staticmethod
    def soft_max(x, m, k):
        return torch.log(torch.exp(k * x) + torch.exp(k * m)) / k

    @staticmethod
    def soft_clip(x, min, max, k):
        a = torch.exp(k * x)
        b = torch.exp(torch.FloatTensor([k * min])).item()
        c = torch.exp(torch.FloatTensor([-k * max])).item()
        return torch.log(1.0 / (a + b) + c) / -k

    def __init__(self, run_parameters, version='fft', SX=256, SY=256):
        # set device
        if torch.cuda.is_available():
            self.use_gpu = True
        else:
            self.use_gpu = False

        # init state
        self.cells = torch.clamp(run_parameters["init_state"], 0.0, 1.0)

        # pytorch model to perform one step in Lenia
        if version == 'fft':
            self.model = LeniaStepFFT(run_parameters, self.use_gpu, SX, SY)
        elif version == 'conv2d':
            self.model = LeniaStepConv2d(run_parameters, self.use_gpu)
        else:
            raise ValueError('Lenia pytorch automaton step calculation can be done with fft or conv 2d')
        if self.use_gpu:
            self.model = self.model.cuda()

    def calc_once(self):
        A = self.cells.unsqueeze(0).unsqueeze(0)
        A_new = self.model(A)
        A_new = A_new[0, 0, :, :]
        self.cells = A_new


class Lenia(BasePythonSystem):
    CONFIG_DEFINITION = [
        AutoDiscParameter(
                    name="version", 
                    type=ParameterTypesEnum.get('STRING'), 
                    values_range=["pytorch_fft", "pytorch_conv2d"], 
                    default="pytorch_fft"),
        AutoDiscParameter(
                    name="SX", 
                    type=ParameterTypesEnum.get('INTEGER'), 
                    values_range=[1, np.inf], 
                    default=256),
        AutoDiscParameter(
                    name="SY", 
                    type=ParameterTypesEnum.get('INTEGER'), 
                    values_range=[1, np.inf], 
                    default=256),
        AutoDiscParameter(
                    name="final_step", 
                    type=ParameterTypesEnum.get('INTEGER'), 
                    values_range=[1, np.inf], 
                    default=200),
    ]

    INPUT_SPACE_DEFINITION = [
        AutoDiscParameter(
                    name="init_state", 
                    type=ParameterTypesEnum.get('SPACE'),
                    default=AutoDiscSpaceDefinition(
                        dims=[ConfigParameterBinding("SX"), ConfigParameterBinding("SY")],
                        bounds=[0, 1],
                        type=ParameterTypesEnum.get('FLOAT')
                    ),
                    modifiable=False),
        AutoDiscParameter(
                    name="kn",
                    type=ParameterTypesEnum.get('SPACE'),
                    default=AutoDiscSpaceDefinition(
                        dims=[],
                        bounds=[1, 4],
                        type=ParameterTypesEnum.get('INTEGER')
                    )), 
        AutoDiscParameter(
                    name="gn", 
                    type=ParameterTypesEnum.get('SPACE'),
                    default=AutoDiscSpaceDefinition(
                        dims=[],
                        bounds=[1, 3],
                        type=ParameterTypesEnum.get('INTEGER')
                    )), 
        AutoDiscParameter(  
                    name="R", 
                    type=ParameterTypesEnum.get('SPACE'),
                    default=AutoDiscSpaceDefinition(
                        dims=[],
                        bounds=[1, 10], # TODO: CHANGE
                        type=ParameterTypesEnum.get('INTEGER')
                    )), 
        AutoDiscParameter(
                    name="T", 
                    type=ParameterTypesEnum.get('SPACE'),
                    default=AutoDiscSpaceDefinition(
                        dims=[],
                        bounds=[1, 10], # TODO: CHANGE
                        type=ParameterTypesEnum.get('INTEGER')
                    )), 
        AutoDiscParameter(
                    name="b", 
                    type=ParameterTypesEnum.get('SPACE'),
                    default=AutoDiscSpaceDefinition(
                        dims=[1],
                        bounds=[1, 10], # TODO: CHANGE
                        type=ParameterTypesEnum.get('INTEGER')
                    )),
        AutoDiscParameter(
                    name="m", 
                    type=ParameterTypesEnum.get('SPACE'),
                    default=AutoDiscSpaceDefinition(
                        dims=[],
                        bounds=[1, 10], # TODO: CHANGE
                        type=ParameterTypesEnum.get('INTEGER')
                    )),
        AutoDiscParameter(
                    name="s", 
                    type=ParameterTypesEnum.get('SPACE'),
                    default=AutoDiscSpaceDefinition(
                        dims=[1],
                        bounds=[1, 10], # TODO: CHANGE
                        type=ParameterTypesEnum.get('FLOAT')
                    )),
    ]

    OUTPUT_SPACE_DEFINITION = [
        AutoDiscParameter(
                    name="states", 
                    type=ParameterTypesEnum.get('SPACE'),
                    default=AutoDiscSpaceDefinition(
                        dims=[ConfigParameterBinding("final_step"),
                        ConfigParameterBinding("SX"), 
                        ConfigParameterBinding("SY")],
                        bounds=[0, 1],
                        type=ParameterTypesEnum.get('FLOAT')
                    ),
                    modifiable=False),
        AutoDiscParameter(
                    name="timepoints", 
                    type=ParameterTypesEnum.get('SPACE'),
                    default=AutoDiscSpaceDefinition(
                        dims=[ConfigParameterBinding("final_step")],
                        bounds=[1, ConfigParameterBinding("final_step")],
                        type=ParameterTypesEnum.get('INTEGER')
                    ),
                    modifiable=False),
    ]

    STEP_OUTPUT_SPACE_DEFINITION = [
        AutoDiscParameter(
                    name="state", 
                    type=ParameterTypesEnum.get('SPACE'),
                    default=AutoDiscSpaceDefinition(
                        dims=[ConfigParameterBinding("SX"), ConfigParameterBinding("SY")],
                        bounds=[0, 1],
                        type=ParameterTypesEnum.get('FLOAT')
                    ),
                    modifiable=False)
    ]

    def reset(self, run_parameters):
        if self.config.version.lower() == 'pytorch_fft':
            self.automaton = AutomatonPytorch(run_parameters, version='fft', SX=self.config.SX, SY=self.config.SY)
        elif self.config.version.lower() == 'pytorch_conv2d':
            self.automaton = AutomatonPytorch(run_parameters, version='conv2d')
        else:
            raise ValueError('Unknown lenia version (config.version = {!r})'.format(self.config.version))

        self._observations = AttrDict()
        self._observations.timepoints = list(range(self.config.final_step))
        self._observations.states = torch.empty((self.config.final_step, self.config.SX, self.config.SY))
        self._observations.states[0] = self.automaton.cells

        self.step_idx = 0

        current_observation = AttrDict()
        current_observation.state = self._observations.states[0]
        return current_observation

    def step(self, action=None):
        if self.step_idx >= self.config.final_step:
            raise Exception("Final step already reached, please reset the system.")

        self.step_idx += 1
        self.automaton.calc_once()
        self._observations.states[self.step_idx] = self.automaton.cells

        current_observation = AttrDict()
        current_observation.state = self._observations.states[self.step_idx]

        return current_observation, 0, self.step_idx < self.config.final_step, None

    def observe(self):
        return self._observations

    def render(self, mode="PIL_image"):
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(self.automaton.cells.cpu().detach().numpy(), cmap='gray', vmin=0, vmax=1)
        if mode == "human":
            return plt.show()
        elif mode == "plt_figure":
            return fig
        elif mode == "PIL_image":
            colormap = create_colormap(np.array([[255,255,255], [119,255,255],[23,223,252],[0,190,250],[0,158,249],[0,142,249],[81,125,248],[150,109,248],[192,77,247],[232,47,247],[255,9,247],[200,0,84]])/255*8)
            im = im_from_array_with_colormap(self.automaton.cells.cpu().detach().numpy(), colormap)
            return im
        else:
            raise NotImplementedError  

    def close(self):
        pass

import numpy as np
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