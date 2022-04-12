from collections import namedtuple
import math
import os
import torch
from torch import nn
from torch.nn.init import kaiming_normal_, kaiming_uniform_, xavier_normal_, xavier_uniform_, uniform_, eye_
from torch.utils.data import Dataset
from typing import Any
import pickle
import numbers
import numpy as np
import random
import torch.nn.functional as F
from torchvision import transforms

PI = torch.acos(torch.zeros(1)).item() * 2

""" ========================================================================================
SPARSE TENSOR HELPERS
======================================================================================== """

def to_sparse_tensor(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    if x_typename == "BoolTensor":
        sparse_tensortype = getattr(torch.sparse, "IntTensor")
    else:
        sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]

    if x_typename == "BoolTensor":
        values = values.int()
    sparse_tensor = sparse_tensortype(indices, values, x.size()).coalesce()

    if x_typename == "BoolTensor":
        sparse_tensor = sparse_tensor.bool()

    return sparse_tensor

""" ========================================================================================
MODULE HELPERS
======================================================================================== """
class Flatten(nn.Module):
    """Flatten the input """

    def forward(self, input):
        return input.view(input.size(0), -1)


class Channelize(nn.Module):
    """Channelize a flatten input to the given (C,D,H,W) or (C,H,W) output """

    def __init__(self, n_channels, out_size):
        nn.Module.__init__(self)
        self.n_channels = n_channels
        self.out_size = out_size

    def forward(self, input):
        out_size = (input.size(0), self.n_channels,) + self.out_size
        return input.view(out_size)

class SphericPad(nn.Module):
    """Pads spherically the input on all sides with the given padding size."""

    def __init__(self, padding_size):
        super(SphericPad, self).__init__()
        if isinstance(padding_size, int) or (isinstance(padding_size, torch.Tensor) and padding_size.shape==()):
            self.pad_left = self.pad_right = self.pad_top = self.pad_bottom = padding_size
        elif (isinstance(padding_size, tuple) or isinstance(padding_size, torch.Tensor)) and len(padding_size) == 2:
            self.pad_left = self.pad_right = padding_size[0]
            self.pad_top = self.pad_bottom = padding_size[1]
        elif (isinstance(padding_size, tuple) or isinstance(padding_size, torch.Tensor)) and len(padding_size) == 4:
            self.pad_left = padding_size[0]
            self.pad_top = padding_size[1]
            self.pad_right = padding_size[2]
            self.pad_bottom = padding_size[3]
        else:
            raise ValueError('The padding size shoud be: int, torch.IntTensor  or tuple of size 2 or tuple of size 4')

    def forward(self, input):

        output = torch.cat([input, input[:, :, :self.pad_bottom, :]], dim=2)
        output = torch.cat([output, output[:, :, :, :self.pad_right]], dim=3)
        output = torch.cat([output[:, :, -(self.pad_bottom + self.pad_top):-self.pad_bottom, :], output], dim=2)
        output = torch.cat([output[:, :, :, -(self.pad_right + self.pad_left):-self.pad_right], output], dim=3)

        return output

def conv_output_sizes(input_size, n_conv=0, kernels_size=1, strides=1, pads=0, dils=1):
    """Returns the size of a tensor after a sequence of convolutions"""
    assert n_conv == len(kernels_size) == len(strides) == len(pads) == len(dils), print(
        'The number of kernels ({}), strides({}), paddings({}) and dilatations({}) has to match the number of convolutions({})'.format(
            len(kernels_size), len(strides), len(pads), len(dils), n_conv))

    spatial_dims = len(input_size)  # 2D or 3D
    in_sizes = list(input_size)
    output_sizes = []

    for conv_id in range(n_conv):
        if type(kernels_size[conv_id]) is not tuple:
            kernel_size = tuple([kernels_size[conv_id]] * spatial_dims)
        if type(strides[conv_id]) is not tuple:
            stride = tuple([strides[conv_id]] * spatial_dims)
        if type(pads[conv_id]) is not tuple:
            pad = tuple([pads[conv_id]] * spatial_dims)
        if type(dils[conv_id]) is not tuple:
            dil = tuple([dils[conv_id]] * spatial_dims)

        for dim in range(spatial_dims):
            in_sizes[dim] = math.floor(
                ((in_sizes[dim] + (2 * pad[dim]) - (dil[dim] * (kernel_size[dim] - 1)) - 1) / stride[dim]) + 1)

        output_sizes.append(tuple(in_sizes))

    return output_sizes


def convtranspose_get_output_padding(input_size, output_size, kernel_size=1, stride=1, pad=0):
    assert len(input_size) == len(output_size)
    spatial_dims = len(input_size)  # 2D or 3D
    out_padding = []

    if type(kernel_size) is not tuple:
        kernel_size = tuple([kernel_size] * spatial_dims)
    if type(stride) is not tuple:
        stride = tuple([stride] * spatial_dims)
    if type(pad) is not tuple:
        pad = tuple([pad] * spatial_dims)

    out_padding = []
    for dim in range(spatial_dims):
        out_padding.append(output_size[dim] + 2 * pad[dim] - kernel_size[dim] - (input_size[dim] - 1) * stride[dim])

    return tuple(out_padding)


""" ========================================================================================
FUNCTIONAL HELPERS
======================================================================================== """
def complex_mult_torch(X, Y):
    """ Computes the complex multiplication in Pytorch when the tensor last dimension is 2: 0 is the real component and 1 the imaginary one"""
    assert X.shape[-1] == 2 and Y.shape[-1] == 2, 'Last dimension must be 2'
    return torch.stack(
        (X[..., 0] * Y[..., 0] - X[..., 1] * Y[..., 1],
         X[..., 0] * Y[..., 1] + X[..., 1] * Y[..., 0]),
        dim=-1)


def roll_n(X, axis, n):
    """ Rolls a tensor with a shift n on the specified axis"""
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None)
                  for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None)
                  for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


def soft_max(x, m, k):
    return torch.log(torch.exp(k * x) + torch.exp(k * m)) / k


def soft_clip(x, min, max, k):
    a = torch.exp(k * x)
    b = torch.exp(torch.FloatTensor([k * min])).item()
    c = torch.exp(torch.FloatTensor([-k * max])).item()
    return torch.log(1.0 / (a + b) + c) / -k

""" ========================================================================================
DATASET HELPERS
======================================================================================== """
class ExperimentHistoryDataset(Dataset):
    """ Represents an abstract dataset that uses the Experiment DB History.

    Input params:
        transform: PyTorch transform to apply on-the-fly to every data tensor instance (default=None).
    """
    def __init__(self, access_history_fn, key, history_ids, filter=None, transform=None):

        self.subkeys = key.split('.')
        self.access_history_fn = access_history_fn
        self.dataset_ids = []
        for idx in history_ids:
            data = self.access_history_fn(index=idx)[0]
            for subkey in self.subkeys:
                data = data[subkey]
            if filter is not None and not filter(data):
                self.dataset_ids.append(idx)
            else:
                pass

        self.transform = transform

    def __len__(self):
        return len(self.dataset_ids)

    def __getitem__(self, idx):
        data = self.access_history_fn(index=self.dataset_ids[idx])[0]
        for subkey in self.subkeys:
            data = data[subkey]

        if self.transform is not None:
            data = self.transform(data)

        return {"obs": data, "label": torch.Tensor([-1]) , "index": idx}


""" ========================================================================================
MODEL WRAPPER HELPERS
======================================================================================== """
# pylint: disable = abstract-method
class ModelWrapper(torch.nn.Module):
    """
    Wrapper class for model with dict/list rvalues.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """
        Init call.
        """
        super().__init__()
        self.model = model

    def forward(self, input_x: torch.Tensor) -> Any:
        """
        Wrap forward call.
        """
        data = self.model(input_x)

        if isinstance(data, dict):
            data_named_tuple = namedtuple("ModelEndpoints", sorted(data.keys()))  # type: ignore
            data = data_named_tuple(**data)  # type: ignore

        elif isinstance(data, list):
            data = tuple(data)

        return data

""" ========================================================================================
WEIGHTS INIT HELPERS
======================================================================================== """
def get_weights_init(initialization_name, initialization_parameters={}):
    '''
    initialization_name: string such that the function called is weights_init_<initialization_name>
    '''
    initialization_name = initialization_name.lower().capitalize()
    initialization_class = eval("{}Weights(**{})".format(initialization_name, initialization_parameters))
    return initialization_class.__call__

class PretrainedWeights():
    def __init__(self, checkpoint_filepath, checkpoint_keys=[]):
        assert os.path.exists(checkpoint_filepath)
        with open(checkpoint_filepath, "rb") as f:
            network_dict = pickle.load(f)
        if isinstance(checkpoint_keys, str):
            checkpoint_keys = checkpoint_keys.split(".")
        for key in checkpoint_keys:
            network_dict = network_dict[key]
        self.network_dict = network_dict

    def __call__(self, m):
        for n, sub_m in m.named_parameters():
            if n in self.network_dict.keys():
                data = self.network_dict[n].data
                assert sub_m.data.shape == data.shape
                sub_m.data = data.type(sub_m.data.dtype).to(sub_m.device)


class NullWeights():
    @staticmethod
    def __call__(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') != -1) or (classname.find('Linear') != -1) or (classname.find('BatchNorm') != -1):
            m.weight.data.fill_(0)
            if m.bias is not None:
                m.bias.data.fill_(0)


class IdentityWeights():
    @staticmethod
    def __call__(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.fill_(1)  # for 1*1 convolution is equivalent to identity
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            eye_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm') != -1:
            m.reset_parameters()

class UniformWeights():
    def __init__(self, a=0., b=0.01):
        self.a = a
        self.b = b

    def __call__(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            uniform_(m.weight.data, self.a, self.b)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            uniform_(m.weight.data, self.a, self.b)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm') != -1:
            m.reset_parameters()

class PytorchWeights():
    @staticmethod
    def __call__(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') != -1) or (classname.find('Linear') != -1) or (classname.find('BatchNorm') != -1):
            m.reset_parameters()


class XavierUniformWeights():
    @staticmethod
    def __call__(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') != -1) or (classname.find('Linear') != -1):
            xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm') != -1:
            m.reset_parameters()


class XavierNormalWeights():
    @staticmethod
    def __call__(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') != -1) or (classname.find('Linear') != -1):
            xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm') != -1:
            m.reset_parameters()


class KaimingUniformWeights():
    @staticmethod
    def __call__(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') != -1) or (classname.find('Linear') != -1):
            kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm') != -1:
            m.reset_parameters()


class KaimingNormalWeights():
    @staticmethod
    def __call__(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') != -1) or (classname.find('Linear') != -1):
            kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm') != -1:
            m.reset_parameters()


""" ========================================================================================
PREPROCESS DATA HELPERS
======================================================================================== """
to_PIL = transforms.ToPILImage()
to_Tensor = transforms.ToTensor()

class TensorRandomFlip(object):
    def __init__(self, p=0.5, dim_flip=-1):
        self.p = p
        self.dim_flip = dim_flip

    def __call__(self, x):
        if torch.rand(()) < self.p:
            x = x.flip(self.dim_flip)
        return x


class TensorRandomGaussianBlur(object):
    def __init__(self, p=0.5, kernel_radius=5, max_sigma=5, n_channels=1, spatial_dims=2):
        self.p = p
        self.kernel_size = 2 * kernel_radius + 1
        self.padding_size = int((self.kernel_size - 1) / 2)
        self.max_sigma = max_sigma
        self.n_channels = n_channels
        self.spatial_dims = spatial_dims

    def gaussian_kernel(self, kernel_size, sigma):
        """
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        mesh_grids = torch.stack([x_grid, y_grid], dim=-1)
        """

        # implementation of meshgrid in torch of shape (kernel_size, kernel_size, kernel_size if Z, 2)
        mesh_coords = [torch.arange(kernel_size)] * kernel_size
        mesh_grids = [None] * self.spatial_dims
        for dim in range(self.spatial_dims):
            view_size = [1, 1, 1]
            view_size[dim] = -1
            repeat_size = [kernel_size, kernel_size, kernel_size]
            repeat_size[dim] = 1
            mesh_grids[dim] = mesh_coords[dim].view(tuple(view_size)).repeat(repeat_size)
        mesh_grids = torch.stack(mesh_grids, dim=-1)

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1. / (2. * PI * variance)) * torch.exp(-torch.sum((mesh_grids - mean) ** 2., dim=-1) / (2 * variance))
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, *gaussian_kernel.size())
        gaussian_kernel = gaussian_kernel.repeat(self.n_channels, 1, *([1]*self.spatial_dims))
        return gaussian_kernel

    def __call__(self, x):
        if torch.rand(()) < self.p:
            sigma = int((torch.rand(()) * (1.0 - self.max_sigma) + self.max_sigma).round())
            x = F.pad(x.unsqueeze(0), pad=self.padding_size, mode='reflect')
            kernel = self.gaussian_kernel(self.kernel_size, sigma)
            if self.spatial_dims == 2:
                x = F.conv2d(x, kernel, groups=self.n_channels).squeeze(0)
            elif self.spatial_dims == 3:
                x = F.conv3d(x, kernel, groups=self.n_channels).squeeze(0)
        return x


class TensorRandomSphericalRotation(object):
    def __init__(self, p=0.5, max_degrees=20, img_size=(64, 64)):
        self.p = p
        self.spatial_dims = len(img_size)
        radius = max(img_size) / 2
        self.padding_size = int(np.sqrt(2 * np.power(radius, 2)) - radius)
        # max rotation needs padding of [sqrt(2*128^2)-128 = 53.01]
        self.spheric_pad = SphericPad(padding_size=self.padding_size)
        if self.spatial_dims == 2:
            self.max_degrees = float(max_degrees)
        elif self.spatial_dims == 3:
            if isinstance(max_degrees, numbers.Number):
                self.max_degrees = (max_degrees, max_degrees, max_degrees)
            elif isinstance(max_degrees, tuple) or isinstance(max_degrees, list):
                assert len(max_degrees) == 3, "the number of rotation is 3, must provide tuple of length 3"
                self.max_degrees = tuple(max_degrees)

    def __call__(self, x):
        if np.random.random() < self.p:
            x = x.unsqueeze(0)
            x = self.spheric_pad(x)
            if self.spatial_dims == 2:
                theta = float(torch.empty(1).uniform_(-self.max_degrees,  self.max_degrees).item()) * math.pi / 180.0
                R = torch.tensor([[math.cos(theta), -math.sin(theta)],
                                    [math.sin(theta), math.cos(theta)]])
                grid = F.affine_grid(torch.cat([R, torch.zeros(2, 1)], dim=-1).unsqueeze(0), size=x.size()).type(x.dtype)
                x = F.grid_sample(x, grid)
                x = x[:, :, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size].squeeze(0)

            elif self.spatial_dims == 3:
                theta_x = float(torch.empty(1).uniform_(-float(self.max_degrees[0]), float(self.max_degrees[0])).item()) * math.pi / 180.0
                theta_y = float(torch.empty(1).uniform_(-float(self.max_degrees[1]), float(self.max_degrees[1])).item()) * math.pi / 180.0
                theta_z = float(torch.empty(1).uniform_(-float(self.max_degrees[2]), float(self.max_degrees[2])).item()) * math.pi / 180.0
                R_x = torch.tensor([[1., 0., 0.],
                                    [0., math.cos(theta_x), -math.sin(theta_x)],
                                    [0., math.sin(theta_x), math.cos(theta_x)]])
                R_y = torch.tensor([[math.cos(theta_y), 0., math.sin(theta_y)],
                                    [0., 1.0, 0.0],
                                    [-math.sin(theta_y), 0.0, math.cos(theta_y)]])
                R_z = torch.tensor([[math.cos(theta_z), -math.sin(theta_z), 0.0],
                                    [math.sin(theta_z), math.cos(theta_z), 0.0],
                                    [0., 0., 1.]])
                R = R_z.matmul(R_y.matmul(R_x)) # batch_size = 1
                grid = F.affine_grid(torch.cat([R, torch.zeros(3,1)], dim=-1).unsqueeze(0), size=x.size()).type(x.dtype)
                x = F.grid_sample(x, grid)
                x = x[:, :, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size , self.padding_size:-self.padding_size].squeeze(0)

        return x


class TensorRandomRoll(object):
    def __init__(self, p=(0.5, 0.5), max_delta=(0.5, 0.5), spatial_dims=2):
        self.spatial_dims = spatial_dims

        if isinstance(p, numbers.Number):
            self.p = tuple([p] * self.spatial_dims)
        else:
            self.p = p

        if isinstance(max_delta, numbers.Number):
            self.max_delta = tuple([max_delta] * self.spatial_dims)
        else:
            self.max_delta = max_delta

        assert len(self.p) == len(self.max_delta) == self.spatial_dims

    def __call__(self, x):

        for dim in range(-self.spatial_dims, 0):

            if np.random.random() < self.p[dim]:

                shift_dim = int(np.round(np.random.uniform(-self.max_delta[dim] * x.shape[dim], self.max_delta[dim] * x.shape[dim]))) #x: C*D*H*W
                # import matplotlib.pyplot as plt
                # fig, axarr = plt.subplots(2,1)
                # axarr[0].imshow(x[0], cmap="gray")
                # axarr[1].imshow(roll_n(x, range(len(x.shape))[dim], shift_dim)[0], cmap="gray")
                # plt.show()
                x = roll_n(x, range(len(x.shape))[dim], shift_dim)


        return x


class TensorRandomResizedCrop(object):
    """
    Reimplementation of torchvision.transforms.RandomResizedCrop to deal with 2D or 3D tensors
    """

    def __init__(self, p, size, scale=(1., 1.), ratio_x=(1., 1.), ratio_y=(1., 1.), interpolation='bilinear'):
        self.p = p
        if (scale[0] > scale[1]) or (ratio_x[0] > ratio_x[1]) or (ratio_y[0] > ratio_y[1]):
            raise ValueError("Scale and ratio should be of kind (min, max)")
        self.scale = scale
        self.ratio_x = ratio_x
        self.ratio_y = ratio_y
        self.out_size = size
        self.spatial_dims = len(self.out_size)
        self.interpolation = interpolation

    def __call__(self, x):
        if np.random.random() < self.p:

            area = torch.prod(torch.tensor(self.out_size)).item()
            log_ratio_x = tuple(torch.log(torch.tensor(self.ratio_x)))
            if self.spatial_dims == 3:
                log_ratio_y = tuple(torch.log(torch.tensor(self.ratio_y)))

            patch_size = [None] * self.spatial_dims
            for _ in range(10):
                target_area = area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
                ratio_x = torch.exp(torch.empty(1).uniform_(log_ratio_x[0], log_ratio_x[1])).item()
                patch_xsize = int(round(math.pow(target_area * ratio_x, 1./self.spatial_dims)))
                if self.spatial_dims == 2:
                    ratio_y = 1. / ratio_x
                    patch_ysize = int(round(math.pow(target_area * ratio_y, 1./self.spatial_dims)))
                    if 0 < patch_xsize <= x.shape[-1] and 0 < patch_ysize <= x.shape[-2]:
                        patch_size = [patch_ysize, patch_xsize]
                elif self.spatial_dims == 3:
                    ratio_y = torch.exp(torch.empty(1).uniform_(log_ratio_y[0], log_ratio_y[1])).item()
                    patch_ysize = int(round(math.pow(target_area * ratio_y, 1./self.spatial_dims)))
                    ratio_z = 1. / (ratio_x * ratio_y)
                    patch_zsize = int(round(math.pow(target_area * ratio_z, 1./self.spatial_dims)))
                    if 0 < patch_xsize <= x.shape[-1] and 0 < patch_ysize <= x.shape[-2] and 0 < patch_zsize <= x.shape[-3]:
                        patch_size = [patch_zsize, patch_ysize, patch_xsize]

            if None in patch_size:
                for dim in range(len(patch_size)):
                    patch_size[dim] = x.shape[1+dim]

            return random_crop_preprocess(x, patch_size, out_size=self.out_size, interpolation=self.interpolation)

        else:
            return x

class TensorRandomCentroidCrop(object):
    def __init__(self, p, size, scale=(1., 1.), ratio_x=(1., 1.), ratio_y=(1.,1.), interpolation='bilinear'):
        self.p = p
        if (scale[0] > scale[1]) or (ratio_x[0] > ratio_x[1]) or (ratio_y[0] > ratio_y[1]):
            raise ValueError("Scale and ratio should be of kind (min, max)")
        self.scale = scale
        self.ratio_x = ratio_x
        self.ratio_y = ratio_y
        self.out_size = size
        self.spatial_dims = len(self.out_size)
        self.interpolation = interpolation


    def __call__(self, x):
        if np.random.random() < self.p:

            area = torch.prod(torch.tensor(self.out_size)).item()
            log_ratio_x = tuple(torch.log(torch.tensor(self.ratio_x)))
            if self.spatial_dims == 3:
                log_ratio_y = tuple(torch.log(torch.tensor(self.ratio_y)))

            patch_size = [None] * self.spatial_dims
            for _ in range(10):
                target_area = area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
                ratio_x = torch.exp(torch.empty(1).uniform_(log_ratio_x[0], log_ratio_x[1])).item()
                patch_xsize = int(round(math.pow(target_area * ratio_x, 1./self.spatial_dims)))
                if self.spatial_dims == 2:
                    ratio_y = 1. / ratio_x
                    patch_ysize = int(round(math.pow(target_area * ratio_y, 1./self.spatial_dims)))
                    if 0 < patch_xsize <= x.shape[-1] and 0 < patch_ysize <= x.shape[-2]:
                        patch_size = [patch_ysize, patch_xsize]
                elif self.spatial_dims == 3:
                    ratio_y = torch.exp(torch.empty(1).uniform_(log_ratio_y[0], log_ratio_y[1])).item()
                    patch_ysize = int(round(math.pow(target_area * ratio_y, 1./self.spatial_dims)))
                    ratio_z = 1. / (ratio_x * ratio_y)
                    patch_zsize = int(round(math.pow(target_area * ratio_z, 1./self.spatial_dims)))
                    if 0 < patch_xsize <= x.shape[-1] and 0 < patch_ysize <= x.shape[-2] and 0 < patch_zsize <= x.shape[-3]:
                        patch_size = [patch_zsize, patch_ysize, patch_xsize]

            if None in patch_size:
                for dim in range(len(patch_size)):
                    patch_size[dim] = x.shape[1+dim]

            return centroid_crop_preprocess(x, patch_size, out_size=self.out_size, interpolation=self.interpolation)

        else:
            return x


def resized_crop(x, bbox, out_size, mode='bilinear'):
    """
    arg: x, tensor Cx(D)xHxW
    Reimplementation of torchvision.transforms.functional.resized_crop to deal with 2D or 3D images
    """
    x = crop(x, bbox)
    x = torch.nn.functional.interpolate(x.unsqueeze(0), out_size, mode=mode).squeeze(0)
    return x


def crop(x, bbox):
    """
    arg: x, tensor Cx(D)xHxW
    """
    spatial_dims = len(x.size()[1:])
    if spatial_dims == 2:
        return x[:, bbox[0]:bbox[0]+bbox[2], bbox[1]:bbox[1]+bbox[3]]
    elif spatial_dims == 3:
        return x[:, bbox[0]:bbox[0]+bbox[3], bbox[1]:bbox[1]+bbox[4], bbox[2]:bbox[2]+bbox[5]]


def centroid_crop_preprocess(x, patch_size, out_size=None, interpolation='bilinear'):
    """
    arg: x, tensor Cx(D)xHxW
    """

    img_size = tuple(x.size()[1:])
    spatial_dims = len(img_size)

    padding_size = round(max(*[patch_size[dim] / 2 for dim in range(spatial_dims)]))

    # crop around center of mass (mY and mX describe the position of the centroid of the image)
    image = x.numpy()
    meshgrids = np.meshgrid(*[range(img_size[dim]) for dim in range(spatial_dims)])

    m00 = np.sum(image)
    bbox = [None]*(2*spatial_dims) #y0,x0,h,w for 2D or z0,y0,x0,d,h,w for 3D
    for dim in range(spatial_dims):
        dim_power1_image = meshgrids[dim] * image
        if m00 == 0:
            m_dim = (img_size[dim] - 1) / 2.0
        else:
            m_dim = np.sum(dim_power1_image) / m00
        m_dim += padding_size
        bbox[dim] = int(m_dim - patch_size[dim] / 2)
        bbox[spatial_dims+dim] = int(patch_size[dim])

    spheric_pad = SphericPad(padding_size=padding_size)
    x = spheric_pad(x.unsqueeze(0)).squeeze(0)

    if out_size is not None:
        patch = resized_crop(x, tuple(bbox), out_size, mode=interpolation)
    else:
        patch = crop(x, tuple(bbox))

    return patch


def random_crop_preprocess(x, patch_size, out_size=None, interpolation='bilinear'):
    '''
    arg: x, tensor Cx(D)xHxW
    '''

    img_size = tuple(x.size()[1:])
    spatial_dims = len(img_size)

    # set the seed as mX*mY(*mZ) for reproducibility ((mZ,mY,mX) describe the position of the centroid of the image)
    local_seed = 1.0
    image = x.numpy()
    meshgrids = np.meshgrid(*[range(img_size[dim]) for dim in range(spatial_dims)])
    m00 = np.sum(image)
    for dim in range(spatial_dims):
        dim_power1_image = meshgrids[dim] * image
        if m00 == 0:
            m_dim = (img_size[dim] - 1) / 2.0
        else:
            m_dim = np.sum(dim_power1_image) / m00
        local_seed *= m_dim

    ## raw set seed
    global_rng_state = random.getstate()
    random.seed(local_seed)

    n_trials = 0
    best_patch_activation = 0
    selected_patch = False

    activation = m00 / torch.prod(torch.tensor(img_size)).item()
    while 1:
        bbox = [None] * spatial_dims + patch_size #y0,x0,h,w for 2D or z0,y0,x0,d,h,w for 3D
        # random sampling of origin crop
        for dim in range(spatial_dims):
            bbox[dim] = random.randint(0, img_size[dim] - patch_size[dim])
        if out_size is not None:
            patch = resized_crop(x, tuple(bbox), out_size, mode=interpolation)
        else:
            patch = crop(x, tuple(bbox))
        patch_activation = patch
        for dim in range(spatial_dims):
            patch_activation = patch_activation.sum(-1) / patch_size[-(dim+1)]

        if patch_activation > (activation * 0.5):
            selected_patch = patch
            break

        if patch_activation >= best_patch_activation:
            best_patch_activation = patch_activation
            selected_patch = patch

        n_trials += 1
        if n_trials == 20:
            break

    ## reput global random state
    random.setstate(global_rng_state)

    return selected_patch