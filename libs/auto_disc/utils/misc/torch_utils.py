from collections import namedtuple
import math
import os
import torch
from torch import nn
from torch.nn.init import kaiming_normal_, kaiming_uniform_, xavier_normal_, xavier_uniform_, uniform_, eye_
from torch.utils.data import Dataset
from typing import Any
import pickle

PI = torch.acos(torch.zeros(1)).item() * 2


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


class ExperimentHistoryDataset(Dataset):
    """ Represents an abstract dataset that uses the Experiment DB History.

    Input params:
        transform: PyTorch transform to apply on-the-fly to every data tensor instance (default=None).
    """
    def __init__(self, access_history_fn, key, history_ids, filter=None, transform=None):

        subkeys = key.split('.')
        history = [access_history_fn()[subkeys[0]][i] for i in history_ids]
        self.dataset = []
        for data in history:
            for subkey in subkeys[1:]:
                data = data[subkey]
            if filter is not None and not filter(data):
                self.dataset.append(data)
            else:
                self.dataset.append(data)

        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

        if self.transform is not None:
            data = self.transform(data)

        return {"obs": data, "label": torch.Tensor([-1]) , "index": idx}


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

"""===================================================================
Weights init utils
====================================================================="""
def get_weights_init(initialization_name):
    '''
    initialization_name: string such that the function called is weights_init_<initialization_name>
    '''
    initialization_name = initialization_name.lower()
    return eval("weights_init_{}".format(initialization_name))

def weights_init_pretrain(checkpoint_filepath, chekpoint_keys=[]):
    if os.path.exists(checkpoint_filepath):
        with open(checkpoint_filepath, "rb") as f:
            network_dict = pickle.load(f)
        if isinstance(chekpoint_keys, str):
            checkpoint_keys = chekpoint_keys.split(".")
        for key in checkpoint_keys:
            network_dict = network_dict[key]
    else:
        print("WARNING: the checkpoint filepath for a pretrain initialization has not been found, skipping the initialization")
    return network_dict


def weights_init_null(m):
    """
    For HOLMES: initialize zero net (child born with no knowledge) and identity connections from parent (child start by copying parent)
    """
    classname = m.__class__.__name__
    if (classname.find('Conv') != -1) or (classname.find('Linear') != -1) or (classname.find('BatchNorm') != -1):
        m.weight.data.fill_(0)
        if m.bias is not None:
            m.bias.data.fill_(0)


def weights_init_connections_identity(m):
    """
    For HOLMES: initialize identity connections
    """
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


def weights_init_uniform(m, a=0., b=0.01):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        uniform_(m.weight.data, a, b)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        uniform_(m.weight.data, a, b)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.reset_parameters()


def weights_init_pytorch(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') != -1) or (classname.find('Linear') != -1) or (classname.find('BatchNorm') != -1):
        m.reset_parameters()


def weights_init_xavier_uniform(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') != -1) or (classname.find('Linear') != -1):
        xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.reset_parameters()


def weights_init_xavier_normal(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') != -1) or (classname.find('Linear') != -1):
        xavier_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.reset_parameters()


def weights_init_kaiming_uniform(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') != -1) or (classname.find('Linear') != -1):
        kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.reset_parameters()


def weights_init_kaiming_normal(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') != -1) or (classname.find('Linear') != -1):
        kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.reset_parameters()


def weights_init_custom_uniform(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # m.weight.data.uniform_(-1,1)
        m.weight.data.uniform_(-1 / (m.weight.size(2)), 1 / (m.weight.size(2)))
        if m.bias is not None:
            m.bias.data.uniform_(-0.1, 0.1)
    elif classname.find('Linear') != -1:
        m.weight.data.uniform_(-1 / math.sqrt(m.weight.size(0)), 1 / math.sqrt(m.weight.size(0)))
        if m.bias is not None:
            m.bias.data.uniform_(-0.1, 0.1)
    elif classname.find('BatchNorm') != -1:
        m.reset_parameters()

