import math
import torch
from torch import nn
from torch.utils.data import Dataset

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
    def __init__(self, access_history_fn, history_ids, wrapped_input_space_key, transform=None, **kwargs):


        self.access_history_fn = access_history_fn
        self.history_ids = history_ids
        self.wrapped_input_space_key = wrapped_input_space_key
        self.transform = transform

    def __len__(self):
        return len(self.history_ids)

    def __getitem__(self, idx):
        rel_idx = self.history_ids[idx]
        data = self.access_history_fn()['input'][rel_idx][self.wrapped_input_space_key]

        if self.transform is not None:
            data = self.transform(data)

        return {"obs": data, "label": torch.Tensor([-1]) , "index": idx}