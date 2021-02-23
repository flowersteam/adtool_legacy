import torch
from torch import nn
PI = torch.acos(torch.zeros(1)).item() * 2

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


def get_regions_masks(env_size, n_sections, n_orientations, include_out_of_maxradius=False):
    regions_masks = []
    # create sectors
    RX = env_size[0] // 2
    RY = env_size[1] // 2
    R = min(RX,RY)
    section_regions = [(ring_idx / n_sections * R,
                        (ring_idx + 1) / n_sections * R)
                       for ring_idx in range(n_sections)]

    # concatenate first and last regions
    orientation_regions = [(wedge_idx / n_orientations * 2*PI,
                            (wedge_idx + 1) / n_orientations * 2*PI)
                            for wedge_idx in range(n_orientations)]
    orientation_regions = [(region[0] - PI, region[1] - PI) for region in orientation_regions]

    grid_x, grid_y = torch.meshgrid(torch.range(-RX, RX - 1, 1), torch.range(-RY, RY - 1, 1))
    grid_r = (grid_x ** 2 + grid_y ** 2).sqrt()
    grid_theta = torch.atan2(grid_y, grid_x)

    # fill feature vector
    for section_idx, section_region in enumerate(section_regions):
        r1 = section_region[0]
        r2 = section_region[1]

        if (section_idx == (len(section_region) - 1)) and include_out_of_maxradius:
            r2 = torch.sqrt(torch.tensor(pow(RX,2) + pow(RY,2), dtype=float)).item()

        for orientation_region in orientation_regions:
            theta1 = orientation_region[0]
            theta2 = orientation_region[1]

            region_mask = (grid_r >= r1) & (grid_r < r2) & (grid_theta >= theta1) & (grid_theta < theta2)
            regions_masks.append(to_sparse_tensor(region_mask))

    return regions_masks

class SphericPad(nn.Module):
    """Pads spherically the input on all sides with the given padding size."""

    def __init__(self, padding_size):
        super(SphericPad, self).__init__()
        if isinstance(padding_size, int):
            self.pad_left = self.pad_right = self.pad_top = self.pad_bottom = padding_size
        elif isinstance(padding_size, tuple) and len(padding_size) == 2:
            self.pad_left = self.pad_right = padding_size[0]
            self.pad_top = self.pad_bottom = padding_size[1]
        elif isinstance(padding_size, tuple) and len(padding_size) == 4:
            self.pad_left = padding_size[0]
            self.pad_top = padding_size[1]
            self.pad_right = padding_size[2]
            self.pad_bottom = padding_size[3]
        else:
            raise ValueError('The padding size shoud be: int, tuple of size 2 or tuple of size 4')

    def forward(self, input):

        output = torch.cat([input, input[:, :, :self.pad_bottom, :]], dim=2)
        output = torch.cat([output, output[:, :, :, :self.pad_right]], dim=3)
        output = torch.cat([output[:, :, -(self.pad_bottom + self.pad_top):-self.pad_bottom, :], output], dim=2)
        output = torch.cat([output[:, :, :, -(self.pad_right + self.pad_left):-self.pad_right], output], dim=3)

        return output

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