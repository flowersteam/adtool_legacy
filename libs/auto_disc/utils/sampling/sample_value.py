import torch

def sample_value(space):
    space_type = space.type
    space_dimensions = space.dims
    space_bounds = space.bounds

    if space_type.name == "Float": # continuous
        val = (space_bounds[0] - space_bounds[1]) * torch.rand(tuple(space_dimensions)) + space_bounds[1]
    elif space_type.name == "Integer": # Dicrete
        val = torch.randint(space_bounds[0], space_bounds[1], tuple(space_dimensions))
    elif space_type.name == "Boolean": # Dicrete
        val = torch.rand(tuple(space_dimensions)) > 0.5
    else:
        raise ValueError('Unknown parameter type {!r} for sampling!', space_type)

    if space_dimensions == []:
        val = val.item()

    return val