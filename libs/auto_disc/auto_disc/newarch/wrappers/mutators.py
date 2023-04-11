import torch
from typing import Any


def add_gaussian_noise(input_tensor: torch.Tensor,
                       mean: torch.Tensor = torch.tensor([0.]),
                       std: torch.Tensor = torch.tensor([1.]),
                       ) -> torch.Tensor:
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, dtype=float)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std, dtype=float)
    noise_unit = torch.randn(input_tensor.size())
    noise = noise_unit*std + mean
    return input_tensor + noise


def call_mutate_method(input_object: Any):
    """
    If parameters are given as a custom object, then the object must have a 
    `mutate` method in order to mutate the underlying parameters controlling
    the object.
    """
    # quick test to check if `mutate` method returns a new object or
    # modifies in place the input_object
    # if it does something else, this is a user error
    capture_out = input_object.mutate()
    if capture_out is not None:
        return capture_out
    else:
        return input_object
