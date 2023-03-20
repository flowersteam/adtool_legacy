import torch


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
