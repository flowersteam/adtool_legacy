import torch
from auto_disc.utils.mutators import BaseMutator
import numbers

class GaussianMutator(BaseMutator):
    """
        description    : apply a gauss function to the data
        x              : list, tuple or torch of integer or bool; Input data, will be modified
        space          : AutoDiscMutationDefinition object      ; defined mutation parameters...
        mutation_factor: float
    """

    def __init__(self, mean, std):
        self._mean = mean
        self._std = std

    def init_shape(self, shape=None):
        super().init_shape(shape)
        if shape:
            if isinstance(self._mean, numbers.Number):
                self._mean = torch.full(shape, self._mean, dtype=torch.float64)
            if isinstance(self._std, numbers.Number):
                self._std = torch.full(shape, self._std, dtype=torch.float64)
        self.mean = torch.as_tensor(self._mean, dtype=torch.float64)
        self.std = torch.as_tensor(self._std, dtype=torch.float64)

    def __call__(self, x, mutate_mask):
        noise = torch.normal(self.mean, self.std)
        x = x.type(torch.float64) + mutate_mask * noise
        return x

    