from leaf.leaf import Leaf
from leaf.locators import FileLocator
from typing import Dict, Callable, Tuple
import torch
from copy import deepcopy


class BoxProjector(Leaf):
    """
    Projects its input into a box space, i.e., the Cartesian product of N
    real-valued intervals, where the dimension N is set by the input dimension.

    Note that it's `map()` method essentially just passes the received input,
    but also adds a `sampler` key to the dict which an `Explorer` can use to
    sample from the space.
    """

    def __init__(self,
                 wrapped_key: str,
                 bound_upper: torch.Tensor = torch.tensor([float('inf')]),
                 bound_lower: torch.Tensor = torch.tensor([-float('inf')]),
                 init_low: torch.Tensor = None,
                 init_high: torch.Tensor = None,
                 tensor_shape: Tuple = None) -> None:
        super().__init__()
        self.locator = FileLocator()
        self.wrapped_key = wrapped_key
        self.bound_upper = bound_upper
        self.bound_lower = bound_lower

        # initialize data_shape if known
        if init_low is not None:
            self.tensor_shape = init_low.size()
        else:
            self.tensor_shape = tensor_shape

        self.low = init_low
        self.high = init_high

    def map(self, input: Dict) -> Dict:
        """
        Passes `input`, adding a `sampler` Callable item which `Explorer` can
        use to sample from the box space.
        """
        output = deepcopy(input)

        tensor_data = output[self.wrapped_key]

        # set tensor_shape dynamically
        if self.tensor_shape is None:
            self.tensor_shape = tensor_data.size()

        tensor_data = self._clamp_and_truncate(tensor_data)
        self._update_low_high(tensor_data)

        return output

    def sample(self) -> torch.Tensor:
        dim = self.tensor_shape
        rand_nums = torch.rand(dim)

        dim_lengths = self.high - self.low
        sample = rand_nums * dim_lengths + self.low

        return sample

    def _clamp_and_truncate(self, data: torch.Tensor) -> torch.Tensor:
        clamped_data = torch.min(
            torch.max(data, self.bound_lower), self.bound_upper)
        # TODO: truncate dimensions
        return clamped_data

    def _update_low_high(self, data: torch.Tensor) -> None:
        """
        Update self.low and self.high which record the highest and lowest
        feature observations in the box space.
        """
        if self.low is None:
            self.low = torch.zeros_like(data)
        if self.high is None:
            self.high = torch.zeros_like(data)

        low_mask = torch.less(data, self.low)
        high_mask = torch.greater(data, self.high)

        # views are not used here, so it is deep-copied
        self.low[low_mask] = data[low_mask]
        self.high[high_mask] = data[high_mask]

        return