from leaf.leaf import Leaf
from typing import Dict, Callable
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
                 bound_upper: float = float('inf'),
                 bound_lower: float = -float('inf')) -> None:
        super().__init__()
        self.bound_upper = bound_upper
        self.bound_lower = bound_lower

        self.low = None
        self.high = None

    def map(self, input: Dict) -> Dict:
        """
        Passes `input`, adding a `sampler` Callable item which `Explorer` can
        use to sample from the box space.
        """
        output = deepcopy(input)

        self._update_low_high(output["output"])

#        output["sampler"] = self._generate_sampler()

        return output

    def sample(self) -> torch.Tensor:
        dim = self.low.size()
        rand_nums = torch.rand(dim)

        dim_lengths = self.high - self.low
        sample = rand_nums * dim_lengths + self.low

        return sample

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

    def _generate_sampler(self) -> Callable[[], torch.Tensor]:
        def func():
            dim = self.low.size()
            rand_nums = torch.rand(dim)

            dim_lengths = self.high - self.low
            sample = rand_nums * dim_lengths + self.low

            return sample
        return func
