import numbers

import torch
from auto_disc.utils.spaces import BaseSpace


class MultiBinarySpace(BaseSpace):
    """
    An n-shape binary space.

    The argument to MultiBinarySpace defines n, which could be a number or a `list` of numbers.

    Example Usage:

    >> self.genome_space = spaces.MultiBinarySpace(5)

    >> self.genome_space.sample()

        array([0,1,0,1,0], dtype =int8)

    >> self.genome_space = spaces.MultiBinarySpace([3,2])

    >> self.genome_space.sample()

        array([[0, 0],
               [0, 1],
               [1, 1]], dtype=int8)

    """

    def __init__(self, n, indpb=1.0):
        self._indpb = indpb

        super(MultiBinarySpace, self).__init__((n,), torch.int8)

    def initialize(self, parent_obj):
        # Apply potential binding
        super().initialize(parent_obj)

        if isinstance(self._indpb, numbers.Number):
            self._indpb = torch.full(self.shape, self._indpb, dtype=torch.float64)
        self.indpb = torch.as_tensor(self._indpb, dtype=torch.float64)

    def sample(self):
        return torch.randint(low=0, high=2, size=self.shape, dtype=self.dtype)

    def mutate(self, x):
        mutate_mask = torch.rand(self.shape) < self.indpb
        x = torch.where(mutate_mask, (~x.bool()).type(self.dtype), x)
        if not self.contains(x):
            return self.clamp(x)
        else:
            return x

    def contains(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = torch.tensor(x)  # Promote list to array for contains check
        if self.shape != x.shape:
            return False
        return ((x == 0) | (x == 1)).all()

    def clamp(self, x):
        # TODO?
        return x

    def __repr__(self):
        return "MultiBinarySpace({})".format(self.shape[0])

    def __eq__(self, other):
        return isinstance(other, MultiBinarySpace) and self.shape[0] == other.shape[0]
