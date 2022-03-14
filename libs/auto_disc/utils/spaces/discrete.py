import torch
from auto_disc.utils.spaces import BaseSpace
from auto_disc.utils.spaces.utils import distance
from copy import deepcopy

class DiscreteSpace(BaseSpace):
    r"""A discrete space in :math:`\{ 0, 1, \\dots, n-1 \}`.
    /!\ mutation is gaussian by default: please create custom space inheriting from discrete space for custom mutation functions

    Example::

        >>> DiscreteSpace(2)

    """

    def __init__(self, n, mutator=None, indpb=1.0):
        assert n >= 0
        self._n = n
        self._indpb = indpb
        
        super(DiscreteSpace, self).__init__((),torch.int64, mutator)

    def initialize(self, parent_obj):
        # Apply potential binding
        super().initialize(parent_obj)
        self.n = self.apply_binding_if_existing(self._n, parent_obj)

        # indpb – independent probability for each attribute to be mutated.
        self.indpb = torch.as_tensor(self._indpb, dtype=torch.float64)

    def sample(self):
        return torch.randint(self.n, ())

    def mutate(self, x):
        if self.mutator:
            mutate_mask = torch.rand(self.shape) < self.indpb
            x = self.mutator(x, mutate_mask)
            x = torch.floor(x).type(self.dtype)
            if not self.contains(x):
                return self.clamp(x)
            else:
                return x
        else:
            return x

    def crossover(self, x1, x2):
        # return x1, x2 (as dim=1), overwrite to blend values
        return deepcopy(x1), deepcopy(x2)

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif not x.dtype.is_floating_point and (x.shape == ()):  # integer or size 0
            as_int = int(x)
        else:
            return False
        return 0 <= as_int < self.n

    def clamp(self, x):
        x.data = torch.max(x.data, torch.as_tensor(0, dtype=self.dtype, device=x.device))
        x.data = torch.min(x.data, torch.as_tensor(self.n - 1, dtype=self.dtype, device=x.device))
        return x

    def calc_distance(self, x1, x2):
        x2 = torch.stack(x2)
        dist = distance.calc_l2(x1, x2)

        return dist

    def expand(self, x):
        x = x.type(self.dtype)
        if x > self.n:
            self.n = x

    def __repr__(self):
        return "DiscreteSpace(%d)" % self.n

    def __eq__(self, other):
        return isinstance(other, DiscreteSpace) and self.n == other.n

    def to_json(self):
        dict = super().to_json()
        dict['n'] = self._n
        dict['indpb'] = self._indpb
        return dict
