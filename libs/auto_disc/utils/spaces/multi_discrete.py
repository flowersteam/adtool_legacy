import numbers

import torch
from auto_disc.utils.spaces import BaseSpace
from auto_disc.utils.spaces.utils import distance
from copy import deepcopy

class MultiDiscreteSpace(BaseSpace):
    """
    - The multi-discrete space consists of a series of discrete spaces with different number of possible instances in eachs
    - Can be initialized as

        MultiDiscreteSpace([ 5, 2, 2 ])

    """

    def __init__(self, nvec, mutator=None, indpb=1.0):

        """
        nvec: vector of counts of each categorical variable
        """
        self._nvec = nvec
        self._indpb = indpb

        super(MultiDiscreteSpace, self).__init__(nvec, torch.int64, mutator)

    def initialize(self, parent_obj):
        # Apply potential binding
        super().initialize(parent_obj)
        self._nvec = self.apply_binding_if_existing(self._nvec, parent_obj)
        self.shape = self._nvec
        assert (torch.tensor(self._nvec) > 0).all(), 'nvec (counts) have to be positive'
        self.nvec = torch.as_tensor(self._nvec, dtype=torch.int64)

        # indpb – independent probability for each attribute to be mutated.
        if isinstance(self._indpb, numbers.Number):
            self._indpb = torch.full(self.nvec.shape, self._indpb, dtype=torch.float64)
        self.indpb = torch.as_tensor(self._indpb, dtype=torch.float64)

    def sample(self):
        return (torch.rand(self.nvec.shape) * self.nvec).type(self.dtype)

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
        child_1 = deepcopy(x1)
        child_2 = deepcopy(x2)
        if self.shape != ():
            crossover_mask = (torch.rand(self.shape) < self.indpb)
            switch_parent_mask = crossover_mask & torch.randint(2, self.shape, dtype=torch.bool)
            # mix parents
            child_1[switch_parent_mask] = x2[switch_parent_mask]
            child_2[switch_parent_mask] = x1[switch_parent_mask]
        return child_1, child_2

    def contains(self, x):
        if isinstance(x, list):
            x = torch.tensor(x)  # Promote list to array for contains check
        # if nvec is uint32 and space dtype is uint32, then 0 <= x < self.nvec guarantees that x
        # is within correct bounds for space dtype (even though x does not have to be unsigned)
        return x.shape == self.shape and (0 <= x).all() and (x < self.nvec).all()

    def clamp(self, x):
        x = torch.max(x, torch.as_tensor(0, dtype=self.dtype, device=x.device))
        x = torch.min(x, torch.as_tensor(self.nvec - 1, dtype=self.dtype, device=x.device))
        return x

    def calc_distance(self, x1, x2):
        x2 = torch.stack(x2)
        dist = distance.calc_l2(x1, x2)

        return dist

    def expand(self, x):
        x = x.type(self.dtype)
        assert len(x.shape) == len(self.nvec)
        self.nvec = [max(n, x[i]) for i, n in enumerate(self.nvec)]
        self.shape = self.nvec

    def __repr__(self):
        return "MultiDiscreteSpace({})".format(self.nvec)

    def __eq__(self, other):
        return isinstance(other, MultiDiscreteSpace) and torch.all(self.nvec == other.nvec)

    def to_json(self):
        dict = super().to_json()
        dict['nvec'] = self._nvec
        dict['indpb'] = self._indpb
        return dict
