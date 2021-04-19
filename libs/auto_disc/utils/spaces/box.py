import numbers

import torch
from auto_disc.utils.spaces import BaseSpace


class BoxSpace(BaseSpace):
    """
    A (possibly unbounded) box in R^n. Specifically, a Box represents the
    Cartesian product of n closed intervals. Each interval has the form of one
    of [a, b], (-oo, b], [a, oo), or (-oo, oo).

    There are two common use cases:

    * Identical bound for each dimension::
        >>> BoxSpace(low=-1.0, high=2.0, shape=(3, 4), dtype=torch.float32)
        Box(3, 4)

    * Independent bound for each dimension::
        >>> BoxSpace(low=torch.tensor([-1.0, -2.0]), high=torch.tensor([2.0, 4.0]), dtype=torch.float32)
        Box(2,)

    """

    def __init__(self, low, high, mutator=None, shape=None, dtype=torch.float32, indpb=1.0):
        assert dtype is not None, 'dtype must be explicitly provided. '
    
        # determine shape if it isn't provided directly
        if shape is not None:
            shape = tuple(shape)
            assert isinstance(low, numbers.Number) or low.shape == shape, "low.shape doesn't match provided shape"
            assert isinstance(high, numbers.Number) or high.shape == shape, "high.shape doesn't match provided shape"
        elif not isinstance(low, numbers.Number):
            shape = low.shape
            assert isinstance(high, numbers.Number) or high.shape == shape, "high.shape doesn't match low.shape"
        elif not isinstance(high, numbers.Number):
            shape = high.shape
            assert isinstance(low, numbers.Number) or low.shape == shape, "low.shape doesn't match high.shape"
        else:
            raise ValueError("shape must be provided or inferred from the shapes of low or high")

        self._low = low
        self._high = high
        self._indpb = indpb

        super(BoxSpace, self).__init__(shape, dtype, mutator)

    def initialize(self, parent_obj):
        # Apply potential binding
        super().initialize(parent_obj)
        self._low = self.apply_binding_if_existing(self._low, parent_obj)
        self._high = self.apply_binding_if_existing(self._high, parent_obj)

        if isinstance(self._low, numbers.Number):
            self._low = torch.full(self.shape, self._low, dtype=self.dtype)

        if isinstance(self._high, numbers.Number):
            self._high = torch.full(self.shape, self._high, dtype=self.dtype)

        self.low = self._low.type(self.dtype)
        self.high = self._high.type(self.dtype)

        # Boolean arrays which indicate the interval type for each coordinate
        self.bounded_below = ~torch.isneginf(self.low)
        self.bounded_above = ~torch.isposinf(self.high)

        # indpb – independent probability for each attribute to be mutated.
        if isinstance(self._indpb, numbers.Number):
            self._indpb = torch.full(self.shape, self._indpb, dtype=torch.float64)
        self.indpb = torch.as_tensor(self._indpb, dtype=torch.float64)

    def is_bounded(self, manner="both"):
        below = torch.all(self.bounded_below)
        above = torch.all(self.bounded_above)
        if manner == "both":
            return below and above
        elif manner == "below":
            return below
        elif manner == "above":
            return above
        else:
            raise ValueError("manner is not in {'below', 'above', 'both'}")

    def sample(self):
        """
        Generates a single random sample inside of the Box.

        In creating a sample of the box, each coordinate is sampled according to
        the form of the interval:

        * [a, b] : uniform distribution
        * [a, oo) : shifted exponential distribution
        * (-oo, b] : shifted negative exponential distribution
        * (-oo, oo) : normal distribution
        """
        high = self.high.type(torch.float64) if self.dtype.is_floating_point else self.high.type(torch.int64) + 1
        sample = torch.empty(self.shape, dtype=torch.float64)

        # Masking arrays which classify the coordinates according to interval
        # type
        unbounded = ~self.bounded_below & ~self.bounded_above
        upp_bounded = ~self.bounded_below & self.bounded_above
        low_bounded = self.bounded_below & ~self.bounded_above
        bounded = self.bounded_below & self.bounded_above

        # Vectorized sampling by interval type
        sample[unbounded] = torch.randn(unbounded[unbounded].shape, dtype=torch.float64)

        sample[low_bounded] = (-torch.rand(low_bounded[low_bounded].shape, dtype=torch.float64)).exponential_() + \
                              self.low[low_bounded]

        sample[upp_bounded] = self.high[upp_bounded] - (
            -torch.rand(upp_bounded[upp_bounded].shape, dtype=torch.float64)).exponential_()

        sample[bounded] = (self.low[bounded] - high[bounded]) * torch.rand(bounded[bounded].shape,
                                                                           dtype=torch.float64) + high[bounded]

        if not self.dtype.is_floating_point:  # integer
            sample = torch.floor(sample)

        return sample.type(self.dtype)

    def mutate(self, x):
        if self.mutator:
            mutate_mask = (torch.rand(self.shape) < self.indpb).type(torch.float64)
            x = self.mutator(x, mutate_mask)
            if not self.dtype.is_floating_point:  # integer
                x = torch.floor(x)
            x = x.type(self.dtype)
            if not self.contains(x):
                return self.clamp(x)
            else:
                return x
        else:
            return x

    def contains(self, x):
        if isinstance(x, list):
            x = torch.tensor(x)  # Promote list to array for contains check
        return x.shape == self.shape and torch.all(x >= self.low) and torch.all(x <= self.high)

    def clamp(self, x):
        if self.is_bounded(manner="below"):
            x = torch.max(x, torch.as_tensor(self.low, dtype=self.dtype, device=x.device))
        if self.is_bounded(manner="above"):
            x = torch.min(x, torch.as_tensor(self.high, dtype=self.dtype, device=x.device))
        return x

    def __repr__(self):
        return "BoxSpace({}, {}, {}, {})".format(self.low.min(), self.high.max(), self.shape, self.dtype)

    def __eq__(self, other):
        return isinstance(other, BoxSpace) and (self.shape == other.shape) and torch.allclose(self.low,
                                                                                              other.low) and torch.allclose(
            self.high, other.high)

    def to_json(self):
        dict = super().to_json()
        dict['low'] = self._low
        dict['high'] = self._high
        dict['indpb'] = self._indpb
        return dict
