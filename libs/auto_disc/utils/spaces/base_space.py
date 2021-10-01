from auto_disc.utils.spaces.utils import ConfigParameterBinding
class BaseSpace(object):
    """
    Defines the init_space, genome_space and intervention_space of a system
    """

    def __init__(self, shape=None, dtype=None, mutator=None):
        self.shape = None if shape is None else tuple(shape)
        self.dtype = dtype
        self.mutator = mutator

    def initialize(self, parent_obj):
        """
        Initialize the space."""
        if self.shape is not None:
            new_shape = []
            for elem in self.shape:
                new_shape.append(int(self.apply_binding_if_existing(elem, parent_obj)))
            self.shape = tuple(new_shape)
            if self.mutator:
                self.mutator.init_shape(self.shape)

    def apply_binding_if_existing(self, var, lookup_obj):
        if isinstance(var, ConfigParameterBinding):
            value = var.__get__(lookup_obj)
        else:
            value = var
            
        return value

    def sample(self):
        """
        Randomly sample an element of this space.
        Can be uniform or non-uniform sampling based on boundedness of space."""
        raise NotImplementedError

    def mutate(self, x):
        """
        Randomly mutate an element of this space.
        """
        raise NotImplementedError

    def crossover(self, x1, x2):
        """
        Mate 2 elements of this space
        """
        raise NotImplementedError

    def contains(self, x):
        """
        Return boolean specifying if x is a valid
        member of this space
        """
        raise NotImplementedError

    def clamp(self, x):
        """
        Return a valid clamped value of x inside space's bounds
        """
        raise NotImplementedError

    def calc_distance(self, x1, x2):
        """
        Returns the distance between a point x1 and a list of points x2
        """
        raise NotImplementedError

    def expand(self, x):
        """
        expands the space bounds to include a point x
        """
        raise NotImplementedError

    def __contains__(self, x):
        return self.contains(x)

    def to_json(self):
        shape = []
        if self.shape is not None:
            for element in self.shape:
                if isinstance(element, ConfigParameterBinding):
                    shape.append(element.to_json())
                else:
                    shape.append(element)
        else:
            shape = None
        return {
            'shape': shape
        }
