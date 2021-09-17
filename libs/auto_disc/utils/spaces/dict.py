from addict import Dict
from auto_disc.utils.spaces import BaseSpace


class DictSpace(BaseSpace):
    """
    A Dict dictionary of simpler spaces.

    Example usage:
    self.genome_space = spaces.DictSpace({"position": spaces.Discrete(2), "velocity": spaces.Discrete(3)})

    Example usage [nested]:
    self.nested_genome_space = spaces.DictSpace({
        'sensors':  spaces.DictSpace({
            'position': spaces.Box(low=-100, high=100, shape=(3,)),
            'velocity': spaces.Box(low=-1, high=1, shape=(3,)),
            'front_cam': spaces.Tuple((
                spaces.Box(low=0, high=1, shape=(10, 10, 3)),
                spaces.Box(low=0, high=1, shape=(10, 10, 3))
            )),
            'rear_cam': spaces.Box(low=0, high=1, shape=(10, 10, 3)),
        }),
        'ext_controller': spaces.MultiDiscrete((5, 2, 2)),
        'inner_state':spaces.DictSpace({
            'charge': spaces.Discrete(100),
            'system_checks': spaces.MultiBinary(10),
            'job_status': spaces.DictSpace({
                'task': spaces.Discrete(5),
                'progress': spaces.Box(low=0, high=100, shape=()),
            })
        })
    })
    """

    def __init__(self, spaces=None, **spaces_kwargs):
        assert (spaces is None) or (
            not spaces_kwargs), 'Use either DictSpace(spaces=dict(...)) or DictSpace(foo=x, bar=z)'
        if spaces is None:
            spaces = spaces_kwargs
        if isinstance(spaces, list):
            spaces = Dict(spaces)
        self.spaces = spaces
        for space in spaces.values():
            assert isinstance(space, BaseSpace), 'Values of the attrdict should be instances of gym.Space'
        super().__init__(None, None)  # None for shape and dtype, since it'll require special handling

    def initialize(self, parent_obj):
        for _, space in self.spaces.items():
            space.initialize(parent_obj)

    def sample(self):
        return Dict([(k, space.sample()) for k, space in self.spaces.items()])

    def mutate(self, x):
        return Dict([(k, space.mutate(x[k])) for k, space in self.spaces.items()])

    def contains(self, x):
        if not isinstance(x, dict) or len(x) != len(self.spaces):
            return False
        for k, space in self.spaces.items():
            if k not in x:
                return False
            if not space.contains(x[k]):
                return False
        return True

    def clamp(self, x):
        return Dict([(k, space.clamp(x[k])) for k, space in self.spaces.items()])

    def calc_distance(self, x1, x2):
        return Dict([(k, space.calc_distance(x1[k], [x[k] for x in x2])) for k, space in self.spaces.items()])

    def expand(self, x):
        return Dict([(k, space.expand(x[k])) for k, space in self.spaces.items()])

    def __getitem__(self, key):
        return self.spaces[key]

    def __setitem__(self, key, value):
        self.spaces[key] = value

    def __delitem__(self, key):
        del self.spaces[key]

    def __iter__(self):
        for key in self.spaces:
            yield key

    def __repr__(self):
        return "DictSpace(" + ", ".join([str(k) + ":" + str(s) for k, s in self.spaces.items()]) + ")"

    def __eq__(self, other):
        return isinstance(other, DictSpace) and self.spaces == other.spaces

    def __len__(self):
        return len(self.spaces)

    def to_json(self):
        dict = {}
        for key, space in self.spaces.items():
            dict[key] = space.to_json()
        return dict