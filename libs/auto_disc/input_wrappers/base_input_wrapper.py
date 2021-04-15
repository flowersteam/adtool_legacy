from addict import Dict
from auto_disc.utils.spaces import DictSpace
from copy import deepcopy

class BaseInputWrapper():
    """ Base class to map the parameters sent by the explorer to the system's input space
    """

    config = Dict()
    input_space = DictSpace()

    def __init__(self, wrapped_output_space_key=None):
        self.input_space = deepcopy(self.input_space)
        self.input_space.initialize(self)
        self.wrapped_output_space_key = wrapped_output_space_key
        self.initial_input_space_keys = [key for key in self.input_space]

    def initialize(self, output_space):
        self._output_space = output_space
        for key in iter(output_space):
            if key != self.wrapped_output_space_key:
                self.input_space[key] = output_space[key]

    def map(self, parameters, **kwargs):
        raise NotImplementedError