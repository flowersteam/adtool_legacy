from addict import Dict
from auto_disc.utils.spaces import DictSpace

class BaseInputWrapper():
    """ Base class to map the parameters sent by the explorer to the system's input space
    """

    config = Dict()
    input_space = DictSpace()

    def __init__(self):
        self.input_space.initialize(self)

    def initialize(self, output_space):
        self._output_space = output_space

    def map(self, parameters, **kwargs):
        raise NotImplementedError