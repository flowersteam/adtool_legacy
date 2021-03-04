from libs.utils.AttrDict import AttrDict
from libs.utils.auto_disc_parameters.AutoDiscParameter import get_default_values

class BaseInputWrapper():
    """ Base class to map the parameters sent by the explorer to the system's input space
    """

    CONFIG_DEFINITION = []
    INPUT_SPACE_DEFINITION = []

    def __init__(self, **kwargs):
        self.config = get_default_values(self, self.CONFIG_DEFINITION)
        self.config.update(kwargs)

        self.input_space = get_default_values(self, self.INPUT_SPACE_DEFINITION)
        self.input_space.update(kwargs)

    def initialize(self, output_space):
        self._output_space = output_space

    def map(self, parameters, **kwargs):
        raise NotImplementedError