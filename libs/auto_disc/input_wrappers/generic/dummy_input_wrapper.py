from libs.utils import AttrDict
from libs.auto_disc.input_wrappers import BaseInputWrapper

class DummyInputWrapper(BaseInputWrapper):
    '''
    Empty InputWrapper used when no wrapper should be used.
    '''
    def initialize(self, output_space):
        super().initialize(output_space)
        self.input_space = output_space

    def map(self, parameters, **kwargs):
        return parameters