from libs.utils.AttrDict import AttrDict
from libs.auto_disc.input_wrappers.BaseInputWrapper import BaseInputWrapper

class DummyInputWrapper(BaseInputWrapper):
    def initialize(self, output_space):
        super().initialize(output_space)
        self.input_space = output_space

    def map(self, parameters, **kwargs):
        return parameters