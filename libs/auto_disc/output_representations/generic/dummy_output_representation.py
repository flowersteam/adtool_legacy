from libs.utils import AttrDict
from libs.auto_disc.output_representations import BaseOutputRepresentation
from libs.utils import LossFunc

class DummyOutputRepresentation(BaseOutputRepresentation):
    '''
    Empty OutputRepresentation used when no representation of the system's output mut be used.
    '''
    def initialize(self, input_space):
        super().initialize(input_space)
        self.output_space = input_space

    def map(self, observations, **kwargs):
        return observations

    def calc_distance(self, embedding_a, embedding_b, **kwargs):
        # L2 + add regularizer to avoid dead outcomes
        dist = LossFunc.l2(embedding_a, embedding_b)
        return dist
