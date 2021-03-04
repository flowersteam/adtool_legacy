from libs.utils.AttrDict import AttrDict
from libs.auto_disc.output_representations.BaseOutputRepresentation import BaseOutputRepresentation

class DummyOutputRepresentation(BaseOutputRepresentation):
    def initialize(self, input_space):
        super().initialize(input_space)
        self.output_space = input_space

    def map(self, observations, **kwargs):
        return observations

    def calc_distance(self, embedding_a, embedding_b, **kwargs):
        # L2 + add regularizer to avoid dead outcomes
        dist = (embedding_a - embedding_b).pow(2).sum(-1).sqrt() - 0.5 * embedding_b.pow(2).sum(-1).sqrt()
        return dist
