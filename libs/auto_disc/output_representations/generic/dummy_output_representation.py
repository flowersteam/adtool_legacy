from auto_disc.output_representations import BaseOutputRepresentation
from auto_disc.utils.spaces.utils import distance

class DummyOutputRepresentation(BaseOutputRepresentation):
    '''
    Empty OutputRepresentation used when no representation of the system's output mut be used.
    '''
    def map(self, observations, is_output_new_discovery, **kwargs):
        return observations

    def calc_distance(self, embedding_a, embedding_b, **kwargs):
        # L2 + add regularizer to avoid dead outcomes
        dist = distance.calc_l2(embedding_a, embedding_b)
        return dist
