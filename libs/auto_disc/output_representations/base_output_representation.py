from addict import Dict
from auto_disc.utils.spaces import DictSpace

class BaseOutputRepresentation ():
    """ Base class to map the observations of a system to an embedding vector (BC characterization)
    """

    config = Dict()
    output_space = DictSpace()

    def __init__(self):
        self.output_space.initialize(self)

    def initialize(self, input_space):
        self._input_space = input_space

    def map(self, observations, **kwargs):
        """ Maps the observations of a system to an embedding
            #TODO: space of possible embeddings as in https://github.com/openai/gym/tree/master/gym/spaces
            #TODO: allow to calc batch of observations
            Args:
                observations (AttrDict):
            Returns
                embeddings (AttrDict): generally vector but we might need AttrDict structures, for instance for IMGEP-HOLMES
        """
        raise NotImplementedError

    def calc_distance(self, embedding_a, embedding_b, **kwargs):
        """ Compute the distance between 2 embedding
        """
        raise NotImplementedError