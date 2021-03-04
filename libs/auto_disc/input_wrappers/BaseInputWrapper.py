from libs.utils.AttrDict import AttrDict
from libs.utils.auto_disc_parameters.AutoDiscParameter import get_default_values

class BaseInputWrapper ():
    """ Base class to map the parameters sent by the explorer to the system's input space
    """

    CONFIG_DEFINITION = []

    def __init__(self, **kwargs):
        self.config = get_default_values(self, self.CONFIG_DEFINITION)
        self.config.update(kwargs)

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