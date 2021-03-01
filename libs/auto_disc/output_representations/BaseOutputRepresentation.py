from libs.utils.AttrDict import AttrDict

class BaseOutputRepresentation ():
    """ Base class to map the observations of a system to an embedding vector (BC characterization)
    """

    CONFIG_DEFINITION = []

    @classmethod
    def get_default_config(cls):
        default_config = AttrDict()
        for param in cls.CONFIG_DEFINITION:
            param_dict = param.to_dict()
            default_config[param_dict['name']] = param_dict['default']
        
        return default_config

    def __init__(self, **kwargs):
        self.config = self.get_default_config()
        self.config.update(kwargs)

    def calc(self, observations, **kwargs):
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