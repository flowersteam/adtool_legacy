from auto_disc import BaseAutoDiscModule
from auto_disc.utils.spaces import DictSpace
from copy import deepcopy

class BaseOutputRepresentation (BaseAutoDiscModule):
    """ Base class to map the observations of a system to an embedding vector (BC characterization)
    """

    output_space = DictSpace()

    def __init__(self, wrapped_input_space_key: str=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.output_space = deepcopy(self.output_space)
        self.output_space.initialize(self)
        self.wrapped_input_space_key = wrapped_input_space_key
        self.initial_output_space_keys = [key for key in self.output_space]

    def initialize(self, input_space: DictSpace) -> None:
        '''
        Defines input and output space for the output representation.
        '''
        self.input_space = input_space
        for key in iter(input_space):
            if key != self.wrapped_input_space_key:
                self.output_space[key] = input_space[key]
        

    def map(self, observations, is_output_new_discovery, **kwargs):
        """
            Maps the observations of a system to an embedding vector

            Args:
                parameters: input parameters
                is_output_new_discovery: indicates if it is a new discovery
        """
        raise NotImplementedError

    def calc_distance(self, embedding_a, embedding_b, **kwargs):
        """
            Compute the distance between 2 embeddings in the latent space
            /!\ batch mode embedding_a and embedding_b can be N*M or M
        """
        raise NotImplementedError