from auto_disc.output_representations import BaseOutputRepresentation
from auto_disc.utils.config_parameters import IntegerConfigParameter, StringConfigParameter
from auto_disc.utils.spaces.utils import ConfigParameterBinding
from auto_disc.utils.spaces import DictSpace, BoxSpace
from copy import deepcopy

@IntegerConfigParameter(name="timestep", default=-1) #TODO: slice range instead of just frame
#TODO @StringConfigParameter(name="wrapped_output_space_key", default="slice", possible_values="all")
# TODO: replace "image" with self.config.wrapped_output_space_key

class TensorSliceSelector(BaseOutputRepresentation):
    '''
        Selects a single slice from Lenia states output.
    '''

    #output_spaces = {ConfigParameterBinding("wrapped_output_space_key"): BoxSpace(low=0, high=10, shape=())}
    #output_space = DictSpace(spaces=output_spaces)
    output_space = DictSpace(
        image = BoxSpace(low=0, high=10, shape=()),
    )# TODO: we dont know shape before initialize here

    def initialize(self, input_space):
        super().initialize(input_space)
        slice_shape = (1, ) #TODO: slice range instead of just frame
        output_shape = slice_shape + self.input_space[self.wrapped_input_space_key].shape[1:]
        self.output_space["image"] = BoxSpace(low=0, high=10, shape=output_shape)


    def map(self, observations, is_output_new_discovery, **kwargs):
        output = deepcopy(observations)
        output["image"] = output[self.wrapped_input_space_key][self.config.timestep]
        del output[self.wrapped_input_space_key]
        return output
