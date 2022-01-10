from auto_disc.output_representations import BaseOutputRepresentation
from auto_disc.utils.config_parameters import IntegerConfigParameter, StringConfigParameter
from auto_disc.utils.spaces.utils import ConfigParameterBinding
from auto_disc.utils.spaces import DictSpace, BoxSpace
from copy import deepcopy

@IntegerConfigParameter(name="timestep", default=-1)
class SliceSelector(BaseOutputRepresentation):
    '''
        Selects a single slice from Lenia states output.
    '''

    output_space = DictSpace(
        slice = BoxSpace(low=0.0, high=1.0, shape=()),
    )# TODO: we dont know shape before initialize here

    def initialize(self, input_space):
        super().initialize(input_space)
        # quick fix
        output_shape = self.input_space[self.wrapped_input_space_key].shape[1:]
        del self.output_space["slice"]
        self.output_space[f"slice_{self.wrapped_input_space_key}"] = BoxSpace(low=0., high=1.0, shape=output_shape)


    def map(self, observations, is_output_new_discovery, **kwargs):
        output = deepcopy(observations)
        output[f"slice_{self.wrapped_input_space_key}"] = output[self.wrapped_input_space_key][self.config.timestep]
        del output[self.wrapped_input_space_key]
        return output
