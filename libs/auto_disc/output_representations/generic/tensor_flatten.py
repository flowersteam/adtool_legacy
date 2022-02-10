from auto_disc.output_representations import BaseOutputRepresentation
from auto_disc.utils.config_parameters import BooleanConfigParameter
from auto_disc.utils.spaces import DictSpace, BoxSpace
from copy import copy
import torch

@BooleanConfigParameter(name="expand_output_space", default=True)
class Flatten(BaseOutputRepresentation):
    '''
        Selects a single slice from Lenia states output.
    '''

    from addict import Dict  # TODO: put config init in Base Module
    CONFIG_DEFINITION = {}
    config = Dict()

    output_space = DictSpace(
        flattened = BoxSpace(low=0.0, high=0.0, shape=()),
    )# TODO: we dont know shape before initialize here

    def init(self, wrapped_input_space_key=None, **kwargs):
        super().init(wrapped_input_space_key, **kwargs)

    def initialize(self, input_space):
        super().initialize(input_space)
        # quick fix
        output_shape = (self.input_space[self.wrapped_input_space_key].shape[0], torch.prod(torch.LongTensor(self.input_space[self.wrapped_input_space_key].shape[1:])).item())
        del self.output_space["flattened"]
        self.output_space[f"flattened_{self.wrapped_input_space_key}"] = BoxSpace(low=0., high=1.0, shape=output_shape)
        self.output_space[f"flattened_{self.wrapped_input_space_key}"].initialize(self)


    def map(self, observations, is_output_new_discovery, **kwargs):
        output = copy(observations)
        output[f"flattened_{self.wrapped_input_space_key}"] = output[self.wrapped_input_space_key].view(output[self.wrapped_input_space_key].shape[0], -1)
        del output[self.wrapped_input_space_key]

        if self.config.expand_output_space:
            self.output_space[f"flattened_{self.wrapped_input_space_key}"].expand(output[f"flattened_{self.wrapped_input_space_key}"].cpu().detach())

        return output
