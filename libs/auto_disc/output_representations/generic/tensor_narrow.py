from addict import Dict

from copy import copy

from auto_disc.output_representations import BaseOutputRepresentation
from auto_disc.utils.config_parameters import IntegerConfigParameter, BooleanConfigParameter
from auto_disc.utils.spaces import DictSpace, BoxSpace

import torch

@IntegerConfigParameter(name="slice_dim", default=0)
@IntegerConfigParameter(name="slice_start", default=-1)
@IntegerConfigParameter(name="slice_length", default=1)
@BooleanConfigParameter(name="expand_output_space", default=True)
class Narrow(BaseOutputRepresentation):

    CONFIG_DEFINITION = {}
    config = Dict()

    output_space = DictSpace(
        narrowed = BoxSpace(low=0.0, high=0.0, shape=()),
    )# TODO: we dont know shape before initialize here


    def __init__(self, wrapped_input_space_key=None, **kwargs):
        super().__init__(wrapped_input_space_key, **kwargs)

    def initialize(self, input_space):
        super().initialize(input_space)
        # quick fix
        output_shape = tuple(self.input_space[self.wrapped_input_space_key].shape[:self.config.slice_dim]) + \
                       (self.config.slice_length, ) + \
                       tuple(self.input_space[self.wrapped_input_space_key].shape[self.config.slice_dim + 1:])
        del self.output_space["narrowed"]
        self.output_space[f"narrowed_{self.wrapped_input_space_key}"] = BoxSpace(low=0., high=0.0, shape=output_shape)
        self.output_space[f"narrowed_{self.wrapped_input_space_key}"].initialize(self)


    def map(self, observations, is_output_new_discovery, **kwargs):
        output = copy(observations)
        output[f"narrowed_{self.wrapped_input_space_key}"] =torch.narrow(output[self.wrapped_input_space_key],
                                                                      self.config.slice_dim,
                                                                      self.config.slice_start,
                                                                      self.config.slice_length)
        del output[self.wrapped_input_space_key]

        if self.config.expand_output_space:
            self.output_space[f"narrowed_{self.wrapped_input_space_key}"].expand(output[f"narrowed_{self.wrapped_input_space_key}"].cpu().detach())

        return output
