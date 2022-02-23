from copy import copy

from auto_disc.output_representations import BaseOutputRepresentation
from auto_disc.utils.config_parameters import BooleanConfigParameter, StringConfigParameter
from auto_disc.utils.spaces import DictSpace, BoxSpace

import torch

@StringConfigParameter(name="shape", default="() ")
@BooleanConfigParameter(name="expand_output_space", default=True)
class Reshape(BaseOutputRepresentation):
    CONFIG_DEFINITION = {}

    output_space = DictSpace(
        reshaped = BoxSpace(low=0.0, high=0.0, shape=()),
    )# TODO: we dont know shape before initialize here


    def __init__(self, wrapped_input_space_key=None, **kwargs):
        super().__init__(wrapped_input_space_key, **kwargs)

    def initialize(self, input_space):
        super().initialize(input_space)
        # quick fix
        del self.output_space["reshaped"]
        self.output_space[f"reshaped_{self.wrapped_input_space_key}"] = BoxSpace(low=0., high=0.0, shape=eval(self.config.shape))
        self.output_space[f"reshaped_{self.wrapped_input_space_key}"].initialize(self)

    def map(self, observations, is_output_new_discovery, **kwargs):
        output = copy(observations)
        output[f"reshaped_{self.wrapped_input_space_key}"] =torch.reshape(output[self.wrapped_input_space_key],
                                                                      eval(self.config.shape))
        del output[self.wrapped_input_space_key]

        if self.config.expand_output_space:
            self.output_space[f"reshaped_{self.wrapped_input_space_key}"].expand(output[f"reshaped_{self.wrapped_input_space_key}"].cpu().detach())

        return output
