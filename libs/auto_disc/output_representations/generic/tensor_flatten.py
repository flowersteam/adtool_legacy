from addict import Dict
from auto_disc.output_representations import BaseOutputRepresentation
from auto_disc.utils.config_parameters import IntegerConfigParameter, BooleanConfigParameter
from auto_disc.utils.spaces import DictSpace, BoxSpace
from copy import copy
import torch

@IntegerConfigParameter(name="start_dim", default=0)
@IntegerConfigParameter(name="end_dim", default=-1)
@BooleanConfigParameter(name="expand_output_space", default=True)
class Flatten(BaseOutputRepresentation):
    CONFIG_DEFINITION = {}
    config = Dict()

    output_space = DictSpace(
        flattened = BoxSpace(low=0.0, high=0.0, shape=()),
    )# TODO: we dont know shape before initialize here

    def __init__(self, wrapped_input_space_key=None, **kwargs):
        super().__init__(wrapped_input_space_key, **kwargs)

    def initialize(self, input_space):
        super().initialize(input_space)
        # quick fix
        input_shape = self.input_space[self.wrapped_input_space_key].shape
        if self.config.end_dim == -1:
            end_dim = len(input_shape)
        else:
            end_dim = self.config.end_dim + 1
        output_shape = input_shape[:self.config.start_dim] + \
                       (torch.prod(torch.LongTensor(input_shape)[self.config.start_dim:end_dim]).item(), ) + \
                       input_shape[end_dim:]
        del self.output_space["flattened"]
        self.output_space[f"flattened_{self.wrapped_input_space_key}"] = BoxSpace(low=0., high=0.0, shape=output_shape)
        self.output_space[f"flattened_{self.wrapped_input_space_key}"].initialize(self)


    def map(self, observations, is_output_new_discovery, **kwargs):
        output = copy(observations)
        output[f"flattened_{self.wrapped_input_space_key}"] = torch.flatten(output[self.wrapped_input_space_key],
                                                                            start_dim=self.config.start_dim,
                                                                            end_dim=self.config.end_dim)
        del output[self.wrapped_input_space_key]

        if self.config.expand_output_space:
            self.output_space[f"flattened_{self.wrapped_input_space_key}"].expand(output[f"flattened_{self.wrapped_input_space_key}"].cpu().detach())

        return output
