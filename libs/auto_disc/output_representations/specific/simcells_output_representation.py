from auto_disc.output_representations import BaseOutputRepresentation
from auto_disc.utils.spaces import DictSpace, BoxSpace
from copy import deepcopy
import torch

class SimCellsMatRenderToRGB(BaseOutputRepresentation):
    CONFIG_DEFINITION = {}

    output_space = DictSpace(
        matrender_rgb = BoxSpace(low=0., high=1., shape=(1, )),
    )# TODO: we dont know shape before initialize here

    def __init__(self, wrapped_input_space_key=None, **kwargs):
        super().__init__(wrapped_input_space_key="MatRender", **kwargs)

    def initialize(self, input_space):
        super().initialize(input_space)
        input_shape = self.input_space[self.wrapped_input_space_key].shape
        timepoints, n_channels, SX, SY = input_shape[0], 3, input_shape[1], input_shape[2],
        self.output_space["matrender_rgb"].shape = (timepoints, n_channels, SX, SY)


    def map(self, observations, is_output_new_discovery, **kwargs):
        output = deepcopy(observations)

        matrender = output[self.wrapped_input_space_key].int()
        bbb = ((matrender & 0x00FF0000) >> 16).float() / 255.0
        ggg = ((matrender & 0x0000FF00) >> 8).float() / 255.0
        rrr = ((matrender & 0x000000FF)).float() / 255.0
        color_array = torch.cat([bbb, ggg, rrr], dim=1)
        color_array[(color_array == 0.0).all(1).unsqueeze(1).repeat(1,3,1,1)] = 1.0

        output["matrender_rgb"] = color_array  # NWHC
        del output[self.wrapped_input_space_key]
        return output
