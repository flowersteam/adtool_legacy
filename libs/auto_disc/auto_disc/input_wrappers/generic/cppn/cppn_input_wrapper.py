from typing import Any, Dict
import torch
from auto_disc.utils.spaces import DictSpace
from auto_disc.input_wrappers.generic.cppn.utils import CPPNGenomeSpace
from auto_disc.input_wrappers.generic.cppn import pytorchneat
from auto_disc.utils.config_parameters import IntegerConfigParameter
from leaf.leaf import Leaf
from copy import deepcopy


@IntegerConfigParameter(name="n_passes", default=2, min=1)
class CppnInputWrapper(Leaf):
    CONFIG_DEFINITION = {}

    input_space = DictSpace(
        genome=CPPNGenomeSpace()
    )

    def __init__(self, wrapped_key: str, output_space: DictSpace = None, **kwargs) -> None:
        # TODO: logger kwarg is not yet handled
        super().__init__()

        self._wrapped_key = wrapped_key

        self.initialize_spaces(output_space)

    def initialize_spaces(self, output_space: DictSpace = None) -> None:
        wrapped_key = self._wrapped_key

        self.input_space = deepcopy(self.input_space)
        self._initial_input_space_keys = [key for key in self.input_space]

        self._wrapped_key = wrapped_key

        if output_space is not None:
            self.output_space = output_space
            for key in iter(output_space):
                if key != self._wrapped_key:
                    self.input_space[key] = output_space[key]

    def map(self, input: Dict[str, Any], **kwargs) -> Dict[str, torch.Tensor]:
        """
            Map the input parameters (from the explorer) 
            to the cppn output parameters (sytem input)

            Args:
                parameters: cppn input parameters
                is_input_new_discovery: indicates if it is a new discovery
            Returns:
                parameters: parameters after map to match system input
        """
        # input
        cppn_genome = input['genome']

        # initialization of CPPN
        initialization_cppn = pytorchneat.rnn.RecurrentNetwork.create(
            cppn_genome, self.input_space['genome'].neat_config)
        cppn_output_height = int(
            self.output_space[self._wrapped_key].shape[1])
        cppn_output_width = int(
            self.output_space[self._wrapped_key].shape[0])

        # forward
        cppn_input = pytorchneat.utils.create_image_cppn_input(
            (cppn_output_height, cppn_output_width), is_distance_to_center=True, is_bias=True)
        cppn_output = initialization_cppn.activate(
            cppn_input, self.config.n_passes)
        cppn_net_output = (1.0 - cppn_output.abs()).squeeze()

        # output
        output = deepcopy(input)
        del output["genome"]
        output[self._wrapped_key] = cppn_net_output

        return output
