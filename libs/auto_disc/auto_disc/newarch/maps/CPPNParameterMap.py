from typing import Any, Dict, Tuple
import torch
from auto_disc.utils.spaces import DictSpace
from auto_disc.input_wrappers.generic.cppn.utils import CPPNGenomeSpace
from auto_disc.input_wrappers.generic.cppn import pytorchneat
from auto_disc.input_wrappers import BaseInputWrapper
from auto_disc.utils.config_parameters import IntegerConfigParameter

from leaf.Leaf import Leaf
import os
import neat
from copy import deepcopy


class CPPNParameterMap(Leaf):
    """ Base class to map the parameters sent by the explorer to the system's input space
    """

    def __init__(self, premap_key: str = "genome",
                 postmap_key: str = "init_state",
                 postmap_dim: Tuple[int, int] = (8, 8),
                 n_passes: int = 2,
                 config_path: str = "./config.cfg") -> None:
        super().__init__()
        self.premap_key = premap_key
        self.postmap_key = postmap_key
        self.postmap_dim = postmap_dim
        self.n_passes = n_passes

        # set global configuration parameters for NEAT
        self.neat_config = neat.Config(
            pytorchneat.selfconnectiongenome.SelfConnectionGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )

        # put for compatibility with old interface
        # TODO: probably change the function signature for sampling
        self.output_shape = [None]

    def map(self, input: Dict) -> Dict:
        """
            Map the input parameters (from the explorer) to the cppn output parameters (sytem input)

            Args:
                parameters: cppn input parameters
                is_input_new_discovery: indicates if it is a new discovery
            Returns:
                parameters: parameters after map to match system input
        """
        intermed_dict = deepcopy(input)

        # always overrides "genome" with new sample
        intermed_dict[self.premap_key] = self.sample(self.output_shape)

        cppn_genome = intermed_dict[self.premap_key]
        initialization_cppn = pytorchneat.rnn.RecurrentNetwork.create(
            cppn_genome, self.neat_config)

        # configure output size
        cppn_output_height = int(self.postmap_dim[0])
        cppn_output_width = int(self.postmap_dim[1])

        cppn_input = pytorchneat.utils.create_image_cppn_input(
            (cppn_output_height, cppn_output_width),
            is_distance_to_center=True,
            is_bias=True
        )
        cppn_output = initialization_cppn.activate(
            cppn_input, self.n_passes)
        cppn_net_output = (1.0 - cppn_output.abs()).squeeze()

        # finish transformation of input dictionary
        intermed_dict[self.postmap_key] = cppn_net_output
        del intermed_dict['genome']

        return intermed_dict

    def sample(self, data_shape: Tuple) -> torch.Tensor:
        genome = self.neat_config.genome_type(0)
        # randomly initializes the genome
        genome.configure_new(self.neat_config.genome_config)
        return genome
