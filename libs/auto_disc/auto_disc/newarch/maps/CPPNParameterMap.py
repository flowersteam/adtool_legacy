from typing import Any, Dict, Tuple
import torch
from auto_disc.utils.spaces import DictSpace
from auto_disc.input_wrappers.generic.cppn.utils import CPPNGenomeSpace
from auto_disc.input_wrappers.generic.cppn import pytorchneat
from auto_disc.input_wrappers import BaseInputWrapper
from auto_disc.utils.config_parameters import IntegerConfigParameter

from leaf.Leaf import Leaf
from leaf.locators.locators import BlobLocator
import os
import neat
from copy import deepcopy


class CPPNParameterMap(Leaf):
    """ Base class to map the parameters sent by the explorer to the system's input space
    """

    def __init__(self, premap_key: str = "genome",
                 postmap_key: str = "init_state",
                 postmap_shape: Tuple[int, int] = (8, 8),
                 n_passes: int = 2,
                 config_path: str = "./config.cfg") -> None:
        super().__init__()
        self.locator = BlobLocator()
        self.premap_key = premap_key
        self.postmap_key = postmap_key
        self.postmap_shape = postmap_shape
        self.n_passes = n_passes

        # set global configuration parameters for NEAT
        self.neat_config = neat.Config(
            pytorchneat.selfconnectiongenome.SelfConnectionGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )

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
        intermed_dict[self.premap_key] = self._sample_genome()

        # generate init state output tensor from CPPN
        cppn_genome = intermed_dict[self.premap_key]
        cppn_net_output = self._generate_init_state(
            cppn_genome,
            shape=(self.postmap_shape[0], self.postmap_shape[1])
        )

        # finish transformation of input dictionary
        intermed_dict[self.postmap_key] = cppn_net_output
        del intermed_dict['genome']

        return intermed_dict

    def sample(self, data_shape: Tuple[int, int]) -> torch.Tensor:
        genome = self._sample_genome()
        # generates init_state from genome
        init_state = self._generate_init_state(genome, shape=data_shape)
        return init_state

    def _sample_genome(self) -> Any:
        genome = self.neat_config.genome_type(0)
        # randomly initializes the genome
        genome.configure_new(self.neat_config.genome_config)
        return genome

    def _generate_init_state(self, cppn_genome, shape: Tuple[int, int]
                             ) -> torch.Tensor:
        initialization_cppn = pytorchneat.rnn.RecurrentNetwork.create(
            cppn_genome, self.neat_config)

        # configure output size
        cppn_output_height = int(shape[0])
        cppn_output_width = int(shape[0])

        cppn_input = pytorchneat.utils.create_image_cppn_input(
            (cppn_output_height, cppn_output_width),
            is_distance_to_center=True,
            is_bias=True
        )
        cppn_output = initialization_cppn.activate(
            cppn_input, self.n_passes)
        cppn_net_output = (1.0 - cppn_output.abs()).squeeze()
        return cppn_net_output
