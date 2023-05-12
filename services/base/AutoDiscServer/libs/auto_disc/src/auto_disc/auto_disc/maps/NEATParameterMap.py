from typing import Any, Dict, Tuple
import torch
from auto_disc.legacy.utils.spaces import DictSpace
from auto_disc.auto_disc.maps.Map import Map
from auto_disc.legacy.input_wrappers.generic.cppn.utils import CPPNGenomeSpace
from auto_disc.legacy.input_wrappers.generic.cppn import pytorchneat
from auto_disc.legacy.input_wrappers import BaseInputWrapper
from auto_disc.legacy.utils.config_parameters import IntegerConfigParameter

from auto_disc.utils.leaf.Leaf import Leaf
from auto_disc.utils.leaf.locators.locators import BlobLocator
import os
import neat
from copy import deepcopy


class NEATParameterMap(Map):
    """ 
    Base class to map the parameters sent by the explorer to the system's input space
    """

    def __init__(self, premap_key: str = "genome",
                 config_path: str = "./config.cfg") -> None:
        super().__init__()
        self.locator = BlobLocator()
        self.premap_key = premap_key

        # set global configuration parameters for NEAT
        self.neat_config = neat.Config(
            pytorchneat.selfconnectiongenome.SelfConnectionGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )

    def map(self, input: Dict, override_existing: bool = True) -> Dict:
        intermed_dict = deepcopy(input)

        # check if either "genome" is not set or if we want to override
        if ((override_existing and self.premap_key in intermed_dict)
                or (self.premap_key not in intermed_dict)):
            # overrides "genome" with new sample
            intermed_dict[self.premap_key] = self.sample()
        else:
            # passes "genome" through if it exists
            pass

        # also passes the neat_config needed to initialize the genome into
        # an initial state tensor
        intermed_dict["neat_config"] = self.neat_config

        return intermed_dict

    def sample(self) -> Any:
        """
        Samples a genome.
        """
        genome = self._sample_genome()
        return genome

    def mutate(self, dict: Dict) -> Dict:
        """
        Mutates the genome in the provided dict.
        """
        intermed_dict = deepcopy(dict)

        genome = intermed_dict[self.premap_key]
        genome.mutate(self.neat_config.genome_config)

        intermed_dict[self.premap_key] = genome

        return intermed_dict

    def _sample_genome(self) -> Any:
        genome = self.neat_config.genome_type(0)
        # randomly initializes the genome
        genome.configure_new(self.neat_config.genome_config)
        return genome
