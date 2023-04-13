from auto_disc.newarch.maps import UniformParameterMap, NEATParameterMap
from auto_disc.newarch.systems.Lenia import (LeniaParameters,
                                             LeniaDynamicalParameters)
from auto_disc.newarch.wrappers.mutators import add_gaussian_noise
from auto_disc.newarch.wrappers.CPPNWrapper import CPPNWrapper
from leaf.locators.locators import BlobLocator
from leaf.Leaf import Leaf
import torch
from typing import Dict, Tuple, Union
import dataclasses
from dataclasses import dataclass, asdict
from copy import deepcopy
from functools import partial


@dataclass
class LeniaHyperParameters:
    """ Holds parameters to initialize Lenia model."""
    tensor_low: torch.Tensor = LeniaDynamicalParameters().to_tensor()
    tensor_high: torch.Tensor = LeniaDynamicalParameters().to_tensor()
    tensor_bound_low: torch.Tensor = torch.tensor(
        [0., 1., 0., 0.001, 0., 0., 0., 0.])
    tensor_bound_high: torch.Tensor = torch.tensor(
        [20., 20., 1., 0.3, 1., 1., 1., 1.])
    init_state_dim: Tuple[int, int] = (10, 10)
    cppn_n_passes: int = 2


@dataclass
class LeniaParameterMapConfig:
    hp: LeniaHyperParameters = LeniaHyperParameters()
    neat_config_path: str = "./config.cfg"


class LeniaParameterMap(Leaf):
    """
    Due to the complexities of initializing Lenia parameters,
    it's easier to make this custom parameter map.
    """

    def __init__(self,
                 premap_key: str = "params",
                 SX: int = 10,
                 SY: int = 10,
                 config: LeniaParameterMapConfig = LeniaParameterMapConfig()
                 ):
        super().__init__()
        self.locator = BlobLocator()

        self.premap_key = premap_key

        self.uniform = UniformParameterMap(
            premap_key=f"tensor_{self.premap_key}",
            tensor_low=config.hp.tensor_low,
            tensor_high=config.hp.tensor_high,
            tensor_bound_low=config.hp.tensor_bound_low,
            tensor_bound_high=config.hp.tensor_bound_high
        )
        self.neat = NEATParameterMap(premap_key=f"genome_{self.premap_key}",
                                     config_path=config.neat_config_path)

        # multi-dimensional "ragged" Gaussian noise
        # based upon the tensor representation of LeniaDynamicalParameters
        self.uniform_mutator = partial(
            add_gaussian_noise,
            mean=torch.tensor([0.]),
            std=torch.tensor([0.5, 0.5, 0.1, 0.05, 0.1, 0.1, 0.1, 0.1])
        )

        self.SX = SX
        self.SY = SY
        self.cppn_n_passes = config.hp.cppn_n_passes

    def map(self, input: Dict, override_existing: bool = True) -> Dict:
        """ 
        Takes input dictionary and overrides the
        premap_key with generated parameters for Lenia.
        """
        intermed_dict = deepcopy(input)

        # check if either "params" is not set or if we want to override
        if ((override_existing and self.premap_key in intermed_dict)
                or (self.premap_key not in intermed_dict)):
            # overrides "params" with new sample
            intermed_dict[self.premap_key] = self.sample()
        else:
            # passes "params" through if it exists
            pass

        return intermed_dict

    def sample(self) -> Dict:
        """ 
        Samples from the parameter map to yield the parameters ready
        for processing by the model (Lenia), i.e., the genome has been
        used to generate the `init_state` of Lenia.
        """
        # sample dynamical parameters
        p_dyn_tensor = self.uniform.sample()

        # sample init_state from genome
        genome = self.neat.sample()
        neat_config = self.neat.neat_config
        init_state = self._cppn_map_genome(genome, neat_config)

        # convert to parameter objects
        dp = LeniaDynamicalParameters().from_tensor(p_dyn_tensor)
        p = LeniaParameters(dynamic_params=dp, init_state=init_state)

        return asdict(p)

    def mutate(self, parameter_dict: Dict) -> Dict:
        """ 
        Takes the dictionary of only parameters outputted by
        the explorer and mutates them. 
        """
        intermed_dict = deepcopy(parameter_dict)

        # mutate dynamic parameters
        dp = LeniaDynamicalParameters(**parameter_dict["dynamic_params"])
        dp_tensor = dp.to_tensor()
        mutated_dp_tensor = self.uniform_mutator(dp_tensor)

        # mutate CPPN genome
        genome = intermed_dict["genome"]
        genome.mutate(self.neat.neat_config.genome_config)

        # reassemble parameter_dict
        intermed_dict["genome"] = genome
        intermed_dict["dynamic_params"] = \
            asdict(
            LeniaDynamicalParameters().from_tensor(mutated_dp_tensor)
        )

        return intermed_dict

    def _cppn_map_genome(self, genome, neat_config) -> torch.Tensor:
        cppn_input = {}
        cppn_input["genome"] = genome
        cppn_input["neat_config"] = neat_config

        cppn_out = CPPNWrapper(postmap_shape=(self.SX, self.SY),
                               n_passes=self.cppn_n_passes).map(cppn_input)
        init_state = cppn_out["init_state"]
        return init_state
