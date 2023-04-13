from auto_disc.newarch.systems.Lenia import Lenia
from auto_disc.newarch.wrappers.CPPNWrapper import CPPNWrapper
from leaf.Leaf import Leaf
from leaf.locators.locators import BlobLocator
from auto_disc.utils.config_parameters import (StringConfigParameter,
                                               IntegerConfigParameter)
from typing import Dict, Union
from copy import deepcopy
from dataclasses import dataclass, asdict
import torch


@StringConfigParameter(name="version", possible_values=["pytorch_fft", "pytorch_conv2d"], default="pytorch_fft")
@IntegerConfigParameter(name="SX", default=256, min=1)
@IntegerConfigParameter(name="SY", default=256, min=1)
@IntegerConfigParameter(name="final_step", default=200, min=1, max=1000)
@IntegerConfigParameter(name="scale_init_state", default=1, min=1)
@IntegerConfigParameter(name="cppn_n_passes", default=2, min=1)
class LeniaCPPN(Leaf):
    CONFIG_DEFINITION = {}

    def __init__(self):
        super().__init__()
        self.locator = BlobLocator()

        self.lenia = Lenia(version=self.config["version"],
                           SX=self.config["SX"],
                           SY=self.config["SY"],
                           final_step=self.config["final_step"],
                           scale_init_state=self.config["scale_init_state"])

        self.cppn = CPPNWrapper(postmap_shape=(self.lenia.config["SY"],
                                               self.lenia.config["SX"]),
                                n_passes=self.config["cppn_n_passes"]
                                )

    def map(self, input: Dict) -> Dict:
        intermed_dict = deepcopy(input)

        # turns genome into init_state
        # as CPPNWrapper is a wrapper, it operates on the lowest level
        intermed_dict["params"] = self.cppn.map(intermed_dict["params"])

        # pass params to Lenia
        intermed_dict = self.lenia.map(intermed_dict)

        return intermed_dict
