from addict import Dict
from auto_disc.utils.spaces import DictSpace, BoxSpace
from auto_disc.utils.config_parameters import IntegerConfigParameter
from auto_disc.base_autodisc_module import BaseAutoDiscModule
from leaf.leaf import Leaf
from copy import deepcopy
import numpy as np


@IntegerConfigParameter(name="n", default=1)
class TimesNInputWrapper(Leaf):
    CONFIG_DEFINITION = {}

    input_space = DictSpace(
        input_parameter=BoxSpace(low=-np.inf, high=np.inf, shape=())
    )

    def __init__(self, wrapped_output_space_key: str) -> None:
        self._wrapped_output_space_key = wrapped_output_space_key

    def map(self, input: Dict) -> Dict:
        output = deepcopy(input)
        output[self._wrapped_output_space_key] = output[self._wrapped_output_space_key] * self.config["n"]
        return output
