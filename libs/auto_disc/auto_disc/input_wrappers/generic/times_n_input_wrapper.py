from auto_disc.utils.spaces import DictSpace, BoxSpace
from auto_disc.utils.config_parameters import IntegerConfigParameter
from leaf.leaf import Leaf
from copy import deepcopy
import numpy as np
from addict import Dict

from leaf.tests.test_leaf import DummyLocator
from leaf.leaf import Locator


@IntegerConfigParameter(name="n", default=1)
class TimesNInputWrapper(Leaf):
    CONFIG_DEFINITION = {}

    input_space = DictSpace(
        input_parameter=BoxSpace(low=-np.inf, high=np.inf, shape=())
    )

    def __init__(self, wrapped_key: str) -> None:
        super().__init__()
        self.input_space = deepcopy(self.input_space)
        # self.input_space.initialize(self)
        self._wrapped_key = wrapped_key
        self._initial_input_space_keys = [key for key in self.input_space]

    def map(self, input: Dict) -> Dict:
        # must do because dicts are mutable types
        output = deepcopy(input)

        output[self._wrapped_key] = \
            output[self._wrapped_key] * self.config["n"]

        return output
