import io
from leaf.Leaf import Leaf
from leaf.locators.locators import BlobLocator
from auto_disc.systems.System import System
from auto_disc_legacy.utils.config_parameters import DecimalConfigParameter, IntegerConfigParameter
from copy import deepcopy
from typing import Dict, Tuple
import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


@DecimalConfigParameter(name="sequence_max", default=100.)
@IntegerConfigParameter(name="sequence_density", default=100)
class ExponentialMixture(System):
    def __init__(self):
        super().__init__()
        # this module is stateless
        self.locator = BlobLocator()

    def map(self, input: Dict) -> Dict:
        intermed_dict = deepcopy(input)

        sequence_max = self.config["sequence_max"]
        sequence_density = self.config["sequence_density"]
        param_tensor = self._process_dict(input)

        _, y_tensor = \
            self._tensor_map(param_tensor, sequence_max, sequence_density)

        intermed_dict["output"] = y_tensor

        return intermed_dict

    def render(self, input_dict: Dict) -> bytes:
        """
        Renders an image given a dict with the `output` key and relevant config
        """
        sequence_max = self.config["sequence_max"]
        sequence_density = self.config["sequence_density"]

        x_tensor = \
            torch.linspace(start=0., end=sequence_max, steps=sequence_density)
        y_tensor = input_dict["output"]

        output_binary = io.BytesIO()
        plt.plot(x_tensor, y_tensor)
        plt.savefig(output_binary)
        plt.clf()

        return output_binary.getvalue()

    def _process_dict(self,
                      input_dict: Dict) -> torch.Tensor:
        # extract param tensor
        param_tensor = input_dict["params"]
        assert len(param_tensor.size()) == 1

        return param_tensor

    def _tensor_map(self, param_tensor: torch.Tensor,
                    sequence_max: float,
                    sequence_density: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_tensor = torch.linspace(
            start=0., end=sequence_max, steps=sequence_density)

        mixture_tensor = torch.exp(torch.outer(param_tensor, -1*x_tensor))

        y_tensor = torch.sum(mixture_tensor, dim=0)

        return x_tensor, y_tensor
