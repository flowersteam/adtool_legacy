from leaf.leaf import Leaf
from leaf.locators import FileLocator
from copy import deepcopy
from typing import Dict, Tuple
import torch
import matplotlib.pyplot as plt
import io


class ExponentialMixture(Leaf):
    def __init__(self):
        super().__init__()
        # this module is stateless
        self.locator = FileLocator()

    def map(self, input: Dict) -> Dict:
        intermed_dict = deepcopy(input)

        param_tensor, sequence_max, sequence_density = self._process_dict(
            input)

        x_tensor, y_tensor = \
            self._tensor_map(param_tensor, sequence_max, sequence_density)

        del intermed_dict["params"]
        intermed_dict["output"] = y_tensor

        return intermed_dict

    def render(self, input_dict: Dict) -> bytes:
        """
        Renders an image given a dict with the `output` key and relevant config
        """
        sequence_max, sequence_density = \
            self._extract_config_metadata(input_dict)
        x_tensor = \
            torch.linspace(start=0., end=sequence_max, steps=sequence_density)
        y_tensor = input_dict["output"]

        output_binary = io.BytesIO()
        plt.plot(x_tensor, y_tensor)
        plt.savefig(output_binary)

        return output_binary.getvalue()

    def _extract_config_metadata(self, input_dict: Dict) -> Tuple[float, int]:
        # requires specifically structured input
        metadata_dict = input_dict["config"]
        assert metadata_dict["sequence_max"] > 1
        assert metadata_dict["sequence_density"] > 1
        sequence_max = metadata_dict["sequence_max"]
        sequence_density = metadata_dict["sequence_density"]

        return sequence_max, sequence_density

    def _process_dict(self,
                      input_dict: Dict) -> Tuple[torch.Tensor, float, int]:

        sequence_max, sequence_density = self._extract_config_metadata(
            input_dict)

        # extract param tensor
        param_tensor = input_dict["params"]
        assert len(param_tensor.size()) == 1

        return param_tensor, sequence_max, sequence_density

    def _tensor_map(self, param_tensor: torch.Tensor,
                    sequence_max: float,
                    sequence_density: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_tensor = torch.linspace(
            start=0., end=sequence_max, steps=sequence_density)

        mixture_tensor = torch.exp(torch.outer(param_tensor, -1*x_tensor))

        y_tensor = torch.sum(mixture_tensor, dim=0)

        return x_tensor, y_tensor
