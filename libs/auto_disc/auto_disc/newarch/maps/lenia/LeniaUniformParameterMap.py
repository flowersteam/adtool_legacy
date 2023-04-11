from auto_disc.newarch.maps import UniformParameterMap
from leaf.locators.locators import BlobLocator
from leaf.Leaf import Leaf
import torch
from typing import Dict, Tuple

# unfinished


class LeniaUniformParameterMap(Leaf):
    def __init__(self, premap_key: str = "params",):
        super().__init__()
        self.locator = BlobLocator()

    def map(self, input: Dict) -> Dict:
        return super().map(input)

    def sample(self, data_shape: Tuple) -> torch.Tensor:
        return super().sample(data_shape)
