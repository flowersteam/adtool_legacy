#!/usr/bin/env python3
from copy import deepcopy
from typing import Dict

import torch
from auto_disc.auto_disc.maps.Map import Map
from auto_disc.auto_disc.wrappers.BoxProjector import BoxProjector
from auto_disc.utils.leaf.locators.locators import BlobLocator

torch.set_default_dtype(torch.float32)


class IdentityBehaviorMap(Map):
    CONFIG_DEFINITION = {}

    def __init__(
        self,
        premap_key: str = "output",
        postmap_key: str = "output",
    ) -> None:
        super().__init__()
        self.premap_key = premap_key
        self.postmap_key = postmap_key
        self.locator = BlobLocator()

        self.projector = BoxProjector(premap_key=self.postmap_key)

    def map(self, input: Dict, override_existing: bool = True) -> Dict:
        input["raw_output"] = input["output"]
        return input

    def sample(self) -> torch.Tensor:
        return self.projector.sample()