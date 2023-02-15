import torch
from leaf.leaf import Leaf
from leaf.locators import FileLocator
from typing import Dict
from auto_disc.newarch.wrappers.SaveWrapper import SaveWrapper
from auto_disc.newarch.wrappers.BoxProjector import BoxProjector
from copy import deepcopy


class MeanBehaviorMap(Leaf):
    """
    A simple `BehaviorMap` which merely extracts the mean.
    """

    def __init__(self, premap_key: str = "output") -> None:
        super().__init__()
        self.locator = FileLocator()
        self.premap_key = premap_key
        self.history_saver = SaveWrapper()
        self.projector = BoxProjector(premap_key=premap_key)

    def map(self, input: Dict) -> Dict:
        # TODO: does not handle batches
        intermed_dict = deepcopy(input)

        tensor = intermed_dict[self.premap_key].detach().clone()
        # unsqueeze to ensure tensor rank is not 0
        mean = torch.mean(tensor, dim=0).unsqueeze(-1)
        intermed_dict[self.premap_key] = mean

        projected_dict = self.projector.map(intermed_dict)
        behavior_dict = self.history_saver.map(projected_dict)

        return behavior_dict

    def sample(self):
        return self.projector.sample()

    def get_tensor_history(self):
        tensor_history = \
            self.history_saver.buffer[0][self.premap_key].unsqueeze(0)
        for dict in self.history_saver.buffer[1:]:
            tensor_history = torch.cat(
                (tensor_history, dict[self.premap_key].unsqueeze(0)),
                dim=0)
        return tensor_history
