import torch
from leaf.Leaf import Leaf
from leaf.locators.locators import BlobLocator
from typing import Dict, Tuple
from auto_disc.newarch.wrappers.SaveWrapper import SaveWrapper
from auto_disc.newarch.wrappers.BoxProjector import BoxProjector
from copy import deepcopy


class MeanBehaviorMap(Leaf):
    """
    A simple `BehaviorMap` which merely extracts the mean.
    """

    def __init__(self,
                 premap_key: str = "output",
                 postmap_key: str = "output",
                 input_shape: Tuple[int] = (1)) -> None:
        super().__init__()
        self.locator = BlobLocator()
        self.premap_key = premap_key
        self.input_shape = input_shape  # unused by the module itself here
        # self.history_saver = SaveWrapper()
        self.projector = BoxProjector(premap_key=premap_key)

    def map(self, input: Dict) -> Dict:
        # TODO: does not handle batches
        intermed_dict = deepcopy(input)

        # store raw output
        tensor = intermed_dict[self.premap_key].detach().clone()
        raw_output_key = "raw_" + self.premap_key
        intermed_dict[raw_output_key] = tensor

        # flatten to 1D
        tensor_flat = tensor.view(-1)

        # unsqueeze to ensure tensor rank is not 0
        mean = torch.mean(tensor_flat, dim=0).unsqueeze(-1)
        intermed_dict[self.postmap_key] = mean
        # remove original output item
        del intermed_dict[self.premap_key]

        behavior_dict = self.projector.map(intermed_dict)
        # behavior_dict = self.history_saver.map(projected_dict)

        return behavior_dict

    def sample(self):
        return self.projector.sample()

    # def get_tensor_history(self):
    #     tensor_history = \
    #         self.history_saver.buffer[0][self.premap_key].unsqueeze(0)
    #     for dict in self.history_saver.buffer[1:]:
    #         tensor_history = torch.cat(
    #             (tensor_history, dict[self.premap_key].unsqueeze(0)),
    #             dim=0)
    #     return tensor_history
