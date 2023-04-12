import torch
from leaf.Leaf import Leaf
from leaf.locators.locators import BlobLocator
from typing import Dict, Tuple
from auto_disc.newarch.wrappers.SaveWrapper import SaveWrapper
from auto_disc.newarch.wrappers.BoxProjector import BoxProjector
from copy import deepcopy


class UniformParameterMap(Leaf):
    """
    A simple `ParameterMap` which generates parameters according to a uniform
    distribution over a box.
    """

    def __init__(self,
                 premap_key: str = "params",
                 tensor_low: torch.Tensor = torch.tensor([0.]),
                 tensor_high: torch.Tensor = torch.tensor([0.]),
                 tensor_bound_low: torch.Tensor = torch.tensor([float('-inf')]),
                 tensor_bound_high: torch.Tensor = torch.tensor([float('inf')])):

        # TODO: put indication that tensor_low and high must be set
        super().__init__()
        self.locator = BlobLocator()
        self.premap_key = premap_key
        if tensor_low.size() != tensor_high.size():
            raise ValueError("tensor_low and tensor_high must be same shape.")
        if tensor_bound_low.size() != tensor_bound_high.size():
            raise ValueError(
                "tensor_bound_low and tensor_bound_high must be same shape.")
        self.postmap_shape = tensor_low.size()
        # self.history_saver = SaveWrapper()

        self.projector = BoxProjector(premap_key=premap_key,
                                      init_high=tensor_high,
                                      init_low=tensor_low,
                                      bound_lower=tensor_bound_low,
                                      bound_upper=tensor_bound_high
                                      )

    def map(self, input: Dict) -> Dict:
        """
        map() takes an input dict of metadata and adds the
        `params` key with a sample if it does not exist
        """
        intermed_dict = deepcopy(input)

        # always overrides "params" with new sample
        intermed_dict[self.premap_key] = self.sample(self.postmap_shape)
        param_dict = self.projector.map(intermed_dict)

        # params_dict = self.history_saver.map(intermed_dict)

        return param_dict

    def sample(self, data_shape: Tuple) -> torch.Tensor:
        dimensions_to_keep = data_shape[0]
        sample = self.projector.sample()
        return sample[:dimensions_to_keep]

    # def get_tensor_history(self) -> torch.Tensor:
    #     tensor_history = \
    #         self.history_saver.buffer[0][self.premap_key].unsqueeze(0)
    #     for dict in self.history_saver.buffer[1:]:
    #         tensor_history = torch.cat(
    #             (tensor_history, dict[self.premap_key].unsqueeze(0)), dim=0)
    #     return tensor_history
