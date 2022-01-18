from auto_disc.utils.callbacks import BaseCallback

from torch import Tensor

class BaseExpeDBCallback(BaseCallback):
    def __init__(self, base_url, **kwargs):
        super().__init__(**kwargs)
        self.base_url = base_url

    def _serialize_autodisc_space(self, space):
        serialized_space = {}
        for key in space:
            if isinstance(space[key], Tensor):
                serialized_space[key] = space[key].tolist()
            else:
                serialized_space[key] = space[key]
        return serialized_space