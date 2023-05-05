from leaf.Leaf import LeafUID, Leaf
import os
import json
from datetime import datetime
from uuid import uuid1


class SaveLeaf:
    """
    Simple adapter which calls the save_leaf method from a callback
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __call__(self,
                 experiment_id: int,
                 seed: int,
                 module_to_save: Leaf,
                 resource_uri: str
                 ) -> None:

        uid = module_to_save.save_leaf(resource_uri=resource_uri)

        # set uid because can't return anything from callbacks
        module_to_save.uid = uid
        return
