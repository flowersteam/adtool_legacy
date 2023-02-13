from copy import deepcopy
from typing import Dict, List
from leaf.leaf import Leaf, Locator, LeafUID
from leafutils.leafstructs.linear import LinearLocator, Stepper


class TransformWrapper(Leaf):
    """
    Wrapper which does basic processing of input
    Usage example:
        ```
            input = {"in" : 1}
            wrapper = TransformWrapper(wrapped_keys = ["in"], 
                                  posttransform_keys = ["out"])
            output = wrapper.map(input)
            assert output["out"] == 1
        ```
    """

    def __init__(self,
                 wrapped_keys: List[str] = [],
                 posttransform_keys: List[str] = [],
                 ) -> None:
        super().__init__()

        # process key wrapping
        if len(wrapped_keys) != len(posttransform_keys):
            raise ValueError(
                "wrapped_keys and transformed_keys must be same length.")
        else:
            pass

        self.wrapped_keys = wrapped_keys
        self.posttransform_keys = posttransform_keys

    def map(self, input: Dict) -> Dict:
        """
        Transforms the input dictionary with the provided relabelling of keys.
        """
        # must do because dicts are mutable types
        intermed_dict = deepcopy(input)

        output = self._transform_keys(intermed_dict)

        return output

    def _transform_keys(self, old_dict: Dict) -> Dict:
        # initialize empty dict so that key-values will not overwrite
        new_dict = {}
        for (old_key, new_key) in \
                zip(self.wrapped_keys, self.posttransform_keys):

            # allows making conditional transformers that ignore input
            # with no appropriately matching keys
            if old_dict.get(old_key, None) is not None:
                new_dict[new_key] = old_dict[old_key]
                del old_dict[old_key]

        new_dict.update(old_dict)

        return new_dict
