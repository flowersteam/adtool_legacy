from copy import deepcopy
from typing import Dict, List
from leaf.leaf import Leaf


class SaveWrapper(Leaf):
    """
    Wrapper which does basic processing and
    saving of captured history
    Usage example:
        ```
            input = {"in" : 1}
            # default setting saves all specified keys
            wrapper = SaveWrapper(wrapped_keys = ["in"], posttransform_keys = ["out"])
            output = wrapper.map(input)
            assert output["out"] == 1
        ```
    """

    def __init__(self,
                 wrapped_keys: List[str] = [],
                 posttransform_keys: List[str] = [],
                 inputs_to_save: List[str] = [],
                 outputs_to_save: List[str] = []
                 ) -> None:
        super().__init__()

        if len(wrapped_keys) != len(posttransform_keys):
            raise ValueError(
                "wrapped_keys and transformed_keys must be same length.")

        self.wrapped_keys = wrapped_keys
        self.posttransform_keys = posttransform_keys
        self.inputs_to_save = inputs_to_save
        self.outputs_to_save = outputs_to_save

        # default save values
        if len(self.inputs_to_save) == 0:
            self.inputs_to_save = self.wrapped_keys
        if len(self.outputs_to_save) == 0:
            self.outputs_to_save = self.posttransform_keys

    def map(self, input: Dict) -> Dict:
        # must do because dicts are mutable types
        intermed_dict = deepcopy(input)
        output = self._transform_keys(intermed_dict)

        return output

    def _transform_keys(self, old_dict: Dict) -> Dict:

        # initialize empty dict so that key-values will not overwrite
        new_dict = {}
        for (old_key, new_key) in \
                zip(self.wrapped_keys, self.posttransform_keys):
            new_dict[new_key] = old_dict[old_key]
            del old_dict[old_key]
        new_dict.update(old_dict)

        return new_dict
