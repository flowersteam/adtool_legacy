from copy import deepcopy
from typing import Dict, List
from leaf.leaf import Leaf, Locator, LeafUID
from leafutils.leafstructs.linear import LinearLocator, Stepper
from auto_disc.newarch.wrappers.TransformWrapper import TransformWrapper


class SaveWrapper(TransformWrapper):
    """
    Wrapper which does basic processing and
    saving of captured *input* history
    Usage example:
        ```
            input = {"in" : 1}
            # default setting saves all specified `wrapped_keys`
            wrapper = SaveWrapper(wrapped_keys = ["in"], 
                                  posttransform_keys = ["out"])
            output = wrapper.map(input)
            assert output["out"] == 1
        ```
    """

    def __init__(self,
                 wrapped_keys: List[str] = [],
                 posttransform_keys: List[str] = [],
                 inputs_to_save: List[str] = [],
                 ) -> None:
        super().__init__(wrapped_keys=wrapped_keys,
                         posttransform_keys=posttransform_keys)

        # resource_uri should be defined when SaveWrapper is bound to a
        # container, or manually initialized
        self.locator = LinearLocator("")

        # MUST define with the name `buffer` to obey LinearStorage interface
        self.buffer = []

        # save all inputs by default
        if len(inputs_to_save) == 0:
            self.inputs_to_save = wrapped_keys
        else:
            self.inputs_to_save = inputs_to_save

    def map(self, input: Dict) -> Dict:
        """
        WARN: This wrapper's .map() is stateful.

        Transforms the input dictionary with the provided relabelling of keys,
        saving inputs and passing outputs
        """
        # must do because dicts are mutable types
        intermed_dict = deepcopy(input)

        if len(self.inputs_to_save) > 0:
            self._store_saved_inputs_in_buffer(intermed_dict)
        else:
            self._store_all_inputs_in_buffer(intermed_dict)

        output = self._transform_keys(intermed_dict)

        return output

    def save_leaf(self, resource_uri: str, leaf_uid: int = -1) -> 'LeafUID':
        # leaf_uid is passed for specifying the parent node,
        # when passed to LinearLocator
        uid = super().save_leaf(resource_uri, leaf_uid)

        # clear cache
        self.buffer = []
        return uid

    def _store_saved_inputs_in_buffer(self, intermed_dict: Dict) -> None:
        saved_input = {}
        for key in self.inputs_to_save:
            saved_input[key] = intermed_dict[key]
        self.buffer.append(saved_input)
        return

    def _store_all_inputs_in_buffer(self, intermed_dict: Dict) -> None:
        saved_input = deepcopy(intermed_dict)
        self.buffer.append(saved_input)
        return

    def _transform_keys(self, old_dict: Dict) -> Dict:
        return super()._transform_keys(old_dict)