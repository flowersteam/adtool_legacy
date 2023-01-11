from copy import deepcopy
from typing import Dict, List
from leaf.leaf import Leaf, Locator, LeafUID
from leafutils.leafstructs.linear import LinearStorage, Stepper
from uuid import uuid1
import sqlite3


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

        # process key wrapping
        if len(wrapped_keys) != len(posttransform_keys):
            raise ValueError(
                "wrapped_keys and transformed_keys must be same length.")

        self.wrapped_keys = wrapped_keys
        self.posttransform_keys = posttransform_keys

        # process saving spec
        if len(inputs_to_save) > 0 and len(outputs_to_save) > 0:
            raise ValueError(
                '''
                Saving both inputs and outputs of this wrapper is not supported.
                Compose multiple wrappers for this functionality.
                '''
            )
        elif len(inputs_to_save) == 0 and len(outputs_to_save) == 0:
            # save inputs by default
            self.inputs_to_save = wrapped_keys
            self.outputs_to_save = outputs_to_save
        else:
            self.inputs_to_save = inputs_to_save
            self.outputs_to_save = outputs_to_save

        self.input_buffer: list = []
        self.output_buffer: list = []

    def map(self, input: Dict) -> Dict:
        """
        WARN: This wrappers .map() is stateful.

        Transforms the input dictionary with the provided relabelling of keys,
        saving inputs and outputs to an instance variable.
        """
        # must do because dicts are mutable types
        intermed_dict = deepcopy(input)

        if len(self.inputs_to_save) > 0:
            saved_input = {}
            for key in self.inputs_to_save:
                saved_input[key] = intermed_dict[key]
            self.input_buffer.append(saved_input)

        output = self._transform_keys(intermed_dict)

        if len(self.outputs_to_save) > 0:
            saved_output = {}
            for key in self.outputs_to_save:
                saved_output[key] = output[key]
            self.output_buffer.append(saved_output)

        return output

    def serialize(self) -> bytes:
        """
        Save desired buffer through a Stepper module,
        as required by LinearStorage
        """
        stepper = Stepper()
        if len(self.inputs_to_save) > 0:
            stepper.buffer = self.input_buffer
        elif len(self.outputs_to_save) > 0:
            stepper.buffer = self.output_buffer
        return stepper.serialize()

    def save_leaf(self, resource_uri: str, leaf_uid: int = -1) -> 'LeafUID':
        uid = super().save_leaf(resource_uri, leaf_uid)

        # clear cache
        self.input_buffer = []
        self.output_buffer = []
        return uid

    @classmethod
    def create_locator(cls, resource_uri: str = "", leaf_uid: int = -1
                       ) -> 'Locator':
        return LinearStorage(resource_uri, leaf_uid)

    def _transform_keys(self, old_dict: Dict) -> Dict:

        # initialize empty dict so that key-values will not overwrite
        new_dict = {}
        for (old_key, new_key) in \
                zip(self.wrapped_keys, self.posttransform_keys):
            new_dict[new_key] = old_dict[old_key]
            del old_dict[old_key]
        new_dict.update(old_dict)

        return new_dict
