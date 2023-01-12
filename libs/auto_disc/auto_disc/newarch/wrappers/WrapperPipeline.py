from leaf.leaf import Leaf
from typing import List, Dict
from copy import deepcopy


class WrapperPipeline(Leaf):
    """
    Module for composing various wrappers during the input or output processing
    of the experiment.
    Usage example:
        ```
            input = {"in" : 1}
            a = alpha_wrapper()
            b = beta_wrapper()
            wrapper_list = [a, b]
            all_wrappers = WrapperPipeline(wrappers=wrapper_list, 
                            inputs_to_save=["in"], outputs_to_save=["out"])
            assert all_wrappers.map(input) == b.map(a.map(input))
        ```
    """

    def __init__(self, wrappers: List['Leaf'] = [],
                 inputs_to_save: List[str] = [],
                 outputs_to_save: List[str] = []):
        super().__init__()

        # self.inputs_to_save = inputs_to_save
        # self.outputs_to_save = outputs_to_save

        # set wrappers as submodules
        # NOTE: wrappers are therefore not individually
        # accessible by public methods
        for (i, el) in enumerate(wrappers):
            # do not need _set_attr_override as dicts are mutable
            self._modules[i] = el

    def map(self, input: Dict) -> Dict:
        working_input = deepcopy(input)
        pipeline_length = len(self._modules)
        for i in range(pipeline_length):
            intermed_output = self._modules[i].map(working_input)
            working_input = intermed_output
        return working_input
