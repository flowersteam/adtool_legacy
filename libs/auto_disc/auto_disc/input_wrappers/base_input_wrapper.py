from auto_disc import BaseAutoDiscModule
from auto_disc.utils.spaces import DictSpace
from copy import deepcopy
from typing import Dict


class BaseInputWrapper(BaseAutoDiscModule):
    """ 
    Base class to map the parameters sent by the explorer to the system's input space
    """

    input_space = DictSpace()

    def __init__(self, wrapped_output_space_key: str = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_space = deepcopy(self.input_space)
        self.input_space.initialize(self)
        self._wrapped_output_space_key = wrapped_output_space_key
        self._initial_input_space_keys = [key for key in self.input_space]

    def initialize(self, output_space: DictSpace) -> None:
        """
        Sets input and output space for the input wrapper based on the wrapped_output_space-key
        """
        self.output_space = output_space
        for key in iter(output_space):
            if key != self._wrapped_output_space_key:
                self.input_space[key] = output_space[key]

    def map(self, input: Dict, is_input_new_discovery: bool, **kwargs) -> Dict:
        """
            Map the input parameters (from the explorer) to the output parameters (sytem input)

            #### Args:
            - input: input parameters
            - is_input_new_discovery: indicates if it is a new discovery
        """
        raise NotImplementedError
