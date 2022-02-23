from auto_disc.input_wrappers import BaseInputWrapper
from auto_disc.utils.spaces import DictSpace, BoxSpace
from auto_disc.utils.config_parameters import IntegerConfigParameter
import numpy as np

@IntegerConfigParameter(name="n", default=1)
class TimesNInputWrapper(BaseInputWrapper):
    CONFIG_DEFINITION = {}
    
    input_space = DictSpace(
        input_parameter = BoxSpace(low=-np.inf, high=np.inf, shape=())
    )

    def __init__(self, wrapped_output_space_key, **kwargs):
        super().__init__(wrapped_output_space_key, **kwargs)
        assert len(self.input_space) == 1
        if not isinstance(wrapped_output_space_key, str):
            raise TypeError("wrapped_output_space_key must be a single string indicating the key of the space to wrap.")

        # Change key name to avoid issues with multiple same wrappers stacked
        new_key = 'Times{0}_{1}'.format(self.config['n'], wrapped_output_space_key)
        self.input_space[new_key] = self.input_space[self.initial_input_space_keys[0]]
        del self.input_space[self.initial_input_space_keys[0]]
        self.initial_input_space_keys = [new_key]

    def map(self, parameters, is_input_new_discovery, **kwargs):
        parameters[self.wrapped_output_space_key] = parameters[self.initial_input_space_keys[0]] * self.config['n']
        del parameters[self.initial_input_space_keys[0]]
        return parameters