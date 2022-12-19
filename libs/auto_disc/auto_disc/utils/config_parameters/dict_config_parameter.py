from typing import Dict
from auto_disc.utils.config_parameters import BaseConfigParameter


class DictConfigParameter(BaseConfigParameter):
    """
    Decorator to add a dict config parameter to a class.
    #### Usage:
    ```
    @DictConfigParameter(name="dict", default={})
    class A():
        CONFIG_DEFINITION = {}
        def __init__(self):
            pass
        ...
    ```
    """

    def __init__(self, name: str, default: Dict = {}) -> None:
        """
            Init a dict config parameter.

            #### Args:
            - name: name of config parameter
            - default: default value of the config parameter
        """
        super().__init__(name, default)

    def check_value_to_set(self, value: Dict) -> bool:
        """
            Check if the value is indeed a dict

            #### Args:
            - value: current value of the config parameter

            #### Returns:
            - bool: The return value is True if the value is a dict
        """
        assert isinstance(value, dict), 'Passed value is not a dict'
        return True

    def __call__(self, original_class):
        new_class = super().__call__(original_class)
        new_class.CONFIG_DEFINITION[self._name]['type'] = 'DICT'

        return new_class
