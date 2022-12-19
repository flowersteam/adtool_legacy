from typing import List
from auto_disc.utils.config_parameters import BaseConfigParameter


class StringConfigParameter(BaseConfigParameter):
    """
    Decorator to add a dict config parameter to a class.
    Uses a list of possible values to choose among.
    #### Usage:
    ```
    @StringConfigParameter(name="string", default="a", possible_values=["a", "b", "c"])
    class A():
        CONFIG_DEFINITION = {}
        def __init__(self):
            pass
        ...
    ```
    """

    def __init__(self, name: str, default: str = "", possible_values: List[str] = None) -> None:
        """
            Init a str config parameter. Define the list of all possible values.

            #### Args:
            - name: name of config parameter
            - default: default value of the config parameter
            - possible_values: List of all possible string
        """
        self._possible_values = possible_values
        if possible_values is not None and default not in possible_values:
            raise Exception("Default value not in possible values.")

        super().__init__(name, default)

    def check_value_to_set(self, value: str) -> bool:
        """
            Check if the value is one of the possible values

            #### Args:
            - value: current value of the config parameter

            #### Returns:
            - bool: The return value is True if the value is in possible values
        """
        assert isinstance(value, str), 'Passed value is not a string'
        if self._possible_values is not None:
            if value in self._possible_values:
                return True
            else:
                raise Exception('Chosen value ({0}) does not belong to the authorized list ({1}).'.format(
                    value, self._possible_values))

        return True

    def __call__(self, original_class: type) -> type:
        new_class = super().__call__(original_class)
        new_class.CONFIG_DEFINITION[self._name]['type'] = 'STRING'
        new_class.CONFIG_DEFINITION[self._name]['possible_values'] = self._possible_values

        return new_class
