from auto_disc.utils.config_parameters import DecimalConfigParameter


class IntegerConfigParameter(DecimalConfigParameter):
    """
    Decorator to add a dict config parameter to a class.
    Uses a 'min' and 'max' to bound the authorized values.
    If a bound is set to 'None', this means no bound.
    If a decimal value is passed, it will be rounded.
    #### Usage:
    ```
    @IntegerConfigParameter(name="integer", default=10, min=0, max=100)
    class A():
        CONFIG_DEFINITION = {}
        def __init__(self):
            pass
        ...
    ```
    """

    def __init__(self, name: str, default: int, min: int = None, max: int = None) -> None:
        """
            Init a int config parameter. Define the bounds of the value

            #### Args:
            - name: name of config parameter
            - default: default value of the config parameter
            - min: the lower bound
            - max: the upper limit
        """
        super().__init__(name,
                         round(default),
                         min=round(min) if min else min,
                         max=round(max) if max else max)

    def check_value_to_set(self, value: int) -> bool:
        """
            Check if the value is between the two bounds

            #### Args:
            - value: current value of the config parameter

            #### Returns:
            - bool: The return value is True if the value is between the two bounds else an exception is thrown 
        """
        return super().check_value_to_set(round(value))

    def __call__(self, original_class: type) -> type:
        new_class = super().__call__(original_class)
        new_class.CONFIG_DEFINITION[self._name]['type'] = 'INTEGER'

        return new_class
