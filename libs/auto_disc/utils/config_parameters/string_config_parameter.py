from auto_disc.utils.config_parameters import BaseConfigParameter

class StringConfigParameter(BaseConfigParameter):
    '''
    Decorator to add a string config parameter to a class.
    Uses a list of possible values to choose among.
    '''
    def __init__(self, name, default="", possible_values=None):
        self._possible_values = possible_values
        if possible_values is not None and default not in possible_values:
            raise Exception("Default value not in possible values.")

        super().__init__(name, default)

    def check_value_to_set(self, value):
        assert isinstance(value, str), 'Passed value is not a string'
        if self._possible_values is not None:
            if value in self._possible_values:
                return True
            else:
                raise Exception('Chosen value ({0}) does not belong to the authorized list ({1}).'.format(value, self._possible_values))

        return True

    def __call__(self, original_class):
        new_class = super().__call__(original_class)
        new_class.CONFIG_DEFINITION[self._name]['type'] = 'STRING'
        new_class.CONFIG_DEFINITION[self._name]['possible_values'] = self._possible_values

        return new_class