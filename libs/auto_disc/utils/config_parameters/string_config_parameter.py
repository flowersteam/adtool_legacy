from auto_disc.utils.config_parameters import BaseConfigParameter

class StringConfigParameter(BaseConfigParameter):
    '''
    Decorator to add a string config parameter to a class.
    Uses a list of possible values to choose among.
    '''
    def __init__(self, name, possible_values, default):
        self._possible_values = possible_values
        super().__init__(name, default)

    def check_value_to_set(self, value):
        if value in self._possible_values:
            return True
        else:
            raise Exception('Chosen value ({0}) does not belong to the authorized list ({1}).'.format(value, self._possible_values))

    def __call__(self, original_class):
        new_class = super().__call__(original_class)
        new_class.CONFIG_DEFINITION[self._name]['type'] = 'STRING'
        new_class.CONFIG_DEFINITION[self._name]['possible_values'] = self._possible_values

        return new_class