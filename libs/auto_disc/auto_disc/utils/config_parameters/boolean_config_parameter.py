from auto_disc.utils.config_parameters import BaseConfigParameter

class BooleanConfigParameter(BaseConfigParameter):
    '''
    Decorator to add a boolean config parameter to a class.
    '''
    def __init__(self, name, default):
        self._possible_values = [True, False]
        super().__init__(name, default)

    def check_value_to_set(self, value):
        if value in self._possible_values:
            return True
        else:
            raise Exception('Chosen value ({0}) is not a boolean.'.format(value))

    def __call__(self, original_class):
        new_class = super().__call__(original_class)
        new_class.CONFIG_DEFINITION[self._name]['type'] = 'BOOLEAN'
        new_class.CONFIG_DEFINITION[self._name]['possible_values'] = self._possible_values

        return new_class