from auto_disc.utils.config_parameters import BaseConfigParameter

class DecimalConfigParameter(BaseConfigParameter):
    '''
    Decorator to add a decimal config parameter to a class.
    Uses a 'min' and 'max' to bound the authorized values.
    If a bound is set to 'None', this means no bound.
    '''
    def __init__(self, name, default, min=None, max=None):
        self._min = min
        self._max = max
        super().__init__(name, default)

    def check_value_to_set(self, value):
        if self._min is not None and value < self._min:
            raise Exception('Chosen value ({0}) is lower than the minimum authorized ({1}).'.format(value, self._min))

        if self._max is not None and value > self._max:
            raise Exception('Chosen value ({0}) is lower than the maximum authorized ({1}).'.format(value, self._max))

        return True

    def __call__(self, original_class):
        new_class = super().__call__(original_class)
        new_class.CONFIG_DEFINITION[self._name]['type'] = 'DECIMAL'
        new_class.CONFIG_DEFINITION[self._name]['min'] = self._min
        new_class.CONFIG_DEFINITION[self._name]['max'] = self._max

        return new_class