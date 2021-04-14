from auto_disc.utils.config_parameters import DecimalConfigParameter

class IntegerConfigParameter(DecimalConfigParameter):
    '''
    Decorator to add an integer config parameter to a class.
    Uses a 'min' and 'max' to bound the authorized values.
    If a bound is set to 'None', this means no bound.
    If a decimal value is passed, it will be rounded.
    '''
    def __init__(self, name, default, min=None, max=None):
        super().__init__(name, 
                        round(default), 
                        min=round(min) if min else min, 
                        max=round(max) if max else max)

    def check_value_to_set(self, value):
        return super().check_value_to_set(round(value))

    def __call__(self, original_class):
        new_class = super().__call__(original_class)
        new_class.CONFIG_DEFINITION[self._name]['type'] = 'INTEGER'

        return new_class