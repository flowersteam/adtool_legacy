from auto_disc.utils.config_parameters import BaseConfigParameter

class DictConfigParameter(BaseConfigParameter):
    '''
    Decorator to add a dict config parameter to a class.
    '''
    def __init__(self, name, default={}):
        super().__init__(name, default)

    def check_value_to_set(self, value):
        assert isinstance(value, dict), 'Passed value is not a dict'
        return True

    def __call__(self, original_class):
        new_class = super().__call__(original_class)
        new_class.CONFIG_DEFINITION[self._name]['type'] = 'DICT'

        return new_class