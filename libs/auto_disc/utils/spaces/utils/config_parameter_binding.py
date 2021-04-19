import json

class ConfigParameterBinding():
    '''
    Allows to bind some properties of a space to the value of a config parameter that belongs to the config of the instance.
    '''
    def __init__(self, parameter_name):
        self.parameter_name = parameter_name
        self._other = []

    def __get__(self, obj, objtype=None):
        value = obj.config[self.parameter_name]
        for other in self._other:
            if isinstance(other, ConfigParameterBinding):
                other_value = other.__get__(obj)
            else:
                other_value = other
            value *= other_value
        
        return value

    def __mul__(self, other):
        self._other.append(other)
        return self

    def __rmul__(self, other):
        self._other.append(other)
        return self

    def to_json(self):
        binding = self.parameter_name
        for other in self._other:
            if isinstance(other, ConfigParameterBinding):
                binding = binding + '*' + other.parameter_name
            else:
                binding = binding + '*' + other
                
        return {'binding': binding}
