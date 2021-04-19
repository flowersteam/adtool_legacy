import json
import math

class ConfigParameterBinding():
    '''
    Allows to bind some properties of a space to the value of a config parameter that belongs to the config of the instance.
    '''
    def __init__(self, parameter_name):
        self.parameter_name = parameter_name
        self._operations = []

    def _apply_operation(self, val1, val2, operator):
        if operator == '+':
            return val1 + val2
        elif operator == '-':
            return val1 - val2
        elif operator == '*':
            return val1 * val2
        elif operator == '/':
            return val1 / val2

    def __get__(self, obj, objtype=None):
        value = obj.config[self.parameter_name]
        for operation in self._operations:
            other = operation[0]
            operator = operation[1]
            if isinstance(other, ConfigParameterBinding):
                other = other.__get__(obj)
            value = self._apply_operation(value, other, operator)
        
        return value

    def __add__(self, other):
        self._operations.append((other, '+'))
        return self

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        self._operations.append((other, '-'))
        return self

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        self._operations.append((other, '*'))
        return self
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        self._operations.append((other, '/'))
        return self

    def __rdiv__(self, other):
        return self.__div__(other)

    def to_json(self):
        binding = 'binding.' + self.parameter_name
        for operation in self._operations:
            other = operation[0]
            operator = operation[1]
            if isinstance(other, ConfigParameterBinding):
                other = other.to_json()
            binding = "({0}, {1}, {2})".format(binding, other, operator)                
        return binding
