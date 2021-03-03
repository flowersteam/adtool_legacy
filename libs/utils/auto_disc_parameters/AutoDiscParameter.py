from libs.utils.AttrDict import AttrDict

class AutoDiscParameter():
    def __init__(self, name, type, values_range=None, default=None, modifiable=True):
        self._name = name
        self._type = type
        self._values_range = values_range
        self._default = default
        self._modifiable = modifiable

    def to_dict(self):
        return {
            "name": self._name,
            "type": self._type,
            "values_range": self._values_range,
            "default": self._default,
            "modifiable": 1 if self._modifiable else 0
        }


class ParameterBinding():
    def __init__(self, parameter_name):
        self._parameter_name = parameter_name

def get_default_values(parameters_definition):
    default_params = AttrDict()
    for param in parameters_definition:
        param_dict = param.to_dict()
        default_params[param_dict['name']] = param_dict['default']

    return default_params
    

