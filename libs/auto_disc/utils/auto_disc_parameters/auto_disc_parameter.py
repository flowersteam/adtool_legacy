from addict import Dict

class AutoDiscParameter():
    '''
    Hyperparameter a class exposes. It is used to configure an instance of this class.
    '''
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


class ConfigParameterBinding():
    '''
    Allows to bind some properties of an AutoDiscParameter to the value of another parameter that belongs to the config of the instance.
    '''
    def __init__(self, parameter_name):
        self.parameter_name = parameter_name
        self.key = None
    def __getitem__(self, key):
        self.key = key
        return self

def recursive_binding_search(config_dict, parameter_name, parameter_key=None):
    for key, value in config_dict.items():
        if isinstance(value, dict):
            return recursive_binding_search(value, parameter_name)
        else:
            if key == parameter_name:
                if parameter_key is not None:
                    return value[parameter_key]
                else:
                    return value
    raise Exception("Binding of parameter {} not found !".format(parameter_name))


def get_default_values(instance, parameters_definition):
    from auto_disc.utils.auto_disc_parameters import AutoDiscSpaceDefinition
    default_params = Dict()
    for param in parameters_definition:
        param_dict = param.to_dict()
        default_value = param_dict['default']
        if param_dict['type'].name == "Space":
            if isinstance(default_value, AutoDiscSpaceDefinition):
                # Dims
                new_dims = []
                for current_val in default_value.dims:
                    if isinstance(current_val, ConfigParameterBinding):
                       new_dims.append(recursive_binding_search(instance.config, current_val.parameter_name, current_val.key))
                    else:
                        new_dims.append(current_val)
                default_value.dims = new_dims
                # Bounds
                new_bounds = []
                for current_val in default_value.bounds:
                    if isinstance(current_val, ConfigParameterBinding):
                       new_bounds.append(recursive_binding_search(instance.config, current_val.parameter_name))
                    else:
                        new_bounds.append(current_val)
                default_value.bounds = new_bounds
            else:
                raise TypeError("Parameter {} has type Space but default value is not an AutoDiscSpaceDefinition.".format(param_dict['name']))
            
        default_params[param_dict['name']] = default_value

    return default_params
    

