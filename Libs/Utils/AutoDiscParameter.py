from enum import Enum

class AutoDiscParameter():
    def __init__(self, name, type, values_range, default=None):
        self._name = name
        self._type = type
        self._values_range = values_range
        self._default = default

    def to_dict(self):
        return {
            "name": self._name,
            "type": self._type,
            "values_range": self._values_range,
            "default": self._default
        }

class ParameterType(object):
    def __init__(self, name):
        self._name = name

class Integer(ParameterType):
    def __init__(self):
        super().__init__("Integer")

class Float(ParameterType):
    def __init__(self):
        super().__init__("Float")

class String(ParameterType):
    def __init__(self):
        super().__init__("String")

class Array(ParameterType):
    def __init__(self, dims):
        super().__init__("Array")
        self._dims = dims

class ParameterTypesEnum(Enum):
    STRING = String
    INTEGER = Integer
    FLOAT = Float
    ARRAY = Array

    @classmethod
    def get(cls, name, **kwargs):
        return ParameterTypesEnum[name].value(**kwargs)

class ParameterBinding():
    def __init__(self, parameter_name):
        self._parameter_name = parameter_name


