from enum import Enum

class ParameterType(object):
    def __init__(self, name):
        self.name = name

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
        self.dims = dims

class Space(ParameterType):
    def __init__(self):
        super().__init__("Space")

class Object(ParameterType):
    def __init__(self):
        super().__init__("Object")

class ParameterTypesEnum(Enum):
    STRING = String
    INTEGER = Integer
    FLOAT = Float
    ARRAY = Array
    SPACE = Space
    OBJECT = Object

    @classmethod
    def get(cls, name, **kwargs):
        return ParameterTypesEnum[name].value(**kwargs)