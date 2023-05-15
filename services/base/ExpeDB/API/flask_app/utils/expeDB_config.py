import os
from os import environ as env
from typing import get_type_hints, Union


def _parse_bool(val: Union[str, bool]) -> bool:  # pylint: disable=E1136
    return val if type(val) == bool else val.lower() in ['true', 'yes', '1']


class ExpeDBConfigError(Exception):
    pass


class ExpeDBConfig():
    FLASK_HOST: str = '0.0.0.0'
    FLASK_PORT: int = 5001
    MONGODB_HOST: str = "localhost"
    MONGODB_PORT: str = "27017"
    MONGODB_USERNAME: str = "autodisc"
    MONGODB_PASSWORD: str = "password"

    def __init__(self):

        for field in self.__annotations__:
            default_value = getattr(self, field, None)
            if default_value is None and env.get(field) is None:
                raise ExpeDBConfigError(
                    'The {} field is required'.format(field))

            var_type = get_type_hints(ExpeDBConfig)[field]

            try:
                if var_type == bool:
                    value = _parse_bool(env.get(field, default_value))
                else:
                    value = var_type(env.get(field, default_value))

                self.__setattr__(field, value)
            except:
                raise ExpeDBConfigError('Unable to cast value of "{}" to type "{}" for "{}" field'.format(
                    env[field],
                    var_type,
                    field
                )
                )
