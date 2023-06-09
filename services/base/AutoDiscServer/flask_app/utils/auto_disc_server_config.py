import os
from os import environ as env
from typing import Union, get_type_hints


def _parse_bool(val: Union[str, bool]) -> bool:  # pylint: disable=E1136
    return val if type(val) == bool else val.lower() in ["true", "yes", "1"]


class AutoDiscServerConfigError(Exception):
    pass


class AutoDiscServerConfig:
    FLASK_HOST: str = "0.0.0.0"
    FLASK_PORT: int = 5002
    SSH_CONFIG_FILE: str = None
    EXPEDB_CALLER_HOST: str = "127.0.0.1"
    EXPEDB_CALLER_PORT: str = "5001"
    APPDB_CALLER_HOST: str = "127.0.0.1"
    APPDB_CALLER_PORT: str = "3000"

    def __init__(self) -> None:
        for field in self.__annotations__:
            default_value = getattr(self, field, None)
            if default_value is None and env.get(field) is None:
                raise AutoDiscServerConfigError(
                    "The {} field is required".format(field)
                )

            var_type = get_type_hints(AutoDiscServerConfig)[field]

            try:
                if var_type == bool:
                    value = _parse_bool(env.get(field, default_value))
                else:
                    value = var_type(env.get(field, default_value))

                self.__setattr__(field, value)
            except:
                raise AutoDiscServerConfigError(
                    'Unable to cast value of "{}" to type "{}" for "{}" field'.format(
                        env[field], var_type, field
                    )
                )
