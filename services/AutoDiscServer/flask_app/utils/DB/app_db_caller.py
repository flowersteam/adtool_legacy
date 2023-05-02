from typing import Callable, Type
import typing
import requests
from enum import Enum


class AppDBMethods(Enum):
    GET = requests.get
    POST = requests.post
    PATCH = requests.patch


class AppDBCaller():
    def __init__(self, url: str) -> None:
        self.url = url

    def __call__(self, route: str, http_method: Callable, request_dict: typing.Dict) -> Type[requests.models.Response]:
        return http_method(self.url + route, json=request_dict)
