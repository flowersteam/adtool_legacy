from typing import Pattern
import requests
from enum import Enum

class AppDBMethods(Enum):
    GET = requests.get
    POST = requests.post
    PATCH = requests.patch

class AppDBCaller():
    def __init__(self, url):
        self.url = url
    
    def __call__(self, route, http_method, request_dict):
        return http_method(self.url + route, json=request_dict)