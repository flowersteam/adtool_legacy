import typing
from typing import Any, Dict
import requests
import json

class ExpeDBCaller():
    def __init__(self, url: str) -> None:
        self.base_url = url
    
    def __call__(self, route: str, request_dict: typing.Dict[str, Any]=None, files: typing.Dict[str, Any]=None) -> Dict:
        response = requests.post(self.base_url + route, json=request_dict, files=files)
        return json.loads(response.text)

   