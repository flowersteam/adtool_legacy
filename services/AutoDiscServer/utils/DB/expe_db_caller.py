from typing import Pattern
import requests
import json

class ExpeDBCaller():
    def __init__(self, url):
        self.base_url = url
    
    def __call__(self, route, request_dict=None, files=None):
        response = requests.post(self.base_url + route, json=request_dict, files=files)
        return json.loads(response.text)

   