import typing
from typing import Any, Dict, Union
import requests
import json
import pickle

class ExpeDBCaller():
    def __init__(self, url: str) -> None:
        self.base_url = url
    
    def __call__(self, route: str, request_dict: typing.Dict[str, Any]=None, files: typing.Dict[str, Any]=None) -> Dict:
        response = requests.post(self.base_url + route, json=request_dict, files=files)
        return json.loads(response.text)

    def read(self, route: str, filter_attribut:dict):
        filter = self.format_filter(filter_attribut)
        response = json.loads(requests.get(self.base_url + route + "?filter=" + filter).content)
        return response
    
    def read_file(self, route: str, response:Union[dict, list], file_name: str):
        if isinstance(response, list):
            for i in range(0, len(response)):
                response[i]["file_"+file_name] = requests.get(self.base_url + route + "/" + response[i]["_id"] + "/" + file_name).content
                try:
                    response[i]["file_"+file_name] = pickle.loads(response[i]["file_"+file_name])
                except:
                    pass
        else:
            response["file_"+file_name] = requests.get(self.base_url + route + "/" + response["_id"] + "/" + file_name).content
            try:
                response["file_"+file_name] = pickle.loads(response["file_"+file_name])
            except:
                pass
        return response

    def format_filter(self, filter_attribut):
        filters = []
        for key in filter_attribut:
            if filter_attribut[key] == None:
                pass
            elif isinstance(filter_attribut[key], list):
                filters.append("{{\"{}\":{{\"$in\":[{}]}} }}".format(key), ', '.join(filter_attribut[key]))
            else:
                if isinstance(filter_attribut[key], str):
                    filters.append("{{\"{}\":\"{}\" }}".format(key, filter_attribut[key]))
                else:
                    filters.append("{{\"{}\":{} }}".format(key, filter_attribut[key]))
        if len(filters) > 1:
            filter = "{\"$and\":["
            for f in filters:
                filter += f +","
            filter = filter[:-1]
            filter += "]}"
        else:
            filter = filters[0]
        return filter