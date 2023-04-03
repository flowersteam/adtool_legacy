import pickle
import os
from datetime import datetime
import json
import torch
from uuid import uuid1
import requests
from typing import Dict, Any


class _TorchTensorJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # catch torch Tensors
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        # pass to usual encoder
        return json.JSONEncoder.default(self, obj)


class SaveDiscoveryInExpeDB:
    def __call__(self,
                 resource_uri: str,
                 experiment_id: int,
                 seed: int,
                 run_idx: int,
                 discovery: Dict[str, Any]
                 ) -> None:

        # extract binaries to separate dictionary
        binaries = {}
        for (k, v) in discovery.items():
            if isinstance(v, bytes):
                binaries[k] = v
        for k in binaries.keys():
            del discovery[k]

        # assemble dict_data
        dict_data = {}
        dict_data["experiment_id"] = experiment_id
        dict_data["run_idx"] = run_idx
        dict_data["seed"] = seed
        dict_data["discovery"] = discovery

        # run through custom JSON encoder
        json_blob = json.dumps(dict_data, cls=_TorchTensorJSONEncoder)
        parsed_dict_data = json.loads(json_blob)

        # post dict_data
        response = requests.post(
            resource_uri + "/discoveries", json=parsed_dict_data)
        discovery_id = json.loads(response.text)["ID"]

        # dump binaries to /files endpoint
        files_to_save = {}
        for (name, data) in binaries.items():
            files_to_save[name] = data
        requests.post(
            resource_uri + f"/discoveries/{discovery_id}/files",
            files=files_to_save)

        return
