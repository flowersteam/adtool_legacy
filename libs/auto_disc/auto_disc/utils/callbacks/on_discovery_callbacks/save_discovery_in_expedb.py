import pickle
import os
from datetime import datetime
import json
import torch
from uuid import uuid1
import requests
from typing import Dict, Any, Type
from auto_disc.utils.callbacks.on_discovery_callbacks.save_discovery import SaveDiscovery
from hashlib import sha1


class SaveDiscoveryInExpeDB(SaveDiscovery):
    def __call__(self, resource_uri: str,
                 experiment_id: int,
                 run_idx: int,
                 seed: int,
                 discovery: Dict[str, Any]
                 ) -> None:
        super().__call__(resource_uri,
                         experiment_id,
                         run_idx,
                         seed,
                         discovery)

        return

    @staticmethod
    def _dump_json(discovery: Dict[str, Any],
                   dir_path: str,
                   json_encoder: Type[json.JSONEncoder],
                   **kwargs
                   ) -> None:
        # converts discovery to JSON blob and calls the _save_binary_callback
        # when needed. Reloads the JSON blob as a dict to push to the DB
        json_blob = json.dumps(discovery, cls=json_encoder)
        parsed_dict_data = json.loads(json_blob)

        # push to DB
        requests.post(dir_path, json=parsed_dict_data)

        return

    @staticmethod
    def _initialize_save_path(resource_uri: str,
                              experiment_id: int,
                              run_idx: int,
                              seed: int
                              ) -> str:
        """
        Pushes metadata to the MongoDB discoveries collection,
        and returns the ID of the newly created document.
        """
        # initial payload
        payload = {"experiment_id": experiment_id,
                   "run_idx": run_idx,
                   "seed": seed}
        response = requests.post(resource_uri + "/discoveries", json=payload)
        doc_id = json.loads(response.text)["ID"]

        document_path = os.path.join(resource_uri, "discoveries", doc_id)
        return document_path

    @classmethod
    def _save_binary_callback(cls: Type, binary: bytes, document_path: str) -> str:
        """
        Pushes binary data to the MongoDB discoveries collection in the form of
        top-level key-value pairs {sha1_hash : binary_data}
        """
        sha1_hash = sha1(binary).hexdigest()
        files_to_save = {sha1_hash: binary}

        requests.post(document_path + "/files",
                      files=files_to_save)
        return sha1_hash


class OldSaveDiscoveryInExpeDB:
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
