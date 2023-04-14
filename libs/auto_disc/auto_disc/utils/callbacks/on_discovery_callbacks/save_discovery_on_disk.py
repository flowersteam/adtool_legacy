import pickle
import os
from datetime import datetime
import json
import torch
import numpy as np
from pydoc import locate
from typing import Type, Dict, Tuple
from leaf.Leaf import Leaf
from hashlib import sha1


def _save_binary_callback(binary: bytes,
                          save_dir: str) -> str:
    file_name = sha1(binary).hexdigest()
    file_path = os.path.join(save_dir, file_name)
    with open(file_path, "wb") as f:
        f.write(binary)
    return file_name


class _JSONEncoderFactory:
    def __call__(self, dir_path: str):
        # return _CustomJSONENcoder but with a class attr dir_path
        cls = _CustomJSONEncoder
        cls.dir_path = dir_path

        return cls


class _CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # catch torch Tensors
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        # catch numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # catch bytes
        if isinstance(obj, bytes):
            return _save_binary_callback(obj, self.dir_path)
        # catch Leaf objects
        if isinstance(obj, Leaf):
            uid = obj.save_leaf(self.dir_path)
            return str(uid)
        # catch python objects not serializable by JSON
        # this is only to comply with legacy code, as others should
        # implement Leaf
        try:
            json.JSONEncoder.default(self, obj)
        except TypeError:
            bin = pickle.dumps(obj)
            return _save_binary_callback(bin, self.dir_path)

        # pass to usual encoder
        return json.JSONEncoder.default(self, obj)


class SaveDiscoveryOnDisk:
    def __call__(self,
                 resource_uri: str,
                 experiment_id: int,
                 seed: int,
                 run_idx: int,
                 discovery: dict
                 ) -> None:
        dir_path = self._construct_save_path(resource_uri,
                                             experiment_id,
                                             run_idx, seed)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        # create JSON encoder
        json_encoder = _JSONEncoderFactory()(dir_path)
        # save dict_data to disk as JSON object
        file_path = os.path.join(dir_path, "discovery.json")
        with open(file_path, "w") as f:
            json.dump(discovery, f, cls=json_encoder)

    @staticmethod
    def _construct_save_path(resource_uri, experiment_id, run_idx, seed):
        dt = datetime.now()
        date_str = dt.isoformat(timespec='minutes')
        disc_path = os.path.join(resource_uri, "discoveries")
        if not os.path.exists(disc_path):
            os.mkdir(disc_path)
        dir_str = f"{date_str}_exp_{experiment_id}_idx_{run_idx}_seed_{seed}"
        dir_path = os.path.join(disc_path, dir_str)
        return dir_path
