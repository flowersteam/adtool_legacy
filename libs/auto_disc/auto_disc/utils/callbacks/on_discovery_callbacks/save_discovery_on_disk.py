import pickle
import os
from datetime import datetime
import json
import torch
from uuid import uuid1


class _TorchTensorJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # catch torch Tensors
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
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
        # construct save_path
        dt = datetime.now()
        date_str = dt.isoformat(timespec='minutes')
        disc_path = os.path.join(resource_uri, "discoveries")
        if not os.path.exists(disc_path):
            os.mkdir(disc_path)
        dir_str = f"{date_str}_exp_{experiment_id}_idx_{run_idx}_{str(uuid1())}"
        dir_path = os.path.join(disc_path, dir_str)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        # extract binaries to separate dictionary
        binaries = {}
        for (k, v) in discovery.items():
            if isinstance(v, bytes):
                binaries[k] = v
        for k in binaries.keys():
            del discovery[k]

        # dump binaries to disk
        for (name, data) in binaries.items():
            file_path = os.path.join(dir_path, name)
            with open(file_path, "wb") as f:
                f.write(data)

        # save dict_data to disk as JSON object
        file_path = os.path.join(dir_path, "discovery.json")
        with open(file_path, "w") as f:
            json.dump(discovery, f, cls=_TorchTensorJSONEncoder)