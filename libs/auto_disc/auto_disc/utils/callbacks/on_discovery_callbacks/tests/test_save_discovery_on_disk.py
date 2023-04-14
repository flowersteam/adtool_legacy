import auto_disc
from auto_disc.utils.callbacks.on_discovery_callbacks.save_discovery_on_disk import SaveDiscoveryOnDisk, _CustomJSONEncoder, _JSONEncoderFactory, _save_binary_callback
import os
import shutil
import pathlib
import torch
import numpy
from leaf.Leaf import Leaf
import leaf
import pytest
import json
from hashlib import sha1
import pickle
from pytest_mock import mocker


def setup_function(function):
    global RESOURCE_URI
    file_path = str(pathlib.Path(__file__).parent.resolve())
    RESOURCE_URI = os.path.join(file_path, "tmp")
    os.mkdir(RESOURCE_URI)
    return


def teardown_function(function):
    global RESOURCE_URI
    if os.path.exists(RESOURCE_URI):
        shutil.rmtree(RESOURCE_URI)
    return


def test_SaveDiscoveryOnDisk__construct_save_path():
    cb = SaveDiscoveryOnDisk()
    dir_path = cb._construct_save_path(RESOURCE_URI, "test_id", 777, 33)

    assert os.path.dirname(dir_path) == RESOURCE_URI + "/discoveries"


def test__save_binary_callback():
    cb = SaveDiscoveryOnDisk()
    dir_path = cb._construct_save_path(RESOURCE_URI, "test_id", 777)
    os.mkdir(dir_path)
    bin = b"123"
    uid = _save_binary_callback(bin, dir_path)
    assert os.path.exists(os.path.join(dir_path, uid))


def test__JSONEncoderFactory():
    fac = _JSONEncoderFactory()
    cls = fac(dir_path="dummy_path")
    assert cls.dir_path == "dummy_path"
    assert cls.__name__ == "_CustomJSONEncoder"
    assert _CustomJSONEncoder.dir_path == "dummy_path"


def test__CustomJSONEncoder(mocker):
    fac = _JSONEncoderFactory()
    cls = fac(dir_path="dummy_path")
    encoder = cls()

    # catch torch Tensors
    obj = torch.Tensor([1, 2, 3])
    encoded = encoder.default(obj)
    assert encoded == [1, 2, 3]

    # catch numpy arrays
    obj = numpy.array([1, 2, 3])
    encoded = encoder.default(obj)
    assert encoded == [1, 2, 3]

    # catch bytes
    obj = b"dummy_bytes"
    callback_name = "auto_disc.utils.callbacks."\
        "on_discovery_callbacks.save_discovery_on_disk."\
        "_save_binary_callback"
    mocker.patch(callback_name,
                 return_value="dummy_uid")
    encoded = encoder.default(obj)
    assert encoded == "dummy_uid"
    assert auto_disc.utils.callbacks.\
        on_discovery_callbacks.save_discovery_on_disk.\
        _save_binary_callback.call_count == 1

    # catch Leaf objects
    obj = Leaf()
    mocker.patch("leaf.Leaf.Leaf.save_leaf", return_value="dummy_uid")
    encoded = encoder.default(obj)
    assert encoded == "dummy_uid"
    assert leaf.Leaf.Leaf.save_leaf.call_count == 1

    # catch python objects not serializable by JSON
    obj = object()
    encoded = encoder.default(obj)
    assert encoded == "dummy_uid"
    assert auto_disc.utils.callbacks.\
        on_discovery_callbacks.save_discovery_on_disk.\
        _save_binary_callback.call_count == 2

    # ensure that the default method is called
    obj = "dummy_string"
    mocker.patch("json.JSONEncoder.default", return_value="dummy_return")
    encoded = encoder.default(obj)
    assert encoded == "dummy_return"
    # it is called twice, because the first is in the try block
    assert json.JSONEncoder.default.call_count == 2


def test___call__():
    data = {
        "params": torch.tensor([1., 2., 3.]),
        "loss": 0.5,
        "model": object(),
        "metadata": {"test": b"test"}
    }
    cb = SaveDiscoveryOnDisk()
    cb(resource_uri=RESOURCE_URI, experiment_id="test_id", seed=33,
       run_idx=777, discovery=data)

    # check if the discovery was saved
    dir_path = cb._construct_save_path(RESOURCE_URI, "test_id", 777, 33)
    assert os.path.exists(dir_path)
    assert os.path.exists(os.path.join(dir_path, "discovery.json"))

    # load the discovery and check if the data is correct
    obj_hash = sha1(pickle.dumps(data["model"])).hexdigest()
    byte_hash = sha1(data["metadata"]["test"]).hexdigest()

    with open(os.path.join(dir_path, "discovery.json"), "r") as f:
        loaded_data = json.load(f)
    assert loaded_data["params"] == [1., 2., 3.]
    assert loaded_data["loss"] == 0.5
    assert loaded_data["model"] == obj_hash
    assert loaded_data["metadata"] == {"test": byte_hash}

    with open(os.path.join(dir_path, obj_hash), "rb") as f:
        loaded_data = f.read()
    assert loaded_data == pickle.dumps(data["model"])
    with open(os.path.join(dir_path, byte_hash), "rb") as f:
        loaded_data = f.read()
    assert loaded_data == b"test"
