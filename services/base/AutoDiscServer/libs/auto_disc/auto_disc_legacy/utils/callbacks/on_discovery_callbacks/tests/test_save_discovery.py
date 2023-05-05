import auto_disc
from auto_disc_legacy.utils.callbacks.on_discovery_callbacks.save_discovery import _CustomJSONEncoder, _JSONEncoderFactory
import torch
import numpy
from leaf.Leaf import Leaf
import leaf
import json
import unittest.mock as mock


def test__JSONEncoderFactory():
    def dummy_cb(x): return x
    fac = _JSONEncoderFactory()
    cls = fac(dir_path="dummy_path", custom_callback=dummy_cb)
    assert cls._dir_path == "dummy_path"
    assert cls._custom_callback == dummy_cb
    assert cls.__name__ == "_CustomJSONEncoder"
    assert _CustomJSONEncoder._dir_path == "dummy_path"
    assert _CustomJSONEncoder._custom_callback == dummy_cb


def test__CustomJSONEncoder(mocker):
    fac = _JSONEncoderFactory()
    mocked_cb = mock.Mock(return_value="dummy_uid")
    cls = fac(dir_path="dummy_path", custom_callback=mocked_cb)
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
    encoded = encoder.default(obj)
    assert encoded == "dummy_uid"
    assert mocked_cb.call_count == 1

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
    assert mocked_cb.call_count == 2

    # ensure that the default method is called
    obj = "dummy_string"
    mocker.patch("json.JSONEncoder.default", return_value="dummy_return")
    encoded = encoder.default(obj)
    assert encoded == "dummy_return"
    # it is called twice, because the first is in the try block
    assert json.JSONEncoder.default.call_count == 2
