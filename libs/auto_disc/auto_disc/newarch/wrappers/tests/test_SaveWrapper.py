from leaf.leaf import StatelessLocator
from auto_disc.newarch.wrappers.SaveWrapper import SaveWrapper
from leafutils.leafstructs.linear import Stepper, LinearLocator
import os
import pathlib
import tempfile


def test___init__():
    input = {"in": 1}
    wrapper = SaveWrapper(wrapped_keys=["in"], posttransform_keys=["out"],
                          inputs_to_save=["in"])
    wrapper_def = SaveWrapper(wrapped_keys=["in"], posttransform_keys=["out"])
    assert isinstance(wrapper.locator, LinearLocator)
    assert isinstance(wrapper_def.locator, LinearLocator)
    assert wrapper.locator.resource_uri == ""
    assert wrapper_def.locator.resource_uri == ""
    del wrapper.locator
    del wrapper_def.locator
    assert wrapper.__dict__ == wrapper_def.__dict__


def test_map():
    input = {"in": 1}
    wrapper = SaveWrapper(wrapped_keys=["in"], posttransform_keys=["out"])
    output = wrapper.map(input)
    assert output["out"] == 1
    assert len(output) == 1
    assert wrapper.buffer == [{"in": 1}]


def test_map_default():
    input = {"data": 1}
    wrapper = SaveWrapper()
    output = wrapper.map(input)
    assert output["data"] == 1
    assert len(output) == 1
    assert wrapper.buffer == [{"data": 1}]


def test_map_minimal():
    input = {"data": 1, "metadata": 0}
    wrapper = SaveWrapper(wrapped_keys=["data"], posttransform_keys=["data"])
    output = wrapper.map(input)
    assert output["data"] == 1
    assert len(output) == 2
    assert wrapper.buffer == [{"data": 1}]


def test_map_complex():
    input = {"a": 1, "b": 2}
    wrapper = SaveWrapper(
        wrapped_keys=["a", "b"], posttransform_keys=["b", "a"])
    output = wrapper.map(input)
    assert output["a"] == 2
    assert output["b"] == 1
    assert wrapper.buffer == [{"a": 1, "b": 2}]

    wrapper.map(output)
    assert wrapper.buffer == [{"a": 1, "b": 2}, {"a": 2, "b": 1}]


def test_serialize():
    input = {"a": 1, "b": 2}
    wrapper = SaveWrapper(
        wrapped_keys=["a", "b"], posttransform_keys=["b", "a"])
    output = wrapper.map(input)
    wrapper.map(output)
    bin = wrapper.serialize()

    linear = LinearLocator()
    _, data_bin = linear._parse_bin(bin)
    a = Stepper().deserialize(data_bin)
    assert a.buffer == wrapper.buffer


def test_saveload_basic():
    """
    This tests saving and loading of a single save step (which saves two
    "map" steps of progress)
    """
    FILE_PATH = str(pathlib.Path(__file__).parent.resolve())
    input = {"a": 1, "b": 2}
    wrapper = SaveWrapper(
        wrapped_keys=["a", "b"], posttransform_keys=["b", "a"])

    output = wrapper.map(input)
    wrapper.map(output)

    leaf_uid = wrapper.save_leaf(FILE_PATH)

    # retrieve from leaf nodes of tree
    new_wrapper = SaveWrapper()
    wrapper_loaded = new_wrapper.load_leaf(leaf_uid, FILE_PATH)
    buffer = wrapper_loaded.buffer

    # unpack and check loaded Stepper
    assert len(buffer) == 2
    assert buffer[0] == {"a": 1, "b": 2}
    assert buffer[1] == {"a": 2, "b": 1}


def test_saveload_advanced():
    """
    This tests saving and loading of multiple save steps (which saves two
    "map" steps of progress)
    """
    FILE_PATH = str(pathlib.Path(__file__).parent.resolve())
    input = {"a": 1, "b": 2}
    wrapper = SaveWrapper(
        wrapped_keys=["a", "b"], posttransform_keys=["b", "a"])
    output = input

    for i in range(2):
        output = wrapper.map(output)
        output = wrapper.map(output)

        # vary save buffer length
        if i == 1:
            output = wrapper.map(output)

        leaf_uid = wrapper.save_leaf(FILE_PATH)

    # retrieve from leaf nodes of tree
    new_wrapper = SaveWrapper()
    wrapper_loaded = new_wrapper.load_leaf(leaf_uid, FILE_PATH)
    buffer = wrapper_loaded.buffer

    # unpack and check loaded Stepper
    assert len(buffer) == 3
    assert buffer[0] == {"a": 1, "b": 2}
    assert buffer[1] == {"a": 2, "b": 1}
    assert buffer[2] == {"a": 1, "b": 2}

    # check metadata
    assert wrapper_loaded.inputs_to_save == ["a", "b"]


def test_saveload_whole_history():
    FILE_PATH = str(pathlib.Path(__file__).parent.resolve())
    input = {"a": 1, "b": 2}
    wrapper = SaveWrapper(
        wrapped_keys=["a", "b"], posttransform_keys=["b", "a"])
    output = input

    for i in range(2):
        output = wrapper.map(output)
        output = wrapper.map(output)

        # vary save buffer length
        if i == 1:
            output = wrapper.map(output)

        leaf_uid = wrapper.save_leaf(FILE_PATH)

    # try retrieval of entire sequence
    new_wrapper = SaveWrapper()
    wrapper_loaded = new_wrapper.load_leaf(leaf_uid, FILE_PATH, 0)
    buffer = wrapper_loaded.buffer

    assert len(buffer) == 5
    assert buffer[0] == {"a": 1, "b": 2}
    assert buffer[1] == {"a": 2, "b": 1}
    assert buffer[2] == {"a": 1, "b": 2}
    assert buffer[3] == {"a": 2, "b": 1}
    assert buffer[4] == {"a": 1, "b": 2}

    # try retrieval of entire sequence with explicit length
    new_wrapper = SaveWrapper()
    wrapper_loaded = new_wrapper.load_leaf(leaf_uid, FILE_PATH, length=2)
    buffer = wrapper_loaded.buffer

    assert len(buffer) == 5
    assert buffer[0] == {"a": 1, "b": 2}
    assert buffer[1] == {"a": 2, "b": 1}
    assert buffer[2] == {"a": 1, "b": 2}
    assert buffer[3] == {"a": 2, "b": 1}
    assert buffer[4] == {"a": 1, "b": 2}
