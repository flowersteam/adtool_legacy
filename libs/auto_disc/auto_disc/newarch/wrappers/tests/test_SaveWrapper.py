from leaf.leaf import StatelessLocator
from auto_disc.newarch.wrappers.SaveWrapper import SaveWrapper
from leafutils.leafstructs.linear import Stepper, LinearLocator
import os
import pathlib
import tempfile


def setup_function(function):
    import sqlite3
    global FILE_PATH, DB_PATH

    FILE_PATH = str(pathlib.Path(__file__).parent.resolve())
    SCRIPT_REL_PATH = "/mockDB.sql"
    SCRIPT_PATH = FILE_PATH + SCRIPT_REL_PATH

    _, DB_PATH = tempfile.mkstemp(suffix=".sqlite", dir=FILE_PATH)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    with open(SCRIPT_PATH) as f:
        query_string = f.read()
        cur.executescript(query_string)
    return


def teardown_function(function):
    global DB_PATH
    os.remove(DB_PATH)
    return


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
    a = Stepper().deserialize(bin)
    assert a.buffer == wrapper.buffer


def test_saveload_basic():
    """
    This tests saving and loading of a single save step (which saves two
    "map" steps of progress)
    """
    db_url = DB_PATH
    input = {"a": 1, "b": 2}
    wrapper = SaveWrapper(
        wrapped_keys=["a", "b"], posttransform_keys=["b", "a"])
    wrapper.resource_uri = db_url
    output = wrapper.map(input)
    wrapper.map(output)
    leaf_uid = wrapper.save_leaf(db_url)

    # retrieve from leaf nodes of tree
    new_wrapper = SaveWrapper()
    stepper_loaded = new_wrapper.load_leaf(leaf_uid, db_url)
    buffer = stepper_loaded.buffer

    # unpack and check loaded Stepper
    assert len(buffer) == 1
    assert Stepper().deserialize(buffer[0]).buffer[0] == {"a": 1, "b": 2}
    assert Stepper().deserialize(buffer[0]).buffer[1] == {"a": 2, "b": 1}


def test_saveload_advanced():
    """
    This tests saving and loading of multiple save steps (which saves two
    "map" steps of progress)
    """
    db_url = DB_PATH
    input = {"a": 1, "b": 2}
    wrapper = SaveWrapper(
        wrapped_keys=["a", "b"], posttransform_keys=["b", "a"])
    output = input
    leaf_uid = -1

    for i in range(2):
        output = wrapper.map(output)
        output = wrapper.map(output)

        # vary save buffer length
        if i == 1:
            output = wrapper.map(output)

        leaf_uid = wrapper.save_leaf(db_url, leaf_uid)

    # retrieve from leaf nodes of tree
    new_wrapper = SaveWrapper()
    stepper_loaded = new_wrapper.load_leaf(leaf_uid, db_url)
    buffer = stepper_loaded.buffer

    # unpack and check loaded Stepper
    assert len(buffer) == 2
    assert Stepper().deserialize(buffer[0]).buffer[0] == {"a": 1, "b": 2}
    assert Stepper().deserialize(buffer[0]).buffer[1] == {"a": 2, "b": 1}
    assert Stepper().deserialize(buffer[1]).buffer[0] == {"a": 1, "b": 2}
    assert Stepper().deserialize(buffer[1]).buffer[1] == {"a": 2, "b": 1}
    assert Stepper().deserialize(buffer[1]).buffer[2] == {"a": 1, "b": 2}