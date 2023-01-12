from auto_disc.newarch.wrappers.SaveWrapper import SaveWrapper
from leafutils.leafstructs.linear import Stepper
import os
import pathlib


def setup_function(function):
    import sqlite3
    global FILE_PATH, DB_PATH

    FILE_PATH = str(pathlib.Path(__file__).parent.resolve())
    DB_REL_PATH = "/tmp.sqlite"
    SCRIPT_REL_PATH = "/mockDB.sql"

    DB_PATH = FILE_PATH + DB_REL_PATH
    SCRIPT_PATH = FILE_PATH + SCRIPT_REL_PATH

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
    assert wrapper.__dict__ == wrapper_def.__dict__


def test_map():
    input = {"in": 1}
    wrapper = SaveWrapper(wrapped_keys=["in"], posttransform_keys=["out"])
    output = wrapper.map(input)
    assert output["out"] == 1
    assert len(output) == 1
    assert wrapper.input_buffer == [{"in": 1}]
    assert wrapper.output_buffer == []


def test_map_default():
    input = {"data": 1}
    wrapper = SaveWrapper()
    output = wrapper.map(input)
    assert output["data"] == 1
    assert len(output) == 1
    assert wrapper.input_buffer == [{"data": 1}]
    assert wrapper.output_buffer == []


def test_map_complex():
    input = {"a": 1, "b": 2}
    wrapper = SaveWrapper(
        wrapped_keys=["a", "b"], posttransform_keys=["b", "a"])
    output = wrapper.map(input)
    assert output["a"] == 2
    assert output["b"] == 1
    assert wrapper.input_buffer == [{"a": 1, "b": 2}]
    assert wrapper.output_buffer == []

    wrapper.map(output)
    assert wrapper.input_buffer == [{"a": 1, "b": 2}, {"a": 2, "b": 1}]
    assert wrapper.output_buffer == []


def test_serialize():
    input = {"a": 1, "b": 2}
    wrapper = SaveWrapper(
        wrapped_keys=["a", "b"], posttransform_keys=["b", "a"])
    output = wrapper.map(input)
    wrapper.map(output)
    bin = wrapper.serialize()
    a = Stepper().deserialize(bin)
    assert a.buffer == wrapper.input_buffer


def test_saveload_basic():
    """
    This tests saving and loading of a single save step (which saves two
    "map" steps of progress)
    """
    db_url = DB_PATH
    input = {"a": 1, "b": 2}
    wrapper = SaveWrapper(
        wrapped_keys=["a", "b"], posttransform_keys=["b", "a"])
    output = wrapper.map(input)
    wrapper.map(output)
    leaf_uid = wrapper.save_leaf(db_url)

    # retrieve from leaf nodes of tree
    stepper_loaded = wrapper.load_leaf(leaf_uid, db_url)
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
    stepper_loaded = wrapper.load_leaf(leaf_uid, db_url)
    buffer = stepper_loaded.buffer

    # unpack and check loaded Stepper
    assert len(buffer) == 2
    assert Stepper().deserialize(buffer[0]).buffer[0] == {"a": 1, "b": 2}
    assert Stepper().deserialize(buffer[0]).buffer[1] == {"a": 2, "b": 1}
    assert Stepper().deserialize(buffer[1]).buffer[0] == {"a": 1, "b": 2}
    assert Stepper().deserialize(buffer[1]).buffer[1] == {"a": 2, "b": 1}
    assert Stepper().deserialize(buffer[1]).buffer[2] == {"a": 1, "b": 2}
