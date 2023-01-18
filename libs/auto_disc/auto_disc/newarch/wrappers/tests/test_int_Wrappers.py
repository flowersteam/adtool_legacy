from auto_disc.newarch.wrappers \
    import IdentityWrapper, SaveWrapper, WrapperPipeline
from leaf.leaf import Leaf
from typing import Dict
from copy import deepcopy
from leafutils.leafstructs.stateless import StatelessService
import pathlib
import tempfile


class IncrementerWrapper(StatelessService):
    def __init__(self) -> None:
        super().__init__()

    def map(self, input: Dict) -> Dict:
        # must do because dicts are mutable types
        output = deepcopy(input)

        output["data"] = output["data"] + 1

        return output


class FakeExperimentPipeline(Leaf):
    def __init__(self, wrappers) -> None:
        super().__init__()
        self.input_wrappers = WrapperPipeline(wrappers, inputs_to_save=["in"])

    def input_transformation(self, input: Dict) -> Dict:
        output = self.input_wrappers.map(input)
        return output


def setup_db():
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


def teardown_db():
    import os
    global DB_PATH
    os.remove(DB_PATH)
    return


def test___init__():
    wrappers = [SaveWrapper(),
                IncrementerWrapper(),
                SaveWrapper(
                    wrapped_keys=["data"], posttransform_keys=["output"],
                    outputs_to_save=["output"])]
    pipeline = FakeExperimentPipeline(wrappers)
    assert isinstance(pipeline._modules["input_wrappers"], WrapperPipeline)


def test_input_transformation():
    input = {"data": 1}
    wrappers = [SaveWrapper(),
                IncrementerWrapper(),
                SaveWrapper(
                    wrapped_keys=["data"], posttransform_keys=["output"],
                    outputs_to_save=["output"])]
    pipeline = FakeExperimentPipeline(wrappers)
    output = pipeline.input_transformation(input)
    assert output == {"output": 2}


def test_saveload():
    setup_db()
    input = {"data": 1}

    wrappers = [SaveWrapper(),
                IncrementerWrapper(),
                SaveWrapper(
                    wrapped_keys=["data"], posttransform_keys=["output"],
                    outputs_to_save=["output"])]
    pipeline = FakeExperimentPipeline(wrappers)
    output = pipeline.input_transformation(input)
    assert output == {"output": 2}
    teardown_db()
