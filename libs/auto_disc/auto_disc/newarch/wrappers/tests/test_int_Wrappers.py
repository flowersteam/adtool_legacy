from auto_disc.newarch.wrappers \
    import IdentityWrapper, SaveWrapper, WrapperPipeline, TransformWrapper
from leaf.leaf import Leaf, Locator, LeafUID, StatelessLocator
from leaf.tests.test_leaf import DummyLocator
from typing import Dict, List
from copy import deepcopy
import pathlib
import tempfile


class IncrementerWrapper(Leaf):
    def __init__(self) -> None:
        super().__init__()

    def map(self, input: Dict) -> Dict:
        # must do because dicts are mutable types
        output = deepcopy(input)

        output["data"] = output["data"] + 1

        return output

    @classmethod
    def create_locator(cls, dict: Dict = None) -> 'Locator':
        return DictLocator(dict)


class FakeExperimentPipeline(Leaf):
    def __init__(self,
                 wrappers: List['Leaf'] = [],
                 locator: 'Locator' = StatelessLocator(),
                 save_db_url: str = "") -> None:
        super().__init__()
        # pass save_db_url to wrappers
        self.input_wrappers = WrapperPipeline(wrappers,
                                              resource_uri=save_db_url)
        # pass locator to WrapperPipeline
        self.input_wrappers.locator = locator

        # use locator also for saving own metadata
        self.locator = locator

    def input_transformation(self, input: Dict) -> Dict:
        output = self.input_wrappers.map(input)
        return output


class DictLocator(Locator):
    def __init__(self, filepath):
        self.filepath = filepath

    def store(self, bin: bytes) -> 'LeafUID':
        uid = LeafUID(self.hash(bin))
        self.table[uid] = bin
        return uid

    def retrieve(self, uid: 'LeafUID') -> bytes:
        return self.table[uid]


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


def teardown_function():
    import os
    global DB_PATH
    os.remove(DB_PATH)
    return


def test___init__():
    wrappers = [SaveWrapper(),
                IncrementerWrapper(),
                TransformWrapper(
                    wrapped_keys=["data"], posttransform_keys=["output"]),
                SaveWrapper()]
    pipeline = FakeExperimentPipeline(wrappers, save_db_url=DB_PATH)
    assert isinstance(pipeline._modules["input_wrappers"], WrapperPipeline)
    assert pipeline.input_wrappers.locator.resource_uri == ""
    assert pipeline.input_wrappers.wrappers[0].locator.resource_uri == DB_PATH
    pipeline = FakeExperimentPipeline(
        wrappers, save_db_url="test", locator=StatelessLocator(DB_PATH))
    assert pipeline.input_wrappers.locator.resource_uri == DB_PATH
    assert pipeline.input_wrappers.wrappers[0].locator.resource_uri == "test"


def test_input_transformation():
    input = {"data": 1}
    wrappers = [SaveWrapper(),
                IncrementerWrapper(),
                TransformWrapper(
                    wrapped_keys=["data"], posttransform_keys=["output"]),
                SaveWrapper()]
    pipeline = FakeExperimentPipeline(wrappers)
    output = pipeline.input_transformation(input)
    assert output == {"output": 2}


def test_saveload():
    db = {}
    input = {"data": 1}
    wrappers = [SaveWrapper(),
                IncrementerWrapper(),
                TransformWrapper(
                    wrapped_keys=["data"], posttransform_keys=["output"]),
                SaveWrapper()]
    pipeline = FakeExperimentPipeline(
        wrappers, save_db_url=DB_PATH, locator=DummyLocator(db))
    assert pipeline.locator.resource_uri == db
    assert pipeline.input_wrappers.locator.resource_uri == db

    output = pipeline.input_transformation(input)

    pipeline_uid = pipeline.save_leaf()
    assert len(db) == 2
