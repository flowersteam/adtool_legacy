from leaf.leaf import Leaf, StatelessLocator
from auto_disc.newarch.wrappers.WrapperPipeline import WrapperPipeline
from leaf.tests.test_leaf import DummyLocator
from copy import deepcopy
from typing import Dict
import pytest


class TestWrapper(Leaf):
    def __init__(self, wrapped_key: str = "in", offset: int = 1) -> None:
        super().__init__()
        self.wrapped_key = wrapped_key
        self.offset = offset

    def map(self, input: Dict) -> Dict:
        output = deepcopy(input)
        output[self.wrapped_key] += self.offset
        return output


def test___init__():
    input = {"in": 1}
    a = TestWrapper(offset=1, wrapped_key="in")
    b = TestWrapper(offset=2, wrapped_key="in")
    wrapper_list = [a, b]
    all_wrappers = WrapperPipeline(wrappers=wrapper_list)
    assert all_wrappers.map(input) == b.map(a.map(input))
    assert all_wrappers.wrappers[0]._container_ptr == all_wrappers
    assert all_wrappers.wrappers[1]._container_ptr == all_wrappers


def test___init__mutually_exclusive():
    storage_db = {}
    input = {"in": 1}
    a = TestWrapper(offset=1, wrapped_key="in")
    b = TestWrapper(offset=2, wrapped_key="in")
    wrapper_list = [a, b]
    with pytest.raises(ValueError):
        all_wrappers = WrapperPipeline(wrappers=wrapper_list,
                                       locator=DummyLocator(storage_db),
                                       resource_uri=storage_db)


def test___init__pass_locator():
    storage_db = {}
    input = {"in": 1}
    a = TestWrapper(offset=1, wrapped_key="in")
    b = TestWrapper(offset=2, wrapped_key="in")
    wrapper_list = [a, b]
    all_wrappers = WrapperPipeline(wrappers=wrapper_list,
                                   locator=DummyLocator(storage_db))
    assert all_wrappers.wrappers[0].locator.resource_uri == storage_db
    assert all_wrappers.wrappers[1].locator.resource_uri == storage_db
    assert isinstance(all_wrappers.wrappers[0].locator, DummyLocator)
    assert isinstance(all_wrappers.wrappers[1].locator, DummyLocator)
    assert isinstance(all_wrappers.locator, DummyLocator)
    assert all_wrappers.locator.resource_uri == storage_db


def test___init__pass_uri():
    input = {"in": 1}
    a = TestWrapper(offset=1, wrapped_key="in")
    b = TestWrapper(offset=2, wrapped_key="in")
    a.locator = DummyLocator("")
    b.locator = DummyLocator("")
    wrapper_list = [a, b]
    all_wrappers = WrapperPipeline(wrappers=wrapper_list,
                                   resource_uri="com.test")
    assert all_wrappers.wrappers[0].locator.resource_uri == "com.test"
    assert all_wrappers.wrappers[1].locator.resource_uri == "com.test"
    assert isinstance(all_wrappers.wrappers[0].locator, DummyLocator)
    assert isinstance(all_wrappers.wrappers[1].locator, DummyLocator)
    assert isinstance(all_wrappers.locator, StatelessLocator)
    assert all_wrappers.locator.resource_uri == "com.test"


def test_saveload():
    storage_db = {}
    input = {"in": 1}
    a = TestWrapper(offset=1, wrapped_key="in")
    b = TestWrapper(offset=2, wrapped_key="in")
    wrapper_list = [a, b]
    all_wrappers = WrapperPipeline(wrappers=wrapper_list,
                                   locator=DummyLocator(storage_db))

    uid = all_wrappers.save_leaf()
    assert len(storage_db) == 3

    loaded_wrappers = WrapperPipeline(locator=DummyLocator(storage_db))
    loaded_wrappers = loaded_wrappers.load_leaf(uid)

    for i in range(2):
        assert all_wrappers.wrappers[i].offset \
            == loaded_wrappers.wrappers[i].offset
        assert all_wrappers.wrappers[i].wrapped_key \
            == loaded_wrappers.wrappers[i].wrapped_key
