from leaf.leaf import Leaf, Locator
from leaf.tests.test_leaf import DummyModule, DummyLocator
import pickle


class DummyPipeline(Leaf):
    def __init__(self, l1=[1, 2, 3, 4], l2=[5, 6, 7, 8], resource_uri=""):
        super().__init__()
        self.locator = DummyLocator(resource_uri)
        self.l1 = DummyModule(l1)
        self.l2 = DummyModule(l2)


class DummyContained(Leaf):
    def __init__(self, s=None):
        super().__init__()
        self.internal_state = s

    def retrieve_metadata(self):
        return self._container_state["metadata"]


class DummyContainer(Leaf):
    def __init__(self, resource_uri):
        super().__init__()
        self.locator = DummyLocator(resource_uri)
        self.l1 = DummyContained([1, 2, 3, 4])
        self.l2 = DummyContained([1, 2, 3, 4])
        self.metadata = 42


def setup_function(function):
    global a, res_uri
    if "pipeline" in function.__name__.split("_"):
        global PipelineDB
        PipelineDB = {}
        res_uri = PipelineDB
        a = DummyPipeline(resource_uri=res_uri)
    elif "container" in function.__name__.split("_"):
        global ContainerDB
        ContainerDB = {}
        res_uri = ContainerDB
        a = DummyContainer(res_uri)
    return


def test_pipeline___init__():
    assert isinstance(a, Leaf)
    assert isinstance(a.l1, Leaf)
    assert isinstance(a.l2, Leaf)
    assert a.l1 == a._modules["l1"]
    assert a.l2 == a._modules["l2"]
    assert a.locator.resource_uri == PipelineDB


def test_pipeline_mutate_data():
    a.l1.internal_state = a.l1.forward(1)
    assert a.l1.internal_state == [2, 3, 4, 5]
    assert a.l2.internal_state == [5, 6, 7, 8]


def test_pipeline_container_ptr():
    assert a.l1._container_ptr == a
    assert a.l2._container_ptr == a


def test_pipeline_container_subleaf_names():
    assert a.l1.name == "l1"
    assert a.l2.name == "l2"


def test_pipeline_serialize_recursively():
    bin = a.serialize()
    obj = pickle.loads(bin)
    assert obj._modules["l1"] == a.l1._get_uid_base_case()
    assert obj._modules["l2"] == a.l2._get_uid_base_case()


def test_pipeline_save_data():
    uid_old = a.save_leaf()
    assert len(res_uri) == 3

    a.l1.internal_state = a.l1.forward(1)
    uid_new = a.save_leaf()
    # only one of the submodules is modified
    assert len(res_uri) == 5

    b = DummyPipeline(l1=[], l2=[])
    b.locator = DummyLocator(res_uri)
    b = b.load_leaf(uid_old, res_uri)
    assert b.l1.internal_state == [1, 2, 3, 4]
    assert b.l1._container_ptr == b
    assert b.l2._container_ptr == b

    c = DummyPipeline(l1=[], l2=[])
    c.locator = DummyLocator(res_uri)
    c = c.load_leaf(uid_new, res_uri)
    assert c.l1.internal_state == [2, 3, 4, 5]
    assert c.l1._container_ptr == c
    assert c.l2._container_ptr == c


def test_pipeline_submodule_pointers():
    a.l3 = a.l2
    assert a.l2 == a.l3


def test_container___init__():
    assert a.l1.locator == a.locator
    assert a.l2.locator == a.locator


def test_container_call_inner():
    assert a.l1.retrieve_metadata() == 42
    a.metadata = 16
    assert a.l1.retrieve_metadata() == 16


def test_container_serialize_recursively():
    bin = a.serialize()
    obj = pickle.loads(bin)
    assert isinstance(obj._modules["l1"], str)
    assert isinstance(obj._modules["l2"], str)
    assert obj._modules["l1"] == a.l1._get_uid_base_case()
    assert obj._modules["l2"] == a.l2._get_uid_base_case()
    assert obj._modules["l1"] != obj._modules["l2"]
