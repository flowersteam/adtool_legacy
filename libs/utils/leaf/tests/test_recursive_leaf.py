from leaf.leaf import Leaf, Locator
from leaf.tests.test_leaf import DummyModule, DummyLocator
import pickle


class DummyPipeline(Leaf):
    def __init__(self, l1=[1, 2, 3, 4], l2=[5, 6, 7, 8]):
        super().__init__()
        self.l1 = DummyModule(l1)
        self.l2 = DummyModule(l2)

    @classmethod
    def create_locator(cls, resource_uri, *args, **kwargs) -> 'Locator':
        # NOTE: we violate the function signature, bc Python globals() is per-module
        # and therefore the namespace lookup fails with imports
        return DummyLocator(resource_uri)


class DummyContained(Leaf):
    def __init__(self, s=None):
        super().__init__()
        self.internal_state = s

    def retrieve_metadata(self):
        return self._container_state["metadata"]

    def create_locator(self, bin):
        return DummyLocator(bin)

    def store_locator(self, loc):
        self.locator_table[self.uid] = loc.serialize()
        return

    @classmethod
    def retrieve_locator(cls, leaf_uid):
        return Locator.deserialize(cls.locator_table[leaf_uid])


class DummyContainer(Leaf):
    def __init__(self):
        super().__init__()
        self.l1 = DummyContained([1, 2, 3, 4])
        self.metadata = 42

    def create_locator(self, bin: bytes) -> 'Locator':
        return DummyLocator(bin)

    def store_locator(self, loc):
        global ContainerDB
        ContainerDB[self.uid] = pickle.dumps(loc)
        return

    @classmethod
    def retrieve_locator(cls, leaf_uid: str) -> 'Locator':
        global ContainerDB
        return pickle.loads(ContainerDB[leaf_uid])


def setup_function(function):
    global a, res_uri
    if "pipeline" in function.__name__.split("_"):
        global PipelineDB
        a = DummyPipeline()
        PipelineDB = {}
        res_uri = PipelineDB
    elif "container" in function.__name__.split("_"):
        global ContainerDB
        a = DummyContainer()
        ContainerDB = {}
        res_uri = ContainerDB
    return


def test_pipeline_init():
    assert isinstance(a, Leaf)
    assert isinstance(a.l1, Leaf)
    assert isinstance(a.l2, Leaf)
    assert a.l1 == a._modules["l1"]
    assert a.l2 == a._modules["l2"]


def test_pipeline_mutate_data():
    a.l1.internal_state = a.l1.forward(1)
    assert a.l1.internal_state == [2, 3, 4, 5]
    assert a.l2.internal_state == [5, 6, 7, 8]


def test_pipeline_container_ptr():
    assert a.l1._container_ptr == a
    assert a.l2._container_ptr == a


def test_pipeline_serialize_recursively():
    bin = a.serialize()
    obj = pickle.loads(bin)
    assert obj._modules["l1"] == a.l1._get_uid_base_case()
    assert obj._modules["l2"] == a.l2._get_uid_base_case()


def test_pipeline_save_data():
    uid_old = a.save_leaf(res_uri)

    a.l1.internal_state = a.l1.forward(1)
    uid_new = a.save_leaf(res_uri)

    b = DummyPipeline(l1=[], l2=[])
    b = b.load_leaf(uid_old, res_uri)
    assert b.l1.internal_state == [1, 2, 3, 4]
    assert b.l1._container_ptr == b
    assert b.l2._container_ptr == b
    b = b.load_leaf(uid_new, res_uri)
    assert b.l1.internal_state == [2, 3, 4, 5]
    assert b.l1._container_ptr == b
    assert b.l2._container_ptr == b


def test_pipeline_submodule_pointers():
    a.l3 = a.l2
    assert a.l2 == a.l3


def test_container_call_inner():
    assert a.l1.retrieve_metadata() == 42
    a.metadata = 16
    assert a.l1.retrieve_metadata() == 16


if __name__ == "__main__":
    setup_function(None)
