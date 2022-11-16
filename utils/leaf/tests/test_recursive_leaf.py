from leaf.leaf import *
from leaf.tests.test_leaf import DummyModule, DummyLocator
import pickle


class DummyPipeline(Leaf):
    def __init__(self):
        super().__init__()
        self.l1 = DummyModule([1, 2, 3, 4])
        self.l2 = DummyModule([5, 6, 7, 8])

    def create_locator(self, bin: bytes) -> 'Locator':
        return DummyLocator(bin)

    def store_locator(self, loc):
        global PipelineDB
        PipelineDB[self.uid] = pickle.dumps(loc)
        return

    @classmethod
    def retrieve_locator(cls, leaf_id: str) -> 'Locator':
        global PipelineDB
        return pickle.loads(PipelineDB[leaf_id])


def setup_function(function):
    global a
    global PipelineDB
    PipelineDB = {}
    a = DummyPipeline()
    return


def test_pipeline_init():
    assert isinstance(a, Leaf)


def test_pipeline_mutate_data():
    a.l1.internal_state = a.l1.forward(1)
    assert a.l1.internal_state == [2, 3, 4, 5]
    assert a.l2.internal_state == [5, 6, 7, 8]


def test_pipeline_save_data():
    a.save_leaf()
    uid_old = a.uid

    # uid updates every save, not when internal state is mutated
    a.l1.internal_state = a.l1.forward(1)
    a.save_leaf()
    uid_new = a.uid

    b = DummyPipeline.load_leaf(uid_old)
    assert b.l1.internal_state == [1, 2, 3, 4]
    b = DummyPipeline.load_leaf(uid_new)
    assert b.l1.internal_state == [2, 3, 4, 5]


if __name__ == "__main__":
    setup_function(None)
