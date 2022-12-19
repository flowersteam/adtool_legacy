from leaf.leaf import Leaf, Locator
import pickle
from hashlib import sha1


class DummyModule(Leaf):
    def __init__(self, s=None):
        super().__init__()
        self.internal_state = s

    def forward(self, x):
        return [x+y for y in self.internal_state]

    # def serialize(self):
    #     """ Dumps state to a pickle object """
    #     return pickle.dumps(self.internal_state)

    # @classmethod
    # def deserialize(cls, bin: bytes) -> 'DummyModule':
    #     """ Restores object from pickle """
    #     loaded_obj = cls(pickle.loads(bin))
    #     return loaded_obj

    @classmethod
    def create_locator(self, resource_uri: str = ""):
        return DummyLocator(resource_uri)


class DummyLocator(Locator):
    def __init__(self, resource_uri):
        self.table = resource_uri

    def store(self, bin):
        uid = self.hash(bin)
        self.table[uid] = bin
        return uid

    def retrieve(self, uid):
        return self.table[uid]


def setup_function(function):
    global a, ResDB, res_uri
    a = DummyModule([1, 2, 3, 4])
    ResDB = {}
    res_uri = ResDB
    return


def test_leaf_init():
    assert a._modules == {}
    assert a.internal_state


def test_leaf_serialize():
    bin = a.serialize()
    b = pickle.loads(bin)
    assert a.__dict__ == b.__dict__


def test_leaf_deserialize():
    bin = a.serialize()
    b = a.deserialize(bin)
    assert a.__dict__ == b.__dict__


def test_leaf_create_locator():
    loc = a.create_locator(res_uri)
    assert isinstance(loc, Locator)
    assert loc.table == ResDB


def test_locator_store_retrieve():
    bin1 = a.serialize()
    loc = a.create_locator(res_uri)
    uid = loc.store(bin1)

    bin2 = loc.retrieve(uid)
    assert bin1 == bin2


def test_leaf_save():
    uid = a.save_leaf(res_uri)

    b = DummyModule()
    b = b.load_leaf(uid, res_uri)
    assert b.internal_state == [1, 2, 3, 4]
    a.internal_state.append(5)
    a.internal_state = a.forward(1)

    # test uid updates after save_leaf
    new_uid = a.save_leaf(res_uri)
    assert new_uid != uid

    c = b.load_leaf(new_uid, res_uri)
    assert c.internal_state == [2, 3, 4, 5, 6]
    assert b.internal_state == [1, 2, 3, 4]
    assert a.internal_state == c.internal_state != b.internal_state


if __name__ == "__main__":
    setup_function(None)
