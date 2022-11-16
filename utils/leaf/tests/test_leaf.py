from leaf.leaf import *
import pickle
from hashlib import sha1


class DummyDB():
    LocDB = {}
    ResDB = {}


class DummyModule(Leaf):
    locator_table = DummyDB.LocDB

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

    def create_locator(self, bin):
        return DummyLocator(bin)

    def store_locator(self, loc):
        self.locator_table[self.uid] = loc.serialize()
        return

    @classmethod
    def retrieve_locator(cls, leaf_id):
        return Locator.deserialize(cls.locator_table[leaf_id])


class DummyLocator(Locator):
    def __init__(self, bin):
        self.table = DummyDB.ResDB
        self.uid = sha1(bin).hexdigest()

    def store(self, bin):
        self.table[self.uid] = bin
        return

    def retrieve(self):
        return self.table[self.uid]


def setup_function(function):
    global a
    a = DummyModule([1, 2, 3, 4])
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
    b = a.__class__.deserialize(bin)
    assert a.__dict__ == b.__dict__


def test_leaf_create_locator():
    loc = a.create_locator(a.serialize())
    assert isinstance(loc, Locator)


def test_leaf_store_locator_retrieve_locator():
    bin = a.serialize()
    a._update_uid(bin)
    # must always set uid before create_locator() call
    assert a.uid is not None
    loc1 = a.create_locator(bin)
    a.store_locator(loc1)
    loc2 = a.retrieve_locator(a.uid)

    assert loc1.__dict__ == loc2.__dict__


def test_locator_store_retrieve():
    bin1 = a.serialize()
    loc = a.create_locator(bin1)
    loc.store(bin1)

    bin2 = loc.retrieve()
    assert bin1 == bin2


def test_leaf_save():
    a.save_leaf()
    uid = a.uid

    b = DummyModule.load_leaf(uid)
    assert b.internal_state == [1, 2, 3, 4]
    a.internal_state.append(5)
    a.internal_state = a.forward(1)

    # test uid updates after save_leaf
    a.save_leaf()
    new_uid = a.uid
    assert new_uid != uid

    c = DummyModule.load_leaf(new_uid)
    assert c.internal_state == [2, 3, 4, 5, 6]
    assert b.internal_state == [1, 2, 3, 4]
    assert a.internal_state == c.internal_state != b.internal_state


if __name__ == "__main__":
    setup_function(None)
    test_leaf_store_locator_retrieve_locator()
