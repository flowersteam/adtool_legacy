from leaf.tests.test_leaf import DummyDB, DummyLocator, DummyModule
from leaf.leaf import Leaf
from leafstructs.service import provide_leaf_as_service


class PlusOffset(Leaf):
    def __init__(self, offset):
        super().__init__()
        self.offset = offset

    def forward(self, x):
        return x + self.offset


def setup_function(function):
    global plusone
    plusone = PlusOffset(1)


def test_init():
    new_plusone = provide_leaf_as_service(plusone, DummyModule)
    assert new_plusone


def test_save_load():
    overloaded_plusone = provide_leaf_as_service(plusone, DummyModule)
    overloaded_plusone.save_leaf()
    saved_uid = overloaded_plusone.uid
    overloaded_plusone.offset = 2

    new_plusone = overloaded_plusone.load_leaf(saved_uid)
    assert new_plusone.offset == 1
    assert new_plusone.forward(2) == 3


def test_dynamic_override_class_vars():
    def override(cls, dict) -> 'type':
        cls.locator_table = dict
        return cls

    new_db = {}
    NewClass = override(DummyModule, new_db)
    assert DummyModule.locator_table == new_db

    override_list = ["create_locator",
                     "store_locator",
                     "retrieve_locator",
                     "locator_table"]

    overloaded_plusone = provide_leaf_as_service(
        plusone, DummyModule, overridden_attr=override_list)

    assert overloaded_plusone.locator_table == new_db
