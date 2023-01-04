from leaf.leaf import Leaf
from leafutils.leafstructs.stateless import StatelessService
from leafutils.leafstructs.service import provide_leaf_as_service


class PlusOffset(Leaf):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + 1


def setup_function(function):
    global plusone, DB
    plusone = PlusOffset()
    DB = {}


def test_StatelessService():
    plus = provide_leaf_as_service(plusone, StatelessService)
    leaf_uid = plus.save_leaf(DB)
    assert leaf_uid == ""
    newplus = PlusOffset()
    newplus_overloaded = provide_leaf_as_service(
        newplus, StatelessService)
    assert newplus_overloaded.forward(1) == newplus_overloaded.forward(1) == 2
