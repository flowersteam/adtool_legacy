from leaf.leaf import Leaf
from auto_disc.newarch.wrappers.WrapperPipeline import WrapperPipeline
from copy import deepcopy
from typing import Dict


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
    all_wrappers = WrapperPipeline(wrappers=wrapper_list,
                                   inputs_to_save=["in"], outputs_to_save=["in"])
    assert all_wrappers.map(input) == b.map(a.map(input))
