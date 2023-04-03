from auto_disc.newarch.wrappers.TransformWrapper import TransformWrapper
from leaf.Leaf import Leaf
from typing import Dict, List


class IdentityWrapper(TransformWrapper):
    """
    Wrapper which passes the input without mutation.
    """

    def __init__(self, premap_keys: List[str] = []) -> None:
        super().__init__()

    def map(self, input: Dict) -> Dict:
        return super().map(input)
