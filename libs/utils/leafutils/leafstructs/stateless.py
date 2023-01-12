from typing import Any, Dict
from leaf.leaf import Leaf, Locator, LeafUID


class StatelessService(Leaf):
    """
    Service implementation for stateless functions. Can be inherited from or
    dynamically injected.
    NOTE: Essentially does nothing except mimick the interface.
    """

    def save_leaf(self, resource_uri: str, *args, **kwargs) -> 'LeafUID':
        return LeafUID('')

    def load_leaf(self, uid: 'LeafUID', resource_uri: str = "") -> 'Leaf':
        return self
