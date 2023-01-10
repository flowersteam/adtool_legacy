from copy import deepcopy
from typing import Dict
from leafutils.leafstructs.stateless import StatelessService


class IdentityWrapper(StatelessService):
    """
    Wrapper which passes the input without mutation.
    """

    def __init__(self, wrapped_key: str = None) -> None:
        super().__init__()
        # wrapped_key is not used

    def map(self, input: Dict) -> Dict:
        # must do because dicts are mutable types
        output = deepcopy(input)

        return output
