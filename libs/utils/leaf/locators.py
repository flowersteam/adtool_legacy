from leaf.leafuid import LeafUID
from hashlib import sha1


class Locator:
    """
    Class which handles data management and the saving/loading of binary data.
    Should have at least an instance variable called `resource_uri`, although
    it may be an empty string.
    """

    def __init__(self, resource_uri: str = '', *args, **kwargs):
        self.resource_uri = resource_uri
        raise NotImplementedError

    def store(self, bin: bytes) -> 'LeafUID':
        """ 
        Stores bin which generated it somewhere, perhaps on network,
        and returns (redundantly) the hash for the object.
        """
        raise NotImplementedError

    def retrieve(self, uid: 'LeafUID') -> bytes:
        """ 
        Retrieve bin identified by uid from somewhere, 
        perhaps on network 
        """
        raise NotImplementedError

    @staticmethod
    def hash(bin: bytes) -> 'LeafUID':
        """ Hashes the bin to give Leaf a uid"""
        return LeafUID(sha1(bin).hexdigest())


class StatelessLocator(Locator):
    """
    Default Locator class, for stateless modules
    """

    def __init__(self, resource_uri: str = '', *args, **kwargs):
        self.resource_uri = resource_uri

    def store(self, bin: bytes) -> 'LeafUID':
        raise Exception("This module is stateless.")

    def retrieve(self, uid: 'LeafUID') -> bytes:
        raise Exception("This module is stateless.")


class DictLocator(Locator):
    """
    Locator only for testing purposes, using a Python dict
    """

    def __init__(self, resource_uri):
        self.resource_uri = resource_uri

    def store(self, bin: bytes) -> 'LeafUID':
        uid = self.hash(bin)
        self.resource_uri[str(uid)] = bin
        return uid

    def retrieve(self, uid: 'LeafUID') -> bytes:
        return self.resource_uri[str(uid)]
