from leaf.leafuid import LeafUID
from hashlib import sha1
import pathlib
import os


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
        raise Exception("This module is either stateless or uninitialized.")

    def retrieve(self, uid: 'LeafUID') -> bytes:
        raise Exception("This module is either stateless or uninitialized.")


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


class FileLocator(Locator):
    """
    Locator which saves modules à la Git, to the filesystem,
    with root directory specified by resource_uri
    """

    def __init__(self, resource_uri: str = ""):
        # set default to relative directory of the caller
        if resource_uri == "":
            self.resource_uri = str(os.getcwd())
        else:
            self.resource_uri = resource_uri

    def store(self, bin: bytes) -> 'LeafUID':
        uid = self.hash(bin)
        save_dir = os.path.join(self.resource_uri, str(uid))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        save_path = os.path.join(save_dir, "metadata")
        with open(save_path, "wb") as f:
            f.write(bin)

        return uid

    def retrieve(self, uid: 'LeafUID') -> bytes:
        save_dir = os.path.join(self.resource_uri, str(uid))

        save_path = os.path.join(save_dir, "metadata")
        with open(save_path, "rb") as f:
            bin = f.read()

        return bin
