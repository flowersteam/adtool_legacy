from typing import Union, Tuple, Dict, Any, NewType
from hashlib import sha1
import pickle
# for dynamic discovery and loading of Python classes
from pydoc import locate

LeafUID = NewType("LeafUID", str)


class Leaf:

    @classmethod
    def create_locator(cls,
                       resource_uri: str = "", *args, **kwargs
                       ) -> 'Locator':
        raise NotImplementedError()

    def __init__(self) -> None:
        self._modules: Dict[str, Union['Leaf', 'LeafUID']] = {}

    def __getattr__(self, name: str) -> Union[object, 'Leaf']:
        """ 
        __getattr__ is called as a fallback in case regular 
        attribute name resolution fails, which will happen with modules 
        and global container state
        """

        # redirect submodule references
        if "_modules" in self.__dict__.keys():
            if name in self._modules.keys():
                return self._modules[name]

        # gives Leaf submodules access to global metadata
        if name == "_container_state":
            return self._container_ptr.__dict__

        # fallback
        raise AttributeError("Could not get attribute.")

    def __setattr__(self, name: str, value: Union[object, 'Leaf']) -> None:

        if isinstance(value, Leaf):

            # set leaf in module dict
            self._modules[name] = value

            # store pointer to parent container
            value._set_attr_override("_container_ptr", self)

        else:

            super().__setattr__(name, value)

        return

    def _set_attr_override(self, name: str, value: Any) -> None:
        # use default __setattr__ to avoid recursion
        super().__setattr__(name, value)
        return

    def __delattr__(self, name: str) -> None:
        if name in self._modules:
            del self._modules[name]
        else:
            super().__delattr__(name)

    def _get_uid_base_case(self) -> 'LeafUID':
        """
        Retrieves the UID of the Leaf, which depends on the Locator,
        non-recursively (i.e., the base case)
        """

        # NOTE: This function is non-recursive, as .serialize() is recursive
        bin = self.serialize()
        loc = self.create_locator()
        uid = loc.hash(bin)

        return uid

    def serialize(self) -> bytes:
        """ 
        Serializes object to pickle, 
        turning all submodules into uniquely identifiable hashes 
        """
        # recursively pointerize all submodules
        old_modules = self._modules
        modules_by_ref = {}
        for (k, v) in old_modules.items():
            if isinstance(v, str):
                modules_by_ref[k] = v
            elif isinstance(v, Leaf):
                modules_by_ref[k] = v._get_uid_base_case()

        self._modules: Dict[str, LeafUID] = dict(modules_by_ref)

        # strip all _container_ptr references,
        # which is necessary to prevent endless recursion
        old_container_ptr = getattr(self, "_container_ptr", None)
        if old_container_ptr is not None:
            del self._container_ptr

        bin = pickle.dumps(self)

        # restore submodules and _container_ptr
        self._set_attr_override("_modules", old_modules)
        if old_container_ptr is not None:
            self._set_attr_override("_container_ptr", old_container_ptr)
        return bin

    def deserialize(self, bin: bytes, resource_uri: str = "") -> 'Leaf':
        """ 
        Restores object from pickle, 
        dereferencing module unique IDs to their 
        respective objects. Note that it is impossible to
        load a contained module and access the data of its
        container.
        """
        container_leaf = pickle.loads(bin)

        # recursively deserialize submodules by pointer indirection
        modules = {}
        for (submodule_str, submodule_ref) in container_leaf._modules.items():

            # dereference submodule pointed by submodule_ref
            submodule = self.load_leaf(submodule_ref, resource_uri)
            modules[submodule_str] = submodule

            # set submodule pointers to container
            submodule._set_attr_override("_container_ptr", container_leaf)

        container_leaf._set_attr_override("_modules", modules)

        return container_leaf

    def save_leaf(self, resource_uri: str, *args, **kwargs) -> 'LeafUID':
        """
        Save entire structure of object. The suggested way to customize 
        behavior is overloading serialize() and create_locator() 
        """
        # recursively save contained leaves
        for m in self._modules.values():
            if isinstance(m, str):
                raise ValueError(
                    "The modules are corrupted and are not of type Leaf.")
            else:
                m.save_leaf(resource_uri)

        # save this leaf
        bin = self.serialize(*args, **kwargs)
        loc = self.create_locator(resource_uri, *args, **kwargs)
        uid = loc.store(bin)

        return uid

    def load_leaf(self, uid: 'LeafUID', resource_uri: str = "") -> 'Leaf':
        """ Load entire structure of object, not mutating self """

        # TODO: this doesn't work(?) if we dynamically overload
        # the locator related methods in a leaf
        # or if necessary metadata is in the instance variables
        # instead of the class variables

        # TODO: perhaps think about the asymmetry of the
        # recursive save/load between this pair and
        # the serialize/deserialize pair

        locator = self.create_locator(resource_uri)

        # TODO: load locator dependencies along with the leaf dependencies.
        # ATTENTION: This should happen automatically if they are within the
        # same module namespace but perhaps needs to do more work otherwise.

        bin = locator.retrieve(uid)
        loaded_obj = self.deserialize(bin, resource_uri)

        return loaded_obj


class Locator:
    def __init__(self,  *args, **kwargs):
        """ 
        Store all stateful information, given in the bin argument, 
        needed to locate the resource in instance variables
        """
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
