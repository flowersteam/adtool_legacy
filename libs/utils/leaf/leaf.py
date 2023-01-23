from typing import Union, Tuple, Dict, Any, NewType, Optional
from hashlib import sha1
import pickle
# for dynamic discovery and loading of Python classes
from pydoc import locate

LeafUID = NewType("LeafUID", str)


class Leaf:

    def __init__(self) -> None:
        self._modules: Dict[str, Union['Leaf', 'LeafUID']] = {}
        self.name: str = ""
        self.locator: Locator = StatelessLocator()

    def __getattr__(self, name: str) -> Union[Any, 'Leaf']:
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

    def __setattr__(self, name: str, value: Union[Any, 'Leaf']) -> None:

        if isinstance(value, Leaf):

            self._bind_submodule_to_self(name, value)

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

    def _bind_submodule_to_self(self,
                                submodule_name: str,
                                submodule: 'Leaf') -> None:
        """
        Sets pointers and default locator initialization of declared submodules
        """
        # set leaf in module dict, and in subleaf instance variable
        self._modules[submodule_name] = submodule
        submodule.name = submodule_name

        # store pointer to parent container
        submodule._set_attr_override("_container_ptr", self)

        # default initialization of locator resource_uri
        if isinstance(submodule.locator, StatelessLocator):
            submodule.locator = submodule._retrieve_parent_locator()

        return

    def _retrieve_parent_locator(self) -> 'Locator':
        module = self
        while getattr(module, "_container_ptr", None) is not None:
            module = module._container_ptr
        return module.locator

    def _get_uid_base_case(self) -> 'LeafUID':
        """
        Retrieves the UID of the Leaf, which depends on the Locator,
        non-recursively (i.e., the base case)
        """

        # NOTE: This function is non-recursive, as .serialize() is recursive
        bin = self.serialize()
        uid = self.locator.hash(bin)

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

        # strip Locator object
        # which is necessary as the meaningful data does not depend on this

        # failsafe to set the default init in case something weird happens
        old_locator = getattr(self, "locator", StatelessLocator())
        if not isinstance(old_locator, StatelessLocator):
            self.locator = StatelessLocator()

        bin = pickle.dumps(self)

        # restore variables
        self._set_attr_override("_modules", old_modules)
        if old_container_ptr is not None:
            self._set_attr_override("_container_ptr", old_container_ptr)
        if not isinstance(old_locator, StatelessLocator):
            self._set_attr_override("locator", old_locator)

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

    def save_leaf(self, resource_uri: str = "", *args, **kwargs) -> 'LeafUID':
        """
        Save entire structure of object. The suggested way to customize 
        behavior is overloading serialize() and create_locator() 
        """
        # check stateless
        if isinstance(self.locator, StatelessLocator):
            return LeafUID('')

        # recursively save contained leaves
        for m in self._modules.values():
            if isinstance(m, str):
                raise ValueError(
                    "The modules are corrupted and are not of type Leaf.")
            else:
                m.save_leaf(resource_uri)

        # save this leaf
        bin = self.serialize()
        # override default initialization in Locator
        if resource_uri != '':
            self.locator.resource_uri = resource_uri
        uid = self.locator.store(bin, *args, **kwargs)
        print(f"Stored {uid}")

        return uid

    def load_leaf(self, uid: 'LeafUID', resource_uri: str = '') -> 'Leaf':
        """ Load entire structure of object, not mutating self """
        # check stateless
        if isinstance(self.locator, StatelessLocator):
            return self

        # TODO: this doesn't work(?) if we dynamically overload
        # the locator related methods in a leaf
        # or if necessary metadata is in the instance variables
        # instead of the class variables

        # TODO: perhaps think about the asymmetry of the
        # recursive save/load between this pair and
        # the serialize/deserialize pair

        # override default initialization in Locator
        if resource_uri != '':
            self.locator.resource_uri = resource_uri

        # TODO: load locator dependencies along with the leaf dependencies.
        # ATTENTION: This should happen automatically if they are within the
        # same module namespace but perhaps needs to do more work otherwise.

        bin = self.locator.retrieve(uid)
        loaded_obj = self.deserialize(bin, resource_uri)

        return loaded_obj


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
