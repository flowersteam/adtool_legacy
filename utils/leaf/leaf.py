from typing import *
from hashlib import sha1
import pickle
# for dynamic discovery and loading of Python classes
from pydoc import locate


def get_qualified_class_path(inst):
    qualified_class_name = inst.__class__.__qualname__
    module_name = inst.__class__.__module__
    class_path = module_name + "." + qualified_class_name
    return class_path


class Leaf:

    def create_locator(self, bin, *args, **kwargs) -> 'Locator':
        raise NotImplementedError()

    def store_locator(self, loc: 'Locator') -> None:
        """ Store Locator in persistent storage with relationship to the Leaf """
        raise NotImplementedError()

    @classmethod
    def retrieve_locator(cls, leaf_id: str) -> 'Locator':
        """ Retrieves all Locator objects associated with the Leaf from persistent storage """
        raise NotImplementedError()

    def __init__(self) -> None:
        self._modules = {}
        self.uid = None

    def __getattr__(self, name: str) -> Union[object, 'Leaf']:
        """ __getattr__ is called as a fallback in case regular attribute name resolution fails, which will happen with modules """
        if "_modules" in self.__dict__.keys():
            if name in self._modules.keys():
                return self._modules[name]
        raise AttributeError("Could not get attribute.")

    def __setattr__(self, name: str, value: Union[object, 'Leaf']) -> None:
        if isinstance(value, Leaf):
            self._modules[name] = value
        else:
            super().__setattr__(name, value)
        return

    def __delattr__(self, name: str) -> None:
        if name in self._modules:
            del self._modules[name]
        else:
            super().__delattr__(name)

    def _parse_leaf_hash(hash: str) -> Tuple[type, str]:
        """ Parses hashes with form module.class|sha1hex """
        class_path, _ = hash.split("|")
        return_class = locate(class_path)
        return (return_class, hash)

    def _update_uid(self, bin: bytes = None) -> None:
        if self.__module__ == "__main__":
            raise Exception(
                "Please don't run anything in __main__ namespace. Try packaging your custom module and running it from a wrapper script.")
        if bin == None:
            bin = self.serialize()
        self.uid = get_qualified_class_path(
            self) + "|" + sha1(bin).hexdigest()

        return self.uid

    def serialize(self) -> bytes:
        """ Serializes object to pickle, turning all submodules into uniquely identifiable hashes """
        # recursively pointerize all submodules
        old_modules = self._modules
        modules_by_ref = {k: v.uid for (k, v) in old_modules.items()}
        self._modules = modules_by_ref

        bin = pickle.dumps(self)

        self._modules = old_modules
        return bin

    @classmethod
    def deserialize(cls, bin: bytes) -> 'Leaf':
        """ Restores object from pickle, dereferencing submodule unique IDs to their respective objects """
        new_leaf = pickle.loads(bin)

        # recursively deserialize submodules by pointer indirection
        modules = {}
        for (m_str, m_ref) in new_leaf._modules.items():
            m = Leaf.load_leaf(m_ref)
            modules[m_str] = m

        new_leaf._modules = modules
        return new_leaf

    def save_leaf(self, *args, **kwargs) -> None:
        """ Save entire structure of object. The suggested way to customize behavior is overloading serialize() and create_locator() """
        # recursively save contained leaves
        for m in self._modules.values():
            m.save_leaf()

        # save this leaf
        bin = self.serialize(*args, **kwargs)
        loc = self.create_locator(bin, *args, **kwargs)
        loc.store(bin)

        # update uid before storing. NOTE: uid is not updated when internal state is changed
        self._update_uid(bin)
        self.store_locator(loc)
        return

    @classmethod
    def load_leaf(cls, hash: str) -> 'Leaf':
        """ Load entire structure of object """
        loaded_cls, hash = Leaf._parse_leaf_hash(hash)

        # TODO: perhaps think about the asymmetry of the recursive save/load between this pair and the serialize/deserialize pair
        locator = loaded_cls.retrieve_locator(hash)
        # TODO: load locator dependencies along with the leaf dependencies. ATTENTION: This should happen automatically if they are within the same module namespace but perhaps needs to do more work otherwise.
        bin = locator.retrieve()
        loaded_obj = loaded_cls.deserialize(bin)
        return loaded_obj


class Locator:
    def __init__(self, *args, **kwargs):
        """ Store all stateful information needed to locate the resource in instance variables """
        raise NotImplementedError

    def store(self, bin: bytes) -> None:
        """ Stores bin which generated it somewhere, perhaps on network """
        raise NotImplementedError

    def retrieve(self) -> bytes:
        """ Retrieve bin from somewhere, perhaps on network """
        raise NotImplementedError

    def serialize(self) -> bytes:
        """ Serializes locator to pickle """
        return pickle.dumps(self)

    @classmethod
    def deserialize(cls, bin: bytes) -> 'Locator':
        """ Restores object from pickle """
        return pickle.loads(bin)
