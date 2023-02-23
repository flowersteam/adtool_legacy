from typing import Union, Tuple, Dict, Any, List
import pickle
# for dynamic discovery and loading of Python classes
from pydoc import locate
from leaf.locators import Locator, StatelessLocator
from leaf.leafuid import LeafUID
from leaf.locators import DictLocator, FileLocator
from leafutils.leafstructs.registration import \
    get_path_from_cls, get_cls_from_path


def prune_state(state_vars: Dict[str, Any]):
    """
    Decorator allowing specification of what instance variables of a Leaf
    to ignore during the serialization call.
    """
    def deco(serialize):
        def inner(self, *args, **kwargs) -> bytes:

            old_vars = {}

            # store old variables
            for name in state_vars.keys():
                attr_ptr = getattr(self, name, None)
                old_vars[name] = attr_ptr
                # clear the namespace
                if attr_ptr is not None:
                    delattr(self, name)

            bin = serialize(self, *args, **kwargs)

            # restore variables
            for (name, attr_ptr) in old_vars.items():

                # if variable was defined, restore it
                if attr_ptr is not None:
                    self._set_attr_override(name, attr_ptr)
                # else, set the default
                else:
                    self._set_attr_override(name, state_vars[name])

            return bin

        return inner
    return deco


class Leaf:

    CONFIG_DEFINITION = {}

    def __init__(self) -> None:
        self._default_leaf_init()

    def _default_leaf_init(self) -> None:
        if getattr(self, "_modules", None) is None:
            self._modules: Dict[str, Union['Leaf', 'LeafUID']] = {}
        if getattr(self, "name", None) is None:
            self.name: str = ""
        if getattr(self, "locator", None) is None:
            self.locator: Locator = StatelessLocator()
        if getattr(self, "_container_ptr", None) is None:
            self._container_ptr: Any = None
        return

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

    @prune_state(state_vars={"_container_ptr": None, "logger": None})
    def serialize(self) -> bytes:
        """ 
        Serializes object to pickle, 
        turning all submodules into uniquely identifiable hashes 
        """

        # recursively pointerize all submodules
        # NOTE: in practice this routine should not be called outside
        #       of testing, as it is already done in .save_leaf()
        old_modules = self._pointerize_submodules()

        # pointerize Locator object, turning into a fully qualified import path
        old_locator = self.locator
        if not isinstance(self.locator, str):
            cls_path = get_path_from_cls(self.locator.__class__)
            self._set_attr_override("locator", cls_path)

        bin = pickle.dumps(self)

        # restore modules
        self._set_attr_override("_modules", old_modules)

        # restore Locator
        self._set_attr_override("locator", old_locator)

        return bin

    def _pointerize_submodules(self) -> Dict[str, LeafUID]:
        """
        This should not be called outside of testing
        """
        old_modules = self._modules
        modules_by_ref = {}
        for (k, v) in old_modules.items():
            if isinstance(v, str):
                modules_by_ref[k] = v
            elif isinstance(v, Leaf):
                modules_by_ref[k] = v._get_uid_base_case()
        self._modules: Dict[str, LeafUID] = dict(modules_by_ref)
        return old_modules

    def deserialize(self, bin: bytes) -> 'Leaf':
        """ 
        Restores object from pickle, without recursively deserializing
        any submodules.
        """
        container_leaf = pickle.loads(bin)
        # reinitialize to default non-set variables
        container_leaf._default_leaf_init()

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
        uid_dict = {}
        for (module_name, module_obj) in self._modules.items():
            if isinstance(module_obj, str):
                raise ValueError(
                    "The modules are corrupted and are not of type Leaf.")
            else:
                module_uid = module_obj.save_leaf(resource_uri)
                uid_dict[module_name] = module_uid

        # replace list of submodules with pointerized list
        old_modules = self._modules
        self._modules = uid_dict

        # save this leaf
        bin = self.serialize()

        # override default initialization in Locator with arg if necessary
        if resource_uri != '':
            self.locator.resource_uri = resource_uri
        uid = self.locator.store(bin, *args, **kwargs)

        # restore old_modules
        self._modules = old_modules

        return uid

    def load_leaf(self,
                  uid: 'LeafUID',
                  resource_uri: str = '',
                  *args,
                  **kwargs) -> 'Leaf':
        """ Load entire structure of object, not mutating self """
        # check stateless
        if isinstance(self.locator, StatelessLocator):
            return self

        # override default initialization in Locator
        if resource_uri != '':
            self.locator.resource_uri = resource_uri

        bin = self.locator.retrieve(uid, *args, **kwargs)
        loaded_obj = self.deserialize(bin)

        # dereference Locator path and initialize a Locator object
        locator_cls = get_cls_from_path(loaded_obj.locator)
        loaded_obj._set_attr_override("locator", locator_cls(resource_uri))

        # bootstrap full object from metadata if locators don't match
        if not isinstance(loaded_obj.locator, type(self.locator)):
            # ensure resource_uri consistency
            loaded_obj.locator.resource_uri = self.locator.resource_uri
            loaded_obj = loaded_obj.load_leaf(uid, resource_uri)

        # recursively deserialize submodules by pointer indirection
        self._load_leaf_submodules_recursively(loaded_obj, resource_uri)

        return loaded_obj

    def _load_leaf_submodules_recursively(self,
                                          container_leaf: 'Leaf',
                                          resource_uri: str):
        """
        Dereference submodule unique IDs to their respective objects. 
        Note that it is impossible to load a contained module and access 
        the data of its container, due to the direction of recursion.

        TODO: This will break in theory if separate deserialization is needed
        for different classes.
        """
        modules = {}
        for (submodule_str, submodule_ref) in container_leaf._modules.items():
            if isinstance(submodule_ref, str):
                # dereference submodule pointed by submodule_ref
                submodule = self.load_leaf(submodule_ref, resource_uri)
                modules[submodule_str] = submodule
                # set submodule pointers to container
                submodule._set_attr_override("_container_ptr", container_leaf)
            else:
                continue
        container_leaf._set_attr_override("_modules", modules)
        return

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
        if isinstance(submodule.locator, StatelessLocator) \
                and submodule.locator.resource_uri == "":
            parent_locator_class = submodule._retrieve_parent_locator_class()
            submodule.locator = parent_locator_class(self.locator.resource_uri)

        return

    def _retrieve_parent_locator_class(self) -> type:
        module = self
        while getattr(module, "_container_ptr", None) is not None:
            module = module._container_ptr
        return module.locator.__class__

    def _get_uid_base_case(self) -> 'LeafUID':
        """
        Retrieves the UID of the Leaf, which depends on the Locator,
        non-recursively (i.e., the base case)
        """

        # NOTE: This function is non-recursive, as .serialize() is recursive
        bin = self.serialize()
        uid = self.locator.hash(bin)

        return uid
