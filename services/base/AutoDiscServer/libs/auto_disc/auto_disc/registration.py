"""
Helper functions for doing namespace resolution. See `test_registration.py` for usage demonstration.
NOTE: Will be deprecated later.
"""
from pydoc import locate as locate_cls


def get_path_from_cls(cls: type) -> str:
    """
    Returns the fully qualified class path, for use with dynamic imports.
    """
    qualified_class_name = cls.__qualname__
    module_name = cls.__module__
    class_path = module_name + "." + qualified_class_name
    return class_path


def get_cls_from_path(cls_path: str) -> object:
    """
    Returns the class pointed to by a fully qualified class path,
    importing along the way.
    """
    return locate_cls(cls_path)
