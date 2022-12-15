from leaf.leaf import Leaf, get_qualified_class_path
from typing import List, Any
from functools import partial


def provide_leaf_as_service(
        object: Any, leaf_cls: Leaf,
        overridden_attr: List['str'] = None
) -> Any:
    if overridden_attr is None:
        overridden_attr = ["create_locator",
                           "store_locator",
                           "retrieve_locator"]

    # override methods in mutable state dict of object
    for name in overridden_attr:
        new_attr = getattr(leaf_cls, name)

        if callable(new_attr):
            # need __get__ here to bound the method from the class to the right object
            setattr(object, name, new_attr.__get__(object))
        else:
            setattr(object, name, new_attr)

    # have to pass an instance of leaf_cls for this to work, may break later?
    class_path_override = get_qualified_class_path(leaf_cls.__new__(leaf_cls))

    # override _update_uid with our provided service
    object.__dict__["_update_uid"] = partial(
        object._update_uid,
        class_path=class_path_override)

    return object
