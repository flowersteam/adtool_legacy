from typing import Any, List, Callable
from dataclasses import dataclass, is_dataclass, asdict
from auto_disc.legacy.utils.expose_config.expose_config import ExposeConfig, _handlers


def defaults(default: Any, domain: List[Any] = None, min: Any = None, max: Any = None):
    """
    The canonical constructor for the _DefaultSetting dataclass,
    means that we don't accidentally expose the expose_config method except
    by subclassing the Defaults class.
    """
    return _DefaultSetting(default, domain, min, max)


class Defaults:
    """ This class is only here for namespacing purposes. """
    @classmethod
    def expose_config(cls) -> Callable:
        """
        This decorator allows exposed config parameters to be exposed via a
        dataclass that inherits from Defaults.
        """
        # manually convert the dataclass to a dict
        # because pre-Python 3.9, decorator syntax is limited
        config_dict = cls._dataclass_to_config_dict()

        # create ExposeConfig objects to chain decorate
        decoration_chain: List[ExposeConfig] = []
        cls._wrap_config_defns(config_dict, decoration_chain)

        # return a big function composition of the decorator function
        return _compose(*decoration_chain)

    @classmethod
    def _wrap_config_defns(cls, config_dict, decoration_chain):
        """
        Takes dict of config definitions (i.e., the dict form of 
        a _DefaultSetting) and converts it into a list of ExposeConfig objects
        """
        # iterate over the dict and create expose_config classes to chain
        for (k, v) in config_dict.items():
            # handle both styles of providing domain
            # this checks either the domain is not set or set to the default
            # value of None by not being constructed in _DefaultSetting
            if v.get("domain", None) is None:
                try:
                    v["domain"] = [v["min"], v["max"]]
                except KeyError:
                    raise ValueError("To expose a config, "
                                     "you must provide either "
                                     "a domain or min/max.")

            # setting up the big function composition but
            # NOTE : it doesn't actually matter what order they're called in,
            # unless the config itself is malformed
            decoration_chain.append(
                ExposeConfig(
                    name=k, default=v["default"],
                    domain=v["domain"], parent=v["parent"]
                )
            )
        return decoration_chain

    @classmethod
    def _dataclass_to_config_dict(cls):
        """ 
        This function takes a (possible recursive) Defaults dataclass and
        converts it to a (flat) dict of config definitions. It's therefore
        the caller's responsibility to avoid key collisions.
        """
        config_dict = {}

        # inner function to recurse through the dataclass
        def recurse(dc: type, parent: str):
            for (k, v) in dc.__dataclass_fields__.items():
                # unwrap from the Field object
                unwrap_v = v.default

                # recurse, noting that past the root level, all values are
                # fields, which are instances of objects and not classes
                if isinstance(unwrap_v, Defaults):
                    # resolve path of recursive modules
                    parent += "." + k
                    recurse(unwrap_v, parent)
                else:
                    # base case, simply load the config dict from Defaults obj
                    if k in config_dict:
                        raise ValueError(
                            f"Config option {k} already exists.")
                    else:
                        config_dict[k] = asdict(unwrap_v)

                        # remove the leading "." from the parent
                        # in a recursive call
                        if len(parent) > 0 and parent[0] is ".":
                            parent = parent[1:]

                        config_dict[k]["parent"] = parent

        recurse(cls, "")
        return config_dict


@dataclass
class _DefaultSetting:
    default: Any
    domain: List[Any]
    min: Any
    max: Any


def _compose(*functions):
    """ Small helper function to compose functions à la pipes in FP. """
    def inner(arg):
        for f in reversed(functions):
            arg = f(arg)
        return arg
    return inner
