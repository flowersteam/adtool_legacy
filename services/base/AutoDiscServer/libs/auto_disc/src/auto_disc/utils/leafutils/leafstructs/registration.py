"""
Helper functions for doing namespace resolution. See `test_registration.py` for usage demonstration.
NOTE: Will be deprecated later.
"""
import importlib
import pkgutil
from pydoc import locate as locate_cls

from mergedeep import merge

_REGISTRATION = {
    'systems': {
        'PythonLenia': "auto_disc.auto_disc.systems.Lenia.Lenia",
        'LeniaCPPN': "auto_disc.auto_disc.systems.LeniaCPPN.LeniaCPPN",
        'ExponentialMixture': "auto_disc.auto_disc.systems.ExponentialMixture.ExponentialMixture",
    },
    'explorers': {
        'IMGEPExplorer': "auto_disc.auto_disc.explorers.IMGEPExplorer.IMGEPFactory",
    },
    'maps': {
        'MeanBehaviorMap': "auto_disc.auto_disc.maps.MeanBehaviorMap.MeanBehaviorMap",
        'UniformParameterMap': "auto_disc.auto_disc.maps.UniformParameterMap.UniformParameterMap",
        'LeniaStatistics': "auto_disc.auto_disc.maps.LeniaStatistics.LeniaStatistics",
    },
    'callbacks': {
        'on_discovery': {
            'expe_db': "auto_disc.auto_disc.utils.callbacks.on_discovery_callbacks.save_discovery_in_expedb.SaveDiscoveryInExpeDB",
            'disk': "auto_disc.auto_disc.utils.callbacks.on_discovery_callbacks.save_discovery_on_disk.SaveDiscoveryOnDisk"
        },
        'on_cancelled': {},
        'on_error': {},
        'on_finished': {},
        'on_saved': {
            'base': "auto_disc.auto_disc.utils.callbacks.on_save_callbacks.save_leaf_callback.SaveLeaf",
            'expe_db': "auto_disc.auto_disc.utils.callbacks.on_save_callbacks.save_leaf_callback_in_expedb.SaveLeafExpeDB",
        },
        'on_save_finished': {
            'base': "auto_disc.auto_disc.utils.callbacks.on_save_finished_callbacks.generate_report_callback.GenerateReport",
        },
        'interact': {},
    },
    'logger_handlers': {
        'logFile': "auto_disc.auto_disc.utils.logger.handlers.file_handler.SetFileHandler"
    }
}


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


def get_custom_modules(submodule: str) -> list:
    lookup_prefix = "adtool_custom." + submodule
    module = importlib.import_module(lookup_prefix)
    path_prefix = lookup_prefix + "."
    it = pkgutil.iter_modules(module.__path__, prefix=path_prefix)
    module_name_list = {el.name.split(".")[-1]: el.name for el in it}
    return module_name_list


def get_default_modules(submodule: str) -> dict:
    # NOTE: we only return the fully qualified module path instead of
    # module itself to avoid polluting the namespace with imports

    # TODO: don't hardcode this
    return _REGISTRATION.get(submodule)


def get_modules(submodule: str):
    return merge(get_default_modules(submodule), get_custom_modules(submodule))
