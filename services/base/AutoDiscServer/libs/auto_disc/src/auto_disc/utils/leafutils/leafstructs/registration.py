"""
Helper functions for doing namespace resolution. See `test_registration.py` for usage demonstration.
NOTE: Will be deprecated later.
"""
import importlib
import pkgutil
from pydoc import locate as locate_cls

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
    module = importlib.import_module("adtool_custom." + submodule)
    it = pkgutil.walk_packages(module.__path__)
    module_name_list = [el.name for el in it]
    return module_name_list


def get_default_modules(submodule: str) -> dict:
    # TODO: don't hardcode this
    return _REGISTRATION.get(submodule)


def get_modules(submodule: str):
    pass
