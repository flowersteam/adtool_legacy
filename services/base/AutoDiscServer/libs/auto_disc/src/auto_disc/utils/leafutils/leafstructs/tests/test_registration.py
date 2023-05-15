from auto_disc.utils.leafutils.leafstructs.registration import (
    get_cls_from_path, get_custom_modules, get_default_modules, get_modules,
    get_path_from_cls)


def test_get_cls_from_path():
    path = "auto_disc.legacy.explorers.imgep_explorer.IMGEPExplorer"
    cls = get_cls_from_path(path)
    from auto_disc.legacy.explorers.imgep_explorer import \
        IMGEPExplorer as compare_cls
    assert cls == compare_cls


def test_get_path_from_cls():
    from auto_disc.legacy.explorers.imgep_explorer import \
        IMGEPExplorer as compare_cls
    compare_path = get_path_from_cls(compare_cls)
    path = "auto_disc.legacy.explorers.imgep_explorer.IMGEPExplorer"
    assert compare_path == path


def test_get_custom_modules():
    assert get_custom_modules("systems") == {}
    assert get_custom_modules("explorers") == {}
    assert get_custom_modules("maps") == {}
    assert get_custom_modules("callbacks") == {}


def test_get_default_modules():
    assert set(get_default_modules("systems").keys()) == \
        set(["ExponentialMixture", "PythonLenia",
             "LeniaCPPN"])
    assert set(get_default_modules("explorers").keys()) == \
        set(["IMGEPExplorer"])
    assert set(get_default_modules("maps").keys()) == \
        set(["MeanBehaviorMap", "UniformParameterMap", "LeniaStatistics"])
    assert set(get_default_modules("callbacks").keys()) == \
        set(["on_discovery",
             "on_cancelled",
             "on_error",
             "on_finished",
             "on_saved",
             "on_save_finished",
             "interact"])


def test_get_modules():
    assert set(get_modules("systems").keys()) == \
        set(["ExponentialMixture", "PythonLenia",
             "LeniaCPPN"])
