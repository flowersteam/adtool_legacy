from auto_disc.utils.leafutils.leafstructs.registration import (
    get_cls_from_name, get_cls_from_path, get_custom_modules,
    get_default_modules, get_modules, get_path_from_cls)


def test_get_cls_from_path():
    path = "auto_disc.legacy.explorers.imgep_explorer.IMGEPExplorer"
    cls = get_cls_from_path(path)
    from auto_disc.legacy.explorers.imgep_explorer import \
        IMGEPExplorer as compare_cls
    assert cls == compare_cls

    path = "auto_disc.auto_disc.maps.lenia.LeniaStatistics"
    cls = get_cls_from_path(path)
    from auto_disc.auto_disc.maps.lenia.LeniaStatistics import \
        LeniaStatistics as compare_cls
    assert cls == compare_cls

    path = "auto_disc.auto_disc.maps.MeanBehaviorMap"
    cls = get_cls_from_path(path)
    from auto_disc.auto_disc.maps import MeanBehaviorMap as compare_cls
    assert cls == compare_cls


def test_get_path_from_cls():
    from auto_disc.legacy.explorers.imgep_explorer import \
        IMGEPExplorer as compare_cls
    compare_path = get_path_from_cls(compare_cls)
    path = "auto_disc.legacy.explorers.imgep_explorer.IMGEPExplorer"
    assert compare_path == path

    from auto_disc.auto_disc.maps.lenia.LeniaStatistics import \
        LeniaStatistics as compare_cls
    compare_path = get_path_from_cls(compare_cls)
    # here we see that `get_path_from_cls` gives an explicit FQDN
    # instead of using imports from `__init__.py` files
    path = "auto_disc.auto_disc.maps.lenia.LeniaStatistics.LeniaStatistics"
    assert compare_path == path

    from auto_disc.auto_disc.maps import MeanBehaviorMap as compare_cls
    compare_path = get_path_from_cls(compare_cls)
    path = "auto_disc.auto_disc.maps.MeanBehaviorMap.MeanBehaviorMap"
    assert compare_path == path


def test_get_cls_from_name():
    from auto_disc.auto_disc.explorers import IMGEPFactory
    from auto_disc.auto_disc.maps.lenia import LeniaStatistics
    from auto_disc.auto_disc.systems.ExponentialMixture import \
        ExponentialMixture
    from auto_disc.auto_disc.utils.callbacks.on_save_callbacks.save_leaf_callback import \
        SaveLeaf

    cls_name = "IMGEPExplorer"
    ad_type_name = "explorers"
    assert get_cls_from_name(
        cls_name, ad_type_name
    ) == IMGEPFactory

    cls_name = "LeniaStatistics"
    ad_type_name = "maps"
    assert get_cls_from_name(
        cls_name, ad_type_name
    ) == LeniaStatistics

    cls_name = "ExponentialMixture"
    ad_type_name = "systems"
    assert get_cls_from_name(
        cls_name, ad_type_name
    ) == ExponentialMixture

    cls_name = "base"
    ad_type_name = "callbacks.on_saved"
    assert get_cls_from_name(
        cls_name, ad_type_name
    ) == SaveLeaf


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
