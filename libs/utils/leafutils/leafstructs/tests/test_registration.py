from leafutils.leafstructs.registration import get_path_from_cls, get_cls_from_path


def test_get_cls_from_path():
    path = "auto_disc.explorers.imgep_explorer.IMGEPExplorer"
    cls = get_cls_from_path(path)
    from auto_disc.explorers.imgep_explorer import IMGEPExplorer as compare_cls
    assert cls == compare_cls


def test_get_path_from_cls():
    from auto_disc.explorers.imgep_explorer import IMGEPExplorer as compare_cls
    compare_path = get_path_from_cls(compare_cls)
    path = "auto_disc.explorers.imgep_explorer.IMGEPExplorer"
    assert compare_path == path
