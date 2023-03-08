from auto_disc.newarch.wrappers.TransformWrapper import TransformWrapper
from leaf.leaf import DictLocator


def test__init__():
    id = TransformWrapper()
    assert isinstance(id, TransformWrapper)


def test_map():
    input = {"a": 1, "b": 2}
    tw = TransformWrapper(
        wrapped_keys=["a", "b"], posttransform_keys=["b", "a"])
    output = tw.map(input)
    assert output == {"a": 2, "b": 1}
    assert input == {"a": 1, "b": 2}


def test_map_missing():
    input = {"data": 1}
    tw = TransformWrapper(
        wrapped_keys=["output"], posttransform_keys=["data"])
    output = tw.map(input)
    assert output == input


def test_saveload():
    tw = TransformWrapper(
        wrapped_keys=["a", "b"], posttransform_keys=["b", "a"])

    db = {}
    tw.locator = DictLocator(db)
    leaf_uid = tw.save_leaf()

    tw2 = tw.load_leaf(leaf_uid)

    del tw.locator
    del tw2.locator
    assert tw.__dict__ == tw2.__dict__
