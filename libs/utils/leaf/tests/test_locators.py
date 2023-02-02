from leaf.leaf import DictLocator, FileLocator
import os
import pathlib


def test_DictLocator___init__():
    db = {}
    locator = DictLocator(db)
    assert db == locator.resource_uri


def test_DictLocator_store():
    db = {}
    locator = DictLocator(db)
    bytestring = b"pasudgfpausdgfpxzucbv"

    uid = locator.store(bytestring)
    assert len(db) == 1


def test_DictLocator_retrieve():
    db = {}
    locator = DictLocator(db)
    bytestring = b"pasudgfpausdgfpxzucbv"
    fake_uid = "abcdefg"
    db[fake_uid] = bytestring

    retrieved_bin = locator.retrieve(fake_uid)

    assert retrieved_bin == bytestring


def test_FileLocator___init__():
    locator = FileLocator()
    assert locator.resource_uri != ""


def test_FileLocator_store():
    locator = FileLocator()
    bytestring = b"pasudgfpausdgfpxzucbv"

    uid = locator.store(bytestring)

    save_dir = os.path.join(os.getcwd(), uid)
    save_path = os.path.join(save_dir, "metadata")

    assert os.path.exists(save_path)
    os.remove(save_path)


def test_FileLocator_retrieve():
    bytestring = b"pasudgfpausdgfpxzucbv"
    fake_uid = "abcdefg"

    res_dir = os.getcwd()
    save_dir = os.path.join(os.getcwd(), fake_uid)
    os.mkdir(save_dir)
    save_path = os.path.join(save_dir, "metadata")

    with open(save_path, "wb") as f:
        f.write(bytestring)

    locator = FileLocator(res_dir)
    bin = locator.retrieve(fake_uid)

    assert bin == bytestring

    os.remove(save_path)
