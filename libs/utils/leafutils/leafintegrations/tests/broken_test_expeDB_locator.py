from leafutils.leafintegrations.expeDB_locator import ExpeDBService, ExpeDBLocator
from leaf.tests.test_leaf import DummyModule
from leafutils.leafstructs.service import provide_leaf_as_service
import requests
import codecs


def setup_function(function):
    global DB_URL, dummy_module
    DB_URL = "http://127.0.0.1:5001"
    dummy_module = DummyModule([1, 2, 3, 4])

    # TODO: mock the ExpeDB responses instead of this


def teardown_function(function):
    # delete checkpoints
    entrypoint_url = DB_URL + "/checkpoint_saves" + "?checkpoint_id=100"
    requests.delete(entrypoint_url)


def _override(dummy_module_instance, db_url):
    """ override with service """
    overridden_vars = ["create_locator",
                       "retrieve_locator", "store_locator", "DB_URL"]
    modify_leaf_cls(ExpeDBService, DB_URL=db_url)
    new_dummy_module = provide_leaf_as_service(
        dummy_module_instance, ExpeDBService, overridden_attr=overridden_vars)
    return new_dummy_module


def test_ExpeDBLocator___init__():
    bin = dummy_module.serialize()
    loc = ExpeDBLocator(bin, "test_label", DB_URL)
    assert loc.db_row_uid


def test_ExpeDBLocator__initial_post():
    loc = ExpeDBLocator.__new__(ExpeDBLocator)
    loc.db_url = DB_URL
    db_row_uid = loc._initial_post()
    assert db_row_uid


def test_ExpeDBLocator_store():
    bin = dummy_module.serialize()
    loc = ExpeDBLocator(bin, "test_label", DB_URL)
    loc.store(bin)

    # test retrieved bin
    entrypoint_url = DB_URL + "/checkpoint_saves/" + loc.db_row_uid + "/test_label"
    response = requests.get(entrypoint_url)
    retrieved_bin = codecs.decode(response.text.encode(), encoding="base64")
    assert bin == retrieved_bin


def test_ExpeDBLocator_retrieve():
    bin = dummy_module.serialize()
    loc = ExpeDBLocator(bin, "test_label", DB_URL)
    loc.store(bin)

    retrieved_bin = loc.retrieve()
    assert bin == retrieved_bin


def test_ExpeDBService_override():
    new_dummy_module = _override(dummy_module, DB_URL)
    assert new_dummy_module.DB_URL == DB_URL


def test_ExpeDBService_create_locator():
    new_dummy_module = _override(dummy_module, DB_URL)
    bin = new_dummy_module.serialize()
    loc = new_dummy_module.create_locator(
        bin, "input_wrapper", new_dummy_module.DB_URL)
    assert isinstance(loc, ExpeDBLocator)


def test_ExpeDBService_store_locator():
    new_dummy_module = _override(dummy_module, DB_URL)

    bin = new_dummy_module.serialize()

    loc = new_dummy_module.create_locator(
        bin, "input_wrapper", new_dummy_module.DB_URL)
    new_dummy_module.store_locator(loc)

    # test retrieved loc
    entrypoint_url = DB_URL + "/checkpoint_saves/" + \
        loc.db_row_uid + "/input_wrapper_ref"
    response = requests.get(entrypoint_url)
    retrieved_bin = codecs.decode(response.text.encode(), encoding="base64")

    assert retrieved_bin == loc.serialize()


def test_ExpeDBService_retrieve_locator():
    new_dummy_module = _override(dummy_module, DB_URL)

    bin = new_dummy_module.serialize()
    loc = new_dummy_module.create_locator(
        bin, "input_wrapper", new_dummy_module.DB_URL)

    data_row_uid = loc.db_row_uid
    leaf_uid = data_row_uid + "/" + loc.label + "_ref"

    new_dummy_module.store_locator(loc)

    # test retrieved loc
    # NOTE: ExpeDBService here has been modified with the proper class
    retrieved_bin = ExpeDBService.retrieve_locator(leaf_uid)

    assert retrieved_bin == loc.serialize()
