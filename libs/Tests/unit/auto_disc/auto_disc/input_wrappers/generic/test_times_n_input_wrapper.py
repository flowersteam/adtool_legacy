from leafstructs.container import provide_leaf_as_service
from auto_disc.input_wrappers.generic.times_n_input_wrapper import DummySaveService
from auto_disc.input_wrappers.generic import TimesNInputWrapper
import os
import sys
import unittest

classToTestFolderPath = os.path.dirname(__file__)
auto_discFolderPath = os.path.abspath(os.path.join(
    classToTestFolderPath, "../"*7 + "/libs/auto_disc/auto_disc"))
sys.path.insert(0, os.path.dirname(auto_discFolderPath))


def test_init():
    wrapped_output_space_key = "a"
    timesNInputWrapper = TimesNInputWrapper(wrapped_output_space_key)
    assert (
        timesNInputWrapper.config == {'n': 1}
        and timesNInputWrapper.CONFIG_DEFINITION == {'n': {'default': 1, 'type': 'INTEGER', 'min': None, 'max': None}}
        and timesNInputWrapper._wrapped_key == "a"
    )


def test_map():
    wrapped_output_space_key = "a"
    timesNInputWrapper = TimesNInputWrapper(wrapped_output_space_key)
    timesNInputWrapper.config["n"] = 10

    parameters = {"a": 5, "b": -1}
    new_parameters = timesNInputWrapper.map(parameters)
    assert new_parameters == {"a": 50, "b": -1}
    assert parameters == {"a": 5, "b": -1}


# def test_dynamic_overloading():
#     wrapped_output_space_key = "a"
#     timesNInputWrapper = TimesNInputWrapper(wrapped_output_space_key)
#     service = LocatorService()

#     # dynamic overloading
#     timesNInputWrapper.create_locator = LocatorService.create_locator.__get__(
#         timesNInputWrapper, TimesNInputWrapper)
#     timesNInputWrapper.store_locator = LocatorService.store_locator.__get__(
#         timesNInputWrapper, TimesNInputWrapper)
#     timesNInputWrapper.retrieve_locator = LocatorService.retrieve_locator.__get__(
#         timesNInputWrapper, TimesNInputWrapper)
#     assert timesNInputWrapper.create_locator is not None
#     assert timesNInputWrapper.store_locator is not None
#     assert timesNInputWrapper.retrieve_locator is not None


def test_save_load():
    wrapped_output_space_key = "a"
    timesNInputWrapper = provide_leaf_as_service(
        TimesNInputWrapper(wrapped_output_space_key), DummySaveService)

    timesNInputWrapper.save_leaf()
    saved_uid = timesNInputWrapper.uid
    del timesNInputWrapper

    new_wrapper = DummySaveService.load_leaf(saved_uid)
    assert (
        new_wrapper.config == {'n': 1}
        and new_wrapper.CONFIG_DEFINITION == {'n': {'default': 1, 'type': 'INTEGER', 'min': None, 'max': None}}
        and new_wrapper._wrapped_key == "a"
    )
