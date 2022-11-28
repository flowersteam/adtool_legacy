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
        and timesNInputWrapper._wrapped_output_space_key == "a"
    )


def test_map():
    wrapped_output_space_key = "a"
    timesNInputWrapper = TimesNInputWrapper(wrapped_output_space_key)
    timesNInputWrapper.config["n"] = 10

    parameters = {"a": 5, "b": -1}
    new_parameters = timesNInputWrapper.map(parameters)
    assert new_parameters == {"a": 50, "b": -1}
    assert parameters == {"a": 5, "b": -1}
