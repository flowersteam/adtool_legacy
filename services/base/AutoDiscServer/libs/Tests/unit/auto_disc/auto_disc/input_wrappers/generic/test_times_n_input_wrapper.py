from auto_disc.input_wrappers.generic import TimesNInputWrapper
import os
import sys
import unittest

classToTestFolderPath = os.path.dirname(__file__)
auto_discFolderPath = os.path.abspath(os.path.join(
    classToTestFolderPath, "../"*7 + "/libs/auto_disc/auto_disc"))
sys.path.insert(0, os.path.dirname(auto_discFolderPath))


def test_init_1():
    wrapped_output_space_key = "a"
    timesNInputWrapper = TimesNInputWrapper(wrapped_output_space_key)
    assert (
        timesNInputWrapper.config == {'n': 1}
        and timesNInputWrapper.CONFIG_DEFINITION == {'n': {'default': 1, 'type': 'INTEGER', 'min': None, 'max': None}}
        and timesNInputWrapper.CURRENT_RUN_INDEX == 0
        and timesNInputWrapper.wrapped_output_space_key == "a"
    )


def test_init_2():
    wrapped_output_space_key = 0
    with unittest.TestCase.assertRaises(Exception, TypeError) as context:
        timesNInputWrapper = TimesNInputWrapper(wrapped_output_space_key)
    assert "wrapped_output_space_key must be a single string indicating the key of the space to wrap." == context.exception.args[
        0]


def test_map_1():
    wrapped_output_space_key = "a"
    timesNInputWrapper = TimesNInputWrapper(wrapped_output_space_key)
    parameters = {"Times1_a": 5}
    parameters = timesNInputWrapper.map(parameters, None)
    assert parameters == {"a": 5}
