from auto_disc.input_wrappers.generic import CppnInputWrapper
from copy import deepcopy
from copyreg import pickle
import os
import sys
import pickle


classToTestFolderPath = os.path.dirname(__file__)
auto_discFolderPath = os.path.abspath(os.path.join(
    classToTestFolderPath, "../"*8 + "/libs/auto_disc/auto_disc"))
sys.path.insert(0, os.path.dirname(auto_discFolderPath))


def setup_function(function):
    global Object, output_space_shape_object, __location__

    Object = lambda **kwargs: type("Object", (), kwargs)()
    def output_space_shape_object(): return Object(shape=(256, 256))
    __location__ = os.path.realpath(os.path.join(
        os.getcwd(), os.path.dirname(__file__)))

    return


def test_map():
    with open(os.path.join(__location__, "parameters.pickle"), "rb") as parametersFile:
        parameters = pickle.load(parametersFile)
    wrapped_output_space_key = "a"
    cppn = CppnInputWrapper(wrapped_output_space_key)
    cppn.output_space = {wrapped_output_space_key: output_space_shape_object()}
    res = cppn.map(deepcopy(parameters), None)
    assert "a" not in parameters and res["genome"] == {}
