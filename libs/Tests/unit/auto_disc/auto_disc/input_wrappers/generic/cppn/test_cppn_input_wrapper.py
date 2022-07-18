from copy import deepcopy
from copyreg import pickle
import os
import sys
import pickle
import unittest

import numpy as np

classToTestFolderPath = os.path.abspath(__file__)
classToTestFolderPath = classToTestFolderPath.split('/')
classToTestFolderPath = classToTestFolderPath[0:classToTestFolderPath.index("AutomatedDiscoveryTool")+1]
auto_discFolderPath = "/".join(classToTestFolderPath) + "/libs/auto_disc/auto_disc"

sys.path.insert(0, os.path.dirname(auto_discFolderPath))
import auto_disc
from auto_disc.input_wrappers.generic import CppnInputWrapper

Object = lambda **kwargs: type("Object", (), kwargs)()
output_space_shape_object = lambda : Object(shape=(256,256))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def test_map():
    with open(os.path.join(__location__, "parameters.pickle"), "rb") as parametersFile :
        parameters = pickle.load(parametersFile)
    wrapped_output_space_key = "a"
    cppn = CppnInputWrapper(wrapped_output_space_key)
    cppn.output_space = {wrapped_output_space_key : output_space_shape_object()}
    res = cppn.map(deepcopy(parameters), None)
    assert "a" not in parameters and res["genome"] == {}