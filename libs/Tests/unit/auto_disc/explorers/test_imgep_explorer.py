#region import
from logging import exception
import os 
import sys
from copy import deepcopy
from unittest import mock
import unittest
from unittest.case import _AssertRaisesContext
from addict import Dict
import torch

classToTestFolderPath = os.path.abspath(__file__)
classToTestFolderPath = classToTestFolderPath.split('/')
classToTestFolderPath = classToTestFolderPath[0:classToTestFolderPath.index("AutomatedDiscoveryTool")+1]
auto_discFolderPath = "/".join(classToTestFolderPath) + "/libs/auto_disc"

sys.path.insert(0, os.path.dirname(auto_discFolderPath))
from auto_disc import explorers
from auto_disc.utils.spaces import DictSpace, BoxSpace
from auto_disc.input_wrappers.generic import DummyInputWrapper
#endregion

#region initialize

def test_initialize_1():
    imgepExplorer = explorers.IMGEPExplorer()
    input_space = Dict(spaces={"a":1})
    output_space = []
    input_distance_fn = lambda : 5
    imgepExplorer.initialize(input_space, output_space, input_distance_fn)
    print('oui')
    assert imgepExplorer._outter_input_space_key == "a"

def test_initialize_2():
    imgepExplorer = explorers.IMGEPExplorer()
    input_space = Dict(spaces={"a":1, "b":2})
    output_space = []
    input_distance_fn = lambda : 5
    imgepExplorer.initialize(input_space, output_space, input_distance_fn)
    print('oui')
    assert imgepExplorer._outter_input_space_key == "a"

def test_initialize_3():
    imgepExplorer = explorers.IMGEPExplorer()
    input_space = [Dict(spaces={"a":1}), Dict(spaces={"a":1})]
    output_space = []
    input_distance_fn = lambda : 5
    with unittest.TestCase.assertRaises(Exception, NotImplementedError) as context: 
        imgepExplorer.initialize(input_space, output_space, input_distance_fn)
    assert "Only 1 vector can be accepted as input space" == context.exception.args[0]

#endregion

#region expand_box_goal_space

def test_expand_goal_space_1():
    imgepExplorer = explorers.IMGEPExplorer()
    space = BoxSpace(low=0, high=0, shape=(17,))
    observations = torch.tensor([0.0, 0.0])
    imgepExplorer.expand_box_goal_space(space, observations)
    #TODO
    assert True
#endregion

#region _get_next_goal

def test__get_next_goal_1():
    imgepExplorer = explorers.IMGEPExplorer()
    input_space = DummyInputWrapper().input_space
    input_space.spaces = {"a":0}
    input_space.sample = lambda : {"a": "sample has been executed"}
    output_space = []
    input_distance_fn = lambda : 5
    imgepExplorer.initialize(input_space, output_space, input_distance_fn)
    res = imgepExplorer._get_next_goal()
    assert res == "sample has been executed"

def test__get_next_goal_2():
    imgepExplorer = explorers.IMGEPExplorer()
    input_space = DummyInputWrapper().input_space
    input_space.spaces = {"a":0}
    input_space.sample = lambda : {"a": "sample has been executed"}
    output_space = []
    input_distance_fn = lambda : 5
    imgepExplorer.initialize(input_space, output_space, input_distance_fn)
    imgepExplorer.config.goal_selection_type = "notRandom"

    with unittest.TestCase.assertRaises(Exception, ValueError) as context: 
        imgepExplorer._get_next_goal()
    assert "Unknown goal generation type 'notRandom' in the configuration!" == context.exception.args[0]
    
#endregion


#region _get_source_policy_idx

def test_get_source_policy_idx_1():
    pass

#endregion