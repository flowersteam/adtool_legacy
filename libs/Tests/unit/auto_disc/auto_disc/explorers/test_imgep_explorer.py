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
auto_discFolderPath = "/".join(classToTestFolderPath) + "/libs/auto_disc/auto_disc"

sys.path.insert(0, os.path.dirname(auto_discFolderPath))
from auto_disc import explorers
from auto_disc.utils.spaces import BoxSpace, DictSpace
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
    space.low = torch.tensor(10)
    space.high = torch.tensor(0)
    observations = torch.tensor([12.0, 13.0])
    imgepExplorer.expand_box_goal_space(space, observations)
    #TODO
    assert (
        torch.equal(space.high, torch.tensor([12.0, 13.0], dtype=torch.float32)) 
        and torch.equal(space.low, torch.tensor([10.0, 10.0], dtype=torch.float32))
    )
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
    imgepExplorer = explorers.IMGEPExplorer()
    input_space = DummyInputWrapper().input_space
    input_space.spaces = {"embedding":0}
    input_space.sample = lambda : {"a": "sample has been executed"}
    output_space = []
    input_distance_fn = lambda x, y: torch.tensor([0,1,2])
    imgepExplorer.initialize(input_space, output_space, input_distance_fn)
    history = [{"embedding":torch.zeros(5)}]
    target_goal = torch.zeros(10)
    source_policy_idx = imgepExplorer._get_source_policy_idx(target_goal, history)
    assert torch.equal(source_policy_idx, torch.tensor(0))

def test_get_source_policy_idx_3():
    imgepExplorer = explorers.IMGEPExplorer()
    input_space = DummyInputWrapper().input_space
    input_space.spaces = {"embedding":0}
    input_space.sample = lambda : {"a": "sample has been executed"}
    output_space = []
    input_distance_fn = lambda x, y: torch.tensor([0,1,2])
    imgepExplorer.initialize(input_space, output_space, input_distance_fn)
    history = [{"embedding":torch.zeros(5)}]
    target_goal = torch.zeros(10)
    imgepExplorer.config.source_policy_selection_type = 'somethingToMakeError'
    with unittest.TestCase.assertRaises(Exception, ValueError) as context: 
        source_policy_idx = imgepExplorer._get_source_policy_idx(target_goal, history)
    assert "Unknown source policy selection type 'somethingToMakeError' in the configuration!" == context.exception.args[0]
#endregion

#region emit

def test_emit_1():
    imgepExplorer = explorers.IMGEPExplorer()
    input_space = DummyInputWrapper().input_space
    input_space.spaces = {"embedding":0}
    input_space.sample = lambda : {"a": "sample has been executed"}
    output_space = DummyInputWrapper().input_space
    output_space.sample = lambda : "sample has been executed"
    input_distance_fn = lambda x, y: torch.tensor([0,1,2])
    imgepExplorer.initialize(input_space, output_space, input_distance_fn)
    run_parameters = imgepExplorer.emit()
    assert run_parameters == "sample has been executed"

def test_emit_2():
    imgepExplorer = explorers.IMGEPExplorer()
    input_space = DummyInputWrapper().input_space
    input_space.spaces = {"embedding":0}
    input_space.sample = lambda : {"a": "sample has been executed"}
    output_space = DummyInputWrapper().input_space
    output_space.sample = lambda : "sample has been executed"
    input_distance_fn = lambda x, y: torch.tensor([0,1,2])
    imgepExplorer.initialize(input_space, output_space, input_distance_fn)

    imgepExplorer._output_space.mutate = lambda x : "mutate has been executed"
    imgepExplorer.CURRENT_RUN_INDEX = 5
    imgepExplorer.config.num_of_random_initialization = 2
    
    imgepExplorer._access_history = lambda :{"input":[{"embedding":torch.zeros(5)}], 0:{'output' : "what's you need"}}
    imgepExplorer._get_next_goal = lambda : torch.zeros(10)
    run_parameters = imgepExplorer.emit()
    
    assert run_parameters == "mutate has been executed"

#endregion