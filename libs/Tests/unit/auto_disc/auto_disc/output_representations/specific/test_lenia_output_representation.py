import os
import pickle
import sys
import unittest

import torch

classToTestFolderPath = os.path.dirname(__file__)
auto_discFolderPath = os.path.abspath(os.path.join(classToTestFolderPath, "../"*7 + "/libs/auto_disc/auto_disc"))
sys.path.insert(0, os.path.dirname(auto_discFolderPath))

from auto_disc.output_representations.specific import LeniaImageRepresentation

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

Object = lambda **kwargs: type("Object", (), kwargs)()

#region map

def test_map_1():
    leniaOutputRepresentation = LeniaImageRepresentation("wrapped_key")
    observations = Object(states=torch.ones((3, 256, 256)))
    res = leniaOutputRepresentation.map(observations, None)
    assert torch.equal(res["embedding"], torch.ones(65536))

def test_map_2():
    leniaOutputRepresentation = LeniaImageRepresentation("wrapped_key")
    observations = Object(states=torch.zeros((3, 256, 256)))
    res = leniaOutputRepresentation.map(observations, None)
    assert torch.equal(res["embedding"], torch.zeros(65536))

def test_map_3():
    leniaOutputRepresentation = LeniaImageRepresentation("wrapped_key")
    torch.manual_seed(0)
    observations = Object(states=torch.rand((3, 256, 256)))
    res = leniaOutputRepresentation.map(observations, None)
    with open(os.path.join(__location__+"/data", "test_map_3.pickle"), "rb") as resFile :
        saved_res = pickle.load(resFile)
    assert torch.equal(res["embedding"], saved_res)

def test_map_4():
    leniaOutputRepresentation = LeniaImageRepresentation("wrapped_key")
    torch.manual_seed(1)
    observations = Object(states=torch.rand((3, 256, 256)))
    res = leniaOutputRepresentation.map(observations, None)
    with open(os.path.join(__location__+"/data", "test_map_4.pickle"), "rb") as resFile :
        saved_res = pickle.load(resFile)
    assert torch.equal(res["embedding"], saved_res)

#endregion

#region calc_distance

def test_calc_distance_1():
    leniaOutputRepresentation = LeniaImageRepresentation("wrapped_key")
    embedding_a = torch.ones(2, 10)
    embedding_b = torch.zeros(2, 10)
    res = leniaOutputRepresentation.calc_distance(embedding_a, embedding_b)
    resWeWant = torch.tensor([3.1622776601683795, 3.1622776601683795], dtype=torch.float64)
    assert torch.equal(res, resWeWant)

def test_calc_distance_2():
    leniaOutputRepresentation = LeniaImageRepresentation("wrapped_key")
    torch.manual_seed(0)
    embedding_a = torch.rand(2, 10)
    embedding_b = torch.rand(2, 10)
    res = leniaOutputRepresentation.calc_distance(embedding_a, embedding_b)
    resWeWant = torch.tensor([0.3797050748081936, 0.11125848646373049], dtype=torch.float64)
    assert torch.equal(res, resWeWant)

def test_calc_distance_3():
    leniaOutputRepresentation = LeniaImageRepresentation("wrapped_key")
    torch.manual_seed(1)
    embedding_a = torch.rand(2, 10)
    embedding_b = torch.rand(2, 10)
    res = leniaOutputRepresentation.calc_distance(embedding_a, embedding_b)
    resWeWant = torch.tensor([0.3789525222714172, 0.16374296050895798], dtype=torch.float64)
    assert torch.equal(res, resWeWant)

def test_calc_distance_4():
    leniaOutputRepresentation = LeniaImageRepresentation("wrapped_key")
    torch.manual_seed(1)
    embedding_a = torch.rand(2, 10)
    embedding_b = torch.rand(2, 10)
    leniaOutputRepresentation.config.distance_function = "NotImplementedFunction"
    with unittest.TestCase.assertRaises(Exception, NotImplementedError) as context: 
       res = leniaOutputRepresentation.calc_distance(embedding_a, embedding_b)
    assert type(context.exception).__name__ == 'NotImplementedError'

#endregion