import os
import pickle
import sys
import unittest

import torch

classToTestFolderPath = os.path.abspath(__file__)
classToTestFolderPath = classToTestFolderPath.split('/')
classToTestFolderPath = classToTestFolderPath[0:classToTestFolderPath.index("AutomatedDiscoveryTool")+1]
auto_discFolderPath = "/".join(classToTestFolderPath) + "/libs/auto_disc/auto_disc"

sys.path.insert(0, os.path.dirname(auto_discFolderPath))

from auto_disc.systems.python_systems.lenia import Lenia, LeniaStepFFT, LeniaStepConv2d, create_colormap, im_from_array_with_colormap
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

Object = lambda **kwargs: type("Object", (), kwargs)()

#region Lenia


#endregion

#region LeniaStepFFT

def test_LeniaStepFFT_forward_1():
    torch.manual_seed(0)
    sx = 5
    sy = 5
    input = torch.rand(1, 1, sx, sy)
    with open(os.path.join(__location__, "run_parameters.pickle"), "rb") as parametersFile :
        run_parameters = pickle.load(parametersFile)
    leniaStepFFT = LeniaStepFFT(SX=sx, SY=sy,**run_parameters)
    res = leniaStepFFT.forward(input)
    assert res.tolist()[0][0] == [
        [0.8940625188122535, 0.6318293814105334, 0.38339246081388156, 0.8447572021055564, 0.5690336375826549],
        [0.7151574100398393, 0.10261569410197593, 0.2751171479865054, 0.5053504374305756, 0.21224540934303368],
        [0.3768783688774575, 0.10080906397343294, 0.27927629112404395, 0.545914765879497, 0.4058579180202637],
        [0.3648100185561107, 0.3312911043760308, 0.12944811540660142, 0.5890476930414752, 0.7088834186028972],
        [0.13437599251610244, 0.6007486404954043, 0.0337447800366764, 0.4477615784243451, 0.1500297220147525]
    ]

def test_LeniaStepFFT_forward_2():
    torch.manual_seed(1)
    sx = 256
    sy = 256
    input = torch.rand(1, 1, sx, sy)
    with open(os.path.join(__location__, "run_parameters.pickle"), "rb") as parametersFile :
        run_parameters = pickle.load(parametersFile)
    leniaStepFFT = LeniaStepFFT(SX=sx, SY=sy,**run_parameters)
    res = leniaStepFFT.forward(input)
    assert list(res.shape) == [1, 1, 256, 256]
    assert res.shape.numel() == 65536
    assert res[0][0][100][100].item() == 0.34667892931432154
#endregion

#region LeniaStepConv2d

def test_LeniaStepConv2d_forward_1():
    torch.manual_seed(0)
    sx = 5
    sy = 5
    input = torch.rand(1, 1, sx, sy)
    with open(os.path.join(__location__, "run_parameters.pickle"), "rb") as parametersFile :
        run_parameters = pickle.load(parametersFile)
    leniaStepConv2d = LeniaStepConv2d(**run_parameters)
    res = leniaStepConv2d.forward(input)
    assert res.tolist()[0][0] == [
        [0.8940625188122535, 0.6318293814105334, 0.38339246081388156, 0.8447572021055564, 0.5690336375826549],
        [0.7151574100398393, 0.10261569410197593, 0.2751171479865054, 0.5053504374305756, 0.21224540934303368],
        [0.3768783688774575, 0.10080906397343294, 0.27927629112404395, 0.545914765879497, 0.4058579180202637],
        [0.3648100185561107, 0.3312911043760308, 0.12944811540660142, 0.5890476930414752, 0.7088834186028972],
        [0.13437599251610244, 0.6007486404954043, 0.0337447800366764, 0.4477615784243451, 0.1500297220147525]
    ]

def test_LeniaStepConv2d_forward_2():
    torch.manual_seed(1)
    sx = 256
    sy = 256
    input = torch.rand(1, 1, sx, sy)
    with open(os.path.join(__location__, "run_parameters.pickle"), "rb") as parametersFile :
        run_parameters = pickle.load(parametersFile)
    leniaStepConv2d = LeniaStepConv2d(SX=sx, SY=sy,**run_parameters)
    res = leniaStepConv2d.forward(input)
    assert list(res.shape) == [1, 1, 256, 256]
    assert res.shape.numel() == 65536
    assert res[0][0][100][100].item() == 0.34667892931432154

#endregion