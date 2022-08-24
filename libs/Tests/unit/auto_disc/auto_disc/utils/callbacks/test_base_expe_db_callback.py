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

from auto_disc.utils.callbacks import BaseExpeDBCallback
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

Object = lambda **kwargs: type("Object", (), kwargs)()

#region serialize_autodisc_space

def test__serialize_autodisc_space():
    torch.manual_seed(0)
    baseExpeDBCallback =BaseExpeDBCallback("dummyURL")
    dummySpace = {
        "a": 10,
        "b": "oui",
        "c": torch.rand(2,3)
    }
    serialized_space = baseExpeDBCallback._serialize_autodisc_space(dummySpace)

    assert serialized_space == {
        'a':10,
        'b':'oui',
        "c": [[0.9700530018065531, 0.707819864399788, 0.45938294312745087], [0.9207476841219603, 0.6450241201227648, 0.7911478921803037]]
    }


#endregion