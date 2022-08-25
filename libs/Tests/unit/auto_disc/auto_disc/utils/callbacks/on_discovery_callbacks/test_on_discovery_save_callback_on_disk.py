import os
import pickle
import sys
import unittest
import io
import imageio
import shutil

import torch

classToTestFolderPath = os.path.abspath(__file__)
classToTestFolderPath = classToTestFolderPath.split('/')
classToTestFolderPath = classToTestFolderPath[0:classToTestFolderPath.index("AutomatedDiscoveryTool")+1]
auto_discFolderPath = "/".join(classToTestFolderPath) + "/libs/auto_disc/auto_disc"

sys.path.insert(0, os.path.dirname(auto_discFolderPath))

from auto_disc.utils.callbacks.on_discovery_callbacks import OnDiscoverySaveCallbackOnDisk
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

Object = lambda **kwargs: type("Object", (), kwargs)()

#region OnDiscoverySaveCallbackOnDisk

def test___call__():
    logger = Object(info=lambda self, x: x)
    test_folder = __location__+"/test_folder/"
    to_save_outputs = ["raw_run_parameters",
                        "run_parameters", 
                        "raw_output", 
                        "output",
                        "rendered_output"]
    dummy_bytes = io.BytesIO()
    imageio.imwrite(dummy_bytes, [[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]], 'png')
    kwargs = {
        "raw_run_parameters": 1,
        "run_parameters": "1",
        "raw_output": "something",
        "output": torch.rand(2),
        "rendered_output": [dummy_bytes,"2","3"],
        "step_observations": 3,
        "experiment_id": 1,
        "seed": 3,
        "run_idx": 10, 
    }
    onDiscoverySaveCallbackOnDisk =OnDiscoverySaveCallbackOnDisk(test_folder, to_save_outputs, logger=logger)
    onDiscoverySaveCallbackOnDisk(**kwargs)
    test_folder = test_folder+"{}/{}".format(1,3)
    assert(
        os.path.isdir(test_folder) and os.path.isdir(test_folder+"/output") and os.path.isdir(test_folder+"/raw_output") and os.path.isdir(test_folder+"/run_parameters") 
        and os.path.isdir(test_folder+"/raw_run_parameters") and os.path.isdir(test_folder+"/rendered_output")
        and os.path.isfile(test_folder+"/output/idx_10.pickle") and os.path.isfile(test_folder+"/raw_output/idx_10.pickle") and os.path.isfile(test_folder+"/run_parameters/idx_10.pickle") 
        and os.path.isfile(test_folder+"/raw_run_parameters/idx_10.pickle") and os.path.isfile(test_folder+"/rendered_output/idx_10.2")
    )
    shutil.rmtree(__location__+"/test_folder/")


#endregion