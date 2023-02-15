from auto_disc.newarch.maps.UniformParameterMap import UniformParameterMap
from auto_disc.newarch.maps.MeanBehaviorMap import MeanBehaviorMap
import torch
import pathlib
import os
import shutil


def setup_function(function):
    global RESOURCE_URI
    file_path = str(pathlib.Path(__file__).parent.resolve())
    RESOURCE_URI = os.path.join(file_path, "tmp")
    os.mkdir(RESOURCE_URI)
    return


def teardown_function(function):
    global RESOURCE_URI
    if os.path.exists(RESOURCE_URI):
        shutil.rmtree(RESOURCE_URI)
    return


def test_map():
    input_dict = {"metadata": 1, "output": torch.rand(10)}
    mean_map = MeanBehaviorMap(premap_key="output")
    output_dict = mean_map.map(input_dict)
    assert output_dict["output"].size() == torch.Size([1])
    assert len(mean_map.history_saver.buffer) == 1
    assert mean_map.projector.low is not None
    assert mean_map.projector.high is not None


def test_sample():
    # redundant test, see test for BoxProjector.map and BoxProjector.sample
    pass


def test_save():
    input_dict = {"metadata": 1, "output": torch.tensor([2., 2., 2., 2., 2.])}
    mean_map = MeanBehaviorMap(premap_key="output")
    output_dict = mean_map.map(input_dict)
    uid = mean_map.save_leaf(resource_uri=RESOURCE_URI)

    new_map = MeanBehaviorMap()
    loaded_map = new_map.load_leaf(uid, resource_uri=RESOURCE_URI)
    assert torch.allclose(loaded_map.projector.low,
                          torch.tensor([0., 0., 0., 0., 0.]))
    assert torch.allclose(loaded_map.projector.high,
                          torch.tensor([2., 2., 2., 2., 2.]))
    assert len(loaded_map.history_saver.buffer) == 1


def test_get_tensor_history():
    def generate_input():
        return {"metadata": 1, "output": torch.rand(10)}
    input_dict = generate_input()
    param_map = MeanBehaviorMap(premap_key="output")
    for _ in range(10):
        param_map.map(input_dict)

    history = param_map.get_tensor_history()

    assert param_map.get_tensor_history().size() == torch.Size([10, 1])
