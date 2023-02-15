from auto_disc.newarch.maps.UniformParameterMap import UniformParameterMap
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


def test_sample():
    param_map = UniformParameterMap(premap_key="params",
                                    tensor_low=torch.tensor([0., 0., 0.]),
                                    tensor_high=torch.tensor([2., 2., 2.]))
    sample_tensor = param_map.sample(param_map.tensor_shape)
    assert torch.all(torch.greater(sample_tensor, torch.tensor([0.])))
    assert torch.all(torch.less(sample_tensor, torch.tensor([2.])))


def test_map():
    input_dict = {"metadata": 1}
    param_map = UniformParameterMap(premap_key="params",
                                    tensor_low=torch.tensor([0., 0., 0.]),
                                    tensor_high=torch.tensor([2., 2., 2.]))
    output_dict = param_map.map(input_dict)
    assert output_dict["params"].size()[0] == 3
    assert torch.all(torch.greater(output_dict["params"],
                                   param_map.projector.low))
    assert torch.all(torch.less(output_dict["params"],
                                param_map.projector.high))
    assert len(param_map.history_saver.buffer) == 1
    assert param_map.history_saver.buffer[0]["metadata"] \
        == output_dict["metadata"]
    assert torch.allclose(param_map.history_saver.buffer[0]["params"],
                          output_dict["params"])


def test_save():
    input_dict = {"metadata": 1}
    param_map = UniformParameterMap(premap_key="params",
                                    tensor_low=torch.tensor([0., 0., 0.]),
                                    tensor_high=torch.tensor([2., 2., 2.]))
    output_dict = param_map.map(input_dict)
    assert len(param_map.history_saver.buffer) == 1
    uid = param_map.save_leaf(resource_uri=RESOURCE_URI)
    assert len(param_map.history_saver.buffer) == 0

    new_map = UniformParameterMap()
    loaded_map = new_map.load_leaf(uid, resource_uri=RESOURCE_URI)
    assert torch.allclose(loaded_map.projector.low,
                          torch.tensor([0., 0., 0.]))
    assert len(loaded_map.history_saver.buffer) == 1


def test_get_tensor_history():
    input_dict = {"metadata": 1}
    param_map = UniformParameterMap(premap_key="params",
                                    tensor_low=torch.tensor([0., 0., 0.]),
                                    tensor_high=torch.tensor([2., 2., 2.]))
    for _ in range(10):
        param_map.map(input_dict)

    assert len(param_map.history_saver.buffer) == 10

    tensor_history = param_map.get_tensor_history()

    assert isinstance(tensor_history, torch.Tensor)
    assert tensor_history.size() == torch.Size([10, 3])
