from auto_disc.newarch.explorers.imgep_explorer import IMGEPExplorer
from auto_disc.newarch.maps.MeanBehaviorMap import MeanBehaviorMap
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


def test___init__():
    mean_map = MeanBehaviorMap(premap_key="output")
    param_map = UniformParameterMap(premap_key="params",
                                    tensor_low=torch.tensor([0., 0., 0.]),
                                    tensor_high=torch.tensor([2., 2., 2.]))
    explorer = IMGEPExplorer(premap_key="output", postmap_key="params",
                             parameter_map=param_map, behavior_map=mean_map,
                             equil_time=2)
    assert explorer.behavior_map == mean_map
    assert explorer.parameter_map == param_map
    assert explorer.equil_time == 2
    assert explorer.timestep == 0


def test__find_closest():
    goal_history = torch.tensor(
        [[1, 2, 3], [4, 5, 6], [-1, -1, -1]], dtype=float)
    goal = torch.tensor([-0.99, -0.99, -0.99])

    mean_map = MeanBehaviorMap(premap_key="output")
    param_map = UniformParameterMap(premap_key="params",
                                    tensor_low=torch.tensor([0., 0., 0.]),
                                    tensor_high=torch.tensor([2., 2., 2.]))
    explorer = IMGEPExplorer(premap_key="output", postmap_key="params",
                             parameter_map=param_map, behavior_map=mean_map,
                             equil_time=2)

    idx = explorer._find_closest(goal, goal_history)
    assert idx == torch.tensor([2])
    assert torch.allclose(goal_history[idx], torch.tensor([-1., -1., -1.]))


def test_observe_results():
    mean_map = MeanBehaviorMap(premap_key="output")
    param_map = UniformParameterMap(premap_key="params",
                                    tensor_low=torch.tensor([0., 0., 0.]),
                                    tensor_high=torch.tensor([2., 2., 2.]))
    explorer = IMGEPExplorer(premap_key="output", postmap_key="params",
                             parameter_map=param_map, behavior_map=mean_map,
                             equil_time=2)

    system_output = {"metadata": 1, "output": torch.tensor([1., 1., 1.])}
    behavior, output_dict = explorer.observe_results(system_output)

    assert behavior == output_dict["output"]
    behavior += 1
    assert torch.allclose(output_dict["output"], torch.tensor([2., 2., 2.]))


def test_map():
    mean_map = MeanBehaviorMap(premap_key="output")
    param_map = UniformParameterMap(premap_key="params",
                                    tensor_low=torch.tensor([0., 0., 0.]),
                                    tensor_high=torch.tensor([6., 6., 6.]))
    explorer = IMGEPExplorer(premap_key="output", postmap_key="params",
                             parameter_map=param_map, behavior_map=mean_map,
                             equil_time=2)
    system_output = {"metadata": 1, "output": torch.tensor([1., 2., 3.])}

    new_params = explorer.map(system_output)
    assert new_params["params"]
    assert new_params.get("output", None) is None
    system_output["metadata"] = 2
    assert new_params["metadata"] == 1
    assert explorer.timestep == 1


def test_suggest_trial_equilibriation():
    mean_map = MeanBehaviorMap(premap_key="output")
    param_map = UniformParameterMap(premap_key="params",
                                    tensor_low=torch.tensor([0., 0., 0.]),
                                    tensor_high=torch.tensor([2., 2., 2.]))
    explorer = IMGEPExplorer(premap_key="output", postmap_key="params",
                             parameter_map=param_map, behavior_map=mean_map,
                             equil_time=2)

    for _ in range(2):
        params_trial = explorer.suggest_trial(torch.Size([3]))
        assert params_trial.size() == torch.Size([3])


def test_suggest_trial_behavioral_diffusion():
    def add_gaussian_noise_test(input_tensor: torch.Tensor,
                                mean: torch.Tensor = torch.tensor([10000.]),
                                std: torch.Tensor = torch.tensor([1.]),
                                ) -> torch.Tensor:
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean, dtype=float)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std, dtype=float)
        noise_unit = torch.randn(input_tensor.size())
        noise = noise_unit*std + mean
        return input_tensor + noise

    mean_map = MeanBehaviorMap(premap_key="output")
    param_map = UniformParameterMap(premap_key="params",
                                    tensor_low=torch.tensor([0., 0., 0.]),
                                    tensor_high=torch.tensor([6., 6., 6.]))
    explorer = IMGEPExplorer(premap_key="output", postmap_key="params",
                             parameter_map=param_map, behavior_map=mean_map,
                             equil_time=2)

    # mock history
    mock_system_output_history = \
        [{"metadata": 1,
          "output": torch.tensor([1., 2., 3.])},
         {"metadata": 1,
          "output": torch.tensor([4., 5., 6.])},
         {"metadata": 1,
          "output": torch.tensor([0., 0., 0.])}]
    mock_param_input_history = \
        [{"metadata": 1,
          "params": torch.tensor([0., 1., 2.])},
         {"metadata": 1,
          "params": torch.tensor([3., 4., 5.])},
         {"metadata": 1,
          "params": torch.tensor([-1., -1., -1.])}]
    explorer.behavior_map.history_saver.buffer = \
        [{"metadata": 1,
          "output": torch.tensor([2.])},
         {"metadata": 1,
          "output": torch.tensor([5.])},
         {"metadata": 1,
          "output": torch.tensor([0.])}]
    explorer.parameter_map.history_saver.buffer = mock_param_input_history
    explorer.behavior_map.projector.low = torch.tensor([0.])
    explorer.behavior_map.projector.high = torch.tensor([5.])
    explorer.behavior_map.projector.tensor_shape = torch.Size([1])
    explorer.timestep = 2
    explorer.mutator = add_gaussian_noise_test

    # actual test
    # NOTE: not deterministic
    params_trial = explorer.suggest_trial(torch.Size([3]))
    assert params_trial.size() == torch.Size([3])
    assert torch.mean(params_trial) > 100
