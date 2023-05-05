from auto_disc.systems.ExponentialMixture import ExponentialMixture
import torch


def test_init():
    system = ExponentialMixture(sequence_max=13, sequence_density=1313)
    assert system.config["sequence_max"] == 13.
    assert system.config["sequence_density"] == 1313

    system = ExponentialMixture()
    assert system.config["sequence_max"] == 100.
    assert system.config["sequence_density"] == 100


def test_map():
    test_params = torch.rand(100)
    sequence_max = 1000.
    sequence_density = 500
    input_dict = {"params": test_params}

    system = ExponentialMixture(sequence_max=sequence_max,
                                sequence_density=sequence_density)
    output_dict = system.map(input_dict)
    assert output_dict["output"].size() == torch.Size([sequence_density])
    assert torch.all(torch.greater(output_dict["output"], 0))


def test_render():
    test_params = torch.rand(100)
    sequence_max = 1000.
    sequence_density = 500
    input_dict = {"params": test_params}

    system = ExponentialMixture(sequence_max=sequence_max,
                                sequence_density=sequence_density)
    output_dict = system.map(input_dict)

    byte_img = system.render(output_dict)

    assert isinstance(byte_img, bytes)
