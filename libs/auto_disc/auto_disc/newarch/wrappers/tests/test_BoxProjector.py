from auto_disc.newarch.wrappers.BoxProjector import BoxProjector
import torch


def test__update_low_high():
    dim = 10
    input = torch.rand(dim)

    box = BoxProjector()
    box._update_low_high(input)

    assert torch.allclose(box.low, torch.zeros_like(input))
    assert torch.allclose(box.high, input)
    old_low = box.low.clone()
    old_high = box.high.clone()

    new_input = torch.rand(dim) * 10 - 5
    box._update_low_high(new_input)
    upper_mask = torch.greater(new_input, old_high)
    lower_mask = torch.less(new_input, old_low)

    assert torch.allclose(box.low[lower_mask], new_input[lower_mask])
    assert torch.allclose(box.high[upper_mask], new_input[upper_mask])

    # test immutability
    new_input += 2
    assert not torch.allclose(box.low[lower_mask], new_input[lower_mask])


def test__generate_sampler():
    """
    NOTE: this test is non-deterministic
    """
    dim = 10
    rand_nums_low = torch.rand(dim) * 2 - 4
    rand_nums_high = torch.rand(dim) * 2 + 4

    box = BoxProjector()
    box.low = rand_nums_low.clone()
    box.high = rand_nums_high.clone()

    f = box._generate_sampler()
    # test if the callable f is still usable
    del box

    for _ in range(100):
        old_sample = f()
        sample = f()

        # samples differ
        assert not torch.all(torch.isclose(sample, old_sample))

        # samples are within the same boundaries
        assert torch.all(torch.greater(sample, rand_nums_low))
        assert torch.all(torch.less(sample, rand_nums_high))


def test_map():
    dim = 10
    input = torch.rand(dim)
    input_dict = {"output": input, "metadata": 1}

    box = BoxProjector()
    output_dict = box.map(input_dict)

    assert output_dict != input
    assert torch.allclose(output_dict["output"], input_dict["output"])
    assert torch.allclose(box.low, torch.zeros_like(input))
    assert torch.allclose(box.high, input)
    assert output_dict["sampler"]
    assert output_dict["metadata"] == 1

    old_low = box.low.clone()
    old_high = box.high.clone()

    new_input = torch.rand(dim) * 10 - 5
    input_dict["output"] = new_input
    output_dict = box.map(input_dict)
    upper_mask = torch.greater(new_input, old_high)
    lower_mask = torch.less(new_input, old_low)

    assert torch.allclose(box.low[lower_mask], new_input[lower_mask])
    assert torch.allclose(box.high[upper_mask], new_input[upper_mask])
