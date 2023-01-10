from auto_disc.newarch.wrappers.SaveWrapper import SaveWrapper


def test___init__():
    input = {"in": 1}
    wrapper = SaveWrapper(wrapped_keys=["in"], posttransform_keys=["out"],
                          inputs_to_save=["in"], outputs_to_save=["out"])
    wrapper_def = SaveWrapper(wrapped_keys=["in"], posttransform_keys=["out"])
    assert wrapper.__dict__ == wrapper_def.__dict__


def test_map():
    input = {"in": 1}
    wrapper = SaveWrapper(wrapped_keys=["in"], posttransform_keys=["out"])
    output = wrapper.map(input)
    assert output["out"] == 1
    assert len(output) == 1
    assert wrapper.input_buffer == [{"in": 1}]
    assert wrapper.output_buffer == [{"out": 1}]


def test_map_complex():
    input = {"a": 1, "b": 2}
    wrapper = SaveWrapper(
        wrapped_keys=["a", "b"], posttransform_keys=["b", "a"])
    output = wrapper.map(input)
    assert output["a"] == 2
    assert output["b"] == 1
    assert wrapper.input_buffer == [{"a": 1, "b": 2}]
    assert wrapper.output_buffer == [{"b": 1, "a": 2}]

    wrapper.map(output)
    assert wrapper.input_buffer == [{"a": 1, "b": 2}, {"a": 2, "b": 1}]
    assert wrapper.output_buffer == [{"b": 1, "a": 2}, {"b": 2, "a": 1}]
