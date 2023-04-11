from auto_disc.newarch.systems.Lenia import Lenia
import torch
from copy import deepcopy


def setup_function(function):
    global dummy_input, dummy_tensor_input
    dummy_input = {"params": {
        "R": torch.tensor(5.),
        "T": torch.tensor(10.),
        "b": torch.tensor([0.1, 0.2, 0.3, 0.4]),
        "m": torch.tensor(0.5),
        "s": torch.tensor(0.1),
        "kn": torch.tensor(0.),
        "gn": torch.tensor(1.),
        "init_state": torch.rand((256, 256))
    }}
    dummy_tensor_input = \
        {"params":
         {"param_tensor":
          torch.tensor(
              [5., 10., 0.5, 0.1, 0., 1., 0.1, 0.2, 0.3, 0.4]),
          "init_state": deepcopy(dummy_input["params"]["init_state"])
          }}
    return


def teardown_function(function):
    pass


def test___init__():
    system = Lenia()
    assert system.orbit.size() == (200, 256, 256)


def test_process_dict():
    system = Lenia()
    dummy_params = system._process_dict(dummy_input)
    assert "R" in dummy_params

    dummy_params_alt = system._process_dict(dummy_tensor_input)

    for (key, val) in dummy_params_alt.items():
        assert isinstance(val, torch.Tensor)
        assert torch.allclose(dummy_params[key], val)

    assert "R" in dummy_params_alt
    assert "b" in dummy_params_alt


def test__generate_automaton():
    system = Lenia()
    dummy_params = system._process_dict(dummy_input)

    # remove init_state
    # as it is usually loaded into the object attributes by _bootstrap
    del dummy_params["init_state"]

    automaton = system._generate_automaton(dummy_params)
    assert isinstance(automaton, torch.nn.Module)


def test__bootstrap():
    system = Lenia()
    dummy_params = system._process_dict(dummy_input)
    system._bootstrap(dummy_params)
    init_state = system.orbit[0]
    assert init_state.size() == (256, 256)


def test__step():
    system = Lenia()
    dummy_params = system._process_dict(dummy_input)
    system._bootstrap(dummy_params)
    init_state = system.orbit[0]
    automaton = system._generate_automaton(dummy_params)
    new_state = system._step(init_state, automaton)

    assert not torch.allclose(new_state, init_state)
    assert automaton.T.grad is None
    assert automaton.m.grad is None
    assert automaton.s.grad is None

    # generate ad graph
    new_state.sum().backward()
    assert automaton.T.grad is not None
    assert automaton.m.grad is not None
    assert automaton.s.grad is not None


def test_map():
    system = Lenia()
    dummy_params = system._process_dict(dummy_input)

    out_dict = system.map(dummy_input)

    assert torch.allclose(out_dict["output"], system.orbit[-1])
    # padded due to the way automaton steps
    assert out_dict["output"].size() == (1, 1, 256, 256)
    assert system.orbit[-1].size() == (256, 256)


def test_render():
    # eyeball test this one
    system = Lenia()
    dummy_params = system._process_dict(dummy_input)

    out_dict = system.map(dummy_input)

    system.render(mode="human")
    assert 1
