from auto_disc.newarch.systems.Lenia import Lenia
import torch
from copy import deepcopy


def setup_function(function):
    global dummy_input
    dummy_input = \
        {
            "params": {
                "dynamic_params":
                {
                    "R": torch.tensor(5.),
                    "T": torch.tensor(10.),
                    "b": torch.tensor([0.1, 0.2, 0.3, 0.4]),
                    "m": torch.tensor(0.5),
                    "s": torch.tensor(0.1)
                },
                "init_state": torch.rand((256, 256))
            }
        }


def teardown_function(function):
    pass


def test___init__():
    system = Lenia()
    assert system.orbit.size() == (200, 256, 256)


def test_process_dict():
    system = Lenia()
    dummy_params = system._process_dict(dummy_input)
    assert dummy_params.dynamic_params.R


def test__generate_automaton():
    system = Lenia()
    dummy_params = system._process_dict(dummy_input)

    automaton = system._generate_automaton(dummy_params.dynamic_params)
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
    automaton = system._generate_automaton(dummy_params.dynamic_params)
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

    system.render(out_dict, mode="human")
    assert 1
