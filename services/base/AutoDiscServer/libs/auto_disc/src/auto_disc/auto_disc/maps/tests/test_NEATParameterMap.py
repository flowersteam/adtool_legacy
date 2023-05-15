from auto_disc.auto_disc.maps.NEATParameterMap import NEATParameterMap
import os
import neat.genome
from copy import deepcopy


def setup_function(function):
    global CONFIG_PATH
    # get current file path
    tests_path = os.path.dirname(os.path.abspath(__file__))
    maps_path = os.path.dirname(tests_path)
    CONFIG_PATH = os.path.join(maps_path, "cppn/config.cfg")

    return


def test_NEATParameterMap___init__():
    neat_map = NEATParameterMap(config_path=CONFIG_PATH)


def test_NEATParameterMap_sample():
    # NOTE: This test is not deterministic.
    neat_map = NEATParameterMap(config_path=CONFIG_PATH)
    genome = neat_map.sample()

    assert isinstance(genome, neat.genome.DefaultGenome)


def test_NEATParameterMap_map():
    input_dict = {"random_data": 1, "metadata": "hello"}
    neat_map = NEATParameterMap(config_path=CONFIG_PATH)
    out = neat_map.map(input_dict)
    assert "genome" in out
    assert "neat_config" in out
    # NOTE: same memory object being pointed to
    assert out["neat_config"] is neat_map.neat_config

    dc = deepcopy(out)
    assert dc["neat_config"] is not neat_map.neat_config

    # TODO: postmap_shape = (1,1) edge case is broken
