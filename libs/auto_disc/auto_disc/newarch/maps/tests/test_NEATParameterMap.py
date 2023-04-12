from auto_disc.newarch.maps.NEATParameterMap import NEATParameterMap
import os
import torch
import neat.genome


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

    # TODO: postmap_shape = (1,1) edge case is broken
