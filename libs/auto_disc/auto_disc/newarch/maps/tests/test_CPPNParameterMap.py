from auto_disc.newarch.maps.CPPNParameterMap import CPPNParameterMap
import os


def setup_function(function):
    global CONFIG_PATH
    # get current file path
    tests_path = os.path.dirname(os.path.abspath(__file__))
    maps_path = os.path.dirname(tests_path)
    CONFIG_PATH = os.path.join(maps_path, "cppn/config.cfg")

    return


def test_CPPNParameterMap___init__():
    cppn = CPPNParameterMap(config_path=CONFIG_PATH)


def test_CPPNParameterMap_sample():
    cppn = CPPNParameterMap(config_path=CONFIG_PATH)
    out = cppn.sample((1))


def test_CPPNParameterMap_map():
    input_dict = {"random_data": 1, "metadata": "hello"}
    cppn = CPPNParameterMap(config_path=CONFIG_PATH, postmap_dim=(10, 10))
    out = cppn.map(input_dict)
    assert "init_state" in out
    assert out["init_state"].size() == (10, 10)

    # TODO: postmap_dim = (1,1) edge case is broken
