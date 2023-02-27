import pathlib
import os
import shutil
import auto_disc.newarch.run as run
from auto_disc.newarch.ExperimentPipeline import ExperimentPipeline
from utils.callbacks.on_save_finished_callbacks.generate_report_callback import GenerateReport


def setup_function(function):
    global RESOURCE_URI, config_json
    file_path = str(pathlib.Path(__file__).parent.resolve())
    RESOURCE_URI = os.path.join(file_path, "tmp")
    os.mkdir(RESOURCE_URI)
    config_json = \
        {
            "experiment": {
                "name": "newarch_demo",
                "config": {
                    "host": "local",
                    "save_location": f"{RESOURCE_URI}",
                    "nb_seeds": 1,
                    "nb_iterations": 20,
                    "save_frequency": 1,
                }
            },
            "system": {
                "name": "auto_disc.newarch.systems.ExponentialMixture",
                "config": {
                    "sequence_max": 1,
                    "sequence_density": 20
                }
            },
            "explorer": {
                "name": "auto_disc.newarch.explorers.IMGEPFactory",
                "config": {
                    "equil_time": 2,
                    "param_dim": 1,
                    "param_init_low": 0.0,
                    "param_init_high": 1.0
                }
            },
            "callbacks": {},
            "logger_handlers": []
        }
    return


def teardown_function(function):
    global RESOURCE_URI
    if os.path.exists(RESOURCE_URI):
        shutil.rmtree(RESOURCE_URI)
    return


def test_create():
    experiment_id = 1
    seed = 1
    pipeline = run.create(config_json, experiment_id=experiment_id, seed=seed)
    assert isinstance(pipeline, ExperimentPipeline)


def test_run():
    experiment_id = 1
    seed = 1
    pipeline = run.create(config_json, experiment_id=experiment_id, seed=seed)

    run.start(pipeline, 10)


def test_save_GenerateReport():
    """
    primarily tests the GenerateReport callback
    """
    config_json["callbacks"] = {
        "on_save_finished": [{"name":
                              "utils.callbacks."
                              "on_save_finished_callbacks."
                              "generate_report_callback."
                              "GenerateReport",
                              "config": {}
                              }]
    }
    experiment_id = 1
    seed = 1
    pipeline = run.create(config_json, experiment_id=experiment_id, seed=seed)

    run.start(pipeline, 10)

    # rough check of file tree
    files = os.listdir(RESOURCE_URI)
    data_dirs = []
    reports = []
    for f in files:
        if f.split(".")[-1] == "json":
            reports.append(f)
        else:
            data_dirs.append(f)

    for r in reports:
        tmp = r.split(".")[0]
        uid = tmp.split("_")[-1]
        assert uid in data_dirs


def test_save_SaveDiscoveryOnDisk():
    config_json["callbacks"] = {
        "on_discovery": [{"name":
                          "utils.callbacks."
                          "on_discovery_callbacks."
                          "save_discovery_on_disk."
                          "SaveDiscoveryOnDisk",
                          "config": {}
                          }]
    }
    experiment_id = 1
    seed = 1
    pipeline = run.create(config_json, experiment_id=experiment_id, seed=seed)

    run.start(pipeline, 10)

    # rough check of file tree
    files = os.listdir(RESOURCE_URI)
    disc_dirs = []

    for f in files:
        tf = f.split("_")
        if (len(tf) > 1) and (tf[-2] == "idx"):
            disc_dirs.append(f)
        else:
            pass

    for dir in disc_dirs:
        each_discovery = os.listdir(os.path.join(RESOURCE_URI, dir))
        assert "rendered_output" in each_discovery
        assert "discovery.json" in each_discovery
