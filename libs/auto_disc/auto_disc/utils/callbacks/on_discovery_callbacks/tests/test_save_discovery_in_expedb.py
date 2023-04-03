from auto_disc.utils.callbacks.on_discovery_callbacks.save_discovery_in_expedb import SaveDiscoveryInExpeDB
import torch


def test___call__():
    binary = bytes(2000)
    discovery = {"input": 1, "output": torch.tensor(
        [[2, 3], [3, 4]]), "rendered_output": binary}
    experiment_id = 162000
    seed = 10
    run_idx = 10

    resource_uri = "http://127.0.0.1:5001"

    callback = SaveDiscoveryInExpeDB()
    callback(resource_uri=resource_uri,
             experiment_id=experiment_id,
             seed=seed,
             run_idx=run_idx,
             discovery=discovery)
