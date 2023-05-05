"""
Helper script which allows creation of `ExperimentPipeline`. 
This file `run.py` can also be run as `__main__`, for example in remote configurations.
"""
from auto_disc_legacy.utils.callbacks import interact_callbacks
import torch
import random
import numpy as np
from auto_disc_legacy.utils.logger import AutoDiscLogger
from auto_disc.ExperimentPipeline import ExperimentPipeline
from auto_disc.registration import get_cls_from_path
import sys
import argparse
import json
import os
from typing import Callable, Dict, List

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../"))


def create(parameters: Dict, experiment_id: int, seed: int,
           additional_callbacks: Dict[str, List[Callable]] = None,
           additional_handlers: List[AutoDiscLogger] = None,
           interactMethod: Callable = None) -> ExperimentPipeline:
    """
        Setup the whole experiment. Set each modules, logger, callbacks and use them to define the experiment pipeline.

        #### Args:
        - parameters: Experiment config (define which systems which explorer which callbacks and all other information needed to set an experiment)
        - experiment_id: Current experiment id
        - seed: current seed number
        - additional_callbacks: callbacks we want use in addition to callbacks from parameters arguments
        - additional_handlers: handlers we want to use in addition to logger_handlers from parameters arguments

        #### Returns:
        - experiment: The experiment we have just defined
    """

    _set_seed(seed)
    save_frequency = parameters['experiment']['config']['save_frequency']

    # Get logger
    handlers = []
    for logger_handler in parameters['logger_handlers']:
        handler_class = get_cls_from_path(logger_handler['name'])
        handler = handler_class(
            **logger_handler['config'], experiment_id=experiment_id)
        handlers.append(handler)
    if additional_handlers is not None:
        handlers.extend(additional_handlers)

    logger = AutoDiscLogger(experiment_id, seed, handlers)

    # Get callbacks
    callbacks = {
        'on_discovery': [],
        'on_save_finished': [],
        'on_finished': [],
        'on_error': [],
        'on_cancelled': [],
        'on_saved': [],
        'interact': {}
    }

    for callback_key in callbacks:
        if additional_callbacks is not None:
            if callback_key != "interact":
                callbacks[callback_key].extend(
                    additional_callbacks[callback_key])
            else:
                callbacks[callback_key].update(
                    additional_callbacks[callback_key])
        for _callback in parameters['callbacks'].get(callback_key, []):
            callback_class = get_cls_from_path(_callback['name'])
            # initialize callbacks with appropriate logger and config
            if callback_key == "interact":
                callbacks[callback_key].update(
                    {_callback['name']: callback_class(
                        interactMethod=interactMethod, **_callback['config'])}
                )
            else:
                callbacks[callback_key].append(
                    callback_class(**_callback['config'])
                )

    # short circuit if "resume_from_uid" is set
    resume_ckpt = parameters["experiment"]["config"].get("resume_from_uid",
                                                         None)
    if (resume_ckpt is not None):
        resource_uri = parameters['experiment']['config']['save_location']
        experiment = \
            ExperimentPipeline().\
            load_leaf(uid=resume_ckpt,
                      resource_uri=resource_uri)

        # set attributes pruned by save_leaf
        experiment.logger = logger
        experiment._on_discovery_callbacks = callbacks['on_discovery']
        experiment._on_save_finished_callbacks = callbacks['on_save_finished']
        experiment._on_finished_callbacks = callbacks['on_finished']
        experiment._on_cancelled_callbacks = callbacks['on_cancelled']
        experiment._on_save_callbacks = callbacks['on_saved']
        experiment._on_error_callbacks = callbacks['on_error']
        experiment._interact_callbacks = callbacks['interact']

        return experiment
    else:
        pass

    # Get explorer factory and generate explorer
    explorer_factory_class = get_cls_from_path(parameters['explorer']['name'])
    explorer_factory = explorer_factory_class(
        **parameters['explorer']['config'])
    explorer = explorer_factory()

    # Get system
    system_class = get_cls_from_path(parameters['system']['name'])
    system = system_class(**parameters['system']['config'])

    # Create experiment pipeline
    experiment = ExperimentPipeline(
        experiment_id=experiment_id,
        seed=seed,
        save_frequency=save_frequency,
        system=system,
        explorer=explorer,
        on_discovery_callbacks=callbacks['on_discovery'],
        on_save_finished_callbacks=callbacks['on_save_finished'],
        on_finished_callbacks=callbacks['on_finished'],
        on_cancelled_callbacks=callbacks['on_cancelled'],
        on_save_callbacks=callbacks['on_saved'],
        on_error_callbacks=callbacks['on_error'],
        interact_callbacks=callbacks['interact'],
        logger=logger,
        resource_uri=parameters['experiment']['config']['save_location']
    )

    return experiment


def start(experiment: ExperimentPipeline, nb_iterations: int) -> None:
    """
        Runs an experiment for a number of iterations

        #### Args:
        - experiment: The experiment we want to launch
        - nb_iterations: the number explorations
    """
    experiment.run(nb_iterations)


def _set_seed(seed: int) -> None:
    """
        Set torch seed to make experiment repeatable.

        #### Args:
        - seed: seed number
    """
    seed = int(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--experiment_id', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--nb_iterations', type=int, required=True)

    args = parser.parse_args()

    with open(args.config_file) as json_file:
        config = json.load(json_file)

    experiment = create(config, args.experiment_id, args.seed)
    start(experiment, args.nb_iterations)
