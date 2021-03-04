import numbers
import numpy as np
import random
import torch


def set_seed(seed):
    seed = int(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sample_value(config=None):
    '''Samples scalar values depending on the provided properties.'''

    val = None

    if isinstance(config, numbers.Number): # works also for booleans
        val = config
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val)

    elif config is None:
        #val = np.random.rand()
        val = torch.rand(())

    elif isinstance(config, tuple):

        if config[0] == 'continuous' or config[0] == 'continous':
            #val = np.random.uniform(config[1], config[2])
            val = (config[1] - config[2]) * torch.rand(()) + config[2]

        elif config[0] == 'discrete':
            #val = np.random.randint(config[1], config[2] + 1)
            val = torch.randint(config[1], config[2] + 1, ())

        elif config[0] == 'function':
            val = config[1](*config[2:])    # call function and give the other elements in the tuple as paramters

        elif len(config) == 2:
            # val = np.random.uniform(config[0], config[1])
            val = (config[0] - config[1]) * torch.rand(()) + config[1]

        else:
            raise ValueError('Unknown parameter type {!r} for sampling!', config[0])

    elif isinstance(config, list):
        #val = np.random.choice(config)
        val = config[torch.randint(len(config), ())]
        if isinstance(val, numbers.Number) and not isinstance(val, torch.Tensor):
            val = torch.tensor(val)

    elif isinstance(config, dict):
        if config['type'] == 'discrete':
            #val = np.random.randint(config['min'], config['max'] + 1)
            val = torch.randint(config['min'], config['max'] + 1, ())

        elif config['type'] == 'continuous':
            #val = np.random.uniform(config['min'], config['max'])
            val = (config['min'] - config['max']) * torch.rand(()) + config['max']

        elif config['type'] == 'boolean':
            #val = bool(np.random.randint(0,1))
            val = torch.rand(()) > 0.5

        elif config['type'] == 'function':
            function_call = config['callname']
            function_kwargs = config
            del function_kwargs['type']
            del function_kwargs['callname']
            val = function_call(**function_kwargs)

        else:
            raise ValueError('Unknown parameter type {!r} for sampling!', config['type'])

    return val


def mutate_value(val, mutation_factor=1.0, config=None, **kwargs):

    # TODO: mutate vector

    new_val = val

    if isinstance(val, list):
        for idx in range(np.shape(val)[0]):
            new_val[idx] = mutate_value(new_val[idx], mutation_factor=mutation_factor, config=config, **kwargs)
    else:

        if config and isinstance(config, dict):

            if 'distribution' in config:
                if config['distribution'] == 'gaussian':
                    #new_val = np.random.normal(val, config['sigma'] * max(0, mutation_factor))
                    std = config['sigma'] * torch.max(torch.tensor([0, mutation_factor]))
                    if std > 0.0:
                        new_val = torch.normal(val, std, ())
                    elif std == 0.0:
                        new_val = torch.tensor(val)
                else:
                    raise ValueError('Unknown parameter distribution {!r} for mutation!', config['distribution'])


            if 'type' in config:
                if config['type'] == 'discrete':
                    #new_val = np.round(new_val)
                    if not isinstance(new_val, torch.Tensor):
                        new_val = torch.tensor(new_val)
                    new_val = torch.round(new_val).int()
                elif config['type'] == 'continuous':
                    pass
                elif config['type'] == 'function':
                    function_call = config['callname']
                    function_kwargs = config
                    del function_kwargs['type']
                    del function_kwargs['callname']
                    new_val = function_call(val, **function_kwargs)
                else:
                    raise ValueError('Unknown parameter type {!r} for mutation!', config['type'])

            if 'min' in config:
                new_val = torch.max(new_val, torch.tensor(config['min'], dtype=new_val.dtype))

            if 'max' in config:
                new_val = torch.min(new_val, torch.tensor(config['max'], dtype=new_val.dtype))

        elif isinstance(config, tuple) and config[0] == 'function':
            new_val = config[1](val, *config[2:])


    return new_val



def sample_vector(config=None):

    val = None

    if isinstance(config, tuple):

        vector_length = int(sample_value(config[0]))

        val = [None]*vector_length
        for idx in range(vector_length):
            val[idx] = sample_value(config[1])

    else:
        raise ValueError('Unknown config type for sampling of a vactor!')

    #return np.array(val)
    return torch.tensor(val)


# TODO: convert in torch
"""
class EpsilonGreedySelection:

    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon


    def do_selection(self, values):

        values = np.array(values)

        not_nan_values = (~np.isnan(values))

        max_value = np.nanmax(values)
        max_inds = (max_value == values)

        n_max_vals = np.sum(max_inds) # number of max values
        n_values = np.sum(not_nan_values)

        probabilities = np.full(len(values), 0.0)
        probabilities[not_nan_values] = self.epsilon / n_values
        probabilities[max_inds] = (1-self.epsilon)/n_max_vals + self.epsilon/n_values

        if n_values == 0:
            choice = np.nan
        else:
            choice = np.where(np.cumsum(probabilities) > np.random.rand())[0][0]

        return choice, probabilities


class SoftmaxSelection:

    def __init__(self, beta=1):
        self.beta = beta


    def do_selection(self, values):

        values = np.array(values)

        buf = np.exp(self.beta * values - self.beta * np.nanmax(values))
        probabilities = buf / np.nansum(buf)

        nan_probabilities = np.isnan(probabilities)
        probabilities[nan_probabilities] = 0.0

        if np.sum(nan_probabilities) == len(values):
            choice = np.nan
        else:
            choice = np.where(np.cumsum(probabilities) > np.random.rand())[0][0]

        return choice, probabilities


class EpsilonSoftmaxSelection:

    def __init__(self, epsilon=0.1, beta=1):
        self.epsilon = epsilon
        self.beta = beta


    def do_selection(self, values):

        values = np.array(values)

        not_nan_values = (~np.isnan(values))

        buf = np.exp(self.beta * values - self.beta * np.nanmax(values))
        probabilities = (1 - self.epsilon) * buf / np.nansum(buf) + self.epsilon / np.sum(not_nan_values)

        probabilities[~not_nan_values] = 0.0

        if np.sum(not_nan_values) == 0:
            choice = np.nan
        else:
            choice = np.where(np.cumsum(probabilities) > np.random.rand())[0][0]

        return choice, probabilities
"""