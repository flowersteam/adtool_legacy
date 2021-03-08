# TODO
# def mutate_value(val, mutation_factor=1.0, config=None, **kwargs):

#     # TODO: mutate vector

#     new_val = val

#     if isinstance(val, list):
#         for idx in range(np.shape(val)[0]):
#             new_val[idx] = mutate_value(new_val[idx], mutation_factor=mutation_factor, config=config, **kwargs)
#     else:

#         if config and isinstance(config, dict):

#             if 'distribution' in config:
#                 if config['distribution'] == 'gaussian':
#                     #new_val = np.random.normal(val, config['sigma'] * max(0, mutation_factor))
#                     std = config['sigma'] * torch.max(torch.tensor([0, mutation_factor]))
#                     if std > 0.0:
#                         new_val = torch.normal(val, std, ())
#                     elif std == 0.0:
#                         new_val = torch.tensor(val)
#                 else:
#                     raise ValueError('Unknown parameter distribution {!r} for mutation!', config['distribution'])


#             if 'type' in config:
#                 if config['type'] == 'discrete':
#                     #new_val = np.round(new_val)
#                     if not isinstance(new_val, torch.Tensor):
#                         new_val = torch.tensor(new_val)
#                     new_val = torch.round(new_val).int()
#                 elif config['type'] == 'continuous':
#                     pass
#                 elif config['type'] == 'function':
#                     function_call = config['callname']
#                     function_kwargs = config
#                     del function_kwargs['type']
#                     del function_kwargs['callname']
#                     new_val = function_call(val, **function_kwargs)
#                 else:
#                     raise ValueError('Unknown parameter type {!r} for mutation!', config['type'])

#             if 'min' in config:
#                 new_val = torch.max(new_val, torch.tensor(config['min'], dtype=new_val.dtype))

#             if 'max' in config:
#                 new_val = torch.min(new_val, torch.tensor(config['max'], dtype=new_val.dtype))

#         elif isinstance(config, tuple) and config[0] == 'function':
#             new_val = config[1](val, *config[2:])


#     return new_val