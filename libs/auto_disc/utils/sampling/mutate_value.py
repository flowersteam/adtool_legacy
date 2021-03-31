import torch

def mutate_value_gauss(val, space_mutation, mutation_factor):
    """
    description    : apply a gauss function to the data
    val            : list, tuple or torch of integer or bool; Input data, will be modified
    space          : AutoDiscMutationDefinition object      ; defined mutation parameters...
    mutation_factor: float
    return         : torch resulting from the val modifiaction
    """
    if space_mutation.distribution == 'gauss':
        std = space_mutation.sigma * torch.max(torch.tensor([0, mutation_factor]))
        if std > 0.0:
            
            val = torch.tensor(val)
            std = torch.full(val.size(), std)
            val = torch.normal(val.float(), std.float()) # .float is needed to correct this error: _th_normal not supported on CPUType for Long 
        elif std == 0.0:
            val = torch.tensor(val)
    else:
        raise ValueError('Unknown parameter distribution {!r} for mutation!', space_mutation.distribution)
    return val


"""
description: associates a mutate function with its name (used in mutate_value)
"""
my_mutate_func_dic = {
    "gauss" : mutate_value_gauss
}

def mutate_value(val, space, mutation_factor=1.0):
    """
    description    : apply a slight random variation to the data
    val            : list, tuple or torch of integer or bool. Input data, will be mdified
    space          : AutoDiscSpaceDefinition object, defined data space. bounds dimensions etc...
    mutation_factor: float
    return         : torch resulting from the val modifiaction
    """
    space_type = space.type
    space_dimensions = space.dims
    space_bounds = space.bounds
    space_mutation = space.mutation

    #defines the function which must be called according to space_mutation.distribution
    #get in "my_mutate_func_dic" dictionary the function associated with the string "space_mutation.distribution"
    my_mutate_func = my_mutate_func_dic[space_mutation.distribution]

    val = my_mutate_func(val, space_mutation, mutation_factor)

    if space_type.name == "Float": # continuous
        pass
    elif space_type.name == "Integer": # Dicrete
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val)
        val = torch.round(val).int()
    elif space_type.name == "Function":
        pass
        #                     function_call = config['callname']
        #                     function_kwargs = config
        #                     del function_kwargs['type']
        #                     del function_kwargs['callname']
        #                     new_val = function_call(val, **function_kwargs)
    else:
        raise ValueError('Unknown parameter type {!r} for mutation!', space_type)

    val = torch.max(val, torch.tensor(space_bounds[0], dtype=val.dtype))
    val = torch.min(val, torch.tensor(space_bounds[1], dtype=val.dtype))


    if space_dimensions == []:
        val = val.item()

    return val