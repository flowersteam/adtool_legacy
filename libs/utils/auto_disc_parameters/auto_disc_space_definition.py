class AutoDiscSpaceDefinition():
    '''
    Configurable space definition.
    '''
    def __init__(self, dims, bounds, type, mutation=None):
        self.dims = dims
        self.bounds = bounds
        self.type = type
        self.mutation = mutation