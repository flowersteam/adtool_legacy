class AutoDiscMutationDefinition():
    '''
    Configurable mutation definition.
    '''
    def __init__(self, distribution, sigma):
        """
        init AutodiscSpaceDefinition instance
        distribution: string; distribution function (ex: gauss)
        sigma: float; paramater of distribution function
        """
        self.distribution = distribution
        self.sigma = sigma