import torch
from auto_disc.output_representations import BaseOutputRepresentation
from auto_disc.utils.config_parameters import IntegerConfigParameter, BooleanConfigParameter
from auto_disc.utils.spaces.utils import ConfigParameterBinding
from auto_disc.utils.spaces import DictSpace, BoxSpace
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import NotFittedError
import numpy as np

@IntegerConfigParameter(name="n_components", default=3, min=1)
@IntegerConfigParameter(name="fit_period", default=10, min=1)
@BooleanConfigParameter(name="expand_output_space", default=True)

class PCA(BaseOutputRepresentation):
    '''
    PCA OutputRepresentation.
    '''
    output_space = DictSpace(
        embedding=BoxSpace(low=0, high=0, shape=(ConfigParameterBinding("n_components"),))
    )

    def __init__(self, wrapped_input_space_key=None):
        super().__init__(wrapped_input_space_key=wrapped_input_space_key)

        self.algorithm = make_pipeline(StandardScaler(),
                                       decomposition.PCA(n_components=self.config.n_components))

    def map(self, observations, is_output_new_discovery, **kwargs):
        input = observations[self.wrapped_input_space_key].reshape(1,-1)
        try:
            output = self.algorithm.transform(input)
            output = torch.from_numpy(output).flatten()
        except NotFittedError as nfe:
            print("The PCA instance is not fitted yet, returning null embedding")
            output = torch.zeros(self.output_space["embedding"].shape, dtype=self.output_space["embedding"].dtype)

        if (self.CURRENT_RUN_INDEX % self.config.fit_period == 0) and (self.CURRENT_RUN_INDEX > 0) and is_output_new_discovery:
            self.fit_update()

        output = {"embedding": output}

        if self.config.expand_output_space:
            self.output_space.expand(output)

        return output

    def fit_update(self):
        history = self._access_history()
        input_library = np.stack([history['input'][i][self.wrapped_input_space_key].flatten() for i in range(len(history['input']))])
        self.algorithm.fit(input_library)
        self._call_output_history_update()
