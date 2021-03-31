from libs.systems.python_systems import PythonLenia
from libs.auto_disc.output_representations.specific import LeniaImageRepresentation
from libs.auto_disc.explorers import IMGEPExplorer

from libs.auto_disc import ExperimentPipeline

from libs.auto_disc.utils.callbacks import CustomSaveCallback
from libs.auto_disc.utils.callbacks import CustomPrintCallback

import numpy as np
from libs.utils.auto_disc_parameters import AutoDiscParameter, ConfigParameterBinding, ParameterTypesEnum, AutoDiscSpaceDefinition, AutoDiscMutationDefinition
from libs.auto_disc.utils.sampling.sample_value import sample_value
from libs.auto_disc.utils.sampling.mutate_value import mutate_value

import asyncio

if __name__ == "__main__":
    experiment = ExperimentPipeline(
        system=PythonLenia(
            config_kwargs={
                'SX':256, 
                'SY':256
            }),
        explorer=IMGEPExplorer(),
        input_wrappers=None,
        output_representations=[LeniaImageRepresentation()],
        #on_exploration_callbacks=[CustomPrintCallback("Newly explored output !")]
        on_exploration_callbacks=[CustomSaveCallback("/home/mperie/project/save_callback/")]
    )

    asyncio.run(experiment.run(100))










# attrs = vars(experiment._system.input_space)
# print("\n".join("%s: %s " % item for item in attrs.items()))

# var = AutoDiscParameter(
#                     name="m", 
#                     type=ParameterTypesEnum.get('SPACE'),
#                     default=AutoDiscSpaceDefinition(
#                         dims=[5],
#                         bounds=[0, 10],
#                         type=ParameterTypesEnum.get('FLOAT'),
#                         mutation=AutoDiscMutationDefinition("gauss", 0.1)
#                     ))


# attrs = vars(var._default)
# print("\n".join("%s: %s " % item for item in attrs.items()))


# b=[[6.5, 0.6, 12.02], [0.5, 0.6, 5]]
# import torch
# b=torch.tensor([6.5, 0.6, 8.02])
# b=(6.5, 0.6, 12.02)
# b=6.5
# b = mutate_value(b, var._default, mutation_factor=5)
# print(b)