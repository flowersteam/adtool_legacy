import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../"))

from auto_disc.systems.python_systems import PythonLenia
from auto_disc.output_representations.specific import LeniaImageRepresentation
from auto_disc.input_wrappers.generic import TimesNInputWrapper, CppnInputWrapper
from auto_disc.explorers import IMGEPExplorer

from auto_disc import ExperimentPipeline

from auto_disc.utils.callbacks import CustomSaveCallback
from auto_disc.utils.callbacks import CustomPrintCallback

if __name__ == "__main__":
    experiment = ExperimentPipeline(
        system=PythonLenia(),
        explorer=IMGEPExplorer(),
        input_wrappers=[CppnInputWrapper('init_state')], # Starting from the explorer !
        output_representations=[LeniaImageRepresentation()], # Starting from the system !
        on_exploration_callbacks=[CustomPrintCallback("Newly explored output !"), CustomSaveCallback("./experiment_results/")]
    )

    experiment.run(100)

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