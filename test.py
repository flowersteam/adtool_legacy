from libs.systems.python_systems.Lenia import Lenia
from libs.auto_disc.output_representations.specific.LeniaOutputRepresentation import LeniaImageRepresentation
from libs.auto_disc.ExperimentPipeline import ExperimentPipeline


experiment = ExperimentPipeline(
    system=Lenia(SX=256, SY=256),
    explorer=None,
    input_wrappers=None,
    output_representations=[LeniaImageRepresentation()],
    on_exploration_callbacks=[lambda params, output: print("Newly explored output !")]
)

experiment.run(5000)