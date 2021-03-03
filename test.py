from libs.systems.python_systems.Lenia import Lenia
from libs.auto_disc.output_representations.specific.LeniaOutputRepresentation import LeniaImageRepresentation

Lenia.CONFIG_DEFINITION
system = Lenia()
input_wrapper = None
output_wrapper = LeniaImageRepresentation()

print(s.config)