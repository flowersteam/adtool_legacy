from auto_disc.systems.python_systems import *
from auto_disc.explorers import *
from auto_disc.input_wrappers.generic import *
from auto_disc.input_wrappers.specific import *
from auto_disc.output_representations.generic import *
from auto_disc.output_representations.specific import *

REGISTRATION = {
    'systems':{
        'PythonLenia': PythonLenia
    },
    'explorers':{
        'IMGEPExplorer': IMGEPExplorer
    },
    'input_wrappers':{
        'generic.CPPN': CppnInputWrapper,
        'generic.TimesN': TimesNInputWrapper 
    },
    'output_representations':{
        'specific.LeniaFlattenImage': LeniaImageRepresentation,
        'specific.LeniaStatistics': LeniaHandDefinedRepresentation
    }
}