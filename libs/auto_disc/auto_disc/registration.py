from auto_disc.systems.python_systems import *
from auto_disc.explorers import *
from auto_disc.input_wrappers.generic import *
from auto_disc.input_wrappers.specific import *
from auto_disc.output_representations.generic import *
from auto_disc.output_representations.specific import *
import auto_disc.utils.callbacks.on_discovery_callbacks as on_discovery_callbacks
import auto_disc.utils.callbacks.on_cancelled_callbacks as on_cancelled_callbacks
import auto_disc.utils.callbacks.on_error_callbacks as on_error_callbacks
import auto_disc.utils.callbacks.on_finished_callbacks as on_finished_callbacks
import auto_disc.utils.callbacks.on_save_callbacks as on_save_callbacks
import auto_disc.utils.callbacks.on_save_finished_callbacks as on_save_finished_callbacks
import auto_disc.utils.callbacks.interact_callbacks as interact_callbacks
from auto_disc.utils.logger.handlers import *

REGISTRATION = {
    'systems':{
        'PythonLenia': PythonLenia,
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
    },
    'callbacks': {
        'on_discovery':{
            'base': on_discovery_callbacks.BaseOnDiscoveryCallback,
            'disk': on_discovery_callbacks.OnDiscoverySaveCallbackOnDisk
        },
        'on_cancelled':{
            'base': on_cancelled_callbacks.BaseOnCancelledCallback
        },
        'on_error':{
            'base': on_error_callbacks.BaseOnErrorCallback
        },
        'on_finished':{
            'base': on_finished_callbacks.BaseOnFinishedCallback
        },
        'on_saved':{
            'base': on_save_callbacks.BaseOnSaveCallback,
            'disk': on_save_callbacks.OnSaveModulesOnDiskCallback
        },
        'on_save_finished':{
            'base': on_save_finished_callbacks.BaseOnSaveFinishedCallback
        },
        'interact':{
            'base': interact_callbacks.BaseInteractCallback,
            'saveExpeDB': interact_callbacks.SaveExpeDBInteractCallback,
            'saveDisk': interact_callbacks.SaveDiskInteractCallback,
        },
    },
    'logger_handlers':{
        'logFile': SetFileHandler
    }
}