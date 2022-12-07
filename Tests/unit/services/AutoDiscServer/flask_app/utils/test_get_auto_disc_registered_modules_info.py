#region import
import os 
import sys
from copy import deepcopy
from unittest import mock

classToTestFolderPath = os.path.dirname(__file__)
classToTestFolderPath = os.path.abspath(os.path.join(classToTestFolderPath, "../"*6 ))

AutoDiscServerPath = classToTestFolderPath + "/services/AutoDiscServer"
auto_discFolderPath = classToTestFolderPath + "/libs/auto_disc/auto_disc"

sys.path.append(os.path.dirname(AutoDiscServerPath))
sys.path.append(os.path.dirname(auto_discFolderPath))

from AutoDiscServer.flask_app.utils.get_auto_disc_registered_modules_info import get_auto_disc_registered_modules_info, get_auto_disc_registered_callbacks, check_jsonify
from auto_disc import REGISTRATION
#endregion

#region test check_jsonify

def test_check_jsonify():
    ## init 
    testDict ={"testInf":float("inf")}
    ## exec
    check_jsonify(testDict)
    ## assert
    assert testDict["testInf"] == "inf"


#endregion

#region test get_auto_disc_registered_modules_info
def test_get_auto_disc_registered_modules_info_output_representations():
    ## init
    info_output = REGISTRATION['output_representations']
    ## exec
    infos = get_auto_disc_registered_modules_info(info_output)
    ## assert
    assert "output_space" in infos[0]
    assert not "input_space" in infos[0]

def test_get_auto_disc_registered_modules_info_input_wrappers():
    ## init
    info_output = REGISTRATION['input_wrappers']
    ## exec
    infos = get_auto_disc_registered_modules_info(info_output)
    ## assert
    assert not "output_space" in infos[0]
    assert "input_space" in infos[0]

#endregion

#region test get_auto_disc_registered_callbacks

def test_get_auto_disc_registered_callbacks():
    ## init
    
    ## exec
    info = get_auto_disc_registered_callbacks(REGISTRATION['callbacks'])
    ## assert
    assert len(info) == 7
    assert info[0]["name_callbacks_category"] == "on_discovery"
    assert info[1]["name_callbacks_category"] == "on_cancelled"
    assert info[2]["name_callbacks_category"] == "on_error"
    assert info[3]["name_callbacks_category"] == "on_finished"
    assert info[4]["name_callbacks_category"] == "on_saved"
    assert info[5]["name_callbacks_category"] == "on_save_finished"
    assert info[6]["name_callbacks_category"] == "interact"

#endregion