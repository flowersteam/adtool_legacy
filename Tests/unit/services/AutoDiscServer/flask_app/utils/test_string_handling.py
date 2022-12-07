#region import
import os 
import sys
from copy import deepcopy
from unittest import mock


classToTestFolderPath = os.path.dirname(__file__)
classToTestFolderPath = os.path.abspath(os.path.join(classToTestFolderPath, "../"*6 ))

AutoDiscServerPath = classToTestFolderPath + "/services/AutoDiscServer"

sys.path.append(os.path.dirname(AutoDiscServerPath))

from AutoDiscServer.flask_app.utils import match_except_number
#endregion

#region test match_except_number

def test_match_except_number():
    ## init
    txt1 = "id : 0, checkpoint : 120"
    txt2 = "id : 115, checkpoint : 10"
    txt3 = "ids : 0, checkpoints : 120"
    txt4 = "id:0, checkpoint:120"
    ## exec
    res1 = match_except_number(txt1, txt1)
    res2 = match_except_number(txt1, txt2)
    res3 = match_except_number(txt1, txt3)
    res4 = match_except_number(txt1, txt4)

    ## assert
    assert res1 == True
    assert res2 == True
    assert res3 == False
    assert res4 == False


#endregion