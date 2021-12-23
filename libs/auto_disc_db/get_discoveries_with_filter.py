import json
import requests

def _get_discoveries_with_filter(_filter, _query=None):
    _filter = "filter="+json.dumps(_filter)
    _query = "&query="+json.dumps(_query) if _query else ""
    return json.loads(requests.get(url = "http://127.0.0.1:5001/discoveries?{}{}".format(_filter, _query)).content.decode())