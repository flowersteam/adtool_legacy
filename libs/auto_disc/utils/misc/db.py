import os
import pickle
import json
import base64
from copy import deepcopy
from typing import Any, Dict, Optional
from tinydb import TinyDB
from tinydb.storages import MemoryStorage


class DB(TinyDB):
    def __init__(self, file_path='./') -> None:
        self.db_file = file_path + 'db.json'
        super().__init__(storage=MemoryStorage)

    def close(self) -> None:
        super().close()
        # os.remove(self.db_file)

    def select_partial(self, documents, keys, new_keys=None):
        '''
            Select only some keys in documents. Use `new_keys` to rename these keys in the document returned.
        '''
        results = []
        result_keys = keys if new_keys is None else new_keys
        assert len(keys) == len(result_keys)

        for document in documents:
            current_result = {}
            for idx, key in enumerate(keys):
                current_result[result_keys[idx]] = document[key]
            results.append(current_result)
        
        return results

# class JSONOrPickleStorage(JSONStorage):
#     def _check_json_or_convert_pickle(self, data):
#         if isinstance(data, dict):
#             for k,v in data.items():
#                 data[k] = self._check_json_or_convert_pickle(v)
#         else:
#             if not self._is_jsonable(data):
#                 bytes = pickle.dumps(data)
#                 encoded = base64.b64encode(bytes)
#                 data = encoded.decode('ascii')
#         return data

#     def _is_jsonable(self, x):
#         try:
#             json.dumps(x)
#             return True
#         except (TypeError, OverflowError):
#             return False

#     def _check_decode_pickle(self, data):
#         if isinstance(data, dict):
#             for k,v in data.items():
#                 data[k] = self._check_decode_pickle(v)
#         else:
#             if isinstance(data, str) and data.isascii():
#                 encoded = base64.b64decode(data)
#                 data = pickle.loads(encoded)
#         return data

#     def write(self, data: Dict[str, Dict[str, Any]]):
#         _data = self._check_json_or_convert_pickle(deepcopy(data))
#         super().write(_data)

#     def read(self) -> Optional[Dict[str, Dict[str, Any]]]:
#         data = super().read()
#         data = self._check_decode_pickle(data)
#         return data