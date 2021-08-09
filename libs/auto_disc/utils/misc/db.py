from auto_disc.utils.misc import History
from tinydb import TinyDB
from tinydb.storages import MemoryStorage


class DB(TinyDB):
    def __init__(self) -> None:
        super().__init__(storage=MemoryStorage)

    def close(self) -> None:
        super().close()

    def to_autodisc_history(self, documents, keys, new_keys=None):
        '''
            Select only some keys in documents and returns a History. Use `new_keys` to rename these keys in the returned History.
        '''
        result_keys = keys if new_keys is None else new_keys
        assert len(keys) == len(result_keys)
        results = History(result_keys)

        for document in documents:
            current_result = {}
            for idx, key in enumerate(keys):
                current_result[result_keys[idx]] = document[key]
            results.append(current_result)
        
        return results


    
    