from auto_disc.utils.misc import History
from tinydb import TinyDB
from tinydb.storages import JSONStorage, MemoryStorage
from tinydb.queries import where, Query


class DB(TinyDB):
    def __init__(self) -> None:
        # super().__init__('./db-cache.json', storage=JSONStorage)
        super().__init__(storage=MemoryStorage)

    def close(self) -> None:
        super().close()

    def to_autodisc_history(self, documents, keys, new_keys=None):
        '''
            Select only some keys in documents and return a History. Use `new_keys` to rename these keys in the returned History.
        '''
        result_keys = keys if new_keys is None else new_keys
        assert len(keys) == len(result_keys)
        results = History(result_keys)

        for document in documents:
            current_result = {}
            add_document = True
            for idx, key in enumerate(keys):
                if key in document:
                    current_result[result_keys[idx]] = document[key]
                else:
                    add_document = False

            if add_document:
                results.append(current_result)
        
        return results
    
    def __getitem__(self, index):
        '''
            Use indexing and slicing over db.
        '''
        if isinstance(index, int):
            if index >= 0:
                return self.search(where("idx") == index)
            else:
                return self.search(where("idx") == len(self) + index)    
        elif isinstance(index, slice):
            db_idx = list(range(len(self)))
            return self.search(Query().idx.test(lambda val: val in db_idx[index]))
        else:
            raise NotImplementedError()

    def save(self):
        '''
            Save DB.
        '''
        return self

    def load(self, saved_dict):
        '''
            Reload DB.
        '''
        pass


    
    