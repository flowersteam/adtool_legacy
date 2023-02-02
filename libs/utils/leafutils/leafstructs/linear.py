from leaf.leaf import Leaf, Locator, LeafUID
from typing import Tuple, List, Union, Any
from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import Engine
from hashlib import sha1
import codecs
import tempfile
import pickle
import os


class Stepper(Leaf):
    def __init__(self):
        super().__init__()
        self.buffer = []


class LinearLocator(Locator):
    """
    Locator which stores branching, linear data
    with minimal redundancies in a SQLite db.

    To use, one should override `deserialize` of
    your Leaf module class to output a serialized Python object `x`
    with `x.buffer` a List type. I.e., any Leaf which uses LinearLocator
    should define an instance variable `buffer` of List type, e.g.,
        ```
        class A(Leaf):

            def __init__(self, buffer = []):
                super().__init__()
                self.buffer = buffer
        ```

    NOTE: This means that the `save_leaf` recursion will not recurse into
        the buffer, and any Leaf types inside will serialize naively.
    """

    def __init__(self, resource_uri: str = ""):
        self.resource_uri = resource_uri
        self.parent_id = -1

    def store(self, bin: bytes, parent_id: int = -1) -> 'LeafUID':
        """
        Stores the bin as a child node of the node given by parent_id.
        #### Returns:
        - leaf_uid (LeafUID): formatted path indicating the DB UID and the 
                              SQLite unique key corresponding
                              to the inserted node
        """

        db_name, data_bin = self._parse_bin(bin)

        # create subfolder if not exist
        subdir = os.path.join(self.resource_uri, db_name)
        if not os.path.exists(subdir):
            os.mkdir(subdir)

        # store metadata binary
        loaded_obj = pickle.loads(data_bin)
        del loaded_obj.buffer
        metadata_bin = pickle.dumps(loaded_obj)
        with open(os.path.join(subdir, "metadata"), "wb") as f:
            f.write(metadata_bin)

        # store data binary
        row_id = self._store_data(data_bin, parent_id, db_name)

        # store leaf_uids in filesystem
        leaf_path = os.path.join(subdir, str(row_id))
        open(leaf_path, "a").close()  # touch empty file

        # return leaf_uid
        leaf_uid = db_name + ":" + row_id

        return leaf_uid

    def _store_data(self,
                    data_bin: bytes,
                    parent_id: int,
                    db_name: str) -> str:
        """
        Store the unpadded binary data (i.e., without tag).
        """

        db_url = self._db_name_to_db_url(db_name)
        # initializes db if it does not exist
        self._init_db(db_url)

        with EngineContext(db_url) as engine:
            # default setting if not set at function call
            if parent_id == -1:
                parent_id = self.parent_id

            # insert node
            delta = self._convert_bytes_to_base64_str(data_bin)
            row_id = self._insert_node(engine, delta, parent_id)

        # update parent_uid in cache
        self.parent_id = row_id
        return str(row_id)

    def retrieve(self, uid: 'LeafUID', length: int = 1) -> bytes:
        """
        Retrieve trajectory of given length of saved data starting from
        the leaf node given by uid, then traversing backwards towards the root.
        NOTE: length=0 corresponds to the entire trajectory
        #### Returns:
        - bin (bytes): trajectory packed as a python object x with
                       x.buffer being an array of binary python objects
        """
        try:
            db_name, row_id = uid.split(":")
        # check in case too many strings are returned
        except ValueError:
            raise ValueError("leaf_uid is not properly formatted.")

        db_url = self._db_name_to_db_url(db_name)
        with EngineContext(db_url) as engine:
            _, trajectory, _ = self._get_trajectory(engine, row_id, length)
            stepper = Stepper()
            stepper.buffer = trajectory
            bin = stepper.serialize()

        return bin

    def _db_name_to_db_url(self, db_name: str) -> str:
        db_url = os.path.join(self.resource_uri, db_name, "lineardb")
        return db_url

    @staticmethod
    def _init_db(db_url: str) -> None:
        create_traj_statement = \
            '''
        CREATE TABLE trajectories (
            id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            content BLOB NOT NULL
            );
            '''
        create_tree_statement = \
            '''
        CREATE TABLE tree (
            id INTEGER NOT NULL REFERENCES trajectories(id),
            child_id INTEGER REFERENCES trajectories(id)
            );
            '''
        if os.path.exists(db_url):
            return
        else:
            with EngineContext(db_url) as engine:
                with engine.begin() as con:
                    con.execute(create_traj_statement)
                    con.execute(create_tree_statement)
            return

    @staticmethod
    def _convert_bytes_to_base64_str(bin: bytes) -> str:
        tmp = codecs.encode(bin, encoding="base64").decode()
        # prune newline
        if tmp[-1] == '\n':
            out_str = tmp[:-1]
        return out_str

    @staticmethod
    def _convert_base64_str_to_bytes(b64_str: str) -> bytes:
        return codecs.decode(b64_str.encode(), encoding="base64")

    @staticmethod
    def _insert_node(engine: Engine, delta: str, parent_id: int = -1) -> int:
        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO trajectories (content) VALUES (:z)"),
                {"z": delta})

            # SQLite exclusive function call to get ID of just inserted row
            res = conn.execute(text("SELECT last_insert_rowid()"))
            id = res.all()[0][0]

            if parent_id != -1:
                conn.execute(
                    text("INSERT INTO tree (id, child_id) VALUES (:y, :z)"),
                    {"y": parent_id, "z": id})
        return id

    @staticmethod
    def _get_trajectory(engine, id: int, trajectory_length: int = 1
                        ) -> Tuple[List[int], List[bytes], List[int]]:
        """
        Retrieves trajectory which has HEAD at id and returns the last 
        trajectory_length elements
        """
        query = \
            '''
            WITH tree_inheritance AS (
                WITH RECURSIVE cte (id, child_id, depth) AS (
                    SELECT :z, NULL, 0
                    UNION ALL
                    SELECT id, child_id, 1
                        FROM tree WHERE tree.child_id = :z
                    UNION
                    SELECT y.id, y.child_id, depth + 1
                        FROM cte AS x INNER JOIN tree AS y ON y.child_id = x.id
                    )
                SELECT * from cte
            )
            SELECT x.id, y.content, x.depth
                FROM
                    tree_inheritance AS x INNER JOIN trajectories AS y
                        ON x.id = y.id
                ORDER BY depth DESC
            '''
        with engine.connect() as conn:
            result = conn.execute(text(query), {"z": id})
            # persist the result into Python list
            result = result.all()

            ids: List[int] = [w for (w, _, _) in result]
            trajectory = [LinearLocator._convert_base64_str_to_bytes(
                w) for (_, w, _) in result]
            depths: List[int] = [w for (_, _, w) in result]

        return ids[-trajectory_length:], \
            trajectory[-trajectory_length:], \
            depths[-trajectory_length:]

    @classmethod
    def _parse_bin(cls, bin: bytes) -> Tuple[str, bytes]:
        if bin[20:24] != bytes.fromhex("deadbeef"):
            raise ValueError("Parsed bin is corrupted.")
        else:
            return bin[0:20].hex(), bin[24:]

    @classmethod
    def _parse_leaf_uid(cls, uid: LeafUID) -> Tuple[str, int]:
        db_name, node_id = uid.split(":")
        return db_name, int(node_id)

    @staticmethod
    def hash(bin: bytes) -> 'LeafUID':
        """ LinearLocator must use SHA1 hashes """
        return LeafUID(sha1(bin).hexdigest())

    @event.listens_for(Engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


class EngineContext:
    def __init__(self, db_url: str = ""):
        self.db_url = db_url

    def __enter__(self):
        self.engine = create_engine(
            f"sqlite+pysqlite:///{self.db_url}", echo=True
        )
        return self.engine

    def __exit__(self, *args):
        self.engine.dispose()
        del self.engine
        return
