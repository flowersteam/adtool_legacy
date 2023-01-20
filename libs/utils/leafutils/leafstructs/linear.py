from leaf.leaf import Leaf, Locator, LeafUID
from typing import Tuple, List, Union, Any
from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import Engine
import codecs
import tempfile


class Stepper(Leaf):
    def __init__(self):
        super().__init__()
        self.buffer = []


class LinearStorage(Locator):
    """
    Locator which stores branching, linear data
    with minimal redundancies in a SQLite db.

    To use, one should override `deserialize` of
    your Leaf module class to output a serialized `Stepper`
    object, for example:
        ```
        class A(Leaf):

            def __init__(self, buffer = []):
                super().__init__()
                self.buffer = buffer

            def serialize(self):
                stepper = Stepper()
                stepper.buffer = self.buffer
                return stepper.serialize()
        ```
    """
    @event.listens_for(Engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

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

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "resource_uri":
            super().__setattr__(name, value)
            super().__setattr__("engine",
                                create_engine(
                                    f"sqlite+pysqlite:///{value}", echo=True)
                                )
        else:
            super().__setattr__(name, value)

        return

    def __init__(self, resource_uri: str = "", leaf_uid: int = -1):
        self.resource_uri = resource_uri
        self.leaf_uid = leaf_uid

        if resource_uri != "":
            self.engine = create_engine(
                f"sqlite+pysqlite:///{resource_uri}", echo=True
            )

    def store(self, bin: bytes, parent_id: int = -1) -> 'LeafUID':
        """
        Stores the bin as a child node of the node given by parent_id.
        #### Returns:
        - leaf_uid (LeafUID): indicating the SQLite unique key corresponding
                              to the inserted node
        """
        #        parent_id, delta = self._get_insertion_tuple(bin)
        if parent_id == -1:
            parent_id = self.leaf_uid
        delta = self._convert_bytes_to_base64_str(bin)
        id = self._insert_node(delta, parent_id)
        return id

    def retrieve(self, uid: 'LeafUID') -> bytes:
        """
        Retrieve entire trajectory of saved data starting from the leaf node
        given by uid, traversing backwards towards the root.
        #### Returns:
        - bin (bytes): trajectory packed as a Stepper bin
        """
        _, trajectory, _ = self._get_trajectory(uid)
        stepper = Stepper()
        stepper.buffer = trajectory
        bin = stepper.serialize()
        return bin

    def _insert_node(self, delta: str, parent_id: int = -1) -> int:
        with self.engine.begin() as conn:
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

    def _get_trajectory(self, id: int
                        ) -> Tuple[List[int], List[bytes], List[int]]:
        """
        Retrieves trajectory which has HEAD at id
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
        with self.engine.connect() as conn:
            result = conn.execute(text(query), {"z": id})
            # persist the result into Python list
            result = result.all()

            ids: List[int] = [w for (w, _, _) in result]
            trajectory = [self._convert_base64_str_to_bytes(
                w) for (_, w, _) in result]
            depths: List[int] = [w for (_, _, w) in result]

        return ids, trajectory, depths

    def _get_heads(self) -> List[int]:
        query = \
            '''
            SELECT id FROM trajectories
                WHERE id NOT IN (
                    SELECT id FROM tree
                )
            '''
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            heads = [x[0] for x in result]

        return heads

    def _match_backwards(self, buffer: list) -> List[int]:
        """
        Searches in the DB for the buffer sequence and returns
        the trajectory in the DB which matches, labelled by id.
        NOTE: The buffer sequence you provide must begin at the t=0 root.
        """

        heads = sorted(self._get_heads(), reverse=True)
        for head in heads:
            db_ids, db_trajectory, _ = self._get_trajectory(head)
            result = self._list_contains(buffer, db_trajectory)
            if result == (-1, -1):
                continue
            else:
                break
        else:
            raise Exception("Fatal error: trajectory not found in DB.")

        beg = result[0]
        end = result[1]
        return_ids = db_ids[beg:end]

        return return_ids

    def _get_insertion_tuple(self, bin: bytes) -> Tuple[int, str]:
        """
        Matches the binary-encoded sequence with the appropriate
        DB primary key of where to insert the latest time step (the parent).
        Returns this key and the binary delta (to insert)
        """
        # TODO: check corner case with len(heads) == 0, need to initialize

        # copy buffer from the binary
        tmp_stepper = Stepper()
        stepper = tmp_stepper.deserialize(bin)
        buffer = stepper.buffer

        # truncate the last element which is to be added
        delta = buffer[-1]
        buffer = buffer[:-1]
        delta = self._convert_bytes_to_base64_str(delta)

        # retrieve the id where to insert
        match_ids = self._match_backwards(buffer)
        parent_id = match_ids[-1]

        return (parent_id, delta)

    @staticmethod
    def _list_contains(small: list, big: list) -> Tuple[int, int]:
        """
        Substring match for lists.
        Note this function never returns boolean True.
        """
        for i in range(len(big)-len(small)+1):
            for j in range(len(small)):
                if big[i+j] != small[j]:
                    break
            # else is triggered if for loop exits without breaking
            else:
                return i, i+len(small)

        # return nonsense if no match is found
        return (-1, -1)
