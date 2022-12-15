from leaf.leaf import Leaf, Locator
from typing import Tuple, List
from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import Engine


class Stepper(Leaf):
    def __init__(self):
        super().__init__()
        self.buffer = []

    def step(self):
        length = len(self.buffer)
        self.buffer.append(length+1)


class LinearStorage(Locator):
    @event.listens_for(Engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    def __init__(self, db_url):
        self.engine = create_engine(f"sqlite+pysqlite:///{db_url}", echo=True)

    def store(self, bin: bytes) -> None:
        parent_id, content = self._get_insertion_tuple(bin)
        id = self._insert_node(content, parent_id)

        # assign the id in the instance so it can be retrieved
        self.retrieval_key = id
        return

    def retrieve(self) -> bytes:
        _, trajectory, _ = self._get_trajectory(self.retrieval_key)
        stepper = Stepper()
        stepper.buffer = trajectory
        bin = stepper.serialize()
        return bin

    def _insert_node(self, content: int, parent_id: int = None) -> int:
        with self.engine.begin() as conn:
            conn.execute(
                text("INSERT INTO trajectories (content) VALUES (:z)"),
                {"z": content})

            # SQLite exclusive function call to get ID of just inserted row
            res = conn.execute(text("SELECT last_insert_rowid()"))
            id = res.all()[0][0]

            if parent_id is not None:
                conn.execute(
                    text("INSERT INTO tree (id, child_id) VALUES (:y, :z)"),
                    {"y": parent_id, "z": id})
        return id

    def _get_trajectory(self, id: int):
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

            ids = [w for (w, _, _) in result]
            trajectory = [w for (_, w, _) in result]
            depths = [w for (_, _, w) in result]

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

    def _match_backwards(self, buffer: list):
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

    def _get_insertion_tuple(self, bin: bytes) -> Tuple[int, int]:
        """
        Matches the binary-encoded sequence with the appropriate
        DB primary key of where to insert the latest time step (i.e., the parent).
        Returns this key and the binary delta (to insert)
        TODO: here it's just an int, for MWE.
        """
        # TODO: check corner case with len(heads) == 0, need to initialize

        # copy buffer from the binary
        tmp_stepper = Stepper()
        stepper = tmp_stepper.deserialize(bin)
        buffer = stepper.buffer

        # truncate the last element which is to be added
        content = buffer[-1]
        buffer = buffer[:-1]

        # retrieve the id where to insert
        match_ids = self._match_backwards(buffer)
        parent_id = match_ids[-1]

        return (parent_id, content)

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
            # else is triggered if for loop exists without breaking
            else:
                return i, i+len(small)

        # return nonsense if no match is found
        return (-1, -1)
