from sqlalchemy import text
import os
import pathlib
import pytest


def setup_function(function):
    import sqlite3
    global FILE_PATH, DB_PATH

    FILE_PATH = str(pathlib.Path(__file__).parent.resolve())
    DB_REL_PATH = "/tmp.sqlite"
    SCRIPT_REL_PATH = "/mockDB.sql"

    DB_PATH = FILE_PATH + DB_REL_PATH
    SCRIPT_PATH = FILE_PATH + SCRIPT_REL_PATH

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    with open(SCRIPT_PATH) as f:
        query_string = f.read()
        cur.executescript(query_string)

    return


def teardown_function(function):
    global DB_PATH
    os.remove(DB_PATH)

    return


def test_LinearLocator__init__():
    from leafutils.leafstructs.linear import LinearLocator
    x = LinearLocator(DB_PATH)
    assert str(x.engine.url) == f"sqlite+pysqlite:///{DB_PATH}"
    y = LinearLocator()
    y.resource_uri = DB_PATH
    assert str(y.engine.url) == f"sqlite+pysqlite:///{DB_PATH}"


def test_LinearLocator__get_heads():
    from leafutils.leafstructs.linear import LinearLocator
    x = LinearLocator(DB_PATH)
    heads = x._get_heads()

    assert len(heads) == 2
    assert 5 in heads
    assert 7 in heads

    return


def test_LinearLocator__get_trajectory():
    from leafutils.leafstructs.linear import LinearLocator
    x = LinearLocator(DB_PATH)
    _, trajectory, depths = x._get_trajectory(5)
    assert trajectory == [bytes(1), bytes(2), bytes(3), bytes(4), bytes(5)]
    _, trajectory, depths = x._get_trajectory(7)
    assert trajectory == [bytes(1), bytes(2), bytes(4), bytes(8)]
    assert len(trajectory) - 1 == depths[0]

    return


def test_LinearLocator__insert_node():
    from leafutils.leafstructs.linear import LinearLocator
    x = LinearLocator(DB_PATH)

    def get_trajectory_table_length(locator):
        with locator.engine.connect() as conn:
            result = conn.execute(text("SELECT * from trajectories"))
            length_table = len(result.all())
        return length_table

    def get_newest_insert(locator):
        with locator.engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM trajectories ORDER BY id DESC LIMIT 1"))
            new_row = result.one()
        return new_row

    length_table = get_trajectory_table_length(x)
    x._insert_node(1)
    new_length_table = get_trajectory_table_length(x)
    new_row = get_newest_insert(x)

    # assert new row is inserted properly into trajectories table
    assert new_length_table == length_table + 1
    assert new_row[1] == 1


def test_LinearLocator__match_backwards():
    from leafutils.leafstructs.linear import LinearLocator
    x = LinearLocator(DB_PATH)

    query_trajectory = [bytes(1), bytes(2), bytes(4), bytes(8)]
    return_ids = x._match_backwards(query_trajectory)
    assert return_ids == [1, 2, 6, 7]

    query_trajectory = [bytes(1), bytes(2), bytes(4)]
    return_ids = x._match_backwards(query_trajectory)
    assert return_ids == [1, 2, 6]

    query_trajectory = [bytes(1), bytes(2), bytes(3)]
    return_ids = x._match_backwards(query_trajectory)
    assert return_ids == [1, 2, 3]

    query_trajectory = [bytes(1), bytes(2), bytes(5)]
    with pytest.raises(Exception):
        return_ids = x._match_backwards(query_trajectory)


def test_LinearLocator__get_insertion_tuple():
    from leafutils.leafstructs.linear import LinearLocator, Stepper
    x = LinearLocator(DB_PATH)

    query_trajectory = [bytes(1), bytes(2), bytes(4), bytes(9)]
    stepper = Stepper()
    stepper.buffer = query_trajectory
    bin = stepper.serialize()

    parent_id, content = x._get_insertion_tuple(bin)
    assert content == x._convert_bytes_to_base64_str(bytes(9))
    assert parent_id == 6


def test_LinearLocator_store():
    from leafutils.leafstructs.linear import LinearLocator, Stepper
    x = LinearLocator(DB_PATH)

    query_trajectory = [bytes(1), bytes(2), bytes(4), bytes(9)]
    parent_id = 6
    stepper = Stepper()
    stepper.buffer = query_trajectory
    bin = stepper.serialize()

    retrieval_key = x.store(bin, 6)

    # assert that retrieval_key is stored
    # and can successfully retrieve trajectory
    ids, trajectory, _ = x._get_trajectory(retrieval_key)
    assert ids == [1, 2, 6, 8]
    assert trajectory == [bytes(1), bytes(2), bytes(4), bin]

    # assert unpacking of stepper object
    new_stepper = stepper.deserialize(bin)
    assert query_trajectory == new_stepper.buffer


def test_LinearLocator_store_instance_var():
    from leafutils.leafstructs.linear import LinearLocator, Stepper
    x = LinearLocator(DB_PATH, leaf_uid=6)

    query_trajectory = [bytes(1), bytes(2), bytes(4), bytes(9)]
    stepper = Stepper()
    stepper.buffer = query_trajectory
    bin = stepper.serialize()

    retrieval_key = x.store(bin)

    # assert that retrieval_key is stored
    # and can successfully retrieve trajectory
    ids, trajectory, _ = x._get_trajectory(retrieval_key)
    assert ids == [1, 2, 6, 8]
    assert trajectory == [bytes(1), bytes(2), bytes(4), bin]


def test_LinearLocator_retrieve():
    from leafutils.leafstructs.linear import LinearLocator, Stepper
    x = LinearLocator(DB_PATH)

    # mock storage of sequence
    retrieval_key = 7

    bin = x.retrieve(retrieval_key)
    tmp_stepper = Stepper()
    stepper = tmp_stepper.deserialize(bin)
    assert stepper.buffer == [bytes(1), bytes(2), bytes(4), bytes(8)]
