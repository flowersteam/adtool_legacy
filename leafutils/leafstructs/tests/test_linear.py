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


def test_linearstorage__init__():
    from leafstructs.linear import LinearStorage
    x = LinearStorage(DB_PATH)
    assert str(x.engine.url) == f"sqlite+pysqlite:///{DB_PATH}"


def test_linearstorage__get_heads():
    from leafstructs.linear import LinearStorage
    x = LinearStorage(DB_PATH)
    heads = x._get_heads()

    assert len(heads) == 2
    assert 5 in heads
    assert 7 in heads

    return


def test_linearstorage__get_trajectory():
    from leafstructs.linear import LinearStorage
    x = LinearStorage(DB_PATH)
    _, trajectory, depths = x._get_trajectory(5)
    assert trajectory == [1, 2, 3, 4, 5]
    _, trajectory, depths = x._get_trajectory(7)
    assert trajectory == [1, 2, 4, 8]
    assert len(trajectory) - 1 == depths[0]

    return


def test_linearstorage__insert_node():
    from leafstructs.linear import LinearStorage
    x = LinearStorage(DB_PATH)

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


def test_linearstorage__match_backwards():
    from leafstructs.linear import LinearStorage
    x = LinearStorage(DB_PATH)

    query_trajectory = [1, 2, 4, 8]
    return_ids = x._match_backwards(query_trajectory)
    assert return_ids == [1, 2, 6, 7]

    query_trajectory = [1, 2, 4]
    return_ids = x._match_backwards(query_trajectory)
    assert return_ids == [1, 2, 6]

    query_trajectory = [1, 2, 3]
    return_ids = x._match_backwards(query_trajectory)
    assert return_ids == [1, 2, 3]

    query_trajectory = [1, 2, 5]
    with pytest.raises(Exception):
        return_ids = x._match_backwards(query_trajectory)


def test_linearstorage__get_insertion_tuple():
    from leafstructs.linear import LinearStorage, Stepper
    x = LinearStorage(DB_PATH)

    query_trajectory = [1, 2, 4, 9]
    stepper = Stepper()
    stepper.buffer = query_trajectory
    bin = stepper.serialize()

    parent_id, content = x._get_insertion_tuple(bin)
    assert content == 9
    assert parent_id == 6


def test_linearstorage_store():
    from leafstructs.linear import LinearStorage, Stepper
    x = LinearStorage(DB_PATH)

    query_trajectory = [1, 2, 4, 9]
    stepper = Stepper()
    stepper.buffer = query_trajectory
    bin = stepper.serialize()

    x.store(bin)

    # assert that retrieval_key is stored
    # and can successfully retrieve trajectory
    assert x.retrieval_key is not None
    ids, _, _ = x._get_trajectory(x.retrieval_key)
    assert ids == [1, 2, 6, 8]


def test_linearstorage_retrieve():
    from leafstructs.linear import LinearStorage, Stepper
    x = LinearStorage(DB_PATH)

    # mock storage of sequence
    x.retrieval_key = 7

    bin = x.retrieve()
    stepper = Stepper.deserialize(bin)
    assert stepper.buffer == [1, 2, 4, 8]
