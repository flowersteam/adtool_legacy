from sqlalchemy import text
import os
import pathlib
import pytest
from leafutils.leafstructs.linear import LinearLocator, EngineContext, Stepper
from leaf.leaf import LeafUID
import pickle
from hashlib import sha1
import shutil


def setup_function(function):
    import sqlite3
    global FILE_PATH, DB_PATH, SCRIPT_PATH, DB_NAME

    SCRIPT_PATH = str(pathlib.Path(__file__).parent.resolve())
    FILE_PATH = os.path.join(SCRIPT_PATH, "tmp")
    os.mkdir(FILE_PATH)
    DB_NAME = '4f700578d4ee31c037562ff8ae00d4863b71c163'
    os.mkdir(os.path.join(FILE_PATH, DB_NAME))
    DB_REL_PATH = f"/{DB_NAME}/lineardb"
    SCRIPT_REL_PATH = "/mockDB.sql"
    DB_PATH = FILE_PATH + DB_REL_PATH
    SCRIPT_PATH = SCRIPT_PATH + SCRIPT_REL_PATH


def teardown_function(function):
    if os.path.exists(FILE_PATH):
        shutil.rmtree(FILE_PATH)

    return


def generate_mock_binary() -> bytes:
    def _pad_binary(bin: bytes) -> bytes:
        """
        Converts the output of Leaf.serialize() into a padded binary format
        with 20-byte SHA1 hash of Leaf metadata, magic byte sequence, and the
        original binary
        """
        stepper = pickle.loads(bin)
        del stepper.buffer
        sha1_hash = LinearLocator.hash(pickle.dumps(stepper))
        output_bin = bytes.fromhex(sha1_hash) + bytes.fromhex("deadbeef") + bin
        return output_bin
    s = Stepper()
    query_trajectory = [bytes(1), bytes(2), bytes(4), bytes(9)]
    s.buffer = query_trajectory
    bin = s.serialize()
    padded_bin = _pad_binary(bin)
    return padded_bin, bin


def generate_fake_data(db_url: str):
    import sqlite3
    con = sqlite3.connect(db_url)
    cur = con.cursor()
    with open(SCRIPT_PATH) as f:
        query_string = f.read()
        cur.executescript(query_string)
    return


def test_LinearLocator___init__():
    x = LinearLocator(FILE_PATH)
    assert x.resource_uri == FILE_PATH


def test_LinearLocator__init_db():
    x = LinearLocator(FILE_PATH)
    x._init_db(DB_PATH)
    assert os.path.exists(DB_PATH)


def test_LinearLocator__insert_node():
    x = LinearLocator(FILE_PATH)
    x._init_db(DB_PATH)

    def get_trajectory_table_length(engine):
        with engine.connect() as conn:
            result = conn.execute(text("SELECT * from trajectories"))
            length_table = len(result.all())
        return length_table

    def get_newest_insert(engine):
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM trajectories ORDER BY id DESC LIMIT 1"))
            new_row = result.one()
        return new_row

    with EngineContext(DB_PATH) as engine:
        length_table = get_trajectory_table_length(engine)
        x._insert_node(engine, 1)
        new_length_table = get_trajectory_table_length(engine)
        new_row = get_newest_insert(engine)

    # assert new row is inserted properly into trajectories table
    assert new_length_table == length_table + 1
    assert new_row[1] == 1


def test_LinearLocator__get_trajectory():
    x = LinearLocator(FILE_PATH)
    x._init_db(DB_PATH)
    generate_fake_data(DB_PATH)

    with EngineContext(DB_PATH) as engine:
        _, trajectory, depths = x._get_trajectory(engine, 5, 0)
        assert trajectory == [bytes(1), bytes(2), bytes(3), bytes(4), bytes(5)]
        _, trajectory, depths = x._get_trajectory(engine, 7, 3)
        assert trajectory == [bytes(2), bytes(4), bytes(8)]
        assert len(trajectory) - 1 == depths[0]


def test_LinearLocator__parse_bin():
    bin, _ = generate_mock_binary()
    sha1_hash, data_bin = LinearLocator._parse_bin(bin)

    stepper = pickle.loads(data_bin)
    assert stepper.buffer == [bytes(1), bytes(2), bytes(4), bytes(9)]

    del stepper.buffer
    assert sha1(pickle.dumps(stepper)).hexdigest() == sha1_hash


def test_LinearLocator__parse_leaf_uid():
    test_str = "asdiufgapsudf:2"
    db_name, node_id = LinearLocator._parse_leaf_uid(test_str)
    assert db_name == "asdiufgapsudf", 2


def test_LinearLocator_store():
    x = LinearLocator(FILE_PATH)
    x._init_db(DB_PATH)
    generate_fake_data(DB_PATH)

    padded_bin, data_bin = generate_mock_binary()

    retrieval_key = x.store(padded_bin, 6)

    # assert that retrieval_key is stored
    # and can successfully retrieve trajectory
    db_name, row_id = LinearLocator._parse_leaf_uid(retrieval_key)
    subdir = os.path.join(FILE_PATH, db_name)
    db_url = os.path.join(subdir, "lineardb")

    with EngineContext(db_url) as engine:
        ids, trajectory, _ = x._get_trajectory(engine, row_id, 0)
        assert ids == [1, 2, 6, 8]
        assert trajectory == [bytes(1), bytes(2), bytes(4), data_bin]
    assert len(os.listdir(FILE_PATH)) == 1
    assert len(os.listdir(subdir)) == 3
    assert os.path.exists(os.path.join(subdir, "8"))


def test_LinearLocator_retrieve():
    x = LinearLocator(FILE_PATH)
    x._init_db(DB_PATH)
    generate_fake_data(DB_PATH)

    # mock storage of sequence
    retrieval_key = DB_NAME + ":7"

    bin = x.retrieve(retrieval_key, 0)
    tmp_stepper = Stepper()
    stepper = tmp_stepper.deserialize(bin)
    assert stepper.buffer == [bytes(1), bytes(2), bytes(4), bytes(8)]
