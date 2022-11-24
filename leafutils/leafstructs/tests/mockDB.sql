BEGIN;

CREATE TABLE trajectories (
    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    content INTEGER NOT NULL
    );
CREATE TABLE tree (
    id INTEGER NOT NULL REFERENCES trajectories(id),
    child_id INTEGER REFERENCES trajectories(id)
    );
INSERT INTO trajectories (content) VALUES (1); 
INSERT INTO trajectories (content) VALUES (2);
INSERT INTO trajectories (content) VALUES (3);
INSERT INTO trajectories (content) VALUES (4);
INSERT INTO trajectories (content) VALUES (5);
INSERT INTO trajectories (content) VALUES (4);
INSERT INTO trajectories (content) VALUES (8);
INSERT INTO tree (id, child_id) VALUES (1, 2);
INSERT INTO tree (id, child_id) VALUES (2, 3);
INSERT INTO tree (id, child_id) VALUES (3, 4);
INSERT INTO tree (id, child_id) VALUES (4, 5);
INSERT INTO tree (id, child_id) VALUES (2, 6);
INSERT INTO tree (id, child_id) VALUES (6, 7);

COMMIT;
