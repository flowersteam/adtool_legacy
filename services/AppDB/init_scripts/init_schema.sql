CREATE TABLE experiments (
	id serial PRIMARY KEY,
	name VARCHAR (255) NOT NULL,
	created_on TIMESTAMP NOT NULL,
	progress INT NOT NULL,
	exp_status INT NOT NULL,
	config jsonb NOT NULL,
	archived BOOLEAN NOT NULL,
	checkpoint_saves_archived BOOLEAN NOT NULL,
	discoveries_archived BOOLEAN NOT NULL,
	remote_run_id VARCHAR (255)
);
CREATE TABLE systems (
	id serial PRIMARY KEY,
	experiment_id INT NOT NULL,
	name VARCHAR (255) NOT NULL,
	config jsonb NOT NULL,
	FOREIGN KEY (experiment_id)
    	REFERENCES experiments (id)
);
CREATE TABLE explorers (
	id serial PRIMARY KEY,
	experiment_id INT NOT NULL,
	name VARCHAR (255) NOT NULL,
	config jsonb NOT NULL,
	FOREIGN KEY (experiment_id)
    	REFERENCES experiments (id)
);
CREATE TABLE input_wrappers (
	id serial PRIMARY KEY,
	experiment_id INT NOT NULL,
	name VARCHAR (255) NOT NULL,
	config jsonb NOT NULL,
	index INT NOT NULL,
	FOREIGN KEY (experiment_id)
    	REFERENCES experiments (id)
);
CREATE TABLE output_representations (
	id serial PRIMARY KEY,
	experiment_id INT NOT NULL,
	name VARCHAR (255) NOT NULL,
	config jsonb NOT NULL,
	index INT NOT NULL,
	FOREIGN KEY (experiment_id)
    	REFERENCES experiments (id)
);
CREATE TABLE checkpoints (
	id serial PRIMARY KEY,
	experiment_id INT NOT NULL,
	parent_id INT,
	status INT NOT NULL,
	-- error_message VARCHAR (8000) NOT NULL,
	FOREIGN KEY (experiment_id)
    	REFERENCES experiments (id)
);

CREATE TABLE log_levels(
	id serial PRIMARY KEY,
	name VARCHAR (255) NOT NULL
);

CREATE TABLE logs (
	id serial PRIMARY KEY,
	experiment_id INT NOT NULL,
	checkpoint_id INT NOT NULL,
	seed INT,
	log_level_id INT NOT NULL,
	name VARCHAR (255) NOT NULL,
	error_message VARCHAR (8000) NOT NULL,
	FOREIGN KEY (experiment_id)
    	REFERENCES experiments (id),
	FOREIGN KEY (checkpoint_id)
    	REFERENCES checkpoints (id),
	FOREIGN KEY (log_level_id)
    	REFERENCES log_levels (id)
);

CREATE TABLE preparing_logs(
	id serial PRIMARY KEY,
	experiment_id INT NOT NULL,
	message VARCHAR (8000) NOT NULL,
	FOREIGN KEY (experiment_id)
    	REFERENCES experiments (id)
);

INSERT INTO log_levels(name) VALUES('NOTSET');
INSERT INTO log_levels(name) VALUES('DEBUG');
INSERT INTO log_levels(name) VALUES('INFO');
INSERT INTO log_levels(name) VALUES('WARNING');
INSERT INTO log_levels(name) VALUES('ERROR');
INSERT INTO log_levels(name) VALUES('CRITICAL');