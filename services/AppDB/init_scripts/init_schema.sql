CREATE TABLE experiments (
	id serial PRIMARY KEY,
	name VARCHAR (255) NOT NULL,
	created_on TIMESTAMP NOT NULL,
	config jsonb NOT NULL
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
	parent_id INT NOT NULL,
	status VARCHAR (255) NOT NULL,
	error_message VARCHAR (255) NOT NULL,
	FOREIGN KEY (experiment_id)
    	REFERENCES experiments (id)
);