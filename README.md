# AutomatedDiscoveryTool
Software for assisted automated discovery and exploration of complex systems.

## Contributing
Please follow these instructions when contributing on the project.

### Installation
#### AutoDisc Lib
1. If you do not already have it, please install [Conda](https://www.anaconda.com/)
2. Create *autoDiscTool* conda environment: `conda env create -f libs/auto_disc/environment.yml `
3. Activate *autoDiscTool* conda environment: `conda activate autoDiscTool`
#### AutoDiscServer
1. Install flask: `pip install flask`
### App DB & Expe DB
1. Install Docker: [`LINK`](https://docs.docker.com/engine/install/)
2. Install Docker-compose: [`LINK`](https://docs.docker.com/compose/install/)
3. Generate containers `docker-compose -f services/docker-compose.yml create`

### Testing the auto_disc lib alone
1. Edit the `libs/tests/AutoDiscExperiment.py` file to configure the experiment
2. Launch the experiment: `python libs/tests/AutoDiscExperiment.py`

### Commits
Please attach every commit to an issue or a merge request. For issues, add #ID at the beginning of your commit message (with ID the id of the issue).

## Startinig the project
#### AutoDiscServer
Launch the flask server: `python services/AutoDiscServer/app.py`.
#### App DB
Start services: `docker-compose up app-db-api`
Add `-d` option for daemon.
#### Expe DB
Start service: `docker-compose up expe-db
Add `-d` option for daemon.