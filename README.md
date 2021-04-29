# AutomatedDiscoveryTool
Software for assisted automated discovery and exploration of complex systems.

## Contributing
Please follow these instructions when contributing on the project.

### Installation
#### AutoDisc Lib
1. If you do not already have it, please install [Conda](https://www.anaconda.com/)
2. Create *autoDiscTool* conda environment: `conda create --name autoDiscTool python=3.7`
3. Activate *autoDiscTool* conda environment: `conda activate autoDiscTool`
4. Install the required conda packages in the environment (one by one to deal with dependencies errors): `while read requirement; do conda install --yes $requirement --channel default --channel anaconda --channel conda-forge --channel pytorch; done < requirements.txt`

#### AutoDiscServer
1. Install flask: `pip install flask`

### App DB
1. Install Docker: [`LINK`](https://docs.docker.com/engine/install/)
2. Install Docker-compose: [`LINK`]https://docs.docker.com/compose/install/

### Commits
Please attach every commit to an issue or a merge request. For issues, add #ID at the beginning of your commit message (with ID the id of the issue).

### Startinig the project
#### AutoDiscServer
Launch the flask server: `python services/AutoDiscServer/app.py`.
#### App DB
Start services: `docker-compose -f services/docker-compose.yml up`