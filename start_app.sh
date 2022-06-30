#!/bin/sh

chmod 777 -R services/JupyterLab/Notebooks/
docker-compose -f services/docker-compose.yml --profile prod up --build
