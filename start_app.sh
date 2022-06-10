#!/bin/sh

chmod 777 -R services/JupyterLab/Notebooks/
sudo docker-compose -f services/docker-compose.yml --profile prod up --build