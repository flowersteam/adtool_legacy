FROM python:3.11-slim

RUN apt-get update -y && apt-get install -y gcc

# AutoDiscServer dependency resolution
# NOTE: can only run in the build context provided in docker-compose.yml
COPY services/AutoDiscServer/flask_app/app.py /tmp/src/services/flask_app/app.py
COPY services/AutoDiscServer/flask_app/requirements.txt /tmp/src/services/flask_app/requirements.txt
COPY libs/auto_disc /tmp/src/libs/auto_disc
WORKDIR /tmp/src/libs/auto_disc
RUN pip install -e . 
RUN pip install -r /tmp/src/services/flask_app/requirements.txt

# expeDB dependency resolution
COPY services/ExpeDB/API/flask_app/requirements.txt /tmp/src/expeDB/requirements.txt
RUN pip install -r /tmp/src/expeDB/requirements.txt

# developer stuff
RUN pip install pdoc pytest pytest-mock autopep8 flake8 mypy
RUN pip install sqlalchemy

CMD bash