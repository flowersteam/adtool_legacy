FROM --platform=linux/amd64 nvidia/cuda:12.1.1-base-ubuntu20.04 as library

# install pip and virtualenv for the system python environment (python3.8)
ENV DEBIAN_FRONTEND="noninteractive"
# TODO: load tz dynamically
# also fixes the issue #210
ENV TZ="Etc/UTC"
RUN apt-get update -y
RUN apt-get install -y python3-pip virtualenv python-is-python3

# bootstrap a virtual env in python3.9 for poetry
RUN apt-get install -y python3.9
ENV POETRY_HOME=/opt/poetry
ENV VIRTUAL_ENV=$POETRY_HOME
RUN virtualenv -p /usr/bin/python3.9 $VIRTUAL_ENV
# activate virtual environment and install poetry
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install poetry==1.5.1

# create and activate the autodisc-env virtualenv for the server
ENV VIRTUAL_ENV=/autodisc-env
RUN virtualenv -p /usr/bin/python3.9 /autodisc-env
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# at this point, the $PATH prefers /autodisc-env, then /opt/poetry, then system

# auto_disc dependency resolution
COPY libs/auto_disc/pyproject.toml /usr/src/libs/auto_disc/pyproject.toml
WORKDIR /usr/src/libs/auto_disc
RUN poetry install --no-root --no-directory
# adtool_default dependency resolution
COPY libs/default_modules/pyproject.toml /usr/src/libs/default_modules/pyproject.toml
WORKDIR /usr/src/libs/default_modules
RUN poetry install --no-root --no-directory
# install root package and path dependencies for auto_disc
COPY libs/auto_disc /usr/src/libs/auto_disc
WORKDIR /usr/src/libs/auto_disc
RUN poetry install --only main
# install root package and path dependencies for adtool_default
COPY libs/default_modules /usr/src/libs/default_modules
WORKDIR /usr/src/libs/default_modules
RUN poetry install --only main

FROM library as server
# nginx setup
RUN apt-get update -y && apt-get install -y nginx
RUN rm /etc/nginx/sites-enabled/default
COPY nginx/nginx.conf /etc/nginx/sites-available/flask
RUN ln -s /etc/nginx/sites-available/flask /etc/nginx/sites-enabled/flask

# AutoDiscServer dependency resolution
# TODO: convert to poetry package
COPY flask_app/app.py /usr/src/services/flask_app/app.py
COPY flask_app/requirements.txt /usr/src/services/flask_app/requirements.txt
RUN pip install -r /usr/src/services/flask_app/requirements.txt

# extra utils needed
# TODO: convert to poetry package
COPY libs/auto_disc_db /usr/src/libs/auto_disc_db
WORKDIR /usr/src/libs/auto_disc_db
RUN pip install -e .

# stage app
COPY flask_app/ /usr/src/services/flask_app/
# stage user custom modules
COPY libs/custom_modules /usr/src/libs/custom_modules
WORKDIR /usr/src/libs/custom_modules
RUN poetry install

# start daemons
WORKDIR /usr/src/services/flask_app
CMD gunicorn -w 4 -b 0.0.0.0:5201 app:app --access-logfile '-' --log-level 'debug' & nginx -g "daemon off;"
