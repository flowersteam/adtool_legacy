FROM --platform=linux/amd64 nvidia/cuda:12.3.2-base-ubuntu22.04 as library

# install pip and virtualenv for the system python environment (python3.8)
ENV DEBIAN_FRONTEND="noninteractive"
# TODO: load tz dynamically, would also fix the issue #210
ENV TZ="Etc/UTC"
RUN apt-get update -y && apt-get install -y python3-pip virtualenv python-is-python3

# bootstrap a virtual env in python3.11 for poetry
RUN apt-get install -y python3.11
ENV POETRY_HOME=/opt/poetry
ENV VIRTUAL_ENV=$POETRY_HOME
RUN virtualenv -p /usr/bin/python3.11 $VIRTUAL_ENV
# activate virtual environment and install poetry
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install poetry==1.*

# create and activate the autodisc-env virtualenv for the server
ENV VIRTUAL_ENV=/autodisc-env
RUN virtualenv -p /usr/bin/python3.11 /autodisc-env
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# at this point, the $PATH prefers /autodisc-env, then /opt/poetry, then system

# autodisc-server-libs dependency resolution
COPY libs/pyproject.toml /usr/src/libs/pyproject.toml
WORKDIR /usr/src/libs
RUN poetry install --no-root --no-directory
# install root package and path dependencies for autodisc-server-libs
# except for custom modules
COPY libs /usr/src/libs
WORKDIR /usr/src/libs
RUN rm -rf adtool_custom
RUN poetry lock; poetry install --only main

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

# stage app
COPY flask_app/ /usr/src/services/flask_app/
# stage user custom modules in adtool_custom
COPY libs/adtool_custom /usr/src/libs/adtool_custom
WORKDIR /usr/src/libs
RUN python build.py # injects custom modules into pyproject.toml
RUN poetry lock; poetry install

# start daemons
WORKDIR /usr/src/services/flask_app
CMD gunicorn -w 4 -b 0.0.0.0:5201 app:app --access-logfile '-' --log-level 'debug' & nginx -g "daemon off;"
