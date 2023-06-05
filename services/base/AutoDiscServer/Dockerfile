FROM --platform=linux/amd64 python:3.9.16-slim as library

# install poetry build tool
RUN pip install 'poetry==1.5.1'
RUN poetry config virtualenvs.create false

# auto_disc dependency resolution
COPY libs/auto_disc/pyproject.toml /usr/src/libs/auto_disc/pyproject.toml
WORKDIR /usr/src/libs/auto_disc
RUN poetry install --no-root --no-directory
# install root package and path dependencies
COPY libs/auto_disc /usr/src/libs/auto_disc
RUN poetry install --no-dev


FROM library as server
# nginx setup
RUN apt-get update -y && apt-get install -y nginx
RUN rm /etc/nginx/sites-enabled/default
COPY nginx/nginx.conf /etc/nginx/sites-available/flask
RUN ln -s /etc/nginx/sites-available/flask /etc/nginx/sites-enabled/flask

# AutoDiscServer dependency resolution
COPY flask_app/app.py /usr/src/services/flask_app/app.py
COPY flask_app/requirements.txt /usr/src/services/flask_app/requirements.txt
RUN pip install -r /usr/src/services/flask_app/requirements.txt

# extra utils needed
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
