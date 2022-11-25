FROM python:3.8-slim

RUN apt-get update -y
RUN apt-get install -y gcc

# ExpeDB dependency resolution
# NOTE: can only run in the build context provided in docker-compose.yml
COPY ./flask_app/app.py /usr/src/services/flask_app/app.py
COPY ./flask_app/requirements.txt /usr/src/services/flask_app/requirements.txt
RUN pip install -r /usr/src/services/flask_app/requirements.txt

# nginx setup
RUN apt-get install -y nginx 
RUN rm /etc/nginx/sites-enabled/default
COPY ./nginx/nginx.conf /etc/nginx/sites-available/flask
RUN ln -s /etc/nginx/sites-available/flask /etc/nginx/sites-enabled/flask

# dev tools
RUN apt-get install -y curl procps vim

# placed here to allowed efficient caching of src changes
COPY ./flask_app/ /usr/src/services/flask_app/
# start daemons
VOLUME ["/usr/src/services"]
WORKDIR /usr/src/services/flask_app

CMD gunicorn -w 4 -b 0.0.0.0:8000 app:app --access-logfile '-' --log-level 'debug' & nginx -g "daemon off;"