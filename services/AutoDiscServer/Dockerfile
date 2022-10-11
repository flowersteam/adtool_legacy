FROM python:3.7.14-slim

RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install -y gcc git

# AutoDiscServer dependency resolution
# NOTE: can only run in the build context provided in docker-compose.yml
COPY services/AutoDiscServer/flask_app /usr/src/services/flask_app
COPY libs/auto_disc /usr/src/libs/auto_disc
WORKDIR /usr/src/libs/auto_disc
RUN pip install -e . 
RUN pip install -r /usr/src/services/flask_app/requirements.txt

# nginx setup
RUN apt-get install -y nginx
RUN rm /etc/nginx/sites-enabled/default
COPY services/AutoDiscServer/nginx/nginx.conf /etc/nginx/sites-available/flask
RUN ln -s /etc/nginx/sites-available/flask /etc/nginx/sites-enabled/flask

# dev tools
RUN apt-get install -y curl procps vim

# start daemons
VOLUME ["/usr/src/services", "/usr/src/libs"]
WORKDIR /usr/src/services/flask_app
#CMD gunicorn -w 1 -b 0.0.0.0:8000 app:app --daemon ; nginx -g "daemon off;"
CMD gunicorn -w 1 -b 0.0.0.0:8000 app:app & nginx -g "daemon off;"
