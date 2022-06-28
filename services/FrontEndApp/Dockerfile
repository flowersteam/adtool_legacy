FROM nginx:stable-alpine
# Setup NGINX reverse proxy
COPY nginx/nginx.conf /etc/nginx/conf.d/default.conf
# Install NPM and Angular CLI
RUN apk add --update npm
RUN npm install -g @angular/cli@latest
# set bash scrpt to manege docker ip
COPY bash/set-ip-from-hostname.sh /usr/src/bash/set-ip-from-hostname.sh
RUN chmod 777 /usr/src/bash/set-ip-from-hostname.sh
# Prepare source folder
WORKDIR /usr/src/angular_app_src
# Launch app: 1) copy sources to another folder to avoid access rights issues, 2) install modules, 3) build app, 4) launch NGINX
CMD cp -r /usr/src/angular_app/* . ; \
    npm install --save --legacy-peer-deps ; \
    source /usr/src/bash/set-ip-from-hostname.sh ; \
    /usr/src/bash/set-ip-from-hostname.sh ; \
    envsubst < /usr/src/angular_app_src/src/assets/env.template.js > /usr/src/angular_app_src/src/assets/env.js ; \
    ng build --output-path=/usr/share/nginx/www/angular_app ; \
    /docker-entrypoint.sh nginx -g "daemon off;"