FROM nginx:stable-alpine
# Setup NGINX reverse proxy
COPY nginx/nginx.conf /etc/nginx/conf.d/default.conf
# Install NPM and Angular CLI
RUN apk add --update npm
RUN npm install -g @angular/cli@latest
# Prepare source folder
WORKDIR /usr/src/angular_app_src
# Launch app: 1) copy sources to another folder to avoid access rights issues, 2) install modules, 3) build app, 4) launch NGINX
CMD cp -r /usr/src/angular_app/* . ; \
    npm install ; \
    ng build --output-path=/usr/share/nginx/www/angular_app ; \
    /docker-entrypoint.sh nginx -g "daemon off;"