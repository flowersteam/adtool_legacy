FROM nginx:stable-alpine

# Setup NGINX reverse proxy
COPY nginx/nginx.conf /etc/nginx/conf.d/default.conf
# Install NPM and Angular CLI
RUN apk add --update npm
RUN npm install -g @angular/cli@v14-lts

# Prepare source folder
WORKDIR /usr/src/angular_app
# Launch app: 1) copy sources to another folder to avoid access rights issues, 2) install modules, 3) build app, 4) launch NGINX
COPY angular_app /usr/src/angular_app
RUN /bin/sh -c "npm install --save --legacy-peer-deps"
# TODO: remove envsubst now that we do it at runtime
RUN /bin/sh -c "export NODE_OPTIONS=--openssl-legacy-provider; \
    envsubst < /usr/src/angular_app/src/assets/env.template.js > /usr/src/angular_app/src/assets/env.js; \
    ng build --output-path=/usr/share/nginx/www/angular_app"
COPY bash/custom-entrypoint.sh /custom-entrypoint.sh
CMD /custom-entrypoint.sh
