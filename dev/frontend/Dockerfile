FROM nginx:stable-alpine
ARG GATEWAY_HOST=127.0.0.1
ARG GATEWAY_PORT=4201
# Setup NGINX reverse proxy
COPY services/FrontEndApp/nginx/nginx.conf /etc/nginx/conf.d/default.conf
# Install NPM and Angular CLI
RUN apk add --update npm
RUN npm install -g @angular/cli@v14-lts

# Prepare source folder
WORKDIR /usr/src/angular_app
# Launch app: 1) copy sources to another folder to avoid access rights issues, 2) install modules, 3) build app, 4) launch NGINX
COPY services/FrontEndApp/angular_app /usr/src/angular_app
CMD ["sh"]
# RUN /bin/sh -c "npm install --save --legacy-peer-deps" 
# RUN /bin/sh -c "export NODE_OPTIONS=--openssl-legacy-provider; \
#     envsubst < /usr/src/angular_app/src/assets/env.template.js > /usr/src/angular_app/src/assets/env.js; \
#     ng build --output-path=/usr/share/nginx/www/angular_app"
# COPY bash/custom-entrypoint.sh /custom-entrypoint.sh
# CMD /custom-entrypoint.sh