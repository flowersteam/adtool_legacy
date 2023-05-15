FROM nginx:1.23.1-alpine-perl

# runtime configuration files
COPY ./entrypoint.sh /entrypoint.sh
COPY ./nginx/templates /etc/nginx/templates
RUN chmod +x /entrypoint.sh

CMD /entrypoint.sh