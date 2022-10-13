FROM nginx:1.23.1-alpine-perl

# parse_hosts
RUN apk add jq
COPY services/Gateway/config_parser/parse_hosts.sh /usr/local/bin/parse_hosts.sh
RUN chmod +x /usr/local/bin/parse_hosts.sh

# runtime configuration files
COPY services/Gateway/entrypoint.sh /entrypoint.sh
COPY services/Gateway/nginx/templates /etc/nginx/templates
VOLUME ["/rtime/userconf.d"]
RUN chmod +x /entrypoint.sh

CMD /entrypoint.sh