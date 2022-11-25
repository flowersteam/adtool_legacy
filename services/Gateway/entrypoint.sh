#!/bin/sh
# convert user-provided hosts.json into correct environment variables and export
parse_hosts.sh /rtime/userconf.d/networking/hosts.json >/rtime/env
while read var; do
    export "${var}"
done </rtime/env

# run docker-entrypoint.sh which does envsubst (since nginx:1.19) via /docker-entrypoint.d script
/docker-entrypoint.sh nginx -g "daemon off;" | grep -vE "\b200\b"
