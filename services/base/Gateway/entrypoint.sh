#!/bin/sh

# run docker-entrypoint.sh which does envsubst (since nginx:1.19) via /docker-entrypoint.d script
/docker-entrypoint.sh nginx -g "daemon off;" | grep -vE "\b200\b"
