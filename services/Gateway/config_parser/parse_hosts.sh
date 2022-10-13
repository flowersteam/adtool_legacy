#!/bin/sh
# parse_hosts.sh: takes the path of hosts.json and outputs it into env variable definitions
#       Dependencies: jq, perl
#       Usage:
#           % ./parse.sh hosts.json > .env
#           % cat .env
#           APP_DB_API_HOST=app-db-api
#           APP_DB_API_PORT=3000
#           EXPE_DB_API_HOST=expe-db-api
#           EXPE_DB_API_PORT=80
#           AUTODISC_SERVER_HOST=autodisc-server
#           AUTODISC_SERVER_PORT=80
#           JUPYTER_HOST=jupyter
#           JUPYTER_PORT=8888
#           APP_HOST=app
#           APP_PORT=80

UNPACK_FILTER='[.upstreams|to_entries[] | [.key, .value]] | .[]'
HOSTS_FILTER='[.[0], "_host=", .[1].host] | add'
PORTS_FILTER='[.[0], "_port=", .[1].port|tostring] | add'
REGEX_FILTER='s/([^=]*)/uc($1)/e;s/\-(?=.*=)/_/g' # uppercases variable name and deletes hyphens before equals sign

cat $1 |
    jq -r "${UNPACK_FILTER}|( (${HOSTS_FILTER}), (${PORTS_FILTER}) )" |
    # TODO: use sed/awk instead of perl
    perl -pe "${REGEX_FILTER}" # uppercase variable name and delete hyphens before equals sign
