#/bin/sh
if [[ -z "${DEBUG}" ]]; then
    DEBUG=0
else
    DEBUG=${DEBUG}
fi

# note: please be in the right workdir before doing this
if [ $DEBUG -eq 0 ]; then
    /docker-entrypoint.sh nginx -g "daemon off;"
else
    export NODE_OPTIONS=--openssl-legacy-provider
    ng serve --configuration development --host 0.0.0.0 --port 80
fi
