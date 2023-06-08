#/bin/sh

# null guards
if [[ -z "${DEBUG}" ]]; then
    DEBUG=0
else
    DEBUG=${DEBUG}
fi
if [[ -z "${GATEWAY_HOST}" ]]; then
    GATEWAY_HOST=127.0.0.1
else
    GATEWAY_HOST=${GATEWAY_HOST}
fi

# runtime injection of $GATEWAY_HOST
# TODO: probably there is something better than this
# Arguments:
#   filename: path to file to be modified.
#   hostname: hostname value to inject
injectenv() {
    local filename="$1"
    local hostname="$2"

    # Replace the value of window["env"]["GATEWAY_HOST"] with ${hostname}
    sed -i -e "s|\(window\[\"env\"\]\[\"GATEWAY_HOST\"\]\) = \"\(.*\)\";|\1 = \"${hostname}\";|" \
        $filename
}

# note: please be in the right workdir before doing this
if [ $DEBUG -eq 0 ]; then
    injectenv "./src/assets/env.js" ${GATEWAY_HOST}
    # serve static web app
    /docker-entrypoint.sh nginx -g "daemon off;"
else
    injectenv "./src/assets/env.js" ${GATEWAY_HOST}
    echo "custom_entrypoint.sh: Injected GATEWAY_HOST=${GATEWAY_HOST}"
    # run angular development server
    export NODE_OPTIONS=--openssl-legacy-provider
    ng serve --configuration development --host 0.0.0.0 --port 80
fi
