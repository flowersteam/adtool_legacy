#!/bin/sh
export EXPEDB_HOST="$(getent hosts $EXPEDB_HOST | awk '{ print $1 }')"
export APPDB_HOST="$(getent hosts $APPDB_HOST | awk '{ print $1 }')"
export AUTODISC_SERVER_HOST="$(getent hosts $AUTODISC_SERVER_HOST | awk '{ print $1 }')"
export JUPYTER_HOST="$(getent hosts $JUPYTER_HOST | awk '{ print $1 }')"