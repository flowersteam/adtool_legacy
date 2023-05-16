#!/usr/bin/env sh
export $(cat $1 | grep --invert-match -E '^#' | xargs)
echo "Successfully exported"
