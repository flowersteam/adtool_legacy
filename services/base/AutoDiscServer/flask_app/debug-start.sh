#!/bin/bash

# exports env vars
set -a
source .env.dev
set +a

# need to do this for module path resolution
python app.py
