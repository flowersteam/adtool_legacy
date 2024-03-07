#!/usr/bin/env bash

echo "Removing AppDB/saved_data..."
rm -rf ../../services/base/AppDB/saved_data;
echo "Removing ExpeDB/saved_data..."
rm -rf ../../services/base/ExpeDB/saved_data;
echo "Removing JupyterLab/Notebooks/Experiments..."
find ../../services/base/JupyterLab/Notebooks/Experiments  -mindepth 1 -type d ! -name '.*'  -exec rm -rf {} +;
echo "Done!"

