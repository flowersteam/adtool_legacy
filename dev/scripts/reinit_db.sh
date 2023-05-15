#!/usr/bin/env bash

echo "Removing AppDB/saved_data..."
rm -rf ../../services/AppDB/saved_data;
echo "Removing ExpeDB/saved_data..."
rm -rf ../../services/ExpeDB/saved_data;
echo "Removing JupyterLab/Notebooks/Experiments..."
find ../../services/JupyterLab/Notebooks/Experiments -type d ! -name '.*' -mindepth 1 -exec rm -rf {} +;
echo "Done!"

