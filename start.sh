#!/bin/bash

# Start script for Render deployment
# This script sets up the environment and starts the application with Gunicorn

# Set Python path
export PYTHONPATH="${PYTHONPATH}:/opt/render/project/src"

# Start the application with Gunicorn
exec gunicorn --config gunicorn.conf.py main:app
