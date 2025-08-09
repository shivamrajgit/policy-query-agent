#!/bin/bash

# Start script for Render deployment
# This script sets up the environment and starts the application with Gunicorn

# Set Python path
export PYTHONPATH="${PYTHONPATH}:/opt/render/project/src"

# Start the application with Gunicorn using simple command
exec gunicorn main:app --bind 0.0.0.0:$PORT --workers 1 --worker-class uvicorn.workers.UvicornWorker --timeout 30 --access-logfile - --error-logfile -
