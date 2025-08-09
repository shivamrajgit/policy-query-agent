# Gunicorn configuration file for Render deployment

import os

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"
backlog = 2048

# Worker processes
workers = int(os.getenv('WEB_CONCURRENCY', '1'))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
# Increased to accommodate PDF downloads + embedding + LLM calls
timeout = int(os.getenv('GUNICORN_TIMEOUT', '90'))
keepalive = 2

# Restart workers after this many requests, to prevent memory leaks
max_requests = 1000
max_requests_jitter = 50

# Logging
loglevel = "info"
accesslog = "-"
errorlog = "-"

# Process naming
proc_name = 'policy-query-api'

# Server mechanics
# Disable preloading so first heavy request has full timeout window
preload_app = False
daemon = False
pidfile = '/tmp/gunicorn.pid'
user = None
group = None
tmp_upload_dir = None

# SSL
keyfile = None
certfile = None
