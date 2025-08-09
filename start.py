#!/usr/bin/env python3
"""
Python-based startup script for Render deployment.
This avoids potential shell parsing issues.
"""

import os
import sys
import subprocess

def main():
    # Set environment variables
    port = os.getenv('PORT', '10000')
    workers = os.getenv('WEB_CONCURRENCY', '1')
    
    # Build the gunicorn command
    cmd = [
        'gunicorn',
        'main:app',
        '--bind', f'0.0.0.0:{port}',
        '--workers', str(workers),
        '--worker-class', 'uvicorn.workers.UvicornWorker',
        '--timeout', '30',
        '--access-logfile', '-',
        '--error-logfile', '-',
        '--preload'
    ]
    
    print(f"Starting application with command: {' '.join(cmd)}")
    
    # Execute gunicorn
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
