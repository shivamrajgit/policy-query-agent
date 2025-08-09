#!/usr/bin/env python3
"""Cross-platform startup script.

On Linux (e.g. Render) -> uses gunicorn.
On Windows (local dev) -> falls back to uvicorn (gunicorn unsupported due to fcntl).
"""

import os
import sys
import subprocess
import platform


def run():
    port = os.getenv('PORT', '8000')
    timeout = os.getenv('GUNICORN_TIMEOUT', '90')
    workers = os.getenv('WEB_CONCURRENCY', '1')
    host = os.getenv('HOST', '0.0.0.0')

    is_windows = platform.system().lower().startswith('win')

    if is_windows:
        # Use uvicorn directly; enable auto-reload for dev convenience.
        cmd = [
            sys.executable, '-m', 'uvicorn', 'main:app',
            '--host', host,
            '--port', port,
            '--workers', '1'
        ]
        if os.getenv('RELOAD', 'true').lower() == 'true':
            cmd.append('--reload')
        print('[start.py] Detected Windows -> using uvicorn (gunicorn not supported).')
    else:
        # Production (Linux) path.
        cmd = [
            'gunicorn', 'main:app',
            '--bind', f'{host}:{port}',
            '--workers', workers,
            '--worker-class', 'uvicorn.workers.UvicornWorker',
            '--timeout', timeout,
            '--access-logfile', '-',
            '--error-logfile', '-'
        ]
        print('[start.py] Using gunicorn for production deployment.')

    print('[start.py] Command:', ' '.join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f'[start.py] Failed to start: {e}')
        sys.exit(e.returncode or 1)


if __name__ == '__main__':
    run()
