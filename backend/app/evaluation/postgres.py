import os
import shlex
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import psycopg
from sqlalchemy import create_engine
from sqlalchemy.engine import URL


@contextmanager
def temporary_postgres():
    """Disposable cluster with a private Unix socket and no TCP listener."""
    binaries = Path('/usr/lib/postgresql/18/bin')
    with tempfile.TemporaryDirectory(prefix='docagent-pg-evaluation-') as directory:
        root = Path(directory)
        data, socket = root / 'data', root / 'socket'
        socket.mkdir(mode=0o700)
        subprocess.run([str(binaries / 'initdb'), '-D', str(data), '--auth-local=trust',
                        '--auth-host=reject', '--no-locale', '--encoding=UTF8'],
                       check=True, capture_output=True, timeout=30)
        options = shlex.join(['-F', '-k', str(socket), '-h', '', '-p', '55439'])
        try:
            subprocess.run([str(binaries / 'pg_ctl'), '-D', str(data), '-l', str(root / 'server.log'),
                            '-o', options, '-w', 'start'], check=True, capture_output=True, timeout=30)
            with psycopg.connect(host=str(socket), port=55439, dbname='postgres', autocommit=True) as connection:
                connection.execute('CREATE DATABASE docagent_ai_evaluation')
            url = URL.create('postgresql+psycopg', database='docagent_ai_evaluation',
                             query={'host': str(socket), 'port': '55439'})
            environment = {'PATH': os.environ.get('PATH', ''), 'HOME': str(Path.home()),
                           'AUTH_ENABLED': 'false', 'DEBUG': 'false',
                           'DATABASE_URL': url.render_as_string(hide_password=False), 'PYTHONDONTWRITEBYTECODE': '1'}
            subprocess.run([sys.executable, '-B', '-m', 'alembic', 'upgrade', 'head'], env=environment,
                           cwd=Path(__file__).resolve().parents[2], check=True, capture_output=True, timeout=30)
            engine = create_engine(url, pool_pre_ping=True)
            try:
                yield engine
            finally:
                engine.dispose()
        finally:
            if (data / 'postmaster.pid').exists():
                subprocess.run([str(binaries / 'pg_ctl'), '-D', str(data), '-m', 'immediate', '-w', 'stop'],
                               check=True, capture_output=True, timeout=30)
