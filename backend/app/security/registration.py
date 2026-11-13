import math
from collections import deque
from functools import lru_cache
from threading import Lock
from time import monotonic

from fastapi import HTTPException


class RegistrationLimiter:
    def __init__(self):
        self._lock = Lock()
        self._attempts = {}
        self._global = deque()

    def check(self, client: str):
        now = monotonic()
        window = 600

        with self._lock:
            for key in list(self._attempts):
                queue = self._attempts[key]
                while queue and queue[0] <= now - window:
                    queue.popleft()

                if not queue:
                    del self._attempts[key]

            while self._global and self._global[0] <= now - window:
                self._global.popleft()

            attempts = self._attempts.get(client, deque())

            if len(attempts) >= 5 or len(self._global) >= 20 or len(self._attempts) >= 512:
                oldest = attempts[0] if len(attempts) >= 5 else self._global[0] if self._global else now
                retry = max(1, math.ceil(oldest + window - now))
                raise HTTPException(429, 'Muitas tentativas de cadastro. Aguarde para tentar novamente.',
                                    headers={'Retry-After': str(retry)})

            attempts.append(now)
            self._attempts[client] = attempts
            self._global.append(now)


@lru_cache
def get_registration_limiter():
    return RegistrationLimiter()
