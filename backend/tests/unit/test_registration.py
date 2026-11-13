import pytest
from fastapi import HTTPException

from app.security.registration import RegistrationLimiter


def test_registration_rate_limits_and_recovers_after_window(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr('app.security.registration.monotonic', lambda: clock[0])
    limiter = RegistrationLimiter()
    for _ in range(5):
        limiter.check('client')
    with pytest.raises(HTTPException) as error:
        limiter.check('client')
    assert error.value.status_code == 429 and error.value.headers['Retry-After'] == '600'
    for index in range(15):
        limiter.check(f'other-{index}')
    with pytest.raises(HTTPException):
        limiter.check('new-client')
    clock[0] += 601
    limiter.check('client')


def test_registration_validation_disabled_flag_and_body_limit(auth_config, db_session):
    from fastapi.testclient import TestClient
    from app.core.config import get_settings
    from app.db.session import get_db
    from app.main import app
    from app.security.registration import get_registration_limiter
    config = auth_config.model_copy(update={'registration_enabled': True})
    limiter = RegistrationLimiter()
    old = app.dependency_overrides.copy()
    app.dependency_overrides.update({get_settings: lambda: config, get_db: lambda: db_session,
                                     get_registration_limiter: lambda: limiter})
    good = {'username': 'reader', 'password': 'Synthetic password 123', 'password_confirmation': 'Synthetic password 123'}
    invalid = [{**good, 'password': 'SENSITIVE'}, {**good, 'password_confirmation': 'SENSITIVE'},
               {**good, 'owner_id': 'SENSITIVE'}, {**good, 'status': 'admin'}, {**good, 'username': '!!SENSITIVE!!'}]
    try:
        with TestClient(app) as client:
            for body in invalid:
                response = client.post('/api/v1/auth/register', json=body)
                assert response.status_code == 422 and 'SENSITIVE' not in response.text
            assert client.post('/api/v1/auth/register', json=good).status_code == 429
            assert client.post('/api/v1/auth/register', content=b'x' * 4097).status_code == 413
            config.registration_enabled = False
            assert client.get('/api/v1/auth/config').json() == {'registration_enabled': False}
            assert client.post('/api/v1/auth/register', json=good).status_code == 403
        db_session.add.assert_not_called()
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(old)
