from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes.auth import router as auth_router
from app.api.routes.documents import router as documents_router
from app.api.routes.health import router as health_router
from app.api.routes.retrieval import router as retrieval_router
from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    debug=settings.debug,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
    allow_methods=['GET', 'POST', 'DELETE'],
    allow_headers=['Authorization', 'Content-Type'],
)

app.include_router(health_router)
app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(retrieval_router)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    if request.url.path in ('/api/v1/search', '/api/v1/ask', '/api/v1/agent/ask'):
        return JSONResponse(status_code=422, content={"detail": "Dados de consulta inválidos."})

    if request.url.path == '/api/v1/auth/register':
        return JSONResponse(status_code=422, content={
            'detail': 'Cadastro inválido. Use usuário de 3 a 64 caracteres simples e senhas iguais de 12 a 128 caracteres.'
        })

    if request.url.path.startswith("/api/v1/auth/"):
        return JSONResponse(status_code=422, content={"detail": "Dados de autenticação inválidos."})

    return await request_validation_exception_handler(request, exc)
