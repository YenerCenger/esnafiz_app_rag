# main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
from datetime import datetime

from core.config import get_settings
from core.exceptions import EsnafizException, handle_esnafiz_exception, handle_http_exception, handle_general_exception
from core.logging import logger, configure_logging
from core.metrics import REQUEST_COUNT, REQUEST_DURATION, ERROR_COUNT

from api.endpoints import health, metrics, chat
from services.dependencies import ServiceManager

configure_logging()
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Esnafiz API starting up", environment=settings.environment)
    try:
        await ServiceManager.initialize_all()
        logger.info("All services initialized successfully")
    except Exception as e:
        ERROR_COUNT.labels(type="startup").inc()
        logger.exception("Service initialization failed", error=str(e))
        raise
    try:
        yield
    finally:
        try:
            await ServiceManager.cleanup_all()
            logger.info("All services cleaned up successfully")
        except Exception as e:
            ERROR_COUNT.labels(type="shutdown").inc()
            logger.exception("Service cleanup failed", error=str(e))

app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
)

# Middleware
if settings.environment == "production":
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["esnafiz.app", "*.esnafiz.app", "api.esnafiz.app"])

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["https://esnafiz.app", "https://www.esnafiz.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router)
app.include_router(metrics.router)
app.include_router(chat.router)

# Error handlers
app.add_exception_handler(EsnafizException, handle_esnafiz_exception)
app.add_exception_handler(HTTPException, handle_http_exception)
app.add_exception_handler(Exception, handle_general_exception)

# Instrumentation middleware (basic)
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    endpoint = request.url.path
    start = datetime.now().timestamp()
    try:
        response = await call_next(request)
        duration = datetime.now().timestamp() - start
        REQUEST_COUNT.labels(method=request.method, endpoint=endpoint, status_code=str(getattr(response, "status_code", 200))).inc()
        REQUEST_DURATION.labels(endpoint=endpoint).observe(duration)
        return response
    except Exception as e:
        ERROR_COUNT.labels(type="unhandled").inc()
        raise

@app.get("/", tags=["root"])
async def root():
    return {
        "service": settings.api_title,
        "version": settings.api_version,
        "status": "running",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
