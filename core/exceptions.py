# core/exceptions.py
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import Dict, Any
import traceback

from core.logging import logger
from core.metrics import ERROR_COUNT
from core.config import get_settings

settings = get_settings()


class EsnafizException(Exception):
    """Base exception for Esnafiz application"""
    
    def __init__(
        self,
        detail: str,
        status_code: int = 500,
        error_code: str = "ESNAFIZ_ERROR",
        metadata: Dict[str, Any] = None
    ):
        self.detail = detail
        self.status_code = status_code
        self.error_code = error_code
        self.metadata = metadata or {}
        super().__init__(detail)


class ValidationError(EsnafizException):
    """Validation related errors"""
    
    def __init__(self, detail: str, field: str = None):
        super().__init__(
            detail=detail,
            status_code=400,
            error_code="VALIDATION_ERROR",
            metadata={"field": field} if field else None
        )


class AuthenticationError(EsnafizException):
    """Authentication related errors"""
    
    def __init__(self, detail: str = "Authentication required"):
        super().__init__(
            detail=detail,
            status_code=401,
            error_code="AUTHENTICATION_ERROR"
        )


class AuthorizationError(EsnafizException):
    """Authorization related errors"""
    
    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(
            detail=detail,
            status_code=403,
            error_code="AUTHORIZATION_ERROR"
        )


class NotFoundError(EsnafizException):
    """Resource not found errors"""
    
    def __init__(self, detail: str = "Resource not found", resource: str = None):
        super().__init__(
            detail=detail,
            status_code=404,
            error_code="NOT_FOUND_ERROR",
            metadata={"resource": resource} if resource else None
        )


class ConflictError(EsnafizException):
    """Resource conflict errors"""
    
    def __init__(self, detail: str = "Resource conflict", resource: str = None):
        super().__init__(
            detail=detail,
            status_code=409,
            error_code="CONFLICT_ERROR",
            metadata={"resource": resource} if resource else None
        )


class RateLimitError(EsnafizException):
    """Rate limiting errors"""
    
    def __init__(self, detail: str = "Rate limit exceeded", retry_after: int = None):
        super().__init__(
            detail=detail,
            status_code=429,
            error_code="RATE_LIMIT_ERROR",
            metadata={"retry_after": retry_after} if retry_after else None
        )


class ServiceUnavailableError(EsnafizException):
    """Service unavailable errors"""
    
    def __init__(self, detail: str = "Service temporarily unavailable", service: str = None):
        super().__init__(
            detail=detail,
            status_code=503,
            error_code="SERVICE_UNAVAILABLE_ERROR",
            metadata={"service": service} if service else None
        )


class ExternalServiceError(EsnafizException):
    """External service integration errors"""
    
    def __init__(self, detail: str, service: str, upstream_status: int = None):
        super().__init__(
            detail=detail,
            status_code=502,
            error_code="EXTERNAL_SERVICE_ERROR",
            metadata={
                "service": service,
                "upstream_status": upstream_status
            }
        )


class DataProcessingError(EsnafizException):
    """Data processing errors"""
    
    def __init__(self, detail: str, operation: str = None):
        super().__init__(
            detail=detail,
            status_code=422,
            error_code="DATA_PROCESSING_ERROR",
            metadata={"operation": operation} if operation else None
        )


class CircuitBreakerError(EsnafizException):
    """Circuit breaker errors"""
    
    def __init__(self, detail: str = "Circuit breaker is open", service: str = None):
        super().__init__(
            detail=detail,
            status_code=503,
            error_code="CIRCUIT_BREAKER_ERROR",
            metadata={"service": service} if service else None
        )


def create_error_response(
    error: EsnafizException,
    request_id: str = None,
    include_metadata: bool = True
) -> Dict[str, Any]:
    """Create standardized error response"""
    
    response = {
        "success": False,
        "error": {
            "code": error.error_code,
            "message": error.detail,
            "timestamp": datetime.now().isoformat()
        }
    }
    
    if request_id:
        response["request_id"] = request_id
    
    if include_metadata and error.metadata:
        response["error"]["metadata"] = error.metadata
    
    # Add debug info in development
    if settings.debug and hasattr(error, '__traceback__'):
        response["error"]["debug"] = {
            "traceback": traceback.format_exc()
        }
    
    return response


async def handle_esnafiz_exception(request: Request, exc: EsnafizException) -> JSONResponse:
    """Handle Esnafiz custom exceptions"""
    
    request_id = getattr(request.state, 'request_id', None)
    
    # Log the error
    logger.error(
        "Esnafiz exception occurred",
        error_code=exc.error_code,
        detail=exc.detail,
        status_code=exc.status_code,
        request_id=request_id,
        path=request.url.path,
        metadata=exc.metadata
    )
    
    # Update metrics
    ERROR_COUNT.labels(error_type=exc.error_code.lower()).inc()
    
    # Create response
    response_data = create_error_response(exc, request_id)
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response_data
    )


async def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTP exceptions"""
    
    request_id = getattr(request.state, 'request_id', None)
    
    logger.warning(
        "HTTP exception occurred",
        status_code=exc.status_code,
        detail=exc.detail,
        request_id=request_id,
        path=request.url.path
    )
    
    ERROR_COUNT.labels(error_type="http_error").inc()
    
    response_data = {
        "success": False,
        "error": {
            "code": "HTTP_ERROR",
            "message": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    }
    
    if request_id:
        response_data["request_id"] = request_id
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response_data
    )


async def handle_general_exception(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions"""
    
    request_id = getattr(request.state, 'request_id', None)
    
    logger.error(
        "Unhandled exception occurred",
        error=str(exc),
        error_type=type(exc).__name__,
        request_id=request_id,
        path=request.url.path,
        traceback=traceback.format_exc() if settings.debug else None
    )
    
    ERROR_COUNT.labels(error_type="unhandled_exception").inc()
    
    # Don't expose internal errors in production
    detail = str(exc) if settings.debug else "Internal server error"
    
    response_data = {
        "success": False,
        "error": {
            "code": "INTERNAL_ERROR",
            "message": detail,
            "timestamp": datetime.now().isoformat()
        }
    }
    
    if request_id:
        response_data["request_id"] = request_id
    
    if settings.debug:
        response_data["error"]["debug"] = {
            "exception_type": type(exc).__name__,
            "traceback": traceback.format_exc()
        }
    
    return JSONResponse(
        status_code=500,
        content=response_data
    )