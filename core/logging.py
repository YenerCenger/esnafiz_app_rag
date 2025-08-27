# core/logging.py
import structlog
import logging
import sys
from typing import Any, Dict
from datetime import datetime

from core.config import get_settings

settings = get_settings()


def configure_logging():
    """Configure structured logging with proper formatting"""
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.DEBUG if settings.debug else logging.INFO,
    )
    
    # Silence noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    
    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        add_timestamp,
        add_environment,
    ]
    
    if settings.debug:
        processors.extend([
            structlog.dev.ConsoleRenderer(colors=True)
        ])
    else:
        processors.extend([
            structlog.processors.JSONRenderer()
        ])
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def add_timestamp(logger, method_name, event_dict):
    """Add timestamp to log events"""
    event_dict["timestamp"] = datetime.now().isoformat()
    return event_dict


def add_environment(logger, method_name, event_dict):
    """Add environment info to log events"""
    event_dict["environment"] = settings.environment
    return event_dict


def sanitize_log_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove sensitive data from logs"""
    sensitive_keys = {
        'password', 'token', 'jwt', 'secret', 'key', 'authorization',
        'credential', 'auth', 'api_key', 'access_token', 'refresh_token'
    }
    
    sanitized = {}
    
    for key, value in data.items():
        key_lower = key.lower()
        
        # Check if key contains sensitive information
        if any(sensitive in key_lower for sensitive in sensitive_keys):
            sanitized[key] = "***REDACTED***"
        elif isinstance(value, dict):
            sanitized[key] = sanitize_log_data(value)
        elif isinstance(value, str) and len(value) > 100:
            # Truncate very long strings
            sanitized[key] = value[:100] + "..."
        else:
            sanitized[key] = value
    
    return sanitized


class EsnafizLogger:
    """Enhanced logger with business context"""
    
    def __init__(self, name: str):
        self.logger = structlog.get_logger(name)
    
    def info(self, message: str, **kwargs):
        """Log info message with sanitized context"""
        sanitized_kwargs = sanitize_log_data(kwargs)
        self.logger.info(message, **sanitized_kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with sanitized context"""
        sanitized_kwargs = sanitize_log_data(kwargs)
        self.logger.warning(message, **sanitized_kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with sanitized context"""
        sanitized_kwargs = sanitize_log_data(kwargs)
        self.logger.error(message, **sanitized_kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with sanitized context"""
        if settings.debug:
            sanitized_kwargs = sanitize_log_data(kwargs)
            self.logger.debug(message, **sanitized_kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with sanitized context"""
        sanitized_kwargs = sanitize_log_data(kwargs)
        self.logger.critical(message, **sanitized_kwargs)
    
    def with_context(self, **kwargs) -> 'EsnafizLogger':
        """Create logger with additional context"""
        bound_logger = EsnafizLogger.__new__(EsnafizLogger)
        bound_logger.logger = self.logger.bind(**sanitize_log_data(kwargs))
        return bound_logger
    
    def log_request(self, method: str, url: str, status_code: int, 
                   duration: float, user_id: str = None, store_id: str = None):
        """Log HTTP request with standardized format"""
        self.info(
            "HTTP request completed",
            method=method,
            url=str(url),
            status_code=status_code,
            duration_ms=round(duration * 1000, 2),
            user_id=user_id,
            store_id=store_id
        )
    
    def log_service_call(self, service: str, operation: str, 
                        duration: float, success: bool, **kwargs):
        """Log external service call"""
        self.info(
            f"{service} service call",
            service=service,
            operation=operation,
            duration_ms=round(duration * 1000, 2),
            success=success,
            **sanitize_log_data(kwargs)
        )
    
    def log_query_analysis(self, query: str, query_type: str, 
                          confidence: float, **kwargs):
        """Log query analysis results"""
        self.info(
            "Query analyzed",
            query_preview=query[:50] + "..." if len(query) > 50 else query,
            query_type=query_type,
            confidence=round(confidence, 3),
            **sanitize_log_data(kwargs)
        )
    
    def log_cache_operation(self, operation: str, key: str, 
                           hit: bool = None, ttl: int = None):
        """Log cache operations"""
        log_data = {
            "cache_operation": operation,
            "cache_key": key[:50] + "..." if len(key) > 50 else key
        }
        
        if hit is not None:
            log_data["cache_hit"] = hit
        
        if ttl is not None:
            log_data["ttl"] = ttl
        
        self.debug("Cache operation", **log_data)


# Configure logging on import
configure_logging()

# Create default logger
logger = EsnafizLogger("esnafiz")