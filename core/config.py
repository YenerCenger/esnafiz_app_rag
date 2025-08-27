# core/config.py
from typing import List, Optional
import secrets
import os
from functools import lru_cache

# Pydantic v2 migration
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # API Configuration
    api_title: str = "Esnafiz AI Assistant API"
    api_version: str = "3.0.0"
    
    # External Services
    ollama_host: str = Field(default="http://ollama:11434", env="OLLAMA_HOST")
    qdrant_url: str = Field(default="http://qdrant:6333", env="QDRANT_URL")
    redis_url: str = Field(default="redis://redis:6379/0", env="REDIS_URL")
    project_id: str = Field(..., env="PROJECT_ID")
    
    # Models and Collections
    collection_name: str = Field(default="esnafiz_v3", env="COLLECTION_NAME")
    embed_model: str = Field(default="intfloat/multilingual-e5-small", env="EMBED_MODEL")
    default_llm: str = Field(default="qwen2.5:7b-instruct", env="DEFAULT_LLM")
    
    # Performance Settings
    max_words: int = Field(default=150, env="MAX_WORDS")
    rate_per_minute: int = Field(default=30, env="RATE_PER_MINUTE")
    cache_ttl: int = Field(default=600, env="CACHE_TTL")
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    
    # Security
    jwt_secret: str = Field(default_factory=lambda: secrets.token_hex(32), env="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(default=1440, env="JWT_EXPIRE_MINUTES")
    
    # Database Connection Pool Settings
    redis_max_connections: int = Field(default=100, env="REDIS_MAX_CONNECTIONS")
    redis_connection_timeout: int = Field(default=5, env="REDIS_CONNECTION_TIMEOUT")
    
    # LLM Settings
    llm_timeout: int = Field(default=90, env="LLM_TIMEOUT")
    llm_max_retries: int = Field(default=3, env="LLM_MAX_RETRIES")
    llm_temperature: float = Field(default=0.3, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=300, env="LLM_MAX_TOKENS")
    
    # Embedding Settings
    embed_batch_size: int = Field(default=32, env="EMBED_BATCH_SIZE")
    embed_cache_ttl: int = Field(default=3600, env="EMBED_CACHE_TTL")
    
    # Circuit Breaker Settings
    circuit_failure_threshold: int = Field(default=5, env="CIRCUIT_FAILURE_THRESHOLD")
    circuit_reset_timeout: int = Field(default=60, env="CIRCUIT_RESET_TIMEOUT")
    
    # CORS Settings
    allowed_origins: List[str] = Field(
        default_factory=lambda: [
            "https://esnafiz.app",
            "https://www.esnafiz.app",
            "https://admin.esnafiz.app"
        ],
        env="ALLOWED_ORIGINS"
    )
    
    @field_validator('jwt_secret')
    @classmethod
    def validate_jwt_secret(cls, v, values):
        # In pydantic v2, to access other fields use info.data or pass through values
        environment = values.get('environment') if isinstance(values, dict) else None
        if environment == "production" and len(v) < 32:
            raise ValueError("JWT_SECRET must be at least 32 characters in production")
        return v
    
    @field_validator('project_id')
    @classmethod
    def validate_project_id(cls, v):
        if not v or not v.strip():
            raise ValueError("PROJECT_ID is required")
        return v.strip()
    
    @field_validator('max_workers')
    @classmethod
    def validate_max_workers(cls, v):
        if v < 1 or v > 20:
            raise ValueError("MAX_WORKERS must be between 1 and 20")
        return v
    
    @field_validator('rate_per_minute')
    @classmethod
    def validate_rate_limit(cls, v):
        if v < 1 or v > 1000:
            raise ValueError("RATE_PER_MINUTE must be between 1 and 1000")
        return v
    
    @field_validator('cache_ttl')
    @classmethod
    def validate_cache_ttl(cls, v):
        if v < 60 or v > 86400:  # 1 minute to 24 hours
            raise ValueError("CACHE_TTL must be between 60 and 86400 seconds")
        return v
    
    @field_validator('allowed_origins', mode='before')
    @classmethod
    def parse_allowed_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    @field_validator('debug', mode='before')
    @classmethod
    def parse_debug(cls, v):
        if isinstance(v, str):
            return v.lower() in ('true', '1', 'on', 'yes')
        return bool(v)
    
    def get_redis_config(self) -> dict:
        """Get Redis connection configuration"""
        return {
            "url": self.redis_url,
            "decode_responses": True,
            "socket_connect_timeout": self.redis_connection_timeout,
            "socket_timeout": self.redis_connection_timeout,
            "retry_on_timeout": True,
            "health_check_interval": 30,
            "max_connections": self.redis_max_connections
        }
    
    def get_llm_config(self) -> dict:
        """Get LLM configuration"""
        return {
            "host": self.ollama_host,
            "timeout": self.llm_timeout,
            "max_retries": self.llm_max_retries,
            "temperature": self.llm_temperature,
            "max_tokens": self.llm_max_tokens
        }
    
    # Pydantic v2 config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings"""
    return Settings()