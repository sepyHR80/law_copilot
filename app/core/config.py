from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Law Copilot"
    app_version: str = "0.1.0"
    environment: str = "development"

    database_url: str = (
        "postgresql+psycopg://"
        "law_copilot:law_copilot@"
        "localhost:5432/"
        "law_copilot"
    )

    # Embedding configuration
    embedding_endpoint: str = "http://localhost:4000/v1"
    embedding_api_key: str = "test-key"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    embedding_timeout: int = 30
    embedding_batch_size: int = 100

    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "law_copilot"
    minio_secret_key: str = "law_copilot_password"
    minio_secure: bool = False
    minio_bucket_name: str = "law-copilot-documents"
    max_document_size_bytes: int = 10 * 1024 * 1024  # 10 MB

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()