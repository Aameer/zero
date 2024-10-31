# app/config/settings.py
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    CACHE_DIR: Path = BASE_DIR / "cache"
    IMAGE_CACHE_DIR: Path = CACHE_DIR / "images"
    EMBEDDING_CACHE_DIR: Path = CACHE_DIR / "embeddings"
    REFRESH_EMBEDDINGS: bool = False

    class Config:
        env_file = ".env"

# Create instance
settings = Settings()
