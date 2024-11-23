# app/config/settings.py
from pydantic_settings import BaseSettings
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Get the absolute path of the project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
APP_DIR = ROOT_DIR / "app"

class Settings(BaseSettings):
    BASE_DIR: Path = ROOT_DIR
    CACHE_DIR: Path = BASE_DIR / "cache"
    IMAGE_CACHE_DIR: Path = CACHE_DIR / "images"
    EMBEDDING_CACHE_DIR: Path = CACHE_DIR / "embeddings"
    REFRESH_EMBEDDINGS: bool = False
    BACKEND_URL: str = "http://localhost:8000"

    class Config:
        # Look for .env in multiple locations
        env_file = str(ROOT_DIR / ".env")  # First try project root
        env_file_encoding = 'utf-8'
        case_sensitive = True

    def __init__(self, **kwargs):
        # Check both possible locations for .env
        env_in_root = ROOT_DIR / ".env"
        env_in_app = APP_DIR / ".env"
        
        if env_in_app.exists():
            logger.warning(f".env file found in app directory ({env_in_app}). "
                         f"Consider moving it to project root ({env_in_root})")
            self.Config.env_file = str(env_in_app)
        elif env_in_root.exists():
            logger.info(f"Using .env from project root: {env_in_root}")
            self.Config.env_file = str(env_in_root)
        else:
            logger.warning(f"No .env file found in either {env_in_root} or {env_in_app}")

        super().__init__(**kwargs)
        
        # Log the actual values being used
        logger.info(f"Using .env file: {self.Config.env_file}")
        logger.info(f"Settings initialized with BACKEND_URL: {self.BACKEND_URL}")
        
        # Show .env contents if file exists
        env_path = Path(self.Config.env_file)
        if env_path.exists():
            with open(env_path) as f:
                logger.info(f".env contents:\n{f.read()}")

# Create settings instance
settings = Settings()