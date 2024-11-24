# app/config/settings.py
from pydantic_settings import BaseSettings
from pathlib import Path
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Get the absolute path of the project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
APP_DIR = ROOT_DIR / "app"

class Settings(BaseSettings):
    # Directory settings
    BASE_DIR: Path = ROOT_DIR
    CACHE_DIR: Path = BASE_DIR / "cache"
    IMAGE_CACHE_DIR: Path = CACHE_DIR / "images"
    EMBEDDING_CACHE_DIR: Path = CACHE_DIR / "embeddings"
    
    # Feature flags
    REFRESH_EMBEDDINGS: bool = False
    
    # API settings
    BACKEND_URL: str = "https://auth.zerotab.app"  # Default production URL
    BACKEND_API_TIMEOUT: int = 30
    BACKEND_API_RETRIES: int = 3
    
    # API paths
    USER_PREFERENCES_PATH: str = "/api/users/{user_id}/"
    
    # Environment-specific settings
    ENV: str = "development"
    DEBUG: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True

    def __init__(self, **kwargs):
        # First check environment variables
        env_backend_url = os.getenv('BACKEND_URL')
        if env_backend_url:
            kwargs['BACKEND_URL'] = env_backend_url
            logger.info(f"Using BACKEND_URL from environment: {env_backend_url}")

        # Check both possible locations for .env
        env_in_root = ROOT_DIR / ".env"
        env_in_app = APP_DIR / ".env"
        
        if env_in_app.exists():
            logger.info(f"Found .env in app directory: {env_in_app}")
            self.Config.env_file = str(env_in_app)
        elif env_in_root.exists():
            logger.info(f"Found .env in root directory: {env_in_root}")
            self.Config.env_file = str(env_in_root)
        else:
            logger.warning("No .env file found, using environment variables and defaults")

        super().__init__(**kwargs)
        
        # Log critical configuration values
        self._log_configuration()
        
        # Ensure directories exist
        self._ensure_directories()

    def _log_configuration(self):
        """Log all critical configuration values"""
        logger.info("=== Current Settings Configuration ===")
        logger.info(f"Environment: {self.ENV}")
        logger.info(f"Debug Mode: {self.DEBUG}")
        logger.info(f"Backend URL: {self.BACKEND_URL}")
        logger.info(f"Base Directory: {self.BASE_DIR}")
        logger.info(f"Cache Directory: {self.CACHE_DIR}")
        logger.info(f"API Timeout: {self.BACKEND_API_TIMEOUT}s")
        logger.info(f"API Retries: {self.BACKEND_API_RETRIES}")
        logger.info("=====================================")

        # Show .env contents if file exists
        env_path = Path(self.Config.env_file)
        if env_path.exists():
            with open(env_path) as f:
                logger.info(f".env contents:\n{f.read()}")

    def _ensure_directories(self):
        """Ensure required directories exist"""
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def get_user_preferences_url(self, user_id: int) -> str:
        """Generate the full URL for user preferences endpoint"""
        return f"{self.BACKEND_URL.rstrip('/')}{self.USER_PREFERENCES_PATH.format(user_id=user_id)}"

# Create settings instance with environment-specific overrides
env = os.getenv("ENV", "development")
debug = os.getenv("DEBUG", "true").lower() == "true"

settings = Settings(
    ENV=env,
    DEBUG=debug
)