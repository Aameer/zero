# app/services/preferences_fetcher.py
import aiohttp
import logging
from typing import Optional, Dict
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class StoredPreferences(BaseModel):
    """Model for stored user preferences from API"""
    brand_affinities: Dict[str, float] = {}
    color_preferences: list[str] = []
    size_preferences: list[str] = []
    fabric_preferences: list[str] = []
    category_preferences: Dict[str, float] = {}
    price_range: Optional[Dict[str, float]] = None
    style_preferences: list[str] = []
    seasonal_preferences: list[str] = []
    purchase_history_categories: Dict[str, int] = {}

class PreferencesFetcher:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = None
        logger.info(f"Initialized PreferencesFetcher with base URL: {base_url}")

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_user_preferences(self, user_id: int, token: str) -> Optional[StoredPreferences]:
        """Fetch user preferences from API"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            headers = {
                "Authorization": f"Token {token}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            async with self.session.get(
                f"{self.base_url}/api/users/{user_id}/",
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    shopping_prefs = data.get("shopping_preferences", {})
                    return StoredPreferences(**shopping_prefs)
                else:
                    import pprint
                    logger.debug(">"*100)
                    pprint.pp(response)
                    logger.error(f"Failed to fetch preferences: {response.status}")
                    return None

        except Exception as e:
            logger.error(f"Error fetching user preferences: {str(e)}")
            return None
        finally:
            if self.session:
                await self.session.close()
                self.session = None