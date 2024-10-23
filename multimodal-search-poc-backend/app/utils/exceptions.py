# app/utils/exceptions.py
from fastapi import HTTPException
from typing import Optional

class SearchException(HTTPException):
    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(status_code=status_code, detail=detail)

class ModelLoadingError(SearchException):
    def __init__(self, model_name: str, detail: Optional[str] = None):
        message = f"Error loading model: {model_name}"
        if detail:
            message += f" - {detail}"
        super().__init__(detail=message, status_code=500)

class EmbeddingError(SearchException):
    def __init__(self, content_type: str, detail: Optional[str] = None):
        message = f"Error generating embeddings for {content_type}"
        if detail:
            message += f" - {detail}"
        super().__init__(detail=message, status_code=400)

class InvalidInputError(SearchException):
    def __init__(self, message: str):
        super().__init__(detail=message, status_code=400)
