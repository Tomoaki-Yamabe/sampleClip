"""
Custom exception classes for the API
"""
from typing import Optional


class APIException(Exception):
    """Base exception for API errors"""
    
    def __init__(
        self,
        message: str,
        code: str,
        status_code: int = 500,
        details: Optional[dict] = None
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(APIException):
    """Validation error (400)"""
    
    def __init__(self, message: str, code: str = "VALIDATION_ERROR", details: Optional[dict] = None):
        super().__init__(message, code, 400, details)


class FileSizeExceededError(ValidationError):
    """File size exceeded error (400)"""
    
    def __init__(self, max_size_mb: float = 5.0):
        super().__init__(
            f"画像ファイルサイズは{max_size_mb}MB以下である必要があります",
            "FILE_SIZE_EXCEEDED"
        )


class UnsupportedMediaTypeError(APIException):
    """Unsupported media type error (415)"""
    
    def __init__(self, supported_formats: str = "JPEG, PNG, WebP"):
        super().__init__(
            f"サポートされている画像形式: {supported_formats}",
            "UNSUPPORTED_IMAGE_FORMAT",
            415
        )


class ModelError(APIException):
    """Model loading or inference error (500)"""
    
    def __init__(self, message: str = "モデル処理中にエラーが発生しました"):
        super().__init__(
            message,
            "MODEL_ERROR",
            500
        )


class DatabaseError(APIException):
    """Database loading or query error (500)"""
    
    def __init__(self, message: str = "データベース処理中にエラーが発生しました"):
        super().__init__(
            message,
            "DATABASE_ERROR",
            500
        )


class ServiceUnavailableError(APIException):
    """Service unavailable error (503)"""
    
    def __init__(self, message: str = "サービスが一時的に利用できません"):
        super().__init__(
            message,
            "SERVICE_UNAVAILABLE",
            503
        )
