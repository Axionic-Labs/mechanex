class MechanexError(Exception):
    """Base exception for the Mechanex library."""
    def __init__(self, message: str, status_code: int = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details

    def __str__(self):
        base = f"{self.message}"
        if self.status_code:
            base = f"[{self.status_code}] {base}"
        return base

class AuthenticationError(MechanexError):
    """Raised when authentication fails."""
    pass

class APIError(MechanexError):
    """Raised when the API returns an error."""
    pass

class NotFoundError(MechanexError):
    """Raised when a resource is not found."""
    pass

class RateLimitError(MechanexError):
    """Raised when rate limits are exceeded."""
    pass

class ValidationError(MechanexError):
    """Raised when request validation fails."""
    pass
