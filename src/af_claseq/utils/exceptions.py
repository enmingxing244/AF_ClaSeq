"""
Common exceptions for AF-ClaSeq pipeline.
"""


class WorkflowError(Exception):
    """Custom exception for workflow errors."""
    pass


class ConfigurationError(Exception):
    """Raised when there's a configuration error."""
    pass


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


