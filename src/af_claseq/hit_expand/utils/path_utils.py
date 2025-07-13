#!/usr/bin/env python3
"""
Path utilities for the MSA pipeline.
Provides centralized path validation for full/absolute paths.
"""

import logging
from pathlib import Path
from typing import Dict, Union

logger = logging.getLogger(__name__)


class PathUtils:
    """Utilities for path validation and management."""
    
    @staticmethod
    def validate_file_exists(file_path: Union[str, Path]) -> Path:
        """
        Validate that a file exists.
        
        Args:
            file_path: Path to file (must be absolute)
            
        Returns:
            Path object
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path
    
    @staticmethod
    def validate_directory_exists(dir_path: Union[str, Path], create: bool = True) -> Path:
        """
        Validate that a directory exists or create it.
        
        Args:
            dir_path: Path to directory (must be absolute)
            create: Whether to create directory if it doesn't exist
            
        Returns:
            Path object
            
        Raises:
            FileNotFoundError: If directory doesn't exist and create=False
        """
        path = Path(dir_path)
        if not path.exists():
            if create:
                path.mkdir(parents=True, exist_ok=True)
            else:
                raise FileNotFoundError(f"Directory not found: {path}")
        elif not path.is_dir():
            raise ValueError(f"Path exists but is not a directory: {path}")
        return path
    
    @staticmethod
    def validate_file_size(file_path: Union[str, Path], max_size_gb: float = 1.0) -> None:
        """
        Validate file size is within limits.
        
        Args:
            file_path: Path to file
            max_size_gb: Maximum allowed size in GB
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is too large or empty
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        file_size = path.stat().st_size
        
        if file_size > max_size_bytes:
            raise ValueError(
                f"File too large: {path} "
                f"({file_size / (1024**3):.2f}GB > {max_size_gb}GB)"
            )
        
        if file_size == 0:
            raise ValueError(f"File is empty: {path}")


def validate_absolute_path(path: Union[str, Path]) -> Path:
    """
    Validate that a path is absolute.
    
    Args:
        path: Path to validate
        
    Returns:
        Path object
        
    Raises:
        ValueError: If path is not absolute
    """
    path_obj = Path(path)
    if not path_obj.is_absolute():
        raise ValueError(f"Path must be absolute: {path}")
    return path_obj


def ensure_absolute_paths(config_data: Dict) -> Dict:
    """
    Ensure all paths in configuration are absolute.
    
    Args:
        config_data: Configuration dictionary
        
    Returns:
        Updated configuration with validated absolute paths
        
    Raises:
        ValueError: If any path is not absolute
    """
    def _validate_path(value):
        if isinstance(value, str) and value:
            # Check if it looks like a path (contains / or \ or ends with file extension)
            if ('/' in value or '\\' in value or 
                (value.count('.') == 1 and len(value.split('.')[-1]) <= 4)):
                # Validate as absolute path
                return str(validate_absolute_path(value))
        elif isinstance(value, dict):
            return {k: _validate_path(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_validate_path(item) for item in value]
        return value
    
    return _validate_path(config_data)