"""
EyeAgent Tools Utilities

This module provides common utility functions used across all EyeAgent tools,
including path handling, image loading, and data processing utilities.

Author: EyeAgent Development Team
Version: 1.0.0
License: MIT
"""

import logging
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Union, Optional, Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class PathHandler:
    """
    Utility class for handling various path formats and protocols.

    Supports:
    - Local file paths
    - file:// protocol
    - http:// and https:// URLs
    - Base64 encoded data URLs
    """

    @staticmethod
    def load_image_from_path(image_path: Union[str, Path]) -> Any:
        """
        Load image from various path formats.

        Args:
            image_path: Image path in various formats:
                - Local path: /path/to/image.jpg or ./image.jpg
                - File protocol: file:///path/to/image.jpg
                - HTTP/HTTPS: https://example.com/image.jpg
                - Data URL: data:image/jpeg;base64,...

        Returns:
            PIL Image object

        Raises:
            ValueError: If image cannot be loaded or format is unsupported
            ImportError: If PIL is not available
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("PIL (Pillow) is required for image loading. Install with: pip install Pillow")

        image_path = str(image_path)

        try:
            if image_path.startswith('file://'):
                # Local file with protocol
                local_path = image_path[7:]  # Remove 'file://'
                return Image.open(local_path).convert('RGB')

            elif image_path.startswith(('http://', 'https://')):
                # Remote URL
                return PathHandler._load_image_from_url(image_path)

            elif image_path.startswith('data:'):
                # Data URL (base64 encoded)
                return PathHandler._load_image_from_data_url(image_path)

            else:
                # Local path
                return Image.open(image_path).convert('RGB')

        except Exception as e:
            logger.error(f"Failed to load image from {image_path}: {e}")
            raise ValueError(f"Cannot load image from {image_path}: {e}")

    @staticmethod
    def _load_image_from_url(url: str) -> Any:
        """
        Load image from HTTP/HTTPS URL.

        Args:
            url: HTTP or HTTPS URL

        Returns:
            PIL Image object
        """
        from PIL import Image

        try:
            with urllib.request.urlopen(url) as response:
                with tempfile.NamedTemporaryFile(delete=False, suffix=PathHandler._get_image_extension(url)) as tmp_file:
                    tmp_file.write(response.read())
                    tmp_file_path = tmp_file.name

                image = Image.open(tmp_file_path).convert('RGB')

                # Clean up temp file
                try:
                    os.unlink(tmp_file_path)
                except OSError:
                    pass  # Ignore cleanup errors

                return image

        except Exception as e:
            logger.error(f"Failed to load image from URL {url}: {e}")
            raise

    @staticmethod
    def _load_image_from_data_url(data_url: str) -> Any:
        """
        Load image from base64 data URL.

        Args:
            data_url: Data URL (e.g., data:image/jpeg;base64,...)

        Returns:
            PIL Image object
        """
        from PIL import Image
        import base64
        import io

        try:
            # Parse data URL
            header, encoded = data_url.split(',', 1)
            if not header.startswith('data:image/'):
                raise ValueError("Invalid data URL format")

            # Decode base64
            image_data = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')

            return image

        except Exception as e:
            logger.error(f"Failed to load image from data URL: {e}")
            raise

    @staticmethod
    def _get_image_extension(url: str) -> str:
        """
        Get appropriate file extension from URL or content type.

        Args:
            url: URL to analyze

        Returns:
            File extension (e.g., '.jpg', '.png')
        """
        # Try to get extension from URL
        parsed = urlparse(url)
        path = parsed.path
        if '.' in path:
            ext = '.' + path.split('.')[-1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
                return ext

        # Default to .jpg
        return '.jpg'

    @staticmethod
    def resolve_path(path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> Path:
        """
        Resolve a path to an absolute Path object.

        Args:
            path: Input path
            base_dir: Base directory for relative paths

        Returns:
            Resolved absolute Path object
        """
        path = Path(path)

        if path.is_absolute():
            return path

        if base_dir:
            return Path(base_dir) / path

        # Use current working directory as base
        return path.resolve()

    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """
        Ensure a directory exists, creating it if necessary.

        Args:
            path: Directory path

        Returns:
            Path object for the directory
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def get_file_info(path: Union[str, Path]) -> dict:
        """
        Get information about a file.

        Args:
            path: File path

        Returns:
            Dictionary with file information
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        stat = path.stat()

        return {
            'path': str(path.absolute()),
            'name': path.name,
            'extension': path.suffix,
            'size': stat.st_size,
            'modified': stat.st_mtime,
            'is_file': path.is_file(),
            'is_dir': path.is_dir()
        }

    @staticmethod
    def validate_image_path(image_path: Union[str, Path]) -> bool:
        """
        Validate if a path points to a valid image file.

        Args:
            image_path: Path to validate

        Returns:
            True if path is valid and points to an image
        """
        try:
            from PIL import Image
        except ImportError:
            logger.warning("PIL not available for image validation")
            return False

        try:
            # Try to load the image (without keeping it in memory)
            with Image.open(str(image_path)) as img:
                img.verify()
            return True
        except Exception:
            return False

# Convenience functions for backward compatibility
def load_image_from_path(image_path: Union[str, Path]) -> Any:
    """
    Load image from various path formats.

    This is a convenience function that delegates to PathHandler.load_image_from_path
    for backward compatibility.

    Args:
        image_path: Image path in various formats

    Returns:
        PIL Image object
    """
    return PathHandler.load_image_from_path(image_path)

def resolve_path(path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolve a path to an absolute Path object.

    Args:
        path: Input path
        base_dir: Base directory for relative paths

    Returns:
        Resolved absolute Path object
    """
    return PathHandler.resolve_path(path, base_dir)

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    return PathHandler.ensure_directory(path)

def get_file_info(path: Union[str, Path]) -> dict:
    """
    Get information about a file.

    Args:
        path: File path

    Returns:
        Dictionary with file information
    """
    return PathHandler.get_file_info(path)

def validate_image_path(image_path: Union[str, Path]) -> bool:
    """
    Validate if a path points to a valid image file.

    Args:
        image_path: Path to validate

    Returns:
        True if path is valid and points to an image
    """
    return PathHandler.validate_image_path(image_path)
