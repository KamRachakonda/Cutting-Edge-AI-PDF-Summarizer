import json
import logging
import os
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, List, Optional, Tuple


# Configure logging
def setup_logger(name: str = "PDFSummarizer", log_level: int = logging.INFO) -> logging.Logger:
    """
    Setup a logger with both file and console handlers.

    Args:
        name: Logger name
        log_level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Avoid adding duplicate handlers if logger is already configured
    if not logger.handlers:
        # File handler
        log_file = log_dir / \
            f"pdf_summarizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


logger = setup_logger()


class ConfigManager:
    """Manage application configuration."""

    DEFAULT_CONFIG = {
        "api_timeout": 30,
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "provider": "openai",  # "openai" or "groq"
        "openai_model": "gpt-3.5-turbo",
        "groq_model": "openai/gpt-oss-20b",
        "temperature": 0.5,
        "max_file_size_mb": 50,
        "supported_formats": [".pdf"],
    }

    @staticmethod
    def load_config(config_path: str = "config.json") -> dict:
        """Load configuration from file or return defaults."""
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    logger.info(f"Configuration loaded from {config_path}")
                    return {**ConfigManager.DEFAULT_CONFIG, **config}
            except Exception as e:
                logger.warning(
                    f"Error loading config: {str(e)}. Using defaults.")
                return ConfigManager.DEFAULT_CONFIG
        else:
            logger.info("Using default configuration")
            return ConfigManager.DEFAULT_CONFIG

    @staticmethod
    def save_config(config: dict, config_path: str = "config.json") -> bool:
        """Save configuration to file."""
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
            logger.info(f"Configuration saved to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving config: {str(e)}")
            return False


class FileValidator:
    """Validate files for processing."""

    @staticmethod
    def validate_pdf_file(file_path: str, max_size_mb: int = 50) -> Tuple[bool, str]:
        """
        Validate if file is a valid PDF.

        Args:
            file_path: Path to file
            max_size_mb: Maximum file size in MB

        Returns:
            Tuple of (is_valid, message)
        """
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"

        # Check file extension
        if not file_path.lower().endswith('.pdf'):
            return False, "File must be a PDF"

        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return False, "File is empty"

        file_size_mb = file_size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return False, f"File size ({file_size_mb:.2f}MB) exceeds maximum ({max_size_mb}MB)"

        # Check if file starts with PDF magic number
        try:
            with open(file_path, 'rb') as f:
                header = f.read(4)
                if header != b'%PDF':
                    return False, "Invalid PDF file (wrong header)"
        except Exception as e:
            return False, f"Error reading file: {str(e)}"

        return True, "File is valid"

    @staticmethod
    def validate_api_key(api_key: str) -> Tuple[bool, str]:
        """
        Validate OpenAI or Groq API key format.

        Args:
            api_key: OpenAI or Groq API key

        Returns:
            Tuple of (is_valid, message)
        """
        if not api_key:
            return False, "API key is empty"

        if not isinstance(api_key, str):
            return False, "API key must be a string"

        api_key = api_key.strip()

        if len(api_key) < 20:
            return False, "API key is too short"

        # Check for OpenAI key format
        if api_key.startswith('sk-'):
            return True, "OpenAI API key is valid"

        # Check for Groq key format
        if api_key.startswith('gsk_'):
            return True, "Groq API key is valid"

        return False, "API key format not recognized (should start with 'sk-' or 'gsk_')"


class TextProcessor:
    """Additional text processing utilities."""

    @staticmethod
    def truncate_text(text: str, max_length: int = 1000) -> str:
        """Truncate text to maximum length."""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."

    @staticmethod
    def count_words(text: str) -> int:
        """Count words in text."""
        return len(text.split())

    @staticmethod
    def count_paragraphs(text: str) -> int:
        """Count paragraphs in text."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        return len(paragraphs)

    @staticmethod
    def estimate_reading_time(text: str, words_per_minute: int = 200) -> str:
        """Estimate reading time for text."""
        word_count = TextProcessor.count_words(text)
        minutes = max(1, word_count // words_per_minute)
        return f"{minutes} minute{'s' if minutes > 1 else ''}"

    @staticmethod
    def get_text_statistics(text: str) -> dict:
        """Get comprehensive text statistics."""
        words = text.split()
        total_words = len(words)
        return {
            "characters": len(text),
            "words": total_words,
            "paragraphs": TextProcessor.count_paragraphs(text),
            "sentences": len([s for s in text.split('.') if s.strip()]),
            "avg_word_length": round(sum(len(w) for w in words) / max(1, total_words), 2),
            "reading_time": TextProcessor.estimate_reading_time(text),
        }


class PerformanceUtils:
    """Performance monitoring utilities."""

    @staticmethod
    def timing_decorator(func):
        """Decorator to measure function execution time."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.info(
                f"{func.__name__} executed in {elapsed_time:.2f} seconds")
            return result
        return wrapper

    @staticmethod
    def measure_time(operation_name: str):
        """Context manager to measure execution time."""
        class Timer:
            def __enter__(self):
                self.start = time.time()
                return self

            def __exit__(self, *args):
                elapsed = time.time() - self.start
                logger.info(f"{operation_name} took {elapsed:.2f} seconds")

        return Timer()


class ErrorHandler:
    """Centralized error handling."""

    class PDFSummarizerError(Exception):
        """Base exception for PDF Summarizer."""
        pass

    class FileError(PDFSummarizerError):
        """File-related errors."""
        pass

    class APIError(PDFSummarizerError):
        """API-related errors."""
        pass

    class ProcessingError(PDFSummarizerError):
        """Processing-related errors."""
        pass

    @staticmethod
    def handle_error(error: Exception, context: str = "") -> str:
        """
        Handle and log errors consistently.

        Args:
            error: Exception object
            context: Additional context about where error occurred

        Returns:
            User-friendly error message
        """
        error_message = str(error)
        full_message = f"[{context}] {error_message}" if context else error_message
        logger.error(full_message, exc_info=True)

        lowered_msg = error_message.lower()
        if "api_key" in lowered_msg or "apikey" in lowered_msg or "groq_api_key" in lowered_msg:
            return "API key error. Please verify your API key is correctly configured."
        elif "rate limit" in lowered_msg:
            return "API rate limit exceeded. Please try again later."
        elif "authentication" in lowered_msg:
            return "Authentication failed. Check your API key."
        elif "connection" in lowered_msg:
            return "Connection error. Check your internet connection."
        else:
            return f"Error: {error_message}"


class CacheManager:
    """Simple caching utilities."""

    CACHE_DIR = Path(".cache")

    @staticmethod
    def setup_cache():
        """Create cache directory."""
        CacheManager.CACHE_DIR.mkdir(exist_ok=True)

    @staticmethod
    def save_to_cache(key: str, data: Any, ttl_seconds: int = 3600) -> bool:
        """
        Save data to cache.

        Args:
            key: Cache key
            data: Data to cache
            ttl_seconds: Time to live in seconds

        Returns:
            Success status
        """
        try:
            CacheManager.setup_cache()
            cache_file = CacheManager.CACHE_DIR / f"{key}.cache"

            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "ttl": ttl_seconds,
                "data": str(data),
            }

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f)

            logger.debug(f"Data cached with key: {key}")
            return True
        except Exception as e:
            logger.warning(f"Cache save failed: {str(e)}")
            return False

    @staticmethod
    def get_from_cache(key: str) -> Optional[str]:
        """
        Retrieve data from cache if valid.

        Args:
            key: Cache key

        Returns:
            Cached data or None if expired/not found
        """
        try:
            cache_file = CacheManager.CACHE_DIR / f"{key}.cache"

            if not cache_file.exists():
                return None

            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            # Check if cache expired
            cache_time = datetime.fromisoformat(cache_data["timestamp"])
            elapsed = (datetime.now() - cache_time).total_seconds()

            if elapsed > cache_data["ttl"]:
                cache_file.unlink(missing_ok=True)  # Delete expired cache
                logger.debug(f"Cache expired for key: {key}")
                return None

            logger.debug(f"Cache hit for key: {key}")
            return cache_data.get("data")
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {str(e)}")
            return None


class EnvironmentSetup:
    """Environment and dependency setup utilities."""

    @staticmethod
    def check_required_modules() -> Tuple[bool, List[str]]:
        """
        Check if all required modules are installed.

        Returns:
            Tuple of (all_installed, missing_modules)
        """
        # Map package display names to importable module names
        required_modules = {
            'streamlit': 'streamlit',
            'langchain': 'langchain',
            'langchain_core': 'langchain_core',
            'langchain_groq': 'langchain_groq',
            'langchain_openai': 'langchain_openai',
            'pypdf': 'pypdf',
            'python-dotenv': 'dotenv',
            'tiktoken': 'tiktoken',
            'pandas': 'pandas',
        }

        missing = []
        for pkg_name, import_name in required_modules.items():
            try:
                __import__(import_name)
            except ImportError:
                missing.append(pkg_name)

        if missing:
            logger.warning(f"Missing modules: {', '.join(missing)}")

        return len(missing) == 0, missing

    @staticmethod
    def load_environment_variables(env_file: str = ".env") -> bool:
        """
        Load environment variables from .env file.

        Args:
            env_file: Path to .env file

        Returns:
            Success status
        """
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
            logger.info(f"Environment variables loaded from {env_file}")
            return True
        except Exception as e:
            logger.warning(f"Could not load environment variables: {str(e)}")
            return False


# Export public API
__all__ = [
    'setup_logger',
    'logger',
    'ConfigManager',
    'FileValidator',
    'TextProcessor',
    'PerformanceUtils',
    'ErrorHandler',
    'CacheManager',
    'EnvironmentSetup',
]
