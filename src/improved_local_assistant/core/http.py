"""
HTTP utilities with proper timeout and retry handling.
"""

from pathlib import Path

from platformdirs import user_cache_dir
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

DEFAULT_TIMEOUT = (3.05, 30)  # (connect, read) seconds

# Use platformdirs for cache directory
CACHE_DIR = Path(user_cache_dir("improved-local-assistant", "hugokos"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def http_session(timeout: tuple[float, float] = DEFAULT_TIMEOUT, retries: int = 3) -> Session:
    """
    Create a requests session with proper timeout and retry configuration.

    Args:
        timeout: (connect_timeout, read_timeout) in seconds
        retries: Number of retries for failed requests

    Returns:
        Configured requests Session
    """
    s = Session()
    retry = Retry(
        total=retries,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"}),
    )
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))

    # Override request method to always include timeout
    original_request = s.request
    s.request = lambda method, url, *args, **kwargs: original_request(
        method, url, *args, timeout=kwargs.pop("timeout", timeout), **kwargs
    )

    return s
