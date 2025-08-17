"""
Safe archive extraction utilities to prevent path traversal attacks.
"""

import tarfile
import zipfile
from pathlib import Path


def _is_within(base: Path, target: Path) -> bool:
    """Check if target path is within base directory."""
    base = base.resolve()
    target = target.resolve()
    try:
        target.relative_to(base)
        return True
    except ValueError:
        return False


def safe_extract_tar(tar: tarfile.TarFile, extract_dir: Path) -> None:
    """
    Safely extract tar file, preventing path traversal attacks.

    Args:
        tar: Open tarfile object
        extract_dir: Directory to extract to

    Raises:
        RuntimeError: If unsafe path detected in archive
    """
    extract_dir = Path(extract_dir)
    for member in tar.getmembers():
        dest = extract_dir / member.name
        if not _is_within(extract_dir, dest):
            raise RuntimeError(f"Unsafe path in tar member: {member.name}")
    tar.extractall(extract_dir)


def safe_extract_zip(zip_file: zipfile.ZipFile, extract_dir: Path) -> None:
    """
    Safely extract zip file, preventing path traversal attacks.

    Args:
        zip_file: Open zipfile object
        extract_dir: Directory to extract to

    Raises:
        RuntimeError: If unsafe path detected in archive
    """
    extract_dir = Path(extract_dir)
    for member in zip_file.infolist():
        dest = extract_dir / member.filename
        if not _is_within(extract_dir, dest):
            raise RuntimeError(f"Unsafe path in zip member: {member.filename}")
    zip_file.extractall(extract_dir)
