#!/usr/bin/env python3
"""
Voice model download script for the Improved Local AI Assistant.

Downloads and sets up Vosk STT models and Piper TTS voices for offline voice processing.
"""

import logging
import sys
import tarfile
import zipfile
from pathlib import Path

import requests

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Model configurations
VOSK_MODELS = {
    "small-en": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
        "size": "40MB",
        "description": "Small English model, good for real-time processing",
    },
    "en": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip",
        "size": "1.8GB",
        "description": "Full English model, better accuracy",
    },
}

PIPER_VOICES = {
    "en_US-libritts-high": {
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/libritts/high/en_US-libritts-high.onnx",
        "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/libritts/high/en_US-libritts-high.onnx.json",
        "size": "63MB",
        "description": "High quality US English voice",
    },
    "en_US-lessac-medium": {
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
        "size": "63MB",
        "description": "Medium quality US English voice, faster",
    },
}


def download_file(url: str, destination: Path, description: str = ""):
    """Download a file with progress indication."""
    try:
        logger.info(f"Downloading {description or url} to {destination}")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        destination.parent.mkdir(parents=True, exist_ok=True)

        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(
                            f"\r  Progress: {percent:.1f}% ({downloaded // 1024 // 1024}MB)",
                            end="",
                            flush=True,
                        )

        print()  # New line after progress
        logger.info(f"Successfully downloaded {destination.name}")
        return True

    except Exception as e:
        logger.error(f"Failed to download {url}: {str(e)}")
        return False


def extract_archive(archive_path: Path, extract_to: Path):
    """Extract zip or tar archive."""
    try:
        logger.info(f"Extracting {archive_path.name}")

        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix in [".tar", ".gz", ".tgz"]:
            with tarfile.open(archive_path, "r:*") as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            logger.error(f"Unsupported archive format: {archive_path.suffix}")
            return False

        logger.info(f"Successfully extracted to {extract_to}")
        return True

    except Exception as e:
        logger.error(f"Failed to extract {archive_path}: {str(e)}")
        return False


def download_vosk_model(model_name: str, models_dir: Path):
    """Download and extract a Vosk model."""
    if model_name not in VOSK_MODELS:
        logger.error(f"Unknown Vosk model: {model_name}")
        return False

    model_info = VOSK_MODELS[model_name]
    model_dir = models_dir / f"vosk-model-{model_name}"

    # Check if model already exists
    if model_dir.exists() and any(model_dir.iterdir()):
        logger.info(f"Vosk model {model_name} already exists at {model_dir}")
        return True

    # Download model
    archive_name = f"vosk-model-{model_name}.zip"
    archive_path = models_dir / archive_name

    logger.info(f"Downloading Vosk model: {model_name} ({model_info['size']})")
    logger.info(f"Description: {model_info['description']}")

    if not download_file(model_info["url"], archive_path, f"Vosk {model_name} model"):
        return False

    # Extract model
    if not extract_archive(archive_path, models_dir):
        return False

    # Clean up archive
    try:
        archive_path.unlink()
        logger.info(f"Cleaned up archive: {archive_name}")
    except Exception as e:
        logger.warning(f"Failed to clean up archive: {str(e)}")

    # Find the extracted directory and rename if needed
    extracted_dirs = [
        d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("vosk-model")
    ]
    if extracted_dirs:
        extracted_dir = extracted_dirs[0]
        if extracted_dir != model_dir:
            extracted_dir.rename(model_dir)
            logger.info(f"Renamed {extracted_dir.name} to {model_dir.name}")

    return True


def download_piper_voice(voice_name: str, voices_dir: Path):
    """Download a Piper TTS voice."""
    if voice_name not in PIPER_VOICES:
        logger.error(f"Unknown Piper voice: {voice_name}")
        return False

    voice_info = PIPER_VOICES[voice_name]
    voice_dir = voices_dir / voice_name
    voice_dir.mkdir(parents=True, exist_ok=True)

    # Check if voice already exists
    model_file = voice_dir / f"{voice_name}.onnx"
    config_file = voice_dir / f"{voice_name}.onnx.json"

    if model_file.exists() and config_file.exists():
        logger.info(f"Piper voice {voice_name} already exists at {voice_dir}")
        return True

    logger.info(f"Downloading Piper voice: {voice_name} ({voice_info['size']})")
    logger.info(f"Description: {voice_info['description']}")

    # Download model file
    if not download_file(voice_info["url"], model_file, f"Piper {voice_name} model"):
        return False

    # Download config file
    return download_file(voice_info["config_url"], config_file, f"Piper {voice_name} config")


def setup_voice_directories():
    """Create voice model directories."""
    project_root = Path(__file__).parent.parent

    models_dir = project_root / "models" / "vosk"
    voices_dir = project_root / "models" / "piper"

    models_dir.mkdir(parents=True, exist_ok=True)
    voices_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Voice models directory: {models_dir}")
    logger.info(f"Voice models directory: {voices_dir}")

    return models_dir, voices_dir


def main():
    """Main function to download voice models."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download voice models for the Improved Local AI Assistant"
    )
    parser.add_argument(
        "--vosk", choices=list(VOSK_MODELS.keys()) + ["all"], help="Download Vosk STT model(s)"
    )
    parser.add_argument(
        "--piper", choices=list(PIPER_VOICES.keys()) + ["all"], help="Download Piper TTS voice(s)"
    )
    parser.add_argument("--all", action="store_true", help="Download all recommended models")
    parser.add_argument("--list", action="store_true", help="List available models")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable Vosk STT Models:")
        for name, info in VOSK_MODELS.items():
            print(f"  {name}: {info['description']} ({info['size']})")

        print("\nAvailable Piper TTS Voices:")
        for name, info in PIPER_VOICES.items():
            print(f"  {name}: {info['description']} ({info['size']})")
        return

    if not any([args.vosk, args.piper, args.all]):
        parser.print_help()
        return

    # Setup directories
    models_dir, voices_dir = setup_voice_directories()

    success = True

    # Download Vosk models
    if args.all or args.vosk:
        vosk_models = (
            ["small-en"]
            if args.all
            else ([args.vosk] if args.vosk != "all" else list(VOSK_MODELS.keys()))
        )

        for model_name in vosk_models:
            if not download_vosk_model(model_name, models_dir):
                success = False

    # Download Piper voices
    if args.all or args.piper:
        piper_voices = (
            ["en_US-lessac-medium"]
            if args.all
            else ([args.piper] if args.piper != "all" else list(PIPER_VOICES.keys()))
        )

        for voice_name in piper_voices:
            if not download_piper_voice(voice_name, voices_dir):
                success = False

    if success:
        logger.info("✅ All voice models downloaded successfully!")
        logger.info("Voice chat functionality is now ready to use.")
    else:
        logger.error("❌ Some voice models failed to download.")
        sys.exit(1)


if __name__ == "__main__":
    main()
