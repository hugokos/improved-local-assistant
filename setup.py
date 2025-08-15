"""
Setup script for the Improved Local AI Assistant.
"""

from setuptools import find_packages
from setuptools import setup

setup(
    name="improved-local-assistant",
    version="1.0.0",
    description="Improved Local AI Assistant with dual-model architecture and knowledge graph integration",
    author="AI Team",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0",
        "websockets>=12.0",
        "pydantic>=2.8.0",
        "pydantic-settings>=2.1.0",
        "ollama>=0.5.1",
        "llama-index[llms-ollama]>=0.12.50",
        "llama-index-embeddings-ollama>=0.6.0",
        "llama-index-llms-ollama>=0.6.2",
        "sentence-transformers>=2.5.0",
        "networkx>=3.2.1",
        "pyvis>=0.3.2",
        "neo4j>=5.14.1",
        "pyyaml>=6.0.1",
        "python-dotenv>=1.0.0",
        "aiofiles>=23.2.1",
        "psutil>=5.9.6",
        "asyncio-throttle>=1.0.2",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "local-assistant=run_app:main",
        ],
    },
)
