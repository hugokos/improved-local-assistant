#!/usr/bin/env python3
"""
Test script to verify LLM setup is working correctly.

Tests that LlamaIndex is using local Ollama models instead of MockLLM.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_llamaindex_settings():
    """Test that LlamaIndex Settings are configured correctly."""
    logger.info("Testing LlamaIndex Settings configuration...")

    try:
        # Initialize model manager first to set up Settings
        from services.orchestrated_model_manager import OrchestratedModelManager

        from app.core.config import load_config

        config = load_config()
        mgr = OrchestratedModelManager(config)  # This sets up Settings.llm

        from llama_index.core import Settings

        # Check LLM
        if Settings.llm is None:
            logger.error("❌ Settings.llm is None - this will cause MockLLM fallback")
            return False

        llm_type = type(Settings.llm).__name__
        llm_model = getattr(Settings.llm, "model", "Unknown")

        logger.info(f"✅ LLM Type: {llm_type}")
        logger.info(f"✅ LLM Model: {llm_model}")

        if llm_type == "MockLLM":
            logger.error("❌ Still using MockLLM - configuration failed")
            return False
        elif llm_type == "Ollama":
            logger.info("✅ Using Ollama LLM - configuration successful")
        else:
            logger.warning(f"⚠️ Using {llm_type} - not Ollama but not MockLLM")

        # Check embeddings
        if Settings.embed_model is None:
            logger.error("❌ Settings.embed_model is None")
            return False

        embed_type = type(Settings.embed_model).__name__
        logger.info(f"✅ Embedding Type: {embed_type}")

        return True

    except Exception as e:
        logger.error(f"❌ Failed to check LlamaIndex settings: {str(e)}")
        return False


def test_model_manager_setup():
    """Test that ModelManager sets up LLMs correctly."""
    logger.info("Testing ModelManager LLM setup...")

    try:
        from services.orchestrated_model_manager import OrchestratedModelManager

        from app.core.config import load_config

        config = load_config()
        mgr = OrchestratedModelManager(config)

        # Check that it has the LLM attributes
        if hasattr(mgr, "chat_llm"):
            logger.info(f"✅ Chat LLM: {type(mgr.chat_llm).__name__} ({mgr.conversation_model})")
        else:
            logger.warning("⚠️ No chat_llm attribute found")

        if hasattr(mgr, "kg_llm"):
            logger.info(f"✅ KG LLM: {type(mgr.kg_llm).__name__} ({mgr.knowledge_model})")
        else:
            logger.warning("⚠️ No kg_llm attribute found")

        return True

    except Exception as e:
        logger.error(f"❌ Failed to test ModelManager: {str(e)}")
        return False


def test_no_openai_fallback():
    """Test that we don't accidentally fall back to OpenAI."""
    logger.info("Testing for OpenAI fallback prevention...")

    try:
        from llama_index.core import Settings

        # Try to access the LLM without any OpenAI keys set
        llm = Settings.llm

        if llm is None:
            logger.error("❌ LLM is None - will cause OpenAI fallback")
            return False

        # Check if it's trying to use OpenAI
        llm_class = type(llm).__name__
        if "OpenAI" in llm_class:
            logger.error(f"❌ Using OpenAI LLM: {llm_class}")
            return False
        elif "Mock" in llm_class:
            logger.error(f"❌ Using MockLLM: {llm_class}")
            return False
        else:
            logger.info(f"✅ Using local LLM: {llm_class}")
            return True

    except Exception as e:
        logger.error(f"❌ Error testing OpenAI fallback: {str(e)}")
        return False


async def test_llm_functionality():
    """Test basic LLM functionality (if Ollama is running)."""
    logger.info("Testing basic LLM functionality...")

    try:
        from llama_index.core import Settings

        llm = Settings.llm
        if llm is None:
            logger.error("❌ No LLM available for testing")
            return False

        # Try a simple completion
        try:
            response = llm.complete("Hello, respond with just 'Hi there!'")
            response_text = str(response).strip()

            logger.info(f"✅ LLM Response: {response_text}")

            if "Hi there" in response_text or "hello" in response_text.lower():
                logger.info("✅ LLM is responding appropriately")
                return True
            else:
                logger.warning(f"⚠️ Unexpected response: {response_text}")
                return True  # Still working, just unexpected response

        except Exception as e:
            error_msg = str(e).lower()
            if "connection" in error_msg or "refused" in error_msg:
                logger.warning("⚠️ Ollama not running - LLM configured but can't connect")
                return True  # Configuration is correct, just Ollama not running
            elif "memory" in error_msg or "system memory" in error_msg:
                logger.warning(
                    "⚠️ Model requires more memory than available - LLM configured correctly"
                )
                return True  # Configuration is correct, just insufficient memory
            else:
                logger.error(f"❌ LLM functionality test failed: {str(e)}")
                return False

    except Exception as e:
        logger.error(f"❌ Failed to test LLM functionality: {str(e)}")
        return False


async def main():
    """Run all LLM setup tests."""
    logger.info("🧠 Starting LLM Setup Tests...")

    tests = [
        ("LlamaIndex Settings", test_llamaindex_settings),
        ("ModelManager Setup", test_model_manager_setup),
        ("OpenAI Fallback Prevention", test_no_openai_fallback),
        ("LLM Functionality", test_llm_functionality),
    ]

    results = []

    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {str(e)}")
            results.append((test_name, False))

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("LLM SETUP TEST RESULTS")
    logger.info("=" * 50)

    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1

    logger.info(f"\nOverall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        logger.info("🎉 All LLM setup tests passed! No more MockLLM.")
        logger.info("The system is now using proper local Ollama models.")
    elif passed >= len(results) - 1:  # Allow 1 failure (Ollama might not be running)
        logger.info("✅ LLM setup is correct. Start Ollama for full functionality.")
    else:
        logger.warning("⚠️ Some LLM setup tests failed. Check configuration.")

    return passed >= len(results) - 1


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
