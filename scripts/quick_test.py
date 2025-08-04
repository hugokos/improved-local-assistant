#!/usr/bin/env python3
"""
Quick test script with timeout and better signal handling.

This script runs a quick test of the knowledge graph functionality
with built-in timeouts and proper signal handling.
"""

import asyncio
import logging
import signal
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global shutdown_requested
    logger.info("🛑 Shutdown requested. Cleaning up...")
    shutdown_requested = True

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

async def quick_test_with_timeout():
    """Run a quick test with timeout."""
    try:
        logger.info("🚀 Starting quick knowledge graph test...")
        
        # Import with timeout
        logger.info("📦 Importing modules...")
        from services.graph_manager import KnowledgeGraphManager
        
        if shutdown_requested:
            return False
        
        # Initialize with minimal config
        logger.info("⚙️  Initializing manager...")
        config = {
            "knowledge_graphs": {
                "prebuilt_directory": "./data/prebuilt_graphs",
                "max_triplets_per_chunk": 2,
            }
        }
        
        manager = KnowledgeGraphManager(model_manager=None, config=config)
        
        if shutdown_requested:
            return False
        
        # Initialize optimizer
        logger.info("🔧 Initializing optimizer...")
        try:
            from services.kg_optimizer import initialize_optimizer
            initialize_optimizer(manager)
            logger.info("✅ Optimizer initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️  Optimizer initialization failed: {e}")
        
        if shutdown_requested:
            return False
        
        # Get statistics (quick operation)
        logger.info("📊 Getting graph statistics...")
        stats = manager.get_graph_statistics()
        logger.info(f"📈 Found {stats['total_graphs']} graphs with {stats['total_nodes']} total nodes")
        
        logger.info("🎉 Quick test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

async def main():
    """Main function with timeout."""
    try:
        # Run test with 30 second timeout
        result = await asyncio.wait_for(quick_test_with_timeout(), timeout=30.0)
        
        if result:
            logger.info("✅ All tests passed!")
            return 0
        else:
            logger.error("❌ Tests failed!")
            return 1
            
    except asyncio.TimeoutError:
        logger.error("⏰ Test timed out after 30 seconds")
        return 1
    except KeyboardInterrupt:
        logger.info("🛑 Test interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted during startup")
        sys.exit(1)