#!/usr/bin/env python3
"""
CLI tool to toggle edge optimization features.

This script allows users to easily enable or disable the edge optimization
system and view the current configuration status.
"""

import argparse
import sys
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import load_config


def load_config_file():
    """Load the configuration file."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return None, None
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config, config_path
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return None, None


def save_config_file(config, config_path):
    """Save the configuration file."""
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        return True
    except Exception as e:
        print(f"❌ Error saving configuration: {e}")
        return False


def show_status(config):
    """Show current edge optimization status."""
    print("🔍 Current Edge Optimization Status:")
    print("=" * 50)
    
    edge_config = config.get("edge_optimization", {})
    enabled = edge_config.get("enabled", False)
    mode = edge_config.get("mode", "production")
    
    print(f"📊 Status: {'✅ ENABLED' if enabled else '❌ DISABLED'}")
    print(f"🎯 Mode: {mode}")
    
    if enabled:
        print("\n🚀 Active Features:")
        
        # Orchestration
        orchestration = config.get("orchestration", {})
        print(f"  • LLM Orchestration: ✅")
        print(f"    - Semaphore timeout: {orchestration.get('llm_semaphore_timeout', 30.0)}s")
        print(f"    - JSON extraction: {orchestration.get('json_mode_for_extraction', True)}")
        print(f"    - Skip on pressure: {orchestration.get('extraction_skip_on_pressure', True)}")
        
        # Working Set Cache
        cache_config = config.get("working_set_cache", {})
        print(f"  • Working Set Cache: ✅")
        print(f"    - Nodes per session: {cache_config.get('nodes_per_session', 100)}")
        print(f"    - Memory limit: {cache_config.get('global_memory_limit_mb', 256)}MB")
        
        # Hybrid Retriever
        retriever_config = config.get("hybrid_retriever", {})
        print(f"  • Hybrid Retriever: ✅")
        print(f"    - Max chunks: {retriever_config.get('budget', {}).get('max_chunks', 12)}")
        weights = retriever_config.get('weights', {})
        print(f"    - Weights: Graph={weights.get('graph', 0.6)}, BM25={weights.get('bm25', 0.2)}, Vector={weights.get('vector', 0.2)}")
        
        # Connection Pool
        pool_config = config.get("connection_pool", {})
        print(f"  • Connection Pool: ✅")
        print(f"    - Max connections: {pool_config.get('max_connections', 10)}")
        print(f"    - Keep-alive: {pool_config.get('keepalive_expiry', 30.0)}s")
        
        # Extraction Pipeline
        extraction_config = config.get("extraction", {})
        print(f"  • Extraction Pipeline: ✅")
        print(f"    - Max time: {extraction_config.get('max_time_seconds', 8.0)}s")
        print(f"    - Max triples: {extraction_config.get('max_triples_per_turn', 10)}")
        
        # Server Environment
        server_env = config.get("server_env", {})
        if server_env:
            print(f"  • Server Environment: ✅")
            print(f"    - OLLAMA_NUM_PARALLEL: {server_env.get('OLLAMA_NUM_PARALLEL', 1)}")
            print(f"    - OLLAMA_MAX_LOADED_MODELS: {server_env.get('OLLAMA_MAX_LOADED_MODELS', 1)}")
    
    else:
        print("\n⚠️  Edge optimization is disabled. Using standard model manager.")
    
    print("\n💡 Use --enable or --disable to change the status.")


def enable_edge_optimization(config):
    """Enable edge optimization."""
    print("🚀 Enabling Edge Optimization...")
    
    # Ensure edge_optimization section exists
    if "edge_optimization" not in config:
        config["edge_optimization"] = {}
    
    config["edge_optimization"]["enabled"] = True
    
    # Add default configurations if they don't exist
    if "orchestration" not in config:
        config["orchestration"] = {
            "llm_semaphore_timeout": 30.0,
            "extraction_skip_on_pressure": True,
            "keep_alive_hermes": "30m",
            "keep_alive_tinyllama": 0,
            "json_mode_for_extraction": True,
            "max_extraction_time": 8.0
        }
    
    if "working_set_cache" not in config:
        config["working_set_cache"] = {
            "nodes_per_session": 100,
            "max_edges_per_query": 40,
            "global_memory_limit_mb": 256,
            "eviction_threshold": 0.8,
            "include_text": False,
            "persist_dir": "./storage"
        }
    
    if "hybrid_retriever" not in config:
        config["hybrid_retriever"] = {
            "weights": {"graph": 0.6, "bm25": 0.2, "vector": 0.2},
            "budget": {"max_chunks": 12, "graph_depth": 2, "bm25_top_k": 3, "vector_top_k": 3},
            "response_mode": "compact",
            "query_fusion": {"enabled": True, "num_queries": 3},
            "vector": {"model": "BAAI/bge-small-en-v1.5", "device": "cpu", "int8": True}
        }
    
    if "connection_pool" not in config:
        config["connection_pool"] = {
            "max_connections": 10,
            "max_keepalive_connections": 5,
            "keepalive_expiry": 30.0,
            "timeout": {"connect": 5.0, "read": 30.0, "write": 30.0, "pool": 5.0},
            "keep_alive_models": ["hermes3:3b"]
        }
    
    if "extraction" not in config:
        config["extraction"] = {
            "max_time_seconds": 8.0,
            "max_tokens": 1024,
            "max_triples_per_turn": 10,
            "skip_on_memory_pressure": True,
            "skip_on_cpu_pressure": True,
            "input_window_turns": 3,
            "format": "json",
            "unload_after_extraction": True
        }
    
    if "server_env" not in config:
        config["server_env"] = {
            "OLLAMA_NUM_PARALLEL": 1,
            "OLLAMA_MAX_LOADED_MODELS": 1
        }
    
    print("✅ Edge optimization enabled successfully!")
    print("💡 Restart the application for changes to take effect.")


def disable_edge_optimization(config):
    """Disable edge optimization."""
    print("⏸️  Disabling Edge Optimization...")
    
    # Ensure edge_optimization section exists
    if "edge_optimization" not in config:
        config["edge_optimization"] = {}
    
    config["edge_optimization"]["enabled"] = False
    
    print("✅ Edge optimization disabled successfully!")
    print("💡 Restart the application for changes to take effect.")


def set_mode(config, mode):
    """Set edge optimization mode."""
    valid_modes = ["development", "production", "raspberry_pi", "iphone"]
    
    if mode not in valid_modes:
        print(f"❌ Invalid mode: {mode}")
        print(f"Valid modes: {', '.join(valid_modes)}")
        return False
    
    if "edge_optimization" not in config:
        config["edge_optimization"] = {}
    
    config["edge_optimization"]["mode"] = mode
    
    print(f"✅ Edge optimization mode set to: {mode}")
    print("💡 Restart the application for changes to take effect.")
    return True


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Toggle edge optimization features for the Improved Local AI Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python cli/toggle_edge_optimization.py --status
    python cli/toggle_edge_optimization.py --enable
    python cli/toggle_edge_optimization.py --disable
    python cli/toggle_edge_optimization.py --mode raspberry_pi
        """
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current edge optimization status"
    )
    
    parser.add_argument(
        "--enable",
        action="store_true",
        help="Enable edge optimization features"
    )
    
    parser.add_argument(
        "--disable",
        action="store_true",
        help="Disable edge optimization features"
    )
    
    parser.add_argument(
        "--mode",
        choices=["development", "production", "raspberry_pi", "iphone"],
        help="Set edge optimization mode"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config, config_path = load_config_file()
    if config is None:
        return 1
    
    # Handle commands
    if args.status or (not args.enable and not args.disable and not args.mode):
        show_status(config)
        return 0
    
    config_changed = False
    
    if args.enable:
        enable_edge_optimization(config)
        config_changed = True
    
    if args.disable:
        disable_edge_optimization(config)
        config_changed = True
    
    if args.mode:
        if set_mode(config, args.mode):
            config_changed = True
        else:
            return 1
    
    # Save configuration if changed
    if config_changed:
        if save_config_file(config, config_path):
            print(f"💾 Configuration saved to: {config_path}")
        else:
            return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)