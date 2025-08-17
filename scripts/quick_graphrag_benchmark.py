#!/usr/bin/env python3
"""
Quick GraphRAG Performance Test

A simple, user-friendly benchmark that tests the complete GraphRAG pipeline
with realistic queries. Shows the breakdown of where time is spent:
- Knowledge retrieval
- Context preparation
- AI response generation

Perfect for users who want to quickly test their system performance.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from improved_local_assistant.app.core.config import load_config  # noqa: E402
from improved_local_assistant.services.conversation_manager import ConversationManager  # noqa: E402
from improved_local_assistant.services.graph_manager import KnowledgeGraphManager  # noqa: E402
from improved_local_assistant.services.model_manager import ModelConfig  # noqa: E402
from improved_local_assistant.services.model_manager import ModelManager  # noqa: E402


async def quick_benchmark():
    """Run a quick GraphRAG performance test."""

    print("🚀 Quick GraphRAG Performance Test")
    print("=" * 50)
    print("Testing the complete AI pipeline including:")
    print("• Knowledge graph retrieval")
    print("• Context assembly")
    print("• AI response generation")
    print()

    try:
        # Initialize system
        print("⚙️  Initializing system...")
        config = load_config()

        # Setup components
        host = config.get("ollama", {}).get("host", "http://localhost:11434")
        model_manager = ModelManager(host)
        kg_manager = KnowledgeGraphManager(config)
        conversation_manager = ConversationManager(
            model_manager=model_manager, kg_manager=kg_manager, config=config
        )

        # Initialize models
        model_config = ModelConfig(
            name="hermes3:3b",
            type="conversation",
            temperature=0.7,
            max_tokens=200,  # Shorter responses for quick test
        )

        init_success = await model_manager.initialize_models(model_config)
        if not init_success:
            print("❌ Failed to initialize models. Make sure Ollama is running.")
            return

        # Initialize knowledge graph (simple initialization)
        try:
            kg_manager.initialize_dynamic_graph()
            print("✅ Knowledge graph initialized!")
        except Exception as e:
            print(f"⚠️  Knowledge graph initialization failed: {e}")
            print("Continuing with model-only testing...")

        print("✅ System initialized successfully!")
        print()

        # Test query
        test_query = "What is artificial intelligence and how does it work?"
        print(f"🔍 Testing query: '{test_query}'")
        print()

        # Create session and run test
        session_id = conversation_manager.create_session()

        # Measure complete pipeline
        start_time = time.time()
        first_token_time = None
        response_text = ""
        retrieval_time = 0

        # Try to measure actual retrieval time
        print("🔍 Checking knowledge graph...")
        retrieval_start = time.time()
        try:
            # Try to query the knowledge graph
            context = await kg_manager.query_graphs(test_query)
            retrieval_time = time.time() - retrieval_start
            if context:
                print(f"📚 Found {len(context)} relevant context items in {retrieval_time:.3f}s")
            else:
                print(f"📚 No specific context found (searched in {retrieval_time:.3f}s)")
        except Exception as e:
            retrieval_time = time.time() - retrieval_start
            print(f"📚 Graph query failed in {retrieval_time:.3f}s: {e}")

        print()
        print("💭 AI Response:")
        print("-" * 30)

        # Stream the response
        response_stream = conversation_manager.process_message(session_id, test_query)

        async for chunk in response_stream:
            if first_token_time is None:
                first_token_time = time.time()
                ttft = first_token_time - start_time
                print(f"⚡ First token in {ttft:.2f}s")
                print()

            if isinstance(chunk, str):
                print(chunk, end="", flush=True)
                response_text += chunk

        end_time = time.time()
        total_time = end_time - start_time

        print()
        print()
        print("📊 Performance Results:")
        print("-" * 30)

        # Calculate metrics
        if first_token_time:
            ttft = first_token_time - start_time
            generation_time = end_time - first_token_time
            token_count = len(response_text.split())

            print(f"⏱️  Time to First Token: {ttft:.2f}s")
            print(f"🏁 Total Response Time: {total_time:.2f}s")
            print(f"📝 Response Length: {token_count} tokens")

            if generation_time > 0:
                throughput = token_count / generation_time
                print(f"🚄 Generation Speed: {throughput:.1f} tokens/sec")

            # Actual breakdown with measured retrieval time
            model_time = total_time - retrieval_time

            print()
            print("🔍 Pipeline Breakdown:")
            print(
                f"   Knowledge Retrieval: {retrieval_time:.3f}s ({retrieval_time/total_time*100:.1f}%)"
            )
            print(f"   AI Processing: {model_time:.3f}s ({model_time/total_time*100:.1f}%)")

        else:
            print(f"⏱️  Total Time: {total_time:.2f}s")
            print("⚠️  No response generated")

        print()
        print("✅ Quick benchmark completed!")

        # Performance assessment
        if first_token_time and ttft < 1.0:
            print("🎉 Excellent response speed!")
        elif first_token_time and ttft < 2.0:
            print("👍 Good response speed")
        elif first_token_time:
            print("⚠️  Response speed could be improved")

        print()
        print("💡 Tips:")
        print("   • Close other applications to free up CPU/RAM")
        print("   • Use smaller models (tinyllama) for faster responses")
        print("   • Run full benchmarks with: python scripts/benchmark_models.py")

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        print()
        print("🔧 Troubleshooting:")
        print("   • Make sure Ollama is running: ollama serve")
        print("   • Check if models are installed: ollama list")
        print("   • Verify the app is properly configured")


if __name__ == "__main__":
    asyncio.run(quick_benchmark())
