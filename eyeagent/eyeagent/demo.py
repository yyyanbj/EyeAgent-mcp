#!/usr/bin/env python3
"""
Demo script to test the multi-agent framework with different types of queries
"""

import asyncio
import sys
import os
from loguru import logger

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from multiagent_framework import run_multiagent_async

async def demo_queries():
    """Run demo queries to showcase different agent capabilities."""

    demo_queries = [
        "Calculate 2 + 3 * 4",  # Should use researcher with calculate tool
        "What's the weather like in Beijing?",  # Should use researcher with get_weather tool
        "Generate an image of a sunset",  # Should use writer with generate_image tool
        "Analyze this image for objects",  # Should use analyst with detect_objects tool
    ]

    logger.info("🎭 Starting Multi-Agent Framework Demo")
    logger.info("=" * 50)

    for i, query in enumerate(demo_queries, 1):
        logger.info(f"📝 Demo Query {i}: {query}")
        logger.info("-" * 30)

        try:
            result = await run_multiagent_async(query)
            logger.success(f"✨ Result: {result}")
        except Exception as e:
            logger.error(f"❌ Error: {e}")

        logger.info("=" * 50)

    logger.success("🎉 Demo completed! Check the logs above to see agent interactions.")

if __name__ == "__main__":
    asyncio.run(demo_queries())
