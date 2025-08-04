"""
Mock conversation data for UI testing
"""

sample_conversations = [
    {
        "id": "conv_001",
        "title": "Survival Shelter Building",
        "messages": [
            {
                "role": "user",
                "content": "How do I build a basic shelter in the wilderness?",
                "timestamp": "2024-01-15 10:30:00"
            },
            {
                "role": "assistant", 
                "content": "Building a basic wilderness shelter involves several key steps:\n\n1. **Location Selection**: Choose a spot that's dry, protected from wind, and away from hazards like dead trees or flood zones.\n\n2. **Basic Lean-to Structure**: Gather a long ridgepole and lean it against a tree or rock at about 45 degrees.\n\n3. **Framework**: Place smaller branches along the ridgepole to create ribs for your shelter.\n\n4. **Insulation**: Layer debris like leaves, pine needles, and bark over the framework for insulation and waterproofing.\n\n5. **Bedding**: Create a thick bed of dry materials inside to insulate you from the ground.\n\nRemember: The shelter should be just big enough for you to lie down - smaller spaces retain heat better.",
                "timestamp": "2024-01-15 10:30:15"
            }
        ]
    },
    {
        "id": "conv_002", 
        "title": "Python Data Processing",
        "messages": [
            {
                "role": "user",
                "content": "What's the best way to process large CSV files in Python?",
                "timestamp": "2024-01-15 14:20:00"
            },
            {
                "role": "assistant",
                "content": "For processing large CSV files in Python, here are the most effective approaches:\n\n**1. Pandas with Chunking:**\n```python\nimport pandas as pd\n\nfor chunk in pd.read_csv('large_file.csv', chunksize=10000):\n    # Process each chunk\n    processed = chunk.groupby('column').sum()\n    # Save or accumulate results\n```\n\n**2. Use Dask for Parallel Processing:**\n```python\nimport dask.dataframe as dd\n\ndf = dd.read_csv('large_file.csv')\nresult = df.groupby('column').sum().compute()\n```\n\n**3. Built-in CSV Module for Memory Efficiency:**\n```python\nimport csv\n\nwith open('large_file.csv', 'r') as file:\n    reader = csv.DictReader(file)\n    for row in reader:\n        # Process row by row\n        pass\n```\n\nChoose based on your needs: pandas for familiar syntax, Dask for parallel processing, or csv module for minimal memory usage.",
                "timestamp": "2024-01-15 14:20:25"
            }
        ]
    }
]

sample_models = [
    {
        "name": "llama3.2:3b",
        "status": "active",
        "response_time": 2.3,
        "memory_usage": "4.2GB",
        "last_used": "2024-01-15 15:30:00"
    },
    {
        "name": "llama3.2:1b", 
        "status": "idle",
        "response_time": 1.8,
        "memory_usage": "2.1GB",
        "last_used": "2024-01-15 12:15:00"
    },
    {
        "name": "qwen2.5:3b",
        "status": "idle", 
        "response_time": 2.1,
        "memory_usage": "3.8GB",
        "last_used": "2024-01-15 11:45:00"
    }
]

sample_graphs = [
    {
        "name": "survivalist",
        "size": "2.3MB",
        "nodes": 1250,
        "edges": 3400,
        "last_updated": "2024-01-15 09:00:00",
        "status": "healthy"
    },
    {
        "name": "general",
        "size": "1.8MB", 
        "nodes": 890,
        "edges": 2100,
        "last_updated": "2024-01-14 16:30:00",
        "status": "healthy"
    },
    {
        "name": "tech",
        "size": "3.1MB",
        "nodes": 1680,
        "edges": 4200,
        "last_updated": "2024-01-15 14:20:00",
        "status": "updating"
    }
]