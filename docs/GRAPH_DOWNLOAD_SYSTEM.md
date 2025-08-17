# Graph Download System Architecture

## Overview

The graph download system is designed to keep the main repository lightweight while providing easy access to large knowledge graph files. This document explains the complete architecture and workflow.

---

## 🎯 Design Goals

### Repository Separation
- **Main Repository**: Contains only code, documentation, and empty graph directories (~50MB)
- **Graph Distribution**: Large graph files hosted separately (100MB+ each)
- **User Choice**: Download only needed knowledge domains
- **Bandwidth Efficiency**: Avoid downloading unused graphs

### User Experience
- **One-Command Download**: Simple script interface
- **Interactive Selection**: Choose specific graphs
- **Progress Tracking**: Visual download progress
- **Integrity Verification**: Automatic checksum validation
- **Error Recovery**: Graceful handling of network issues

---

## 🏗️ System Architecture

### 1. Configuration System

The system is configured via `GRAPHS_CONFIG` in `scripts/download_graphs.py`:

```python
GRAPHS_CONFIG = {
    "source": "github_releases",  # Distribution method
    "repository": "hugokos/improved-local-assistant-graphs",  # Graph repository
    "release_tag": "graphs-v1.0.0",  # Version tag
    "base_url": "https://github.com/hugokos/improved-local-assistant-graphs/releases/download",
    "graphs_dir": "./data/graphs",  # Local installation directory
    "available_graphs": [...]  # Graph catalog
}
```

#### Graph Catalog Structure
Each graph entry contains:
```python
{
    "name": "survivalist",  # Unique identifier
    "filename": "survivalist-knowledge-v1.0.tar.gz",  # Archive filename
    "description": "Survivalist and outdoor knowledge base",  # Human description
    "size": "45MB",  # Compressed size
    "entities": "2,847",  # Number of entities
    "relationships": "8,234",  # Number of relationships
    "checksum": "sha256:abc123..."  # Integrity verification
}
```

### 2. Distribution Methods

The system supports multiple hosting strategies:

#### GitHub Releases (Current)
- **Pros**: Free, integrated with GitHub, version control
- **Cons**: 2GB per release limit, bandwidth limitations
- **Best for**: Open source projects, moderate file sizes

#### Cloud Storage (S3, Google Cloud)
- **Pros**: Unlimited size, global CDN, high bandwidth
- **Cons**: Costs money, requires setup
- **Best for**: Large files, enterprise distribution

#### CDN Distribution (jsDelivr, CloudFlare)
- **Pros**: Global distribution, high performance, caching
- **Cons**: File size limits, dependency on third party
- **Best for**: High-traffic projects, global audience

#### Self-Hosted
- **Pros**: Complete control, custom domains, no limits
- **Cons**: Infrastructure management, bandwidth costs
- **Best for**: Enterprise deployments, custom requirements

### 3. Download Workflow

#### User Interaction
```bash
# Interactive mode - shows available graphs and lets user select
python scripts/download_graphs.py

# Direct download - downloads specific graph
python scripts/download_graphs.py survivalist

# List mode - shows available graphs without downloading
python scripts/download_graphs.py --list
```

#### Internal Process Flow

1. **Configuration Loading**
   ```python
   # Load graph catalog from GRAPHS_CONFIG
   available_graphs = GRAPHS_CONFIG["available_graphs"]
   ```

2. **URL Construction**
   ```python
   def build_download_url(graph):
       base_url = GRAPHS_CONFIG["base_url"]
       release_tag = GRAPHS_CONFIG["release_tag"]
       filename = graph["filename"]
       return f"{base_url}/{release_tag}/{filename}"
   ```

3. **Download Process**
   ```python
   def download_file(url, filepath, description):
       # Stream download with progress tracking
       response = requests.get(url, stream=True)
       total_size = int(response.headers.get('content-length', 0))

       with open(filepath, 'wb') as f:
           for chunk in response.iter_content(chunk_size=8192):
               f.write(chunk)
               # Update progress bar
   ```

4. **Integrity Verification**
   ```python
   def verify_checksum(filepath, expected_checksum):
       hash_type, expected_hash = expected_checksum.split(":", 1)
       hasher = hashlib.sha256()

       with open(filepath, 'rb') as f:
           for chunk in iter(lambda: f.read(4096), b""):
               hasher.update(chunk)

       return hasher.hexdigest() == expected_hash
   ```

5. **Archive Extraction**
   ```python
   def extract_archive(filepath, extract_dir):
       if filepath.suffix == '.gz' and filepath.stem.endswith('.tar'):
           with tarfile.open(filepath, 'r:gz') as tar:
               tar.extractall(extract_dir)
       # Clean up archive after extraction
       filepath.unlink()
   ```

### 4. Directory Structure

#### Before Download
```
data/graphs/
├── README.md          # Usage instructions
└── .gitkeep          # Preserves directory in Git
```

#### After Download
```
data/graphs/
├── README.md          # Usage instructions
├── .gitkeep          # Preserves directory in Git
└── survivalist/      # Downloaded graph
    ├── kg.json       # Property graph data
    ├── graph_meta.json  # Graph metadata
    ├── triples.json  # Compatibility triples
    └── vector_store.json  # Vector embeddings
```

### 5. Integration with Application

#### Automatic Discovery
The application automatically discovers downloaded graphs:

```python
# services/graph_router.py
def discover_graphs():
    graphs_dir = Path("data/graphs")
    available_graphs = []

    for graph_dir in graphs_dir.iterdir():
        if graph_dir.is_dir() and (graph_dir / "graph_meta.json").exists():
            available_graphs.append(graph_dir.name)

    return available_graphs
```

#### Graph Loading
```python
# services/graph_manager/persistence_simple.py
def load_graph(graph_name):
    graph_path = Path(f"data/graphs/{graph_name}")

    # Check for property graph format
    if (graph_path / "kg.json").exists():
        return load_property_graph(graph_path)
    # Fallback to simple graph format
    elif (graph_path / "triples.json").exists():
        return load_simple_graph(graph_path)
```

---

## 🔧 Implementation Details

### Error Handling

#### Network Issues
```python
def download_with_retry(url, filepath, max_retries=3):
    for attempt in range(max_retries):
        try:
            return download_file(url, filepath)
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
```

#### Disk Space Checks
```python
def check_disk_space(required_bytes):
    free_bytes = shutil.disk_usage(".").free
    if free_bytes < required_bytes * 1.2:  # 20% buffer
        raise InsufficientDiskSpaceError()
```

#### Partial Download Recovery
```python
def resume_download(url, filepath):
    if filepath.exists():
        resume_header = {'Range': f'bytes={filepath.stat().st_size}-'}
        response = requests.get(url, headers=resume_header, stream=True)
        mode = 'ab'  # Append mode
    else:
        response = requests.get(url, stream=True)
        mode = 'wb'  # Write mode
```

### Performance Optimizations

#### Concurrent Downloads
```python
import concurrent.futures

def download_multiple_graphs(graph_names):
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(download_graph, name): name
            for name in graph_names
        }

        for future in concurrent.futures.as_completed(futures):
            graph_name = futures[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"Failed to download {graph_name}: {e}")
```

#### Compression Optimization
```python
# Use maximum compression for distribution
tar -czf --best survivalist-knowledge-v1.0.tar.gz survivalist/

# Alternative: 7zip for better compression
7z a -t7z -mx=9 survivalist-knowledge-v1.0.7z survivalist/
```

### Security Considerations

#### Checksum Verification
- All downloads verified with SHA256 checksums
- Prevents corrupted or tampered files
- Automatic retry on checksum mismatch

#### URL Validation
```python
def validate_download_url(url):
    parsed = urlparse(url)
    if parsed.scheme not in ['https']:
        raise SecurityError("Only HTTPS URLs allowed")
    if not parsed.netloc.endswith(('.github.com', '.amazonaws.com')):
        raise SecurityError("Untrusted domain")
```

#### Path Traversal Prevention
```python
def safe_extract(tar, path):
    for member in tar.getmembers():
        if os.path.isabs(member.name) or ".." in member.name:
            raise SecurityError(f"Unsafe path: {member.name}")
    tar.extractall(path)
```

---

## 📊 Monitoring and Analytics

### Download Tracking
```python
def track_download(graph_name, version, success):
    analytics_data = {
        "event": "graph_download",
        "graph": graph_name,
        "version": version,
        "success": success,
        "timestamp": datetime.utcnow().isoformat(),
        "user_agent": "improved-local-assistant/2.0.0"
    }

    # Send to analytics endpoint (optional)
    try:
        requests.post(ANALYTICS_URL, json=analytics_data, timeout=5)
    except:
        pass  # Don't fail download for analytics
```

### Health Monitoring
```python
def check_graph_availability():
    """Monitor if graph download endpoints are accessible."""
    for graph in GRAPHS_CONFIG["available_graphs"]:
        url = build_download_url(graph)
        try:
            response = requests.head(url, timeout=10)
            if response.status_code != 200:
                alert_admin(f"Graph {graph['name']} unavailable: {response.status_code}")
        except requests.RequestException as e:
            alert_admin(f"Graph {graph['name']} unreachable: {e}")
```

---

## 🚀 Future Enhancements

### Planned Features

#### Delta Updates
```python
# Download only changes since last version
python scripts/download_graphs.py survivalist --delta-from v1.0.0
```

#### Peer-to-Peer Distribution
```python
# BitTorrent-style distribution for large graphs
python scripts/download_graphs.py survivalist --p2p
```

#### Graph Streaming
```python
# Stream graph data without full download
python scripts/stream_graph.py survivalist --query "fire starting"
```

#### Automatic Updates
```python
# Check for and download graph updates
python scripts/update_graphs.py --check-all
```

### Scalability Improvements

#### CDN Integration
- Automatic failover between multiple CDNs
- Geographic routing for optimal performance
- Edge caching for frequently accessed graphs

#### Compression Improvements
- Graph-specific compression algorithms
- Incremental compression for updates
- Streaming decompression for large files

#### Bandwidth Optimization
- Adaptive bitrate based on connection speed
- Pause/resume functionality
- Background downloading with priority queues

---

## 🎯 Summary

The graph download system provides:

1. **Separation of Concerns**: Code and data distributed independently
2. **User Choice**: Download only needed knowledge domains
3. **Multiple Distribution Options**: GitHub, cloud storage, CDN, self-hosted
4. **Robust Error Handling**: Network issues, disk space, integrity verification
5. **Performance Optimization**: Concurrent downloads, compression, caching
6. **Security**: Checksum verification, URL validation, path traversal prevention
7. **Monitoring**: Download tracking, health checks, analytics

This architecture enables the main repository to stay lightweight while providing easy access to large knowledge graph files, supporting the project's goal of being both powerful and accessible.

---

*For implementation details, see `scripts/download_graphs.py` and related documentation.*
