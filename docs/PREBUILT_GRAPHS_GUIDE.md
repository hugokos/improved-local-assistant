# Prebuilt Knowledge Graphs Distribution Guide

## Overview

This guide explains how to distribute and use prebuilt knowledge graphs for the Improved Local Assistant. Prebuilt graphs are hosted separately from the main repository to keep the codebase lightweight while providing ready-to-use knowledge bases.

---

## 🎯 Why Separate Distribution?

### Repository Benefits
- **Lightweight codebase**: Main repository stays under 50MB
- **Fast cloning**: Quick setup for developers and users
- **Version control efficiency**: No large binary files in Git history
- **Modular deployment**: Users can choose which graphs to download

### Graph Benefits
- **Large file support**: Knowledge graphs can be 100MB+ each
- **Specialized hosting**: Optimized for large file downloads
- **Version management**: Independent versioning of graphs and code
- **Bandwidth efficiency**: Download only needed graphs

---

## 📦 Distribution Options

### Option 1: GitHub Releases (Recommended)
**Best for**: Open source projects with moderate file sizes (<2GB per release)

#### Setup Process:
1. **Create Release Repository** (optional)
   ```bash
   # Create separate repository for graphs
   git init improved-local-assistant-graphs
   cd improved-local-assistant-graphs
   
   # Create README for graphs repository
   echo "# Improved Local Assistant - Prebuilt Knowledge Graphs" > README.md
   echo "This repository contains prebuilt knowledge graphs for the Improved Local Assistant." >> README.md
   
   git add README.md
   git commit -m "Initial commit"
   git remote add origin https://github.com/hugokos/improved-local-assistant-graphs.git
   git push -u origin main
   ```

2. **Prepare Graph Packages**
   ```bash
   # Navigate to your graphs directory
   cd improved-local-assistant/data/graphs
   
   # Create compressed archives with metadata
   tar -czf survivalist-knowledge-v1.0.tar.gz survivalist/
   tar -czf medical-knowledge-v1.0.tar.gz medical/
   tar -czf technical-docs-v1.0.tar.gz technical/
   
   # Generate checksums
   sha256sum *.tar.gz > checksums.txt
   ```

3. **Create GitHub Release**
   - Go to graphs repository → Releases → Create new release
   - Tag: `graphs-v1.0.0`
   - Title: `Prebuilt Knowledge Graphs v1.0.0`
   - Upload compressed graph files and checksums.txt
   - Include detailed release notes

#### Example Release Notes Template:
```markdown
# Prebuilt Knowledge Graphs v1.0.0

## 📊 Available Graphs

### Survivalist Knowledge Base
- **File**: `survivalist-knowledge-v1.0.tar.gz`
- **Size**: 45MB compressed, 180MB extracted
- **Entities**: 2,847 unique entities
- **Relationships**: 8,234 relationships
- **Content**: Outdoor survival, bushcraft, emergency preparedness, wilderness skills
- **Sources**: Survival manuals, outdoor guides, emergency response documentation

### Medical Knowledge Base  
- **File**: `medical-knowledge-v1.0.tar.gz`
- **Size**: 78MB compressed, 320MB extracted
- **Entities**: 4,521 unique entities
- **Relationships**: 12,847 relationships
- **Content**: Health information, medical procedures, anatomy, pharmacology
- **Sources**: Medical textbooks, health guides, clinical documentation

### Technical Documentation
- **File**: `technical-docs-v1.0.tar.gz`
- **Size**: 92MB compressed, 380MB extracted
- **Entities**: 5,234 unique entities
- **Relationships**: 15,672 relationships
- **Content**: Programming languages, frameworks, APIs, development tools
- **Sources**: Official documentation, technical guides, programming references

## 🔧 Installation

### Automatic Download (Recommended)
```bash
# Download specific graphs
python scripts/download_graphs.py survivalist medical

# Download all graphs
python scripts/download_graphs.py all
```

### Manual Download
1. Download the desired `.tar.gz` files from this release
2. Extract to `improved-local-assistant/data/graphs/`
3. Verify checksums against `checksums.txt`

## ✅ Verification
After installation, verify graphs are working:
```bash
python cli/graphrag_repl.py
> Tell me about fire starting techniques
```

## 📝 Changelog
- Initial release of prebuilt knowledge graphs
- Optimized for GraphRAG retrieval performance
- Comprehensive entity linking and relationship mapping
```

### Option 2: Cloud Storage (AWS S3, Google Cloud, etc.)
**Best for**: Large files, enterprise distribution, or bandwidth optimization

#### AWS S3 Setup:
```bash
# Create S3 bucket
aws s3 mb s3://improved-local-assistant-graphs

# Upload graphs with public read access
aws s3 cp survivalist-knowledge-v1.0.tar.gz s3://improved-local-assistant-graphs/v1.0/ --acl public-read
aws s3 cp medical-knowledge-v1.0.tar.gz s3://improved-local-assistant-graphs/v1.0/ --acl public-read
aws s3 cp technical-docs-v1.0.tar.gz s3://improved-local-assistant-graphs/v1.0/ --acl public-read
aws s3 cp checksums.txt s3://improved-local-assistant-graphs/v1.0/ --acl public-read

# Create index file
echo '{"version": "1.0.0", "graphs": ["survivalist", "medical", "technical"]}' > index.json
aws s3 cp index.json s3://improved-local-assistant-graphs/v1.0/ --acl public-read
```

#### Update Download Script Configuration:
```python
# In scripts/download_graphs.py, update GRAPHS_CONFIG:
GRAPHS_CONFIG = {
    "source": "s3",
    "base_url": "https://improved-local-assistant-graphs.s3.amazonaws.com/v1.0",
    "release_tag": "v1.0",
    # ... rest of config
}
```

### Option 3: CDN Distribution
**Best for**: High-traffic projects, global distribution

#### Setup with jsDelivr (for GitHub releases):
```python
# Update download script for CDN
GRAPHS_CONFIG = {
    "source": "cdn",
    "base_url": "https://cdn.jsdelivr.net/gh/hugokos/improved-local-assistant-graphs@graphs-v1.0.0",
    # ... rest of config
}
```

#### Setup with CloudFlare R2:
```bash
# Upload to R2 with public access
wrangler r2 object put improved-local-assistant-graphs/v1.0/survivalist-knowledge-v1.0.tar.gz --file survivalist-knowledge-v1.0.tar.gz
```

### Option 4: Self-Hosted Distribution
**Best for**: Complete control, custom domains

#### Nginx Configuration:
```nginx
server {
    listen 80;
    server_name graphs.your-domain.com;
    
    location /v1.0/ {
        root /var/www/graphs;
        autoindex on;
        add_header Access-Control-Allow-Origin *;
    }
}
```

---

## 🛠️ Implementation Steps

### 1. Update Main Repository

#### Add to .gitignore:
```gitignore
# Knowledge graphs (downloaded separately)
data/graphs/*/
!data/graphs/.gitkeep
!data/graphs/README.md
```

#### Create data/graphs/README.md:
```markdown
# Knowledge Graphs Directory

This directory contains prebuilt knowledge graphs for the Improved Local Assistant.

## Download Graphs

Use the download script to get prebuilt graphs:

```bash
# Interactive selection
python scripts/download_graphs.py

# Download specific graphs
python scripts/download_graphs.py survivalist medical

# Download all graphs
python scripts/download_graphs.py all
```

## Available Graphs

- **survivalist**: Outdoor survival and bushcraft knowledge
- **medical**: Health and medical information
- **technical**: Programming and technical documentation

## Custom Graphs

To create your own knowledge graphs:

```bash
# Use the kg_builder tool
cd kg_builder
python src/graph_builder.py --input your_documents/ --output ../improved-local-assistant/data/graphs/custom/
```

For more information, see the [Graph Builder Documentation](../../kg_builder/README.md).
```

### 2. Update Installation Documentation

#### Add to CONTRIBUTING.md:
```markdown
### Working with Knowledge Graphs

#### Download Development Graphs
```bash
# Download test graphs for development
python scripts/download_graphs.py survivalist
```

#### Creating New Graphs
```bash
# Build custom graphs for testing
cd kg_builder
python src/graph_builder.py --input test_data/ --output ../improved-local-assistant/data/graphs/test/
```

#### Testing Graph Integration
```bash
# Test GraphRAG with specific graph
python cli/graphrag_repl.py --graph survivalist
```
```

### 3. Create Release Automation

#### GitHub Actions Workflow (.github/workflows/release-graphs.yml):
```yaml
name: Release Knowledge Graphs

on:
  workflow_dispatch:
    inputs:
      version:
        description: 'Graph version (e.g., v1.0.0)'
        required: true
        default: 'v1.0.0'

jobs:
  release-graphs:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Build knowledge graphs
      run: |
        cd kg_builder
        python src/graph_builder.py --input data/survivalist/ --output ../data/graphs/survivalist/
        python src/graph_builder.py --input data/medical/ --output ../data/graphs/medical/
        python src/graph_builder.py --input data/technical/ --output ../data/graphs/technical/
    
    - name: Package graphs
      run: |
        cd data/graphs
        tar -czf survivalist-knowledge-${{ github.event.inputs.version }}.tar.gz survivalist/
        tar -czf medical-knowledge-${{ github.event.inputs.version }}.tar.gz medical/
        tar -czf technical-docs-${{ github.event.inputs.version }}.tar.gz technical/
        sha256sum *.tar.gz > checksums.txt
    
    - name: Create Release
      uses: softprops/action-gh-release@v1
      with:
        tag_name: graphs-${{ github.event.inputs.version }}
        name: Prebuilt Knowledge Graphs ${{ github.event.inputs.version }}
        files: |
          data/graphs/*.tar.gz
          data/graphs/checksums.txt
        body: |
          # Prebuilt Knowledge Graphs ${{ github.event.inputs.version }}
          
          Automatically generated knowledge graphs for the Improved Local Assistant.
          
          ## Installation
          ```bash
          python scripts/download_graphs.py all
          ```
          
          ## Verification
          ```bash
          cd data/graphs
          sha256sum -c checksums.txt
          ```
```

---

## 📊 Graph Packaging Best Practices

### 1. Compression and Optimization
```bash
# Optimize graph files before packaging
python scripts/optimize_graphs.py --input data/graphs/survivalist/ --output data/graphs/survivalist-optimized/

# Create compressed archives with maximum compression
tar -czf survivalist-knowledge-v1.0.tar.gz --best survivalist-optimized/

# Alternative: Use 7zip for better compression
7z a -t7z -mx=9 survivalist-knowledge-v1.0.7z survivalist-optimized/
```

### 2. Metadata and Documentation
```bash
# Create metadata file for each graph
cat > survivalist/metadata.json << EOF
{
  "name": "survivalist",
  "version": "1.0.0",
  "description": "Survivalist and outdoor knowledge base",
  "created_at": "2025-01-04T12:00:00Z",
  "entities": 2847,
  "relationships": 8234,
  "size_mb": 180,
  "sources": [
    "SAS Survival Handbook",
    "Bushcraft 101",
    "Emergency Response Guides"
  ],
  "tags": ["survival", "outdoor", "emergency", "bushcraft"],
  "compatibility": {
    "min_version": "2.0.0",
    "max_version": "2.x.x"
  }
}
EOF
```

### 3. Quality Assurance
```bash
# Validate graphs before packaging
python scripts/validate_graphs.py --input data/graphs/survivalist/
python scripts/test_graph_queries.py --graph survivalist --queries test_queries.txt

# Performance benchmarks
python scripts/benchmark_graph_performance.py --graph survivalist
```

---

## 🔧 Advanced Distribution Strategies

### Multi-Platform Packages
```bash
# Create platform-specific packages
tar -czf survivalist-knowledge-v1.0-linux.tar.gz survivalist/
zip -r survivalist-knowledge-v1.0-windows.zip survivalist/
tar -czf survivalist-knowledge-v1.0-macos.tar.gz survivalist/
```

### Incremental Updates
```bash
# Create delta packages for updates
python scripts/create_graph_delta.py --old v1.0.0 --new v1.1.0 --output survivalist-delta-v1.0.0-to-v1.1.0.tar.gz
```

### Mirror Distribution
```bash
# Upload to multiple mirrors
python scripts/upload_to_mirrors.py --graphs *.tar.gz --mirrors github,s3,cloudflare
```

---

## 📈 Usage Analytics and Monitoring

### Download Tracking
```python
# Add to download script
import requests

def track_download(graph_name, version):
    """Track graph downloads for analytics."""
    try:
        requests.post("https://analytics.your-domain.com/track", json={
            "event": "graph_download",
            "graph": graph_name,
            "version": version,
            "timestamp": datetime.utcnow().isoformat()
        }, timeout=5)
    except:
        pass  # Don't fail download for analytics
```

### Health Monitoring
```bash
# Monitor download endpoints
python scripts/monitor_graph_availability.py --check-all --notify-on-failure
```

---

## 🎯 Summary

This comprehensive guide provides multiple options for distributing prebuilt knowledge graphs:

1. **GitHub Releases**: Best for most open source projects
2. **Cloud Storage**: Best for large files and enterprise use
3. **CDN Distribution**: Best for high-traffic global distribution
4. **Self-Hosted**: Best for complete control

The included Python download script provides a user-friendly interface for downloading and managing graphs, while the automation workflows ensure consistent packaging and distribution.

Choose the distribution method that best fits your project's needs, audience, and infrastructure requirements.

<function_calls>
<invoke name="strReplace">
<parameter name="path">improved-local-assistant/README.md