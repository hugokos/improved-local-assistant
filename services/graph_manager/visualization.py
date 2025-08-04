"""
Visualization module for KnowledgeGraphManager.

Handles HTML and PyVis visualization of knowledge graphs.
"""

import os
import tempfile
from typing import Optional

from pyvis.network import Network


class KnowledgeGraphVisualization:
    """Handles visualization operations for knowledge graphs."""

    def visualize_graph(self, graph_id: str = "all", title: str = "Knowledge Graph") -> str:
        """
        Create an HTML visualization of the knowledge graph using PyVis.

        Args:
            graph_id: ID of the graph to visualize, or "all" for all graphs
            title: Title for the visualization

        Returns:
            str: HTML content of the visualization
        """
        try:
            if not self.enable_visualization:
                return "<p>Graph visualization is disabled in configuration.</p>"

            # Create PyVis network
            net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")

            # Configure physics
            net.set_options("""
            var options = {
              "physics": {
                "enabled": true,
                "stabilization": {"iterations": 100}
              }
            }
            """)

            nodes_added = set()
            edges_added = set()

            # Add nodes and edges from specified graph(s)
            if graph_id == "all":
                # Add from all graphs
                graphs_to_process = list(self.kg_indices.items())
                if self.dynamic_kg:
                    graphs_to_process.append(("dynamic", self.dynamic_kg))
            elif graph_id == "dynamic" and self.dynamic_kg:
                graphs_to_process = [("dynamic", self.dynamic_kg)]
            elif graph_id in self.kg_indices:
                graphs_to_process = [(graph_id, self.kg_indices[graph_id])]
            else:
                return f"<p>Graph '{graph_id}' not found.</p>"

            for gid, kg_index in graphs_to_process:
                try:
                    # Get NetworkX graph
                    G = self._safe_get_networkx_graph(kg_index, gid)
                    if G is None:
                        continue

                    # Add nodes
                    for node, data in G.nodes(data=True):
                        if node not in nodes_added:
                            # Determine node color based on graph source
                            if gid == "dynamic":
                                color = "#ff6b6b"  # Red for dynamic
                            elif "survivalist" in gid:
                                color = "#4ecdc4"  # Teal for survivalist
                            else:
                                color = "#45b7d1"  # Blue for others

                            net.add_node(
                                node,
                                label=str(node)[:50],  # Truncate long labels
                                title=f"Graph: {gid}\nNode: {node}\nData: {str(data)[:200]}",
                                color=color,
                            )
                            nodes_added.add(node)

                    # Add edges
                    for source, target, data in G.edges(data=True):
                        edge_id = f"{source}-{target}"
                        if edge_id not in edges_added:
                            relation = data.get("relation", "connected_to")
                            net.add_edge(
                                source,
                                target,
                                label=relation[:20],  # Truncate long relations
                                title=f"Relation: {relation}\nData: {str(data)[:200]}",
                            )
                            edges_added.add(edge_id)

                except Exception as e:
                    self.logger.error(f"Error processing graph {gid} for visualization: {str(e)}")
                    continue

            if not nodes_added:
                return "<p>No nodes found to visualize.</p>"

            # Generate HTML
            html_file = tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False)
            net.save_graph(html_file.name)
            html_file.close()

            # Read the generated HTML
            with open(html_file.name, "r", encoding="utf-8") as f:
                html_content = f.read()

            # Add title to HTML
            html_content = html_content.replace("<body>", f"<body><h2>{title}</h2>")

            # Clean up temporary file
            try:
                os.remove(html_file.name)
            except (OSError, FileNotFoundError):
                pass

            return html_content

        except Exception as e:
            self.logger.error(f"Error visualizing graph: {str(e)}")
            return f"<p>Error visualizing graph: {str(e)}</p>"