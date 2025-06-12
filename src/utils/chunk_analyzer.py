"""
Chunk Analysis Utility for Federated Learning

This module provides utilities to analyze model chunks, their importance,
and aggregation statistics in federated learning scenarios.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Any
from src.utils.log import Log


class ChunkAnalyzer:
    """Utility class for analyzing model chunks and their importance."""

    def __init__(self, log: Log):
        self.log = log

    def analyze_chunk_distribution(self, clients: List, config) -> Dict[str, Any]:
        """
        Analyze the distribution of important chunks across all clients.
        
        Args:
            clients: List of client objects
            config: Configuration object
            
        Returns:
            Dictionary containing analysis results
        """
        if not config.CHUNKING:
            return {"error": "Chunking is not enabled"}

        analysis = {
            "total_chunks": config.CHUNKING_PARTS,
            "sensitivity_percentage": config.SENSITIVITY_PERCENTAGE,
            "client_analyses": {},
            "chunk_popularity": {},
            "aggregation_coverage": {},
        }

        # Analyze each client's chunk selection
        chunk_selections = {}
        for client in clients:
            if hasattr(client, 'gradients') and client.gradients:
                _, important_indices = client.get_model_chunks()
                chunk_selections[client.id] = important_indices
                
                analysis["client_analyses"][client.id] = {
                    "selected_chunks": important_indices,
                    "num_selected": len(important_indices),
                    "selection_percentage": len(important_indices) / config.CHUNKING_PARTS * 100
                }

        # Analyze chunk popularity (how many clients selected each chunk)
        chunk_counts = {}
        for chunk_idx in range(config.CHUNKING_PARTS):
            count = sum(1 for indices in chunk_selections.values() if chunk_idx in indices)
            chunk_counts[chunk_idx] = count

        analysis["chunk_popularity"] = chunk_counts
        
        # Calculate aggregation coverage
        total_possible_selections = len(clients) * len(chunk_selections.get(list(chunk_selections.keys())[0], []))
        actual_selections = sum(chunk_counts.values())
        
        analysis["aggregation_coverage"] = {
            "total_possible_chunk_selections": total_possible_selections,
            "actual_chunk_selections": actual_selections,
            "coverage_percentage": (actual_selections / total_possible_selections * 100) if total_possible_selections > 0 else 0,
            "chunks_with_contributions": len([c for c in chunk_counts.values() if c > 0]),
            "chunks_without_contributions": len([c for c in chunk_counts.values() if c == 0])
        }

        return analysis

    def log_chunk_analysis(self, analysis: Dict[str, Any]):
        """Log the chunk analysis results."""
        if "error" in analysis:
            self.log.warn(f"Chunk analysis error: {analysis['error']}")
            return

        self.log.info("=== CHUNK ANALYSIS REPORT ===")
        self.log.info(f"Total chunks: {analysis['total_chunks']}")
        self.log.info(f"Sensitivity percentage: {analysis['sensitivity_percentage']}%")
        
        self.log.info("\n--- Client Chunk Selections ---")
        for client_id, client_analysis in analysis["client_analyses"].items():
            self.log.info(
                f"Client {client_id}: {client_analysis['num_selected']} chunks "
                f"({client_analysis['selection_percentage']:.1f}%) - {client_analysis['selected_chunks']}"
            )

        self.log.info("\n--- Chunk Popularity ---")
        popular_chunks = sorted(analysis["chunk_popularity"].items(), key=lambda x: x[1], reverse=True)
        for chunk_idx, count in popular_chunks[:10]:  # Show top 10 most popular chunks
            self.log.info(f"Chunk {chunk_idx}: selected by {count} clients")

        self.log.info("\n--- Aggregation Coverage ---")
        coverage = analysis["aggregation_coverage"]
        self.log.info(f"Coverage: {coverage['coverage_percentage']:.1f}%")
        self.log.info(f"Chunks with contributions: {coverage['chunks_with_contributions']}")
        self.log.info(f"Chunks with no contributions: {coverage['chunks_without_contributions']}")

    def calculate_chunk_overlap(self, client1_chunks: List[int], client2_chunks: List[int]) -> Dict[str, float]:
        """
        Calculate overlap metrics between two clients' chunk selections.
        
        Args:
            client1_chunks: List of chunk indices selected by client 1
            client2_chunks: List of chunk indices selected by client 2
            
        Returns:
            Dictionary with overlap metrics
        """
        set1 = set(client1_chunks)
        set2 = set(client2_chunks)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        jaccard_similarity = intersection / union if union > 0 else 0
        overlap_coefficient = intersection / min(len(set1), len(set2)) if min(len(set1), len(set2)) > 0 else 0
        
        return {
            "intersection_size": intersection,
            "union_size": union,
            "jaccard_similarity": jaccard_similarity,
            "overlap_coefficient": overlap_coefficient
        }

    def get_chunk_importance_distribution(self, client, config) -> Dict[str, Any]:
        """
        Get the importance distribution of chunks for a specific client.
        
        Args:
            client: Client object
            config: Configuration object
            
        Returns:
            Dictionary with importance distribution data
        """
        if not hasattr(client, 'gradients') or not client.gradients:
            return {"error": "No gradients available for analysis"}

        params_vector = torch.nn.utils.parameters_to_vector(client.model.parameters()).detach()
        total_params = len(params_vector)
        chunk_size = total_params // config.CHUNKING_PARTS
        
        grad_vector = torch.tensor([client.gradients[i] for i in range(len(client.gradients))])
        
        chunk_importance = []
        for i in range(config.CHUNKING_PARTS):
            start_idx = i * chunk_size
            if i == config.CHUNKING_PARTS - 1:  # Last chunk
                end_idx = total_params
            else:
                end_idx = start_idx + chunk_size
            
            if end_idx <= len(grad_vector):
                chunk_grads = grad_vector[start_idx:end_idx]
                importance = torch.norm(chunk_grads).item()
            else:
                available_grads = grad_vector[start_idx:min(len(grad_vector), end_idx)]
                importance = torch.norm(available_grads).item() if len(available_grads) > 0 else 0.0
            
            chunk_importance.append(importance)

        importance_array = np.array(chunk_importance)
        
        return {
            "chunk_importance": chunk_importance,
            "mean_importance": float(np.mean(importance_array)),
            "std_importance": float(np.std(importance_array)),
            "min_importance": float(np.min(importance_array)),
            "max_importance": float(np.max(importance_array)),
            "importance_range": float(np.max(importance_array) - np.min(importance_array)),
            "top_chunks": np.argsort(importance_array)[-5:].tolist()[::-1],  # Top 5 chunks
            "bottom_chunks": np.argsort(importance_array)[:5].tolist()  # Bottom 5 chunks
        } 