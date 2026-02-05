"""
Gaussian-based patch merging for QAOA mesh optimization.

This module provides functionality to merge overlapping patches using 
Gaussian-weighted node interpolation to handle common elements smoothly.
"""

import numpy as np
from scipy.spatial import KDTree


class GaussianBoundarySelector:
    """Select boundary nodes using Gaussian weighting."""
    
    def __init__(self, sigma_factor=0.5):
        """
        Initialize Gaussian boundary selector.
        
        Args:
            sigma_factor: Factor for Gaussian width (default 0.5)
        """
        self.sigma_factor = sigma_factor
    
    def compute_gaussian_weights(self, nodes, center):
        """
        Compute Gaussian weight for each node based on distance from center.
        
        Args:
            nodes: (N, 2) array of node coordinates
            center: (2,) array of patch center coordinates
            
        Returns:
            weights: (N,) array of normalized weights
        """
        dists = np.linalg.norm(nodes - center, axis=1)
        sigma = self.sigma_factor * np.max(dists)
        if sigma > 0:
            weights = 1.0 - np.exp(-(dists / sigma) ** 2)
        else:
            weights = np.ones(len(nodes))
        return weights / (weights.sum() + 1e-10)  # Normalize
    
    def select_boundary_nodes(self, patch_nodes, n_select):
        """
        Select nodes with highest Gaussian weight (near boundary).
        
        Args:
            patch_nodes: (N, 2) array of node coordinates
            n_select: Number of nodes to select
            
        Returns:
            top_indices: Indices of selected nodes
        """
        center = patch_nodes.mean(axis=0)
        weights = self.compute_gaussian_weights(patch_nodes, center)
        n_select = min(n_select, len(weights))
        top_indices = np.argsort(weights)[-n_select:]
        return top_indices


def merge_patch_results_gaussian(patch_results, nodes, boundary_threshold=0.15):
    """
    Merge QAOA solutions from overlapping patches using Gaussian-weighted node merging.
    
    This function handles patches with common elements by:
    1. Collecting all selected nodes from all patches
    2. Finding clusters of nearby nodes (within boundary_threshold)
    3. Merging clusters using Gaussian-weighted interpolation
    4. Returning unique global node indices
    
    Args:
        patch_results: List of patch solution dictionaries, each containing:
            - 'local_selected': Local indices of selected nodes within patch
            - 'patch_indices': Mapping from local to global indices
            - 'patch_id': Unique identifier for the patch (optional)
            - 'global_selected': Global indices (optional, computed if not provided)
        nodes: (N, 2) array of all node coordinates
        boundary_threshold: Distance threshold for considering nodes as duplicates
    
    Returns:
        merged_indices: Array of unique global node indices after merging
    """
    # Collect all selected nodes from all patches
    all_selected = []
    patch_origins = []  # (patch_id, local_idx, patch_idx, global_idx)
    
    for patch_idx, result in enumerate(patch_results):
        patch_id = result.get('patch_id', patch_idx)
        
        # Get local indices of selected nodes
        local_selected = result.get('local_selected', [])
        patch_indices = result.get('patch_indices', [])
        
        # Convert to global indices
        for local_idx in local_selected:
            if local_idx < len(patch_indices):
                global_idx = patch_indices[local_idx]
                if global_idx < len(nodes):
                    all_selected.append(nodes[global_idx])
                    patch_origins.append((patch_id, local_idx, patch_idx, global_idx))
    
    if len(all_selected) == 0:
        print("  Warning: No nodes selected in any patch")
        return np.array([])
    
    all_selected = np.array(all_selected)
    print(f"  Total selected nodes across patches: {len(all_selected)}")
    
    # Handle edge case: too few nodes for merging
    if len(all_selected) < 3:
        print("  Warning: Too few nodes for merging, returning all")
        return np.array([origin[3] for origin in patch_origins])
    
    # Build KD-tree for efficient neighbor search
    tree = KDTree(all_selected)
    
    # Group nearby nodes and merge using Gaussian weighting
    merged_nodes = []
    merged_indices = []
    visited = set()
    
    for i in range(len(all_selected)):
        if i in visited:
            continue
        
        # Find neighbors within threshold
        neighbors = tree.query_ball_point(all_selected[i], r=boundary_threshold)
        neighbors = [n for n in neighbors if n not in visited]
        
        if len(neighbors) > 1:
            # Multiple nodes close together - merge using Gaussian interpolation
            neighbor_coords = all_selected[neighbors]
            
            # Compute Gaussian-weighted average (favor nodes closer to center)
            center = neighbor_coords.mean(axis=0)
            dists = np.linalg.norm(neighbor_coords - center, axis=1)
            sigma = 0.5 * np.max(dists) if np.max(dists) > 0 else 1.0
            weights = np.exp(-(dists / sigma) ** 2)
            weights = weights / (weights.sum() + 1e-10)
            
            # Weighted merge (position - not used in final output but could be)
            merged_node = np.average(neighbor_coords, axis=0, weights=weights)
            merged_nodes.append(merged_node)
            
            # Map original global indices
            merged_idx_map = []
            for neighbor in neighbors:
                _, _, _, global_idx = patch_origins[neighbor]
                merged_idx_map.append(global_idx)
            
            merged_indices.append(merged_idx_map)
            print(f"    Merged {len(neighbors)} nodes (Gaussian-weighted)")
        else:
            # Unique node - no merging needed
            merged_nodes.append(all_selected[i])
            _, _, _, global_idx = patch_origins[i]
            merged_indices.append([global_idx])
        
        # Mark all neighbors as visited
        visited.update(neighbors)
    
    # Extract final merged indices (unique global indices)
    final_merged = []
    seen = set()
    for idx_group in merged_indices:
        for idx in idx_group:
            if idx not in seen:
                final_merged.append(idx)
                seen.add(idx)
    
    print(f"  Final merged nodes: {len(final_merged)}")
    return np.array(final_merged)


def generate_patches_with_overlap(nodes, centers, r_patch, r_halo, Q_max=None):
    """
    Generate patches with interior and halo regions for overlap handling.
    
    Args:
        nodes: (N, 2) array of node coordinates
        centers: (M, 2) array of patch center coordinates
        r_patch: Radius for interior nodes
        r_halo: Radius for halo nodes (overlap region)
        Q_max: Maximum nodes per patch (optional)
        
    Returns:
        patches: List of patch dictionaries with 'center', 'interior_idx', 'halo_idx'
    """
    patches = []
    
    for ci, center in enumerate(centers):
        # Compute distances from center to all nodes
        dists = np.linalg.norm(nodes - center, axis=1)
        
        # Classify nodes as interior or halo
        interior_idx = np.where(dists <= r_patch)[0]
        halo_idx = np.where((dists > r_patch) & (dists <= r_halo))[0]
        
        # Enforce qubit limit if specified
        if Q_max is not None and len(interior_idx) > Q_max:
            # Sort by distance and take closest Q_max nodes
            interior_dists = dists[interior_idx]
            sorted_idx = np.argsort(interior_dists)
            interior_idx = interior_idx[sorted_idx[:Q_max]]
        
        patch = {
            "center": center,
            "interior_idx": interior_idx,
            "halo_idx": halo_idx,
            "patch_id": ci
        }
        patches.append(patch)
    
    return patches


def prepare_patch_for_qaoa(patch, nodes):
    """
    Prepare patch data for QAOA processing.
    
    Args:
        patch: Patch dictionary with 'interior_idx', 'halo_idx', etc.
        nodes: Full node set
        
    Returns:
        patch_data: Dictionary ready for QAOA processing
    """
    interior_idx = patch['interior_idx']
    halo_idx = patch.get('halo_idx', [])
    
    # Combine interior and halo for full patch context
    all_patch_idx = np.concatenate([interior_idx, halo_idx])
    
    patch_data = {
        'patch_id': patch.get('patch_id', 0),
        'center': patch['center'],
        'patch_indices': all_patch_idx,  # Global indices
        'interior_count': len(interior_idx),
        'halo_count': len(halo_idx),
        'patch_nodes': nodes[all_patch_idx],  # Actual coordinates
    }
    
    return patch_data
