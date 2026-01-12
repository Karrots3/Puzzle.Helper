"""Matching logic for puzzle pieces."""

from itertools import product

import numpy as np

from pipeline.models import EdgeType, Match, Piece


class PipelineParamsMatching:
    """Parameters for matching pipeline."""
    
    def __init__(
        self,
        max_diff_length_ratio: float = 0.02,
        min_score_threshold: float | None = None,
        normalize_scores: bool = True,
    ):
        self.max_diff_length = max_diff_length_ratio
        self.min_score_threshold = min_score_threshold
        self.normalize_scores = normalize_scores


def _sliding_window_distance(
    emb1: np.ndarray,
    emb2: np.ndarray,
    normalize: bool = True,
) -> tuple[float | None, int, int]:
    """
    Compute best sliding-window alignment distance between two embeddings.
    Embeddings can have different lengths.
    
    Args:
        emb1: First embedding array
        emb2: Second embedding array
        normalize: Whether to normalize scores by embedding length
    
    Returns:
        best_score: Best matching score (lower is better)
        shift_1: How much piece1 should shift relative to piece2
        shift_2: How much piece2 should shift relative to piece1
    """
    if len(emb1) == 0 or len(emb2) == 0:
        return None, -1, -1
    
    # Reverse second edge (puzzle edges meet "face-to-face")
    emb2_reversed = emb2[::-1]
    
    # Identify short vs long BUT also remember who is who
    if len(emb1) < len(emb2_reversed):
        S = np.asarray(emb1, dtype=np.float64)
        L = np.asarray(emb2_reversed, dtype=np.float64)
        short_is_1 = True  # S corresponds to emb1
    else:
        S = np.asarray(emb2_reversed, dtype=np.float64)
        L = np.asarray(emb1, dtype=np.float64)
        short_is_1 = False  # S corresponds to emb2
    
    lenS = len(S)
    lenL = len(L)
    
    if lenS == 0 or lenL < lenS:
        return None, -1, -1
    
    best_score = float("inf")
    best_offset = None
    
    # Slide S along L - vectorized computation
    for offset in range(lenL - lenS + 1):
        window = L[offset : offset + lenS]
        
        # Point-per-point distance (vectorized)
        diff = np.abs(S - window)
        
        # Score: area between curves (trapezoidal integration)
        score = float(np.trapz(diff))
        
        # Normalize by length to make scores comparable across different edge sizes
        if normalize and lenS > 0:
            score = score / lenS
        
        if score < best_score:
            best_score = score
            best_offset = offset
    
    # Compute shift relative to emb1 vs emb2
    if best_offset is None:
        return None, -1, -1
    
    if short_is_1:
        shift_1 = best_offset
        shift_2 = 0
    else:
        shift_1 = 0
        shift_2 = best_offset
    
    return best_score, shift_1, shift_2


def find_match(
    piece1: Piece,
    piece2: Piece,
    params: PipelineParamsMatching,
    verbose: bool = False,
) -> Match:
    """
    Find the best match between two puzzle pieces.
    
    Args:
        piece1: First puzzle piece
        piece2: Second puzzle piece
        params: Matching parameters
        verbose: Whether to print debug information
        
    Returns:
        Match object with best match information
    """
    # Early exit if pieces are the same
    if piece1.piece_id == piece2.piece_id:
        return Match(
            piece_id_1=piece1.piece_id,
            piece_id_2=piece2.piece_id,
            score=None,
            shift_1=-1,
            shift_2=-1,
        )
    
    couples_edges = product(piece1.edge_list, piece2.edge_list)
    
    best_score = float("inf")
    best_shift_1 = -1
    best_shift_2 = -1
    
    for edge1, edge2 in couples_edges:
        # Skip flat edges
        if edge1.edge_type == EdgeType.flat or edge2.edge_type == EdgeType.flat:
            continue
        
        # Skip edges of same type (man-man or woman-woman)
        if edge1.edge_type == edge2.edge_type:
            continue
        
        # Check length compatibility using cached property
        edge1_length = edge1.straight_length
        edge2_length = edge2.straight_length
        max_length = max(edge1_length, edge2_length)
        
        if max_length < 1e-6:  # Avoid division by zero
            continue
        
        diff_length = abs(edge1_length - edge2_length)
        length_ratio = diff_length / max_length
        
        if length_ratio > params.max_diff_length:
            continue
        
        if verbose:
            print(
                f"Edge {edge1.edge_id} of Piece {piece1.piece_id} is compatible with "
                f"Edge {edge2.edge_id} of Piece {piece2.piece_id}"
            )
            print(f"{edge1.edge_type} vs {edge2.edge_type}")
            print(f"Length Edge 1: {edge1_length:.2f}")
            print(f"Length Edge 2: {edge2_length:.2f}")
            print(f"Length Difference Ratio: {length_ratio:.4f}")
        
        # Use cached embeddings
        embedding_edge1 = edge1.get_embedding()
        embedding_edge2 = edge2.get_embedding()
        
        score, shift_1, shift_2 = _sliding_window_distance(
            embedding_edge1,
            embedding_edge2,
            normalize=params.normalize_scores,
        )
        
        if score is None:
            continue
        
        # Apply threshold if specified
        if params.min_score_threshold is not None and score > params.min_score_threshold:
            continue
        
        if score < best_score:
            best_score = score
            best_shift_1 = shift_1
            best_shift_2 = shift_2
    
    # Return match (score will be inf if no match found)
    final_score = best_score if best_score != float("inf") else None
    
    return Match(
        piece_id_1=piece1.piece_id,
        piece_id_2=piece2.piece_id,
        score=final_score,
        shift_1=best_shift_1,
        shift_2=best_shift_2,
    )
