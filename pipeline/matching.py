"""Matching logic for puzzle pieces."""

from itertools import product

import numpy as np

from pipeline.models import EdgeType, Match, Piece


class PipelineParamsMatching:
    """Parameters for matching pipeline."""
    
    def __init__(self, max_diff_length_ratio: float = 0.02):
        self.max_diff_length = max_diff_length_ratio


def _sliding_window_distance(emb1, emb2) -> tuple[float | None, int, int]:
    """
    Compute best sliding-window alignment distance between two embeddings.
    Embeddings can have different lengths.
    
    Returns:
        best_score: Best matching score (lower is better)
        shift_1: How much piece1 should shift relative to piece2
        shift_2: How much piece2 should shift relative to piece1
    """
    # Reverse second edge (puzzle edges meet "face-to-face")
    emb2 = emb2[::-1]
    
    # Identify short vs long BUT also remember who is who
    if len(emb1) < len(emb2):
        S = np.array(emb1)
        L = np.array(emb2)
        short_is_1 = True  # S corresponds to emb1
    else:
        S = np.array(emb2)
        L = np.array(emb1)
        short_is_1 = False  # S corresponds to emb2
    
    lenS = len(S)
    lenL = len(L)
    
    best_score = float("inf")
    best_offset = None
    
    # Slide S along L
    for offset in range(lenL - lenS + 1):
        window = L[offset : offset + lenS]
        
        # Point-per-point distance
        diff = np.abs(S - window)
        
        # Score: area between curves (trapezoidal integration)
        score = float(np.trapz(np.abs(diff)))
        
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
    couples_edges = product(piece1.edge_list, piece2.edge_list)
    
    best_score = float("inf")
    best_shift_1 = -1
    best_shift_2 = -1
    
    for edge1, edge2 in couples_edges:
        # Skip flat edges
        if any([edge1.edge_type == EdgeType.flat, edge2.edge_type == EdgeType.flat]):
            continue
        
        # Skip edges of same type (man-man or woman-woman)
        if edge1.edge_type == edge2.edge_type:
            continue
        
        # Check length compatibility
        edge1_length = np.linalg.norm(
            edge1.edge_contour[0] - edge1.edge_contour[-1]
        ).__float__()
        edge2_length = np.linalg.norm(
            edge2.edge_contour[0] - edge2.edge_contour[-1]
        ).__float__()
        diff_length = abs(edge1_length - edge2_length)
        
        if diff_length / max(edge1_length, edge2_length) > params.max_diff_length:
            continue
        
        if verbose:
            print(
                f"Edge {edge1.edge_id} of Piece {piece1.piece_id} is compatible with "
                f"Edge {edge2.edge_id} of Piece {piece2.piece_id}"
            )
            print(f"{edge1.edge_type} vs {edge2.edge_type}")
            print(f"Length Edge 1: {edge1_length}")
            print(f"Length Edge 2: {edge2_length}")
            print(
                f"Length Difference: {diff_length} which is "
                f"{diff_length/edge1_length} and {diff_length/edge2_length}"
            )
        
        # Calculate distance of the contour of each edge from the line that connects the two extremes
        len_contour_edge_1 = edge1.edge_contour.shape[0]
        line_edge_1 = np.linspace(
            edge1.edge_contour[0], edge1.edge_contour[-1], len_contour_edge_1
        )
        embedding_edge1 = list(map(np.linalg.norm, edge1.edge_contour - line_edge_1))
        
        len_contour_edge_2 = edge2.edge_contour.shape[0]
        line_edge_2 = np.linspace(
            edge2.edge_contour[0], edge2.edge_contour[-1], len_contour_edge_2
        )
        embedding_edge2 = list(map(np.linalg.norm, edge2.edge_contour - line_edge_2))
        
        score, shift_1, shift_2 = _sliding_window_distance(
            embedding_edge1, embedding_edge2
        )
        
        if score is None:
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
