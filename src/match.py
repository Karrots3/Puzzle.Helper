from src.common import Edge, Piece
import numpy as np

def match_edges(edge1: Edge, edge2: Edge):
    if edge1.sign == 0 or edge2.sign == 0:
        return False
    if edge1.sign == edge2.sign:
        return False

    distance = np.linalg.norm(edge1.sample_indices - edge2.sample_indices)
    if distance < 100:
        return True
    return False



def match_edge_points(edge1: Edge, edge2: Edge):
    pass

def match_pieces(piece1: Piece, piece2: Piece):
    if piece1 == piece2:
        return False

    for edge1 in piece1.edges:
        for edge2 in piece2.edges:
            if match_edge_points(edge1, edge2):
                return True

    return False
