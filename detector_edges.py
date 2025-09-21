import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import pickle

class Piece:
    def __init__(self, piece_id, img_thresh, contour, area, perimeter, bounding_rect, min_area_rect, box_points, hull, solidity, aspect_ratio, extent):
        self.piece_id = piece_id
        self.img_thresh = img_thresh
        self.contour = contour
        self.area = area
        self.perimeter = perimeter
        self.bounding_rect = bounding_rect
        self.min_area_rect = min_area_rect
        self.box_points = box_points
        self.hull = hull
        self.solidity = solidity
        self.aspect_ratio = aspect_ratio
        self.extent = extent
      
class Edge:
    def __init__(self, edge_id, contour, length, approx, edge_type):
        self.edge_id = edge_id
        self.contour = contour
        self.length = length
        self.approx = approx
        self.edge_type = edge_type

path_pikle = Path("./data/results/list_pieces.pkl")
with open(path_pikle, 'rb') as f:
  list_pieces = pickle.load(f)

print(list_pieces)
print(f"Loaded {len(list_pieces)} pieces from {path_pikle}")

i=1

piece=list(list_pieces.values())[i][0]


print(piece)