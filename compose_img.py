from pathlib import Path
from pickle import load

import cv2 as cv
import numpy as np

from src.classes import Piece
from src.newutils import plt_np_3ch

# load all pkl in data/pieces
pkl_files = [f for f in Path("data/pieces").glob("*.pkl")]

list_img = []
for pkl_file in pkl_files:
    with open(pkl_file, "rb") as f:
        piece = load(f)
    

    height, width = piece.rgb_img.shape[:2]
    _img = np.zeros((height, width, 3), dtype=np.uint8)
    for edge in piece.edge_list:
        _img = cv.drawContours(
            _img,
            [edge.edge_contour],
            -1,
            edge.edge_color,
            thickness=cv.FILLED,
        )
    list_img.append(_img)

plt_np_3ch(list_img, max_col=10, size_height=2, out_path="compose.pdf")