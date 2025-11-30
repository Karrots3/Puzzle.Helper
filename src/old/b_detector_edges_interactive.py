# %% Imports
import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


path_pieces = Path("../results/list_pieces.pkl")
path_single_piece = Path("../results/single_piece.pkl")

with open(path_single_piece, "rb") as f:
    piece = pickle.load(f)[0]
assert type(piece) == Piece, "Expected a Piece object"

contour = piece.contour

# %%  Find corners via peak distance from center
(cx, cy), cr = cv2.minEnclosingCircle(contour)
centered_contour = contour - np.array([cx, cy])
print(centered_contour)

# ensure peaks are not at start or end of the distances array
distances = np.sum(centered_contour**2, axis=2)[:, 0]
distance_offset = np.argmin(distances)
shifted_distances = np.concatenate(
    [distances[distance_offset:], distances[:distance_offset]]
)

# find peak distances
from scipy.signal import find_peaks

for prominence in range(1400, 1600, 200):
    peak_indices = [
        (distance_idx + distance_offset) % len(distances)
        for distance_idx in find_peaks(shifted_distances, prominence=prominence)[0]
    ]
    peak_indices.sort()
    piece_center = np.array([cx, cy])
    piece_peak_indices = LoopingList(peak_indices)
    print(peak_indices)

    # plot all peaks over the image
    img_contour = cv2.cvtColor(piece.img_thresh, cv2.COLOR_GRAY2BGR)
    for peak_index in peak_indices:
        cv2.circle(
            img_contour,
            (int(contour[peak_index][0][0]), int(contour[peak_index][0][1])),
            20,
            (0, 0, 255),
            -1,
        )
    plt.imshow(img_contour)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


# %%
"""## Filter corners by rectangle geometry"""
from itertools import combinations
from math import sqrt


def compute_rectangle_error(indices):
    # get coordinates of corners
    corners = LoopingList(np.take(contour, sorted(list(indices)), axis=0)[:, 0, :])
    # compute the side lengths and diagonal lengths
    lengths = [
        sqrt(np.sum((corners[i0] - corners[i1]) ** 2))
        for i0, i1 in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
    ]

    def f_error(a, b):
        return abs(b - a) / (a + b)

    return sum(
        [f_error(lengths[i], lengths[j]) for i, j in [(0, 2), (1, 3), (4, 5), (0, 1)]]
    )


# form a good rectangle with peak indices
rectangles = []  # list of (score, [indices])
for indices in combinations(peak_indices, 4):
    error = compute_rectangle_error(indices)
    rectangles.append((error, indices))

error, indices = sorted(rectangles)[0]
rectangle_error = error
corner_indices = LoopingList(indices)
print("Rectangle error:", rectangle_error)
print(corner_indices)


# plot contour and bigger color in the corresponding corner indices
def plot_contour(contour, c="blue", corner_indices=None):
    plt.plot(contour[:, 0, 0], contour[:, 0, 1], c=c)
    if corner_indices is not None:
        plt.scatter(corner_indices[:, 0, 0], corner_indices[:, 0, 1], c="red", s=100)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


plot_contour(contour, corner_indices=np.take(contour, corner_indices, axis=0))


# %%
"""# Compute edges
Extract edges from the contour by splitting it into 4 quarters and computing the edge between the corners of each quarter.
"""
import numpy as np

idx = [
    (corner_indices[0], corner_indices[1]),
    (corner_indices[1], corner_indices[2]),
    (corner_indices[2], corner_indices[3]),
    (corner_indices[3], corner_indices[0]),
]

edge_contours = [
    contour[corner_indices[0] : corner_indices[1], :],
    contour[corner_indices[1] : corner_indices[2], :],
    contour[corner_indices[2] : corner_indices[3], :],
    contour[corner_indices[3] : corner_indices[0] + len(contour), :],
]

# Color edges with different colors
colors = ["red", "green", "blue", "yellow"]

for i, edge_contour in enumerate(edge_contours):
    plt.plot(edge_contour[:, 0, 0], edge_contour[:, 0, 1], c=colors[i])
    plt.gca().set_aspect("equal", adjustable="box")
plt.show()


# %%
import math

(x, y) = np.mean(piece.contour[:, 0, :], axis=0)


def transform_contour(contour):
    p0 = contour[0][0]
    p1 = contour[-1][0]
    dx, dy = p1 - p0
    center = np.mean(contour[:, 0, :], axis=0)
    degrees = math.degrees(math.atan2(dy, dx))
    scale = 1.0

    matrix = cv2.getRotationMatrix2D(center, degrees, scale)
    translate = (0, 0) - center

    return cv2.transform(contour, matrix) + translate


def classify_edge(edge_contour, piece_center, flat_thresh=5.0):
    """
    Classify an edge contour as flat, male, or female.

    Args:
        edge_contour (np.ndarray): Contour points of the edge (Nx1x2).
        piece_center (np.ndarray): (x, y) center of the piece, to determine inward vs outward.
        flat_thresh (float): Max allowed deviation (pixels) to consider an edge flat.

    Returns:
        str: "flat", "male", or "female"
    """
    pts = edge_contour[:, 0, :]  # Nx2
    p0, p1 = pts[0], pts[-1]

    # Vector of the edge baseline
    edge_vec = p1 - p0
    edge_len = np.linalg.norm(edge_vec)
    if edge_len < 1e-6:
        return "flat"

    # Unit normal to the edge
    normal = np.array([-edge_vec[1], edge_vec[0]]) / edge_len

    # Signed distance of each point to baseline
    distances = []
    for p in pts:
        v = p - p0
        signed_dist = np.dot(v, normal)
        distances.append(signed_dist)
    distances = np.array(distances)

    max_dev = np.max(np.abs(distances))

    # Flat check
    if max_dev < flat_thresh:
        return 0

    # Determine sign of deviation at the most extreme point
    idx_max = np.argmax(np.abs(distances))
    extreme_point = pts[idx_max]
    extreme_sign = np.sign(distances[idx_max])

    # Decide male/female based on whether extreme point is closer/farther than center
    vec_center = piece_center - p0
    center_side = np.sign(np.dot(vec_center, normal))

    if extreme_sign == center_side:
        return -1
    else:
        return 1


list_edges: list[Edge] = []
for edg in edge_contours:
    p0 = edg[0][0]
    p1 = edg[-1][0]
    # normalize the contour: first point at (0, 0), last point at (X, 0)
    dx, dy = p1 - p0
    straight_length = math.sqrt(dx**2 + dy**2)
    angle_degrees = math.degrees(math.atan2(dy, dx))

    normalized_piece_contour = transform_contour(edg)
    type_edge = classify_edge(edg, piece_center=np.array([x, y]), flat_thresh=20)

    list_edges.append(
        Edge(
            edg,
            contour,
            normalized_piece_contour,
            straight_length,
            angle_degrees,
            type_edge,
        )
    )

# %% Visualize edges, do not warp axes, keep same scale
if plot:
    colors_gender={0:"red", 1:"green", -1:"blue"}
    for edge in list_edges:
        plt.plot(edge.norm_contour[:, 0, 0], edge.norm_contour[:, 0, 1], c=colors_gender[edge.edge_type])
        plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


# %%
# normalized_edge_contour = sub_contour(normalized_piece_contour, idx0, idx1 + 1)
# normalized_piece_center = transform_point(piece.center, transform)

## compute the sign of the edge
# heights = normalized_edge_contour[:, 0, 1]
# if np.max(np.abs(heights)) > 10:
#    sign = 1 if np.max(heights) > -np.min(heights) else -1
# else:
#    sign = 0

## rotate male contours by 180° for easy match with female contours
# if sign == 1:
#    angle_degrees += 180
#    transform = get_contour_transform(piece.contour, idx1, 0, 0, angle_degrees)
#    normalized_piece_contour = transform_contour(piece.contour, transform)
#    normalized_piece_center = transform_point(piece.center, transform)

# edge = Item(
#    idx0=idx0,
#    idx1=idx1,
#    normalized_piece_contour=normalized_piece_contour,
#    normalized_piece_center=normalized_piece_center,
#    angle_degrees=angle_degrees,
#    sign=sign,
#    straight_length=straight_length,
# )
# edges.append(edge)
#
## %%
# for idx, edge in enumerate(edges):
#    edge.update(prev=edges[idx - 1], next=edges[idx + 1])
#
# piece.update(edges=edges, nb_flats=len([edge for edge in edges if edge.sign == 0]))
#
# print("edge sign:", Counter([edge.sign for piece in pieces for edge in piece.edges]))
# print("nb of flats:", Counter([piece.nb_flats for piece in pieces]))
#
# flat_pieces = [piece for piece in pieces if piece.nb_flats > 0]
#
# for piece in flat_pieces:
#    for edge in piece.edges:
#        if edge.sign == 0 and edge.prev.sign != 0:
#            first_flat = edge
#        if edge.sign == 0 and edge.next.sign != 0:
#            last_flat = edge
#    piece.update(
#        first_flat=first_flat,
#        last_flat=last_flat,
#        before_flat=first_flat.prev,
#        after_flat=last_flat.next,
#    )
#
## Show the pieces having the smallest / highest number of flats
# pieces.sort(key=lambda piece: piece.nb_flats)
#
# sign2color = {-1: "red", 0: "green", 1: "blue"}
#
# for piece in pieces[:1] + pieces[-1:]:
#    plt.title(f"{piece.name}, nb of flats={piece.nb_flats}")
#    for edge in piece.edges:
#        plot_contour(
#            sub_contour(piece.contour, edge.idx0, edge.idx1), c=sign2color[edge.sign]
#        )
#    plt.show()
#
## Show the pieces having the min/max edge straight length
# edge_pieces = [(edge, piece) for piece in pieces for edge in piece.edges]
# edge_pieces.sort(key=lambda ep: ep[0].straight_length)
#
# for edge, piece in edge_pieces[:1] + edge_pieces[-1:]:
#    plt.title(f"{piece.name}, edge straight length={edge.straight_length}")
#    plot_contour(piece.contour)
#    plot_contour(sub_contour(piece.contour, edge.idx0, edge.idx1), c="red")
#    plt.show()
#
## Show some normalized edges
# import random
#
# for piece in random.sample(pieces, 1):
#    for edge in piece.edges[:2]:
#        plt.title(f"{piece.name} normalized edge & center")
#        plot_contour(edge.normalized_piece_contour)
#        plot_point(edge.normalized_piece_center, marker="o", c="red")
#        plt.axhline(0, c="gray", ls=":")
#        plt.axvline(0, c="gray", ls=":")
#        plt.show()
#
## %%
