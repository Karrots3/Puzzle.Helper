# The objective of this script is to detect the edges of each puzzle piece
# Extract each edge contour, normalize it
# Check also length is within a certain tolerance (both straight length and contour length)

import math
import pickle
from itertools import combinations
from math import sqrt
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from src.classes import Edge, EdgeColorType, EdgeType, LoopingList, Piece
from src.myutils import plot_list_images

PROMINENCE = 10500


# TODO(matte): smarter prominence when is not found the correct number of peaks
def get_edges(piece: Piece, prominence=PROMINENCE) -> Piece:
    """
    Detect edges of a puzzle piece, classify them, and store in the Piece object.
    """
    # Default paths for saving images (can be overridden if needed)
    path_contour = "results/02_contour/"
    path_edges = "results/02_edges/"
    path_pipelines = "results/02_pipelines/"

    list_images_pipeline = {}

    # -------------------
    # --- Get contour ---
    # -------------------
    assert type(piece) == Piece, "Expected a Piece object, obtained: " + str(
        type(piece)
    )
    contour = piece.contour

    # --------------------------------------------------
    # --- Find corners via peak distance from center ---
    # --------------------------------------------------
    (cx, cy), _ = cv2.minEnclosingCircle(contour)
    centered_contour = contour - np.array([cx, cy])

    # -- ensure peaks are not at start or end of the distances array --
    distances = np.sum(centered_contour**2, axis=2)[:, 0]
    distance_offset = np.argmin(distances)
    shifted_distances = np.concatenate(
        [distances[distance_offset:], distances[:distance_offset]]
    )

    peak_indices = [
        (distance_idx + distance_offset) % len(distances)
        for distance_idx in find_peaks(shifted_distances, prominence=prominence)[0]
    ]
    peak_indices.sort()

    # plot all peaks over the image
    img_contour = cv2.cvtColor(piece.bw_thresh_fixed, cv2.COLOR_GRAY2BGR)
    for peak_index in peak_indices:
        cv2.circle(
            img_contour,
            (int(contour[peak_index][0][0]), int(contour[peak_index][0][1])),
            20,
            (0, 0, 255),
            -1,
        )
    list_images_pipeline["allpeaks"] = img_contour

    # --------------------------------------------
    # --- Filter corners by rectangle geometry ---
    # --------------------------------------------
    def compute_rectangle_error(indices):
        # -- get coordinates of corners --
        corners = LoopingList(np.take(contour, sorted(list(indices)), axis=0)[:, 0, :])

        # -- compute the side lengths and diagonal lengths --
        lengths = [
            sqrt(np.sum((corners[i0] - corners[i1]) ** 2))
            for i0, i1 in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
        ]

        # -- compute the error of the rectangle --
        def f_error(a, b):
            return abs(b - a) / (a + b)

        return sum(
            [
                f_error(lengths[i], lengths[j])
                for i, j in [(0, 2), (1, 3), (4, 5), (0, 1)]
            ]
        )

    # sort rectangles combinations
    rectangles: list[tuple[float, list[int]]] = []
    for indices in combinations(peak_indices, 4):
        error = compute_rectangle_error(indices)
        rectangles.append((error, indices))

    # -- select the best one --
    if len(rectangles) == 0:
        print(f"[ERROR] piece {piece.piece_id} has no rectangle")
        return piece

    error, indices = sorted(rectangles)[0]
    rectangle_error = error
    corner_indices = LoopingList(indices)
    print("Rectangle error:", rectangle_error)

    (p_cx, p_cy) = np.mean(piece.contour[:, 0, :], axis=0)

    # --- plot all peaks over the image ---
    img_contour = cv2.cvtColor(piece.img_thresh, cv2.COLOR_GRAY2BGR)

    # --- add center points ---
    cv2.circle(img_contour, (int(p_cx), int(p_cy)), 10, (255, 0, 0), -1)

    # -- plot corner points --
    for corner_index in corner_indices:
        x, y = int(contour[corner_index][0][0]), int(contour[corner_index][0][1])
        cv2.circle(
            img_contour,
            (x, y),
            20,
            (0, 0, 255),
            -1,
        )
        # Add text label beside the corner point
        cv2.putText(
            img_contour,
            f"{corner_index}",
            (x + 25, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    # --- draw rectangle lines connecting the corners ---
    for i in range(len(corner_indices)):
        start_idx = corner_indices[i]
        end_idx = corner_indices[(i + 1) % len(corner_indices)]

        start_point = (int(contour[start_idx][0][0]), int(contour[start_idx][0][1]))
        end_point = (int(contour[end_idx][0][0]), int(contour[end_idx][0][1]))

        cv2.line(img_contour, start_point, end_point, (0, 0, 255), 3)

    list_images_pipeline[f"Rectangle error: {rectangle_error:.3f}"] = img_contour

    # ----------------------
    # --- classify edges ---
    # ----------------------
    def classify_edge(edge_contour, piece_center, flat_thresh=5.0) -> EdgeType:
        pts = edge_contour[:, 0, :]  # Nx2
        p0, p1 = pts[0], pts[-1]

        # Vector of the edge baseline
        edge_vec = p1 - p0
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-6:
            return EdgeType.FLAT

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
            return EdgeType.FLAT

        # Determine sign of deviation at the most extreme point
        idx_max = np.argmax(np.abs(distances))
        # extreme_point = pts[idx_max]
        extreme_sign = np.sign(distances[idx_max])

        # Decide male/female based on whether extreme point is closer/farther than center
        vec_center = piece_center - p0
        center_side = np.sign(np.dot(vec_center, normal))

        if extreme_sign == center_side:
            return EdgeType.FEMALE
        else:
            return EdgeType.MALE

    ###############################################################
    # Normalize contours
    ###############################################################
    indices = [
        [corner_indices[0], corner_indices[1]],
        [corner_indices[1], corner_indices[2]],
        [corner_indices[2], corner_indices[3]],
        [corner_indices[3], corner_indices[0] + len(contour)],
    ]
    print("indices", indices)

    # edge_contours = [
    #    contour[corner_indices[0] : corner_indices[1], :],
    #    contour[corner_indices[1] : corner_indices[2], :],
    #    contour[corner_indices[2] : corner_indices[3], :],
    #    contour[corner_indices[3] : corner_indices[0] + len(contour), :],
    # ]
    edge_contours = []
    for i, idx in enumerate(indices):
        if idx[1] + 1 > idx[0]:
            edge_contours.append(contour[idx[0] : idx[1] + 1, :])
        else:
            edge_contours.append(
                np.concatenate([contour[idx[0] :], contour[: idx[1] + 1]])
            )

        # plot this new edge, imshow and after click close the plot
        # Create a blank image to draw the contour on
        img_height, img_width = piece.img_thresh.shape[:2]
        blank_img = np.zeros((img_height, img_width), dtype=np.uint8)

        # Draw the contour on the blank image
        if len(edge_contours[i]) > 0:
            cv2.drawContours(blank_img, [edge_contours[i].astype(np.int32)], -1, 255, 2)

        # cv2.imshow("edge", blank_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    print("edge_contours lengths", [len(edge_contours[i]) for i in range(4)])

    def normalize_contour(contour):
        p0 = contour[0][0]
        p1 = contour[-1][0]
        dx, dy = p1 - p0
        center = np.mean(contour[:, 0, :], axis=0)
        degrees = math.degrees(math.atan2(dy, dx))
        scale = 1.0
        straight_length = math.sqrt(dx**2 + dy**2)

        matrix = cv2.getRotationMatrix2D(center, degrees, scale)
        translate = (0, 0) - center

        norm_contour = cv2.transform(contour, matrix) + translate
        straight_length2 = math.sqrt(
            np.sum((norm_contour[-1][0] - norm_contour[0][0]) ** 2)
        )
        # assert (
        #     abs(straight_length - straight_length2) < straight_length * 0.01
        # ), f"piece {piece.piece_id} has a different straight length {straight_length} {straight_length2}"
        if abs(straight_length - straight_length2) > straight_length * 0.01:
            print(
                f"[ERROR] piece {piece.piece_id} has a different straight length {straight_length} {straight_length2}"
            )

        return norm_contour

    list_edges: list[Edge] = []
    for edg in edge_contours:
        p0 = edg[0][0]
        p1 = edg[-1][0]
        # normalize the contour: first point at (0, 0), last point at (X, 0)
        dx, dy = p1 - p0
        straight_length = math.sqrt(dx**2 + dy**2)
        angle_degrees = math.degrees(math.atan2(dy, dx))
        if straight_length < 400:
            print(
                f"piece {piece.piece_id} has a straight length {straight_length} < 400"
            )

        normalized_edge_contour = normalize_contour(edg)
        type_edge = classify_edge(
            edg, piece_center=np.array([p_cx, p_cy]), flat_thresh=20
        )

        list_edges.append(
            Edge(
                edg,
                contour,
                normalized_edge_contour,
                straight_length,
                angle_degrees,
                type_edge,
            )
        )

    esl = [edge.straight_length for edge in list_edges]

    if min(esl) < 2 / 3 * max(esl):
        print(piece.piece_id, min(esl), max(esl), np.mean(esl))

    ###############################################################
    # Plot edges over the actual contour in the image
    ###############################################################
    # TODO(matte): fix this plot to show also the image, fix list, list contains cv2
    img_contour = cv2.cvtColor(piece.img_thresh, cv2.COLOR_GRAY2BGR)
    for edge in list_edges:
        plt.plot(
            edge.contour[:, 0, 0],
            edge.contour[:, 0, 1],
            # COLORS_GENDER[edge.edge_type],
            c=EdgeColorType[edge.edge_type.name].value,
        )
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig(path_contour + f"{piece.piece_id}.png")
    plt.close()
    list_images_pipeline["contour"] = img_contour

    ###############################################################
    # Visualize normalized edges, do not warp axes, keep same scale
    ###############################################################
    for edge in list_edges:
        plt.plot(
            edge.normalized_contour[:, 0, 0],
            edge.normalized_contour[:, 0, 1],
            c=EdgeColorType[edge.edge_type.name].value,
        )
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig(path_edges + f"{piece.piece_id}.png")
    # plt.show(block=False)
    # plt.waitforbuttonpress()
    plt.close()

    plot_list_images(
        list_images_pipeline, path_save=path_pipelines + f"{piece.piece_id}.png"
    )
    print(piece.piece_id)
    piece.edge_list = list_edges

    return piece


def main():
    # --- Paths ---
    Path_allpeaks = Path("results/02_allpeaks")
    Path_allpeaks.mkdir(parents=True, exist_ok=True)

    Path_edges = Path("results/02_edges")
    Path_edges.mkdir(parents=True, exist_ok=True)

    Path_rectangle = Path("results/02_rectangle")
    Path_rectangle.mkdir(parents=True, exist_ok=True)

    Path_pipelines = Path("results/02_pipelines")
    Path_pipelines.mkdir(parents=True, exist_ok=True)

    Path_contour = Path("results/02_contour")
    Path_contour.mkdir(parents=True, exist_ok=True)

    path_allpeaks: str = str(Path_allpeaks) + "/"
    path_edges = str(Path_edges) + "/"
    path_rectangle = str(Path_rectangle) + "/"
    path_pipelines = str(Path_pipelines) + "/"
    path_contour = str(Path_contour) + "/"

    path_pieces = Path("results/pieces/list_pieces.pkl")
    path_new_pieces = Path("results/02_pieces_with_edges.pkl")

    with open(path_pieces, "rb") as f:
        pieces = pickle.load(f)
    assert type(pieces) == dict, "Expected a dictionary of pieces"

    from multiprocessing import Pool

    nw = 12
    with Pool(nw) as p:
        pieces = p.map(get_edges, list(pieces.values()))

    with open(path_new_pieces, "wb") as f:
        pickle.dump({piece.piece_id: piece for piece in pieces}, f)


if __name__ == "__main__":
    main()

# %%
