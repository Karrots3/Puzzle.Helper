import pickle
from turtle import distance

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splev, splprep

from classes import Edge, Piece


def smoot_contour(
    contour: np.ndarray, smooth_factor: float = 0.5, num_points: int = 300
) -> np.ndarray:
    pts = contour[:, 0, :]
    x, y = pts[:, 0], pts[:, 1]
    try:
        tck, u = splprep([x, y], s=smooth_factor)
        x_smooth, y_smooth = splev(np.linspace(0, 1, num_points), tck)
        smoothed = np.stack((x_smooth, y_smooth), axis=1).astype(np.float32)
        return smoothed.reshape(-1, 1, 2)
    except Exception as e:
        print(f"Smoothing failed: {e}")
        # If spline fails, resample the original contour to the desired number of points
        if len(contour) != num_points:
            # print(f"Resampling contour from {len(contour)} to {num_points} points")
            indices = np.linspace(0, len(contour) - 1, num_points, dtype=int)
            contour = contour[indices]
        return contour


def plot_edges(edge1: Edge, edge2: Edge) -> float:
    """
    Compare two edges and return a score indicating how well they match.
    """
    # edge1.normalized_contour = smoot_contour(edge1.normalized_contour)
    # edge2.normalized_contour = smoot_contour(edge2.normalized_contour)

    # Plot toghether for visual inspection edges.normalized_contour
    plt.plot(
        edge1.normalized_contour[:, 0, 0],
        edge1.normalized_contour[:, 0, 1],
        label=f"Edge {edge1.edge_id} ({edge1.edge_type})",
        color="red",
    )
    plt.plot(
        edge2.normalized_contour[:, 0, 0],
        -edge2.normalized_contour[:, 0, 1],
        label=f"Edge {edge2.edge_id} ({edge2.edge_type})",
        color="blue",
    )
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show(block=False)
    plt.waitforbuttonpress()
    plt.close()


def geometric_match(edge1: Edge, edge2: Edge) -> float:
    """
    Compare two edges by sampling vertical cross-sections and measuring distances.
    edge1, edge2: np.ndarray with shape (N, 1, 2)
    """
    x_min = max(
        edge1.normalized_contour[:, 0, 0].min(), edge2.normalized_contour[:, 0, 0].min()
    )
    x_max = min(
        edge1.normalized_contour[:, 0, 0].max(), edge2.normalized_contour[:, 0, 0].max()
    )

    # cut edges at x_min x_max
    edge1_cut = edge1.normalized_contour[
        np.argmin(np.abs(edge1.normalized_contour[:, 0, 0] - x_min)) : np.argmin(
            np.abs(edge1.normalized_contour[:, 0, 0] - x_max)
        )
    ]
    edge2_cut = edge2.normalized_contour[
        np.argmin(np.abs(edge2.normalized_contour[:, 0, 0] - x_min)) : np.argmin(
            np.abs(edge2.normalized_contour[:, 0, 0] - x_max)
        )
    ]

    # if one of the two has got more points, sample the other to the same length
    if len(edge1_cut) > len(edge2_cut):
        edge2_cut = edge2_cut[
            np.linspace(0, len(edge2_cut) - 1, len(edge1_cut), dtype=int)
        ]
    elif len(edge2_cut) > len(edge1_cut):
        edge1_cut = edge1_cut[
            np.linspace(0, len(edge1_cut) - 1, len(edge2_cut), dtype=int)
        ]
    edge1_smoothed = smoot_contour(edge1_cut, smooth_factor=0.5, num_points=1000)
    edge2_smoothed = smoot_contour(edge2_cut, smooth_factor=0.5, num_points=1000)

    distances = np.linalg.norm(edge1_smoothed - edge2_smoothed, axis=1)
    x_range = np.linspace(x_min, x_max, len(distances))

    # find error from the average
    distances = np.array(distances[:, 0])
    d_avg = np.mean(distances)
    mse = np.sum(np.square((distances - d_avg))) / len(distances)
    sd = np.std(distances)
    max_dist = np.max(distances)
    if mse > 20:
        return mse

    # plot the distances
    plt.plot(
        edge1_cut[:, 0, 0],
        edge1_cut[:, 0, 1],
        label=f"Edge {edge1.edge_id} ({edge1.edge_type})",
        color="red",
    )
    plt.plot(
        edge2_cut[:, 0, 0],
        -edge2_cut[:, 0, 1],
        label=f"Edge {edge2.edge_id} ({edge2.edge_type})",
        color="blue",
    )
    plt.plot(x_range, distances, color="green")
    plt.title(f"MSE: {mse:.6f}, SD: {sd:.6f}, Max: {max_dist:.6f}")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show(block=False)
    plt.waitforbuttonpress()
    plt.close()
    return mse


if __name__ == "__main__":
    path_pieces_edges = "results/02_pieces_with_edges.pkl"
    # Load the single piece from the pickle file
    with open(path_pieces_edges, "rb") as f:
        piece: dict[str, Piece] = pickle.load(f)

    l_pieces = list(piece.values())
    piece1 = l_pieces[8]
    piece2 = l_pieces[365]
    print(piece1)

    def check_len(edge: Edge) -> bool:
        return len(edge.normalized_contour) > 600

    min_score = float("inf")

    for ii, edge1 in enumerate(piece1.edge_list):
        if edge1.edge_type == "straight":
            continue
        if not check_len(edge1):
            print(
                f"[ERROR] Edge {edge1.edge_id[0]} has got len {len(edge1.normalized_contour)}"
            )
            continue

        for ip, piece2 in enumerate(l_pieces):
            for ie, edge2 in enumerate(piece2.edge_list):
                if edge2.edge_type == "straight" or edge1.edge_type == edge2.edge_type:
                    continue
                if not check_len(edge2):
                    print(
                        f"[ERROR] Edge {edge2.edge_id[0]} has got len {len(edge2.normalized_contour)}"
                    )
                    continue

                score = geometric_match(edge1, edge2)
                if score < min_score:
                    min_score = score
                    best_match = (ii, ip, ie)
                    print(
                        f"New best match: {min_score} between edge {edge1.edge_id[0]} and edge {edge2.edge_id[0]}"
                    )

    print(f"Best match: {best_match} with score {min_score}")
    plot_edges(edge1, l_pieces[best_match[0]].edge_list[best_match[1]])
