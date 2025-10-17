import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from classes import (FLAT_EDGE, MAX_SCORE_MSE_DISTANCE, MIN_LEN_EDGE,
                     SAMPLE_POINTS, Edge, Piece)
from myutils import plot_list_images


class MatchData:
    def __init__(
        self,
        edge1: Edge,
        edge2: Edge,
    ):
        self.edge1 = edge1
        self.edge2 = edge2
        (
            self.distances,
            self.mse,
            self.sd,
            self.max_dist,
            self.x_range,
            self.sum_dist,
        ) = self._distance_score()

    def _distance_score(self, NUM_POINTS: int = SAMPLE_POINTS) -> tuple:
        """
        Compare two edges by sampling vertical cross-sections and measuring distances.
        edge1, edge2: np.ndarray with shape (N, 1, 2)
        Returns: (mse, plot_data) or (None, None) if no match
        """
        x_min = max(
            self.edge1.normalized_contour[:, 0, 0].min(),
            self.edge2.normalized_contour[:, 0, 0].min(),
        )
        x_max = min(
            self.edge1.normalized_contour[:, 0, 0].max(),
            self.edge2.normalized_contour[:, 0, 0].max(),
        )

        edge1_cut = self.edge1.normalized_contour[
            np.argmin(
                np.abs(self.edge1.normalized_contour[:, 0, 0] - x_min)
            ) : np.argmin(np.abs(self.edge1.normalized_contour[:, 0, 0] - x_max))
        ]
        edge2_cut = self.edge2.normalized_contour[
            np.argmin(
                np.abs(self.edge2.normalized_contour[:, 0, 0] - x_min)
            ) : np.argmin(np.abs(self.edge2.normalized_contour[:, 0, 0] - x_max))
        ]

        edge1_cut = edge1_cut[np.linspace(0, len(edge1_cut) - 1, NUM_POINTS, dtype=int)]
        edge2_cut = edge2_cut[np.linspace(0, len(edge2_cut) - 1, NUM_POINTS, dtype=int)]

        distances = np.linalg.norm(edge1_cut - edge2_cut, axis=1)
        x_range = np.linspace(x_min, x_max, NUM_POINTS)

        mse = float(np.mean(np.square(self.distances - np.mean(self.distances))))
        sd = float(np.std(self.distances))
        max_dist = np.max(self.distances)
        sum_dist = np.sum(distances)

        return distances, mse, sd, max_dist, x_range, sum_dist

    def plot(self):
        """
        Plot the edges and the distances between them.
        Returns the matplotlib Figure object.
        """
        edge1 = self.edge1
        edge2 = self.edge2
        distances = self.distances
        x_range = self.x_range
        mse = self.mse
        sd = self.sd
        max_dist = self.max_dist

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
        plt.plot(x_range, distances, color="green")
        plt.title(f"MSE: {mse:.6f}, SD: {sd:.6f}, Max: {max_dist:.6f}")
        plt.gca().set_aspect("equal", adjustable="box")
        return plt.gcf()


def plot_edges(edge1: Edge, edge2: Edge):
    """
    plot two edges for visual inspection
    """
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


def plot_list_MatchData(
    list_match_data: list[MatchData],
    suptitle: str = "Match Data",
    path_save: Path | None | str = None,
):
    """
    Plot a list of MatchData objects.
    """
    list_images = {
        f"Match {i}": match_data.plot() for i, match_data in enumerate(list_match_data)
    }
    plot_list_images(list_images, suptitle=suptitle, path_save=path_save)


# def distance_score(edge1: Edge, edge2: Edge) -> tuple:
#    """
#    Compare two edges by sampling vertical cross-sections and measuring distances.
#    edge1, edge2: np.ndarray with shape (N, 1, 2)
#    Returns: (mse, plot_data) or (None, None) if no match
#    """
#    x_min = max(
#        edge1.normalized_contour[:, 0, 0].min(), edge2.normalized_contour[:, 0, 0].min()
#    )
#    x_max = min(
#        edge1.normalized_contour[:, 0, 0].max(), edge2.normalized_contour[:, 0, 0].max()
#    )
#
#    # cut edges at x_min x_max
#    edge1_cut = edge1.normalized_contour[
#        np.argmin(np.abs(edge1.normalized_contour[:, 0, 0] - x_min)) : np.argmin(
#            np.abs(edge1.normalized_contour[:, 0, 0] - x_max)
#        )
#    ]
#    edge2_cut = edge2.normalized_contour[
#        np.argmin(np.abs(edge2.normalized_contour[:, 0, 0] - x_min)) : np.argmin(
#            np.abs(edge2.normalized_contour[:, 0, 0] - x_max)
#        )
#    ]
#
#    # if one of the two has got more points, sample the other to the same length
#    if len(edge1_cut) > len(edge2_cut):
#        edge2_cut = edge2_cut[
#            np.linspace(0, len(edge2_cut) - 1, len(edge1_cut), dtype=int)
#        ]
#    elif len(edge2_cut) > len(edge1_cut):
#        edge1_cut = edge1_cut[
#            np.linspace(0, len(edge1_cut) - 1, len(edge2_cut), dtype=int)
#        ]
#
#    def sample_contour(contour: np.ndarray, num_points: int = 300) -> np.ndarray:
#        """
#        Idea was to smooth the imagine but in the end I just resample the contour to a fixed number of points.
#        """
#        if len(contour) != num_points:
#            # print(f"Resampling contour from {len(contour)} to {num_points} points")
#            indices = np.linspace(0, len(contour) - 1, num_points, dtype=int)
#            contour = contour[indices]
#        return contour
#
#    edge1_smoothed = sample_contour(edge1_cut, num_points=1000)
#    edge2_smoothed = sample_contour(edge2_cut, num_points=1000)
#
#    distances = np.linalg.norm(edge1_smoothed - edge2_smoothed, axis=1)
#    x_range = np.linspace(x_min, x_max, len(distances))
#
#    # find error from the average
#    distances = np.array(distances[:, 0])
#    d_avg = np.mean(distances)
#    mse = np.sum(np.square((distances - d_avg))) / len(distances)
#    sd = float(np.std(distances))
#    max_dist = np.max(distances)
#
#    # plot the distances
#    return mse, PlotData(
#        edge1=edge1,
#        edge2=edge2,
#        distances=distances,
#        x_range=x_range,
#        mse=mse,
#        sd=sd,
#        max_dist=max_dist,
#    )


if __name__ == "__main__":
    path_pieces_edges = "results/02_pieces_with_edges.pkl"
    # Load the single piece from the pickle file
    with open(path_pieces_edges, "rb") as f:
        dict_pieces: dict[str, Piece] = pickle.load(f)

    # l_pieces = list(piece.values())
    # piece1 = l_pieces[8]
    # piece2 = l_pieces[365]
    # print(piece1)

    def check_len(edge: Edge) -> bool:
        return len(edge.normalized_contour) > MIN_LEN_EDGE

    # Create output directory for edge comparison plots
    output_dir = "results/edge_comparisons"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

    dict_matches = {}

    for ip1, piece1 in enumerate(dict_pieces.values()):
        print(f"Processing piece {ip1}...")
        for ie1, edge1 in enumerate(piece1.edge_list):
            if edge1.edge_type == FLAT_EDGE:
                continue
            if not check_len(edge1):
                print(
                    f"[ERROR] Edge {edge1.edge_id[0]} has got len {len(edge1.normalized_contour)}"
                )
                continue

            # Collect all plot data for this edge1
            list_match_data = []
            min_score = float("inf")
            best_match = None

            for ip2, piece2 in enumerate(dict_pieces.values()):
                if ip2 <= ip1:
                    continue

                for ie2, edge2 in enumerate(piece2.edge_list):
                    if edge2.edge_type == 0 or edge1.edge_type == edge2.edge_type:
                        continue
                    if not check_len(edge2):
                        print(
                            f"[ERROR] Edge {edge2.edge_id[0]} has got len {len(edge2.normalized_contour)}"
                        )
                        continue

                    match_data = MatchData(edge1, edge2)
                    if (
                        match_data.mse is not None
                        and match_data.mse < MAX_SCORE_MSE_DISTANCE
                    ):
                        list_match_data.append(match_data)

            # Create and save multiplot for this edge1

    # TODO(matte): fix plotting
    # if list_match_data::
    #    create_multiplot_for_edge(edge1, all_plot_data, output_dir, ip1, ie1)
    #    print(
    #        f"Edge {edge1.edge_id} from piece {ip1}: {len(all_plot_data)} comparisons, best score: {min_score:.6f}"
    #    )
    # else:
    #    print(f"Edge {edge1.edge_id} from piece {ip1}: No valid comparisons found")

    # TODO(matte): fix matching, create set, look at pieces which match with others
