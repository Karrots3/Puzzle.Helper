import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import pickle
    from itertools import product
    from pathlib import Path

    import cv2 as cv
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from src.classes import EdgeType, Match, Piece
    from src.newutils import plt_np_3ch


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    # Classes
    """
    )
    return


@app.class_definition
class PipelineParamsMatching:
    def __init__(self, max_diff_length_ratio: float = 0.02):
        self.max_diff_length = max_diff_length_ratio


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    # Functions
    """
    )
    return


@app.function
def _sliding_window_distance(emb1, emb2) -> tuple[float | None, int, int]:
    """
    Compute best sliding-window alignment distance between two embeddings.
    Embeddings can have different lengths.

    Also returns:
        best_shift: how much piece2 should shift relative to piece1
                    (positive = piece2 moves right, negative = left)
    """

    # reverse second edge (puzzle edges meet "face-to-face")
    emb2 = emb2[::-1]

    # identify short vs long BUT also remember who is who
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

        # point-per-point distance
        diff = np.abs(S - window)

        ## check tolerance (percentage)
        # allowed = tolerance * np.maximum(np.abs(S), np.abs(window) + 1e-9)
        # mask = diff <= allowed  # boolean per point

        # score: mean absolute error
        score = np.mean(diff)

        area_between = np.trapezoid(np.abs(diff)).__float__()  # scipy.integration
        score = area_between

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


@app.function
def _filter_edges(
    piece1: Piece,
    piece2: Piece,
    params: PipelineParamsMatching,
    plot: bool = False,
    verbose: bool = False,
):
    couples_edges = product(piece1.edge_list, piece2.edge_list)
    _height1, _width1 = piece1.rgb_img.shape[:2]
    _height2, _width2 = piece2.rgb_img.shape[:2]
    height = max(_height1, _height2)
    width = max(_width1, _width2)

    _list_images = []
    score, shift_1, shift_2 = None, -1, -1
    for edge1, edge2 in couples_edges:
        if any([edge1.edge_type == EdgeType.flat, edge2.edge_type == EdgeType.flat]):
            continue
        if edge1.edge_type == edge2.edge_type:
            continue

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
                f"Edge {edge1.edge_id} of Piece {piece1.piece_id} is compatible with Edge {edge2.edge_id} of Piece {piece2.piece_id}"
            )
            print(f"{edge1.edge_type} vs {edge2.edge_type}")
            print(f"Length Edge 1: {edge1_length}")
            print(f"Length Edge 2: {edge2_length}")
            print(
                f"Length Difference: {diff_length} which is {diff_length/edge1_length} and {diff_length/edge2_length}"
            )

        _bw_img = np.zeros((height, width, 3), dtype=np.uint8)
        cv.drawContours(
            _bw_img, [edge1.edge_contour], -1, edge1.edge_color, thickness=2
        )
        cv.drawContours(
            _bw_img, [edge2.edge_contour], -1, edge2.edge_color, thickness=2
        )
        _list_images.append(_bw_img)

        # calculate distance of the contour of each edge from the line that connect the two extremes
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

        aligned_1 = np.zeros(max(len(embedding_edge1), len(embedding_edge2)))
        aligned_2 = np.zeros(max(len(embedding_edge1), len(embedding_edge2)))

        if shift_1 > 0:  # emb1 is shifted to the right
            aligned_1[shift_1 : shift_1 + len(embedding_edge1)] = embedding_edge1
            aligned_2[: len(embedding_edge2)] = embedding_edge2
        elif shift_2 > 0:  # emb2 is shifted to the right
            aligned_1[: len(embedding_edge1)] = embedding_edge1
            aligned_2[shift_2 : shift_2 + len(embedding_edge2)] = embedding_edge2
        else:
            # same alignment
            aligned_1 = embedding_edge1
            aligned_2 = embedding_edge2

        # ---- Plot embeddings ----
        if plot:
            plt.figure(figsize=(12, 4))
            plt.plot(aligned_1, label="Embedding 1 (aligned)")
            plt.plot(aligned_2, label="Embedding 2 (aligned)")
            plt.title(f"Aligned Embeddings — best_score={score:.4f}")
            plt.xlabel("Index (aligned)")
            plt.ylabel("Distance from line")
            plt.legend()
            plt.grid(True)
            plt.show()

            plt_np_3ch(_list_images)

    return Match(
        piece_id_1=piece1.piece_id,
        piece_id_2=piece2.piece_id,
        score=score,
        shift_1=shift_1,
        shift_2=shift_2,
    )


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    # Run Pipeline tests
    """
    )
    return


@app.function
def PipelineMatchings(
    _piece1: Piece | Path | str,
    _piece2: Piece | Path | str,
    params: PipelineParamsMatching,
    path_output: str | Path | None,
    PLOT: bool = False,
    VERBOSE: bool = False,
):
    if isinstance(_piece1, Piece):
        piece1: Piece = _piece1
    elif isinstance(_piece1, (Path, str)):
        with open(_piece1, "rb") as f:
            piece1: Piece = pickle.load(f)
    else:
        raise ValueError(f"Invalid type for piece1: {type(_piece1)}")

    if isinstance(_piece2, Piece):
        piece2: Piece = _piece2
    elif isinstance(_piece2, (Path, str)):
        with open(_piece2, "rb") as f:
            piece2: Piece = pickle.load(f)
    else:
        raise ValueError(f"Invalid type for piece2: {type(_piece2)}")

    out: Match = _filter_edges(piece1, piece2, params, PLOT, VERBOSE)
    if path_output is not None:
        out_path = (
            f"{path_output}/match_{piece1.piece_id:04d}_{piece2.piece_id:04d}.pkl"
        )
        with open(out_path, "wb") as f:
            pickle.dump(out, f)


@app.cell
def _():
    params = PipelineParamsMatching(max_diff_length_ratio=0.1)

    piece1 = pickle.load(open("./data/pieces/0001.pkl", "rb"))
    piece2 = pickle.load(open("./data/pieces/0002.pkl", "rb"))
    return params, piece1, piece2


@app.cell
def _(piece1, piece2):
    plt_np_3ch([piece1.rgb_img, piece2.rgb_img])

    # plt edges
    height, width = piece1.rgb_img.shape[:2]
    _list_images = []
    for p in [piece1, piece2]:
        _bw_img = np.zeros((height, width, 3), dtype=np.uint8)
        for edge in p.edge_list:
            cv.drawContours(
                _bw_img, [edge.edge_contour], -1, edge.edge_color, thickness=2
            )
        _list_images.append(_bw_img)
    plt_np_3ch(_list_images)
    return height, width


@app.cell
def _(params, piece1, piece2):
    PipelineMatchings(piece1, piece2, params, "./data/match", PLOT=True)
    return


if __name__ == "__main__":
    app.run()
