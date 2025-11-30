import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    from pickle import dump, load
    from src.classes import Piece, Edge, EdgeColor, EdgeType, LoopingList
    from src.newutils import plt_np_3ch
    import numpy as np
    import cv2 as cv
    import matplotlib.pyplot as plt

    from itertools import product


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Classes
    """)
    return


@app.class_definition
class PipelineParamsMatching:
    def __init__(self, max_diff_length_ratio:float=0.02):
        self.max_diff_length = max_diff_length_ratio


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Functions
    """)
    return


@app.function
def sliding_window_distance(emb1, emb2, PLOT:bool=False):
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
        short_is_1 = True   # S corresponds to emb1
    else:
        S = np.array(emb2)
        L = np.array(emb1)
        short_is_1 = False  # S corresponds to emb2

    lenS = len(S)
    lenL = len(L)

    best_score = float('inf')
    best_offset = None

    # Slide S along L
    for offset in range(lenL - lenS + 1):
        window = L[offset : offset + lenS]

        # point-per-point distance
        diff = np.abs(S - window)

        ## check tolerance (percentage)
        #allowed = tolerance * np.maximum(np.abs(S), np.abs(window) + 1e-9)
        #mask = diff <= allowed  # boolean per point

        # score: mean absolute error
        score = np.mean(diff)

        area_between = np.trapezoid(np.abs(diff)) # scipy.integration
        score = area_between

        if score < best_score:
            best_score = score
            best_offset = offset

    # Compute shift relative to emb1 vs emb2
    if best_offset is None:
        return None, None, None

    if short_is_1:
        shift_1 = best_offset
        shift_2 = 0
    else:
        shift_1 = 0
        shift_2 = best_offset

    return best_score, shift_1, shift_2


@app.cell
def _(height, params, width):
    def filter_edges(piece1:Piece, piece2:Piece, PLOT:bool=False):
        couples_edges = product(piece1.edge_list, piece2.edge_list)

        _list_images=[]
        for edge1, edge2 in couples_edges:
            if any([edge1.edge_type == EdgeType.flat, edge2.edge_type == EdgeType.flat]):
                continue
            if edge1.edge_type == edge2.edge_type:
               continue 

            edge1.length = np.linalg.norm(edge1.edge_contour[0] - edge1.edge_contour[-1])
            edge2.length = np.linalg.norm(edge2.edge_contour[0] - edge2.edge_contour[-1])
            diff_length = abs(edge1.length - edge2.length)

            if diff_length/max(edge1.length, edge2.length) > params.max_diff_length:
                continue

            print(f"Edge {edge1.edge_id} of Piece {piece1.piece_id} is compatible with Edge {edge2.edge_id} of Piece {piece2.piece_id}")
            print(f"{edge1.edge_type} vs {edge2.edge_type}")
            print(f"Length Edge 1: {edge1.length}")
            print(f"Length Edge 2: {edge2.length}")
            print(f"Length Difference: {diff_length} which is {diff_length/edge1.length} and {diff_length/edge2.length}")
            _bw_img = np.zeros((height, width, 3), dtype=np.uint8)
            cv.drawContours(_bw_img, [edge1.edge_contour], -1, edge1.edge_color, thickness=2)
            cv.drawContours(_bw_img, [edge2.edge_contour], -1, edge2.edge_color, thickness=2)
            _list_images.append(_bw_img)

            # calculate distance of the contour of each edge from the line that connect the two extremes
            len_contour_edge_1 = edge1.edge_contour.shape[0]
            line_edge_1 = np.linspace(edge1.edge_contour[0], edge1.edge_contour[-1], len_contour_edge_1)
            embedding_edge1= list(map(np.linalg.norm, edge1.edge_contour - line_edge_1))

            len_contour_edge_2 = edge2.edge_contour.shape[0]
            line_edge_2 = np.linspace(edge2.edge_contour[0], edge2.edge_contour[-1], len_contour_edge_2)
            embedding_edge2= list(map(np.linalg.norm, edge2.edge_contour - line_edge_2))

            score, shift_1, shift_2= sliding_window_distance(embedding_edge1, embedding_edge2)

            if score is None:
                print("No acceptable alignment found.")
                continue

            aligned_1 = np.zeros(max(len(embedding_edge1), len(embedding_edge2)))
            aligned_2 = np.zeros(max(len(embedding_edge1), len(embedding_edge2)))

            if shift_1 > 0:   # emb1 is shifted to the right
                aligned_1[shift_1:shift_1+len(embedding_edge1)] = embedding_edge1
                aligned_2[:len(embedding_edge2)] = embedding_edge2
            elif shift_2 > 0:  # emb2 is shifted to the right
                aligned_1[:len(embedding_edge1)] = embedding_edge1
                aligned_2[shift_2:shift_2+len(embedding_edge2)] = embedding_edge2
            else:
                # same alignment
                aligned_1= embedding_edge1
                aligned_2= embedding_edge2

            # ---- Plot embeddings ----
            if PLOT:
                plt.figure(figsize=(12,4))
                plt.plot(aligned_1, label="Embedding 1 (aligned)")
                plt.plot(aligned_2, label="Embedding 2 (aligned)")
                plt.title(f"Aligned Embeddings — best_score={score:.4f}")
                plt.xlabel("Index (aligned)")
                plt.ylabel("Distance from line")
                plt.legend()
                plt.grid(True)
                plt.show()

                plt_np_3ch(_list_images)

        return #TODO return edges and matcheds
    return (filter_edges,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Run Pipeline tests
    """)
    return


@app.cell
def _():
    params=PipelineParamsMatching(max_diff_length_ratio=0.1)

    piece1 = load(open("./data/pieces/0001.pkl", "rb"))
    piece2 = load(open("./data/pieces/0002.pkl", "rb"))
    return params, piece1, piece2


@app.cell
def _(piece1, piece2):
    plt_np_3ch([piece1.rgb_img, piece2.rgb_img])

    # plt edges
    height, width = piece1.rgb_img.shape[:2]
    _list_images=[]
    for p in [piece1,piece2]:
        _bw_img = np.zeros((height, width, 3), dtype=np.uint8)
        for edge in p.edge_list:
            cv.drawContours(_bw_img, [edge.edge_contour], -1, edge.edge_color, thickness=2)
        _list_images.append(_bw_img)
    plt_np_3ch(_list_images)
    return height, width


@app.cell
def _(filter_edges):
    def PipelineMatchings(piece1:Piece, piece2:Piece, params:PipelineParamsMatching, PLOT:bool=False):
       filter_edges(piece1, piece2, PLOT)

    return (PipelineMatchings,)


@app.cell
def _(PipelineMatchings, params, piece1, piece2):
    PipelineMatchings(piece1, piece2, params, PLOT=True)
    return


if __name__ == "__main__":
    app.run()
