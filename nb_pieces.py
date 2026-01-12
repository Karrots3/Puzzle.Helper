import marimo

__generated_with = "0.18.1"
app = marimo.App(width="full")

with app.setup:
    # Initialization code that runs before all other cells
    from itertools import combinations
    from math import sqrt
    from pathlib import Path
    import marimo as mo
    import cv2 as cv

    import numpy as np
    from scipy.signal import find_peaks

    from src.classes import (
        ContourError,
        Edge,
        EdgeColor,
        EdgeType,
        LoopingList,
        Piece,
        PipelineProcessImgParams,
    )
    from src.newutils import plt_np_3ch


@app.cell
def _():
    mo.md(r"""
    # Classes definitions for this notebook
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Definitions for pipeline imagine processing
    """)
    return


@app.function
def _crop_img(
    img: np.ndarray, params: PipelineProcessImgParams, PLOT: bool = False
) -> np.ndarray:
    # Get Params
    radius = params.crop_radius
    shift_x = params.crop_shift_x
    shift_y = params.crop_shift_y

    # Do stuff
    _row, _col = img.shape[:2]
    _center_row, _center_col = _row // 2, _col // 2
    _center_row, _center_col = int(_center_row * shift_y), int(_center_col * shift_x)
    _radius_row, _radius_col = int(_center_row * radius), int(_center_col * radius)
    _radius = min(_radius_row, _radius_col)
    cropped_img = img[
        _center_row - _radius : _center_row + _radius,
        _center_col - _radius : _center_col + _radius,
        :,
    ]

    if PLOT:
        _img = img.copy()
        cv.circle(
            _img,
            (_center_col, _center_row),
            min(_radius_row, _radius_col),
            (0, 255, 0),
            2,
        )
        cv.line(_img, (_center_col, 0), (_center_col, _row), (255, 0, 0), 1)
        cv.line(_img, (0, _center_row), (_col, _center_row), (255, 0, 0), 1)
        plt_np_3ch([img, _img])

    return cropped_img


@app.function
def _treshold_img(
    img: np.ndarray, params: PipelineProcessImgParams, PLOT: bool = False
) -> np.ndarray:
    # Get Params
    thresh = params.th_thresh
    maxvalue = params.th_maxvalue

    # Do stuff
    _, img_fixed = cv.threshold(img, thresh, maxvalue, cv.THRESH_BINARY)
    _, img_otsu = cv.threshold(img, thresh, maxvalue, cv.THRESH_BINARY + cv.THRESH_OTSU)

    if PLOT:
        plt_np_3ch([img, img_fixed, img_otsu], size_height=3, ibw=range(3))

    return img_otsu


@app.function
def _get_contour(
    bw_img: np.ndarray, params: PipelineProcessImgParams, PLOT: bool = False
) -> np.ndarray | None:
    # Get params
    min_area = params.filter_contour_min_area
    # max_area = params.filter_contour_max_area
    # min_perimeter = params.filter_contour_min_perimeter
    # max_perimeter = params.filter_contour_max_perimeter

    # Do stuff
    contours, _ = cv.findContours(bw_img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    filtered_contours = []
    for contour in contours:
        # perimeter = cv.arcLength(contour, True)
        # x, y, w, h = cv.boundingRect(contour)
        # aspect_ratio = (float(w / h) if h > 0 else 0,)
        # box = np.int8(cv.boxPoints(rect))
        # rect = cv.minAreaRect(contour)
        area = cv.contourArea(contour)
        hull = cv.convexHull(contour)
        hull_area = cv.contourArea(hull)

        if area > min_area and area != hull_area:
            filtered_contours.append(contour)

    if PLOT:
        height, width = bw_img.shape
        _plt_images = [bw_img]
        for contour in filtered_contours:
            rgb_img = np.zeros((height, width, 3), dtype=np.uint8)
            cv.drawContours(rgb_img, [contour], 0, (255, 255, 255), 7)
            _plt_images.append(rgb_img)

        plt_np_3ch(_plt_images, ibw=0, size_height=3)

    # Calculate areas for error reporting
    contour_areas = [cv.contourArea(contour) for contour in filtered_contours]

    # Raise exception if we don't have exactly one contour
    if len(filtered_contours) == 0:
        raise ContourError(
            message="No contours found after filtering",
            num_contours=0,
            contour_areas=[],
        )
    elif len(filtered_contours) > 1:
        raise ContourError(
            message=f"Found {len(filtered_contours)} contours, expected exactly 1",
            num_contours=len(filtered_contours),
            contour_areas=contour_areas,
        )

    contour = filtered_contours[0]

    return contour


@app.function
def _get_corner_indices(
    bw_img: np.ndarray,
    contour: np.ndarray,
    params: PipelineProcessImgParams,
    PLOT: bool = False,
) -> np.ndarray:
    # Get Params
    prominence = params.edges_peaks_prominence

    # Do stuff
    width, height = bw_img.shape[:2]

    (_circle_x, _circle_y), _ = cv.minEnclosingCircle(contour)
    centered_contour = contour - np.array([_circle_x, _circle_y])

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

    if PLOT:
        _img = np.zeros((height, width, 3), dtype=np.uint8)
        cv.drawContours(_img, [contour], -1, (255, 255, 255), 2)
        for peak_index in peak_indices:
            x, y = int(contour[peak_index][0][0]), int(contour[peak_index][0][1])
            cv.circle(_img, (x, y), 20, (0, 0, 255), 2)
        plt_np_3ch([_img], size_height=3)

    # --------------------------------------------
    # --- Filter corners by rectangle geometry ---
    # --------------------------------------------
    def _compute_rectangle_error(indices):
        # -- get coordinates of corners --
        peaks_coordinates = LoopingList(
            np.take(contour, sorted(list(indices)), axis=0)[:, 0, :]
        )

        # -- compute the side lengths and diagonal lengths --
        lengths = [
            sqrt(np.sum((peaks_coordinates[i0] - peaks_coordinates[i1]) ** 2))
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
        error = _compute_rectangle_error(indices)
        rectangles.append((error, indices))

    # if PLOT:
    #    imgs=[]
    #    for rectangle in rectangles:
    #        _img = np.zeros((height, width, 3), dtype=np.uint8)
    #        cv.drawContours(_img, [contour], -1, (255, 255, 255), 2)
    #        for i1, i2 in zip(rectangle[1], rectangle[1][1:]):
    #            start_point = (int(contour[i1][0][0]), int(contour[i1][0][1]))
    #            end_point = (int(contour[i2][0][0]), int(contour[i2][0][1]))
    #            cv.line(_img, start_point, end_point, (255, 255, 255), 3)
    #        imgs.append(_img)
    #    plt_np_3ch(imgs, size_height=3, max_col=4)

    # --- Select best rectangle ---
    error, indices = sorted(rectangles)[0]
    # rectangle_error = error
    corner_indices = LoopingList(indices)

    # -- plot corner points --
    if PLOT:
        _bw_img = np.zeros((height, width), dtype=np.uint8)
        for corner_index in corner_indices:
            x, y = int(contour[corner_index][0][0]), int(contour[corner_index][0][1])
            cv.drawContours(_bw_img, [contour], -1, (255, 255), 2)
            cv.circle(
                _bw_img,
                (x, y),
                20,
                (255, 255, 255),
                -1,
            )
            # Add text label beside the corner point
            cv.putText(
                _bw_img,
                f"{corner_index}",
                (x + 25, y - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
        for i in range(len(corner_indices)):
            start_idx = corner_indices[i]
            end_idx = corner_indices[(i + 1) % len(corner_indices)]
            start_point = (int(contour[start_idx][0][0]), int(contour[start_idx][0][1]))
            end_point = (int(contour[end_idx][0][0]), int(contour[end_idx][0][1]))
            cv.line(_bw_img, start_point, end_point, (255, 255, 255), 3)
        plt_np_3ch([_bw_img], ibw=0, size_height=3)

    if PLOT:
        # Draw a close up of the peak corner indices found
        _bw_img = bw_img.copy()
        cv.drawContours(_bw_img, [contour], -1, (0, 255, 0), 2)
        closeup_imgs = []
        zone = 100
        for corner_index in corner_indices:
            x, y = int(contour[corner_index][0][0]), int(contour[corner_index][0][1])
            cv.circle(_bw_img, (x, y), 20, (0, 255, 0), 3, 5)
            x_start, x_end = max(0, x - zone), min(width, x + zone)
            y_start, y_end = max(0, y - zone), min(height, y + zone)
            closeup_img = _bw_img[y_start:y_end, x_start:x_end]
            closeup_imgs.append(closeup_img)
        plt_np_3ch(closeup_imgs, ibw=range(4), size_height=3)

    return corner_indices


@app.function
def _get_edges(
    bw_img: np.ndarray,
    contour: np.ndarray,
    corner_indices: list[int],
    params: PipelineProcessImgParams,
    PLOT: bool,
) -> list[Edge]:
    # Get params
    height, width = bw_img.shape[:2]
    contour_center = np.mean(contour[:, 0, :], axis=0)
    edge_type = params.edge_type
    edge_color = params.edge_color
    flat_threshold = params.flat_threshold

    # Do stuff
    # given 4 peak indices, extrac edges
    corner_indices.sort()
    _pki = corner_indices
    edges = [
        np.concatenate(
            [
                contour[_pki[-1] :, :, :],  # from pki[-1] to end
                contour[: _pki[0], :, :],  # from 0 to pki[0]
            ],
            axis=0,
        ),
        contour[_pki[0] : _pki[1], :, :],
        contour[_pki[1] : _pki[2], :, :],
        contour[_pki[2] : _pki[3], :, :],
    ]

    def classify_edge(edge, piece_center, flat_thresh):
        pts = edge[:, 0, :]
        p0, p1 = pts[0], pts[-1]

        # Vector of the edge baseline
        edge_vec = p1 - p0
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-6:
            return edge_type.flat

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
            return edge_type.flat

        # Determine sign of deviation at the most extreme point
        idx_max = np.argmax(np.abs(distances))
        # extreme_point = pts[idx_max]
        extreme_sign = np.sign(distances[idx_max])

        # Decide male/female based on whether extreme point is closer/farther than center
        vec_center = piece_center - p0
        center_side = np.sign(np.dot(vec_center, normal))

        if extreme_sign == center_side:
            return edge_type.woman
        else:
            return edge_type.man

    edges_type = [classify_edge(edge, contour_center, flat_threshold) for edge in edges]
    edges_color = [edge_color[et.name].value for et in edges_type]

    if PLOT:
        _img = np.zeros((height, width, 3), dtype=np.uint8)
        for i, edge in enumerate(edges):
            cv.drawContours(_img, [edge], -1, edges_color[i], 2)
        plt_np_3ch([_img], size_height=3)

    return [
        Edge(
            edge_id=i,
            edge_type=edges_type[i],
            edge_color=edges_color[i],
            edge_contour=edges[i],
        )
        for i in range(4)
    ]


@app.cell
def _():
    mo.md(r"""
    ## Final Pipeline Imagine Processing
    """)
    return


@app.function
def PipelineProcessImg(path: Path | str, params, PLOT: bool = False):
    bgr_img = cv.imread(str(path))
    bgr_img = _crop_img(bgr_img, params, PLOT)
    rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
    bw_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)

    if PLOT:
        plt_np_3ch([rgb_img, bw_img], ibw=1, size_height=3)

    img = _treshold_img(bw_img, params, PLOT)
    contour = _get_contour(img, params, PLOT)
    corner_indices = _get_corner_indices(bw_img, contour, params, PLOT)
    edges = _get_edges(bw_img, contour, corner_indices, params, PLOT)

    piece = Piece(piece_id=0, rgb_img=rgb_img, contour=contour, edge_list=edges)
    return piece


@app.cell
def _():
    mo.md(r"""
    # Running Pipeline
    """)
    return


@app.cell
def _():
    str_img = "data/0002.JPG"
    params = PipelineProcessImgParams()
    return params, str_img


@app.cell
def _(params, str_img):
    PipelineProcessImg(str_img, params, PLOT=True)
    return


if __name__ == "__main__":
    app.run()
