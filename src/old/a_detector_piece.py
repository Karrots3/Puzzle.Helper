import argparse
import pickle
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt

from src.classes import DEBUG, Piece
from src.myutils import get_pieceId, plot_list_images


def detectPieceFromImagine(
    img_path, img_index, method, path_plot_imagine_pipeline: Path
) -> Piece | None:
    """
    Detect puzzle pieces in an image and return analysis results.
    """
    assert method in ["otsu", "fixed"], "Method must be 'otsu' or 'fixed'"
    assert (
        path_plot_imagine_pipeline is not None
    ), "path_plot_imagine_pipeline must be provided"
    list_images_pipeline = {}  # visualization steps

    # --- Load image ---
    assert img_path.exists(), f"Image path {img_path} does not exist"
    cv2_img_rgb = cv2.imread(str(img_path))

    # list_images_pipeline["00-original"] = img_rgb

    # --- Preprocessing with simple threshold ---
    def _trim_thresh(cv2_img_rgb) -> tuple[np.ndarray, np.ndarray]:
        cv2_img_gray = cv2.cvtColor(cv2_img_rgb, cv2.COLOR_BGR2GRAY)
        h, w = cv2_img_gray.shape[:2]
        center_x, center_y = w // 2, h // 2
        radius = 60
        sz = int((radius / 100) * min(h, w))
        half_sz = sz // 2
        x1, y1, x2, y2 = (
            max(0, center_x - half_sz),
            max(0, center_y - half_sz),
            min(w, center_x + half_sz),
            min(h, center_y + half_sz),
        )
        img_trimmed = cv2_img_gray[y1:y2, x1:x2]
        # list_images_pipeline["01-grey-trimmed"] = img_trimmed

        _, bw_img_thresh_fixed = cv2.threshold(img_trimmed, 110, 255, cv2.THRESH_BINARY)
        _, bw_img_thresh_otsu = cv2.threshold(
            img_trimmed, 110, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return bw_img_thresh_fixed, bw_img_thresh_otsu

    bw_img_thresh_fixed, bw_img_thresh_otsu = _trim_thresh(cv2_img_rgb)

    # list_images_pipeline["02-thresh-fixed"] = img_thresh_fixed
    # list_images_pipeline["02-thresh-otsu"] = img_thresh_otsu

    # plot the image
    plt.imshow(bw_img_thresh_fixed, cmap="gray")
    plt.show()

    # --- Contour extraction ---
    def get_valid_contours(bw_img_thresh) -> list[np.ndarray]:
        contours, _ = cv2.findContours(
            bw_img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        return contours

    cv2_contours_otsu = get_valid_contours(bw_img_thresh_otsu)
    cv2_contours_fixed = get_valid_contours(bw_img_thresh_fixed)

    def contour2piece(contour, piece_index) -> Piece | None:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        rect = cv2.minAreaRect(contour)
        box = np.int8(cv2.boxPoints(rect))
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        return Piece(
            piece_id=get_pieceId(piece_index),
            bw_thresh_fixed=bw_img_thresh_fixed,
            bw_thresh_otsu=bw_img_thresh_otsu,
            contour=contour,
            area=area,
            perimeter=perimeter,
            bounding_rect=(x, y, w, h),
            min_area_rect=rect,
            box_points=box,
            hull=hull,
            solidity=float(area / hull_area) if hull_area > 0 else 0,
            aspect_ratio=float(w / h) if h > 0 else 0,
            extent=float(area / (w * h)) if w * h > 0 else 0,
            edge_list=[],
        )

    pieces_otsu = [contour2piece(c, img_index) for c in cv2_contours_otsu]
    pieces_fixed = [contour2piece(c, img_index) for c in cv2_contours_fixed]

    def filter_pieces(pieces, min_perimeter, max_perimeter, min_area, max_area):
        return [
            p
            for p in pieces
            if p.aspect_ratio != 1
            and p.perimeter > min_perimeter
            and p.perimeter < max_perimeter
        ]

    pieces_otsu = filter_pieces(
        pieces_otsu, min_perimeter=100, max_perimeter=7000, min_area=20, max_area=8000
    )
    pieces_fixed = filter_pieces(
        pieces_fixed, min_perimeter=100, max_perimeter=8000, min_area=20, max_area=10000
    )

    img_otsu_full_contours = cv2.cvtColor(bw_img_thresh_otsu.copy(), cv2.COLOR_GRAY2BGR)
    img_fixed_full_contours = cv2.cvtColor(
        bw_img_thresh_fixed.copy(), cv2.COLOR_GRAY2BGR
    )

    if len(pieces_otsu) == 0 and len(pieces_fixed) == 0:
        print("#####################################################################")
        print(f"Image: {img_path}")
        print(f"num pieces otsu: {len(pieces_otsu)}")
        print(f"num pieces fixed: {len(pieces_fixed)}")
    full_pieces_fixed = pieces_fixed.copy()
    full_pieces_otsu = pieces_otsu.copy()

    # pieces_fixed = filter_pieces(pieces_fixed)
    # pieces_otsu = filter_pieces(pieces_otsu)
    # print(f"num pieces otsu: {len(pieces_otsu)}")
    # print(f"num pieces fixed: {len(pieces_fixed)}")

    try:
        assert len(pieces_fixed + pieces_otsu) > 0
    except:
        print("No pieces found!")
        for p in full_pieces_fixed + full_pieces_otsu:
            if p is None:
                continue
            print(
                f"Piece {p.piece_id}: Area={p.area}, Perimeter={p.perimeter}, BoundingRect={p.bounding_rect}, Solidity={p.solidity:.2f}, AspectRatio={p.aspect_ratio:.2f}, Extent={p.extent:.2f}"
            )
        list_images_pipeline["02b-full-contours-otsu"] = img_otsu_full_contours
        list_images_pipeline["02b-full-contours-fixed"] = img_fixed_full_contours
        print(f"Image: {img_path}")
        return None

    if DEBUG:
        for p in pieces_fixed + pieces_otsu:
            if p is None:
                continue
            print(
                f"Piece {p.piece_id}: Area={p.area}, Perimeter={p.perimeter}, BoundingRect={p.bounding_rect}, Solidity={p.solidity:.2f}, AspectRatio={p.aspect_ratio:.2f}, Extent={p.extent:.2f}"
            )

    img_otsu_contours = cv2.cvtColor(bw_img_thresh_otsu.copy(), cv2.COLOR_GRAY2BGR)
    img_fixed_contours = cv2.cvtColor(bw_img_thresh_fixed.copy(), cv2.COLOR_GRAY2BGR)

    for p in pieces_fixed:
        if p is None:
            continue
        cv2.drawContours(img_fixed_contours, [p.contour], -1, (0, 255, 0), 2)

    for p in pieces_otsu:
        if p is None:
            continue
        cv2.drawContours(img_otsu_contours, [p.contour], -1, (0, 255, 0), 2)

    # visualizations steps
    list_images_pipeline[f"03-contours-otsu--{len(cv2_contours_otsu)}"] = (
        img_otsu_contours
    )
    list_images_pipeline[f"04-contours-fixed--{len(cv2_contours_fixed)}"] = (
        img_fixed_contours
    )

    if method == "otsu" and len(pieces_otsu) > 0:
        assert type(pieces_otsu[0]) == Piece
        final_piece: Piece = pieces_otsu[0]
    elif method == "fixed" and len(pieces_fixed) > 0:
        assert type(pieces_fixed[0]) == Piece
        final_piece: Piece = pieces_fixed[0]
    else:
        final_piece: Piece = (pieces_fixed + pieces_otsu)[0]

    assert (
        type((pieces_fixed + pieces_otsu)[0]) == Piece
        or type((pieces_fixed + pieces_otsu)[0]) == None
    )

    plot_list_images(
        list_images_pipeline,
        path_save=path_plot_imagine_pipeline / f"{final_piece.piece_id}.png",
    )

    return final_piece

    # if len(contours_otsu) >= len(contours_fixed):
    #     contours, threshold_method = contours_otsu, "otsu"
    # else:
    #     contours, threshold_method = contours_fixed, "fixed"

    # if len(contours) < 5:  # fallback with smaller thresholds
    #     for min_area_try in [500, 200]:
    #         contours_otsu = get_valid_contours(img_thresh_otsu, min_area_try)
    #         contours_fixed = get_valid_contours(img_thresh_fixed, min_area_try)
    #         if len(contours_otsu) >= len(contours_fixed):
    #             contours = contours_otsu
    #         else:
    #             contours = contours_fixed
    #         if len(contours) >= 5:
    #             break

    # --- Analyze contours ---
    # pieces = []
    # for i, contour in enumerate(contours_fixed + contours_otsu):
    #     area = cv2.contourArea(contour)
    #     perimeter = cv2.arcLength(contour, True)
    #     x, y, w, h = cv2.boundingRect(contour)
    #     rect = cv2.minAreaRect(contour)
    #     box = np.int8(cv2.boxPoints(rect))
    #     hull = cv2.convexHull(contour)
    #     hull_area = cv2.contourArea(hull)

    #     pieces.append(Piece(
    #         piece_id=i2grid(i),
    #         area=area,
    #         perimeter=perimeter,
    #         bounding_rect=(x, y, w, h),
    #         min_area_rect=rect,
    #         box_points=box,
    #         hull=hull,
    #         solidity=float(area / hull_area) if hull_area > 0 else 0,
    #         aspect_ratio=float(w / h) if h > 0 else 0,
    #         extent=float(area / (w * h)) if w * h > 0 else 0,
    #         contour=contour[0]
    #     ))
    #     print(f"Piece {i}: Area={area}, Perimeter={perimeter}, BoundingRect=({x},{y},{w},{h}), Solidity={pieces[-1].solidity:.2f}, AspectRatio={pieces[-1].aspect_ratio:.2f}, Extent={pieces[-1].extent:.2f}")

    # return pieces[0]


if __name__ == "__main__":
    list_pieces = {}
    list_images_pieces = {}
    # --- Paths ---
    Path_org_images = Path("data")  # sure matches
    Path_org_images = Path("data/puzzle")
    Path_out_pieces = Path("results/01_pieces")
    Path_img_pieces_pipeline = Path("results/01_pieces_img")

    for path in [Path_out_pieces, Path_img_pieces_pipeline]:
        path.mkdir(parents=True, exist_ok=True)

    # --- Parse arguments ---
    parser = argparse.ArgumentParser(description="Detect puzzle pieces in images")
    parser.add_argument(
        "--data-folder", default=Path_org_images, help="Folder containing images"
    )
    parser.add_argument(
        "--output-folder", default=Path_out_pieces, help="Folder to save results"
    )
    parser.add_argument(
        "--output-img-folder",
        default=Path_img_pieces_pipeline,
        help="Folder to save images",
    )
    parser.add_argument("--image", help="Specific image to process (optional)")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    # --- Get sorted list of image paths to maintain creation order ---
    image_paths = sorted(args.data_folder.glob("*.JPG"))

    # --- Detect pieces ---
    with Pool(cpu_count()) as pool:
        # --- Create arguments with image index for coordinate system ---
        args_with_index = [
            (img_path, i, "otsu", args.output_img_folder)
            for i, img_path in enumerate(image_paths)
        ]
        results: list[Piece | None] = pool.starmap(
            detectPieceFromImagine, args_with_index
        )

    dict_pieces: dict[str, Piece] = {}
    for p in results:
        if p is None:
            continue
        dict_pieces[p.piece_id] = p

    with open(args.output_folder / "dict_pieces.pkl", "wb") as f:
        pickle.dump(dict_pieces, f)
