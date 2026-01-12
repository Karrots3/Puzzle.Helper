import json
import multiprocessing as mp
import os
from functools import partial
from pathlib import Path
from pickle import dump

from tqdm import tqdm

from match import PipelineMatchings, PipelineParamsMatching
from nb_pieces import PipelineProcessImg, PipelineProcessImgParams
from src.classes import ContourError


def extract_piece_from_img(
    str_img: str | Path,
    path_output: str = "./data/pieces",
    force: bool = False,
    log_file: str = "./data/processing_errors.jsonl",
) -> None:
    name_str_img = str(str_img).split("/")[-1].split(".")[0]
    path_output_img = os.path.join(path_output, f"{name_str_img}.pkl")

    if os.path.exists(path_output_img) and not force:
        return

    try:
        params = PipelineProcessImgParams()
        piece = PipelineProcessImg(str_img, params, PLOT=False)

        with open(path_output_img, "wb") as f:
            dump(piece, f)
    except ContourError as e:
        # Log the error with image path and details
        error_entry = {
            "image_path": str(str_img),
            "error_type": "ContourError",
            "message": e.message,
            "num_contours": e.num_contours,
            "contour_areas": e.contour_areas,
        }
        # Append to log file (JSONL format - one JSON object per line)
        # Using 'a' mode with proper locking would be ideal, but for simplicity
        # we'll use append mode which is generally safe for small writes
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(error_entry) + "\n")
        # Don't re-raise - continue processing other images
    except Exception as e:
        # Log any other unexpected errors
        error_entry = {
            "image_path": str(str_img),
            "error_type": type(e).__name__,
            "message": str(e),
        }
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(error_entry) + "\n")
        # Don't re-raise - continue processing other images


def main():
    ###########################################################################
    # Set Paths - img, pieces, log ############################################
    ###########################################################################
    # -- Paths --
    _puzzle_img_dir = Path("./data")
    puzzle_img_paths = _puzzle_img_dir.glob("*.JPG")

    # -- Pieces --
    puzzle_pieces_pkl_out = Path("./data/pieces")
    puzzle_pieces_pkl_out.mkdir(parents=True, exist_ok=True)

    # -- Matching --
    puzzle_matches_pkl_out = Path("./data/matches")
    puzzle_matches_pkl_out.mkdir(parents=True, exist_ok=True)

    # -- Log --
    log_file = "./data/processing_errors.jsonl"

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if log_path.exists():
        log_path.unlink()  # Clear previous log

    ###########################################################################
    # Piece extraction ########################################################
    ###########################################################################
    force = False  # Set to True to force overwrite existing files
    puzzle_img_paths_list = list(puzzle_img_paths)

    _extract_func = partial(
        extract_piece_from_img,
        path_output=str(puzzle_pieces_pkl_out),
        force=force,
        log_file=log_file,
    )

    with tqdm(total=len(puzzle_img_paths_list), desc="Processing images") as pbar:
        with mp.Pool(mp.cpu_count()) as pool:
            for _ in pool.imap(_extract_func, puzzle_img_paths_list):
                pbar.update(1)

    ###########################################################################
    # Match pieces #############################################################
    ###########################################################################
    from itertools import combinations

    puzzle_matches_pkl_out = Path("./data/matches")
    puzzle_matches_pkl_out.mkdir(parents=True, exist_ok=True)

    pkl_pieces_files = list(puzzle_pieces_pkl_out.glob("*.pkl"))
    combs = list(combinations(pkl_pieces_files, 2))

    _params = PipelineParamsMatching()
    _match_function = partial(
        PipelineMatchings,
        params=_params,
        path_output=puzzle_matches_pkl_out,
        PLOT=False,
        VERBOSE=False,
    )

    with tqdm(total=len(combs), desc="Matching pieces") as pbar:
        with mp.Pool(mp.cpu_count()) as pool:
            for _ in pool.starmap(_match_function, combs):
                pbar.update(1)


if __name__ == "__main__":
    main()

