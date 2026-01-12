"""Main pipeline script for processing puzzle pieces and finding matches."""

import json
import multiprocessing as mp
import os
from functools import partial
from pathlib import Path
from pickle import dump, load

from tqdm import tqdm

from pipeline.matching import PipelineParamsMatching, find_match
from pipeline.models import ContourError, Piece
from pipeline.preprocessing import PipelineProcessImgParams, process_image


def extract_piece_from_img(
    str_img: str | Path,
    path_output: str = "./data/pieces",
    force: bool = False,
    log_file: str = "./data/processing_errors.jsonl",
) -> None:
    """
    Extract puzzle piece from image and save to pickle file.
    
    Args:
        str_img: Path to input image
        path_output: Directory to save extracted pieces
        force: Whether to overwrite existing files
        log_file: Path to error log file
    """
    name_str_img = str(str_img).split("/")[-1].split(".")[0]
    path_output_img = os.path.join(path_output, f"{name_str_img}.pkl")
    
    if os.path.exists(path_output_img) and not force:
        return
    
    try:
        params = PipelineProcessImgParams()
        piece = process_image(str_img, params, plot=False)
        
        os.makedirs(path_output, exist_ok=True)
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
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(error_entry) + "\n")
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


def match_pieces(
    piece1_path: Path | str,
    piece2_path: Path | str,
    params: PipelineParamsMatching,
    path_output: Path | str | None = None,
) -> None:
    """
    Match two puzzle pieces and optionally save result.
    
    Args:
        piece1_path: Path to first piece pickle file
        piece2_path: Path to second piece pickle file
        params: Matching parameters
        path_output: Directory to save match results (optional)
    """
    # Load pieces
    with open(piece1_path, "rb") as f:
        piece1: Piece = load(f)
    
    with open(piece2_path, "rb") as f:
        piece2: Piece = load(f)
    
    # Find match
    match = find_match(piece1, piece2, params, verbose=False)
    
    # Save match if output path provided
    if path_output is not None:
        os.makedirs(path_output, exist_ok=True)
        out_path = (
            f"{path_output}/match_{piece1.piece_id:04d}_{piece2.piece_id:04d}.pkl"
        )
        with open(out_path, "wb") as f:
            dump(match, f)


def main():
    """Main function to run the complete pipeline."""
    ###########################################################################
    # Set Paths - img, pieces, log ############################################
    ###########################################################################
    # -- Paths --
    _puzzle_img_dir = Path("./data")
    puzzle_img_paths = list(_puzzle_img_dir.glob("*.JPG"))
    
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
    
    _extract_func = partial(
        extract_piece_from_img,
        path_output=str(puzzle_pieces_pkl_out),
        force=force,
        log_file=log_file,
    )
    
    print("Processing images to extract puzzle pieces...")
    with tqdm(total=len(puzzle_img_paths), desc="Processing images") as pbar:
        with mp.Pool(mp.cpu_count()) as pool:
            for _ in pool.imap(_extract_func, puzzle_img_paths):
                pbar.update(1)
    
    ###########################################################################
    # Match pieces #############################################################
    ###########################################################################
    from itertools import combinations
    
    pkl_pieces_files = list(puzzle_pieces_pkl_out.glob("*.pkl"))
    combs = list(combinations(pkl_pieces_files, 2))
    
    _params = PipelineParamsMatching()
    _match_function = partial(
        match_pieces,
        params=_params,
        path_output=puzzle_matches_pkl_out,
    )
    
    print("Matching puzzle pieces...")
    with tqdm(total=len(combs), desc="Matching pieces") as pbar:
        with mp.Pool(mp.cpu_count()) as pool:
            for _ in pool.starmap(_match_function, combs):
                pbar.update(1)
    
    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
