"""Main pipeline script for processing puzzle pieces and finding matches."""

import json
import multiprocessing as mp
import os
from functools import partial
from pathlib import Path
from pickle import dump, load

from tqdm import tqdm

from pipeline.matching import PipelineParamsMatching, find_match
from pipeline.models import ContourError, Match, Piece
from pipeline.preprocessing import PipelineProcessImgParams, process_image


def extract_piece_from_img(
    str_img: str | Path,
    path_output: str | Path = "./data/pieces",
    force: bool = False,
    log_file: str | Path = "./data/processing_errors.jsonl",
) -> None:
    """
    Extract puzzle piece from image and save to pickle file.
    
    Args:
        str_img: Path to input image
        path_output: Directory to save extracted pieces
        force: Whether to overwrite existing files
        log_file: Path to error log file
    """
    str_img = Path(str_img)
    path_output = Path(path_output)
    log_file = Path(log_file)
    
    if not str_img.exists():
        error_entry = {
            "image_path": str(str_img),
            "error_type": "FileNotFoundError",
            "message": f"Image file does not exist: {str_img}",
        }
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a") as f:
            f.write(json.dumps(error_entry) + "\n")
        return
    
    name_str_img = str_img.stem
    path_output_img = path_output / f"{name_str_img}.pkl"
    
    if path_output_img.exists() and not force:
        return
    
    try:
        params = PipelineProcessImgParams()
        piece = process_image(str_img, params, plot=False)
        
        # Validate piece has expected structure
        if not piece.edge_list or len(piece.edge_list) != 4:
            raise ValueError(
                f"Expected 4 edges, got {len(piece.edge_list) if piece.edge_list else 0}"
            )
        
        path_output.mkdir(parents=True, exist_ok=True)
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
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a") as f:
            f.write(json.dumps(error_entry) + "\n")
    except Exception as e:
        # Log any other unexpected errors
        error_entry = {
            "image_path": str(str_img),
            "error_type": type(e).__name__,
            "message": str(e),
        }
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a") as f:
            f.write(json.dumps(error_entry) + "\n")


def match_pieces(
    piece1_path: Path | str,
    piece2_path: Path | str,
    params: PipelineParamsMatching,
    path_output: Path | str | None = None,
) -> Match:
    """
    Match two puzzle pieces and optionally save result.
    
    Args:
        piece1_path: Path to first piece pickle file
        piece2_path: Path to second piece pickle file
        params: Matching parameters
        path_output: Directory to save match results (optional)
    
    Returns:
        Match object
    """
    piece1_path = Path(piece1_path)
    piece2_path = Path(piece2_path)
    
    # Validate files exist
    if not piece1_path.exists():
        raise FileNotFoundError(f"Piece file not found: {piece1_path}")
    if not piece2_path.exists():
        raise FileNotFoundError(f"Piece file not found: {piece2_path}")
    
    # Load pieces
    try:
        with open(piece1_path, "rb") as f:
            piece1: Piece = load(f)
    except Exception as e:
        raise ValueError(f"Failed to load piece from {piece1_path}: {e}") from e
    
    try:
        with open(piece2_path, "rb") as f:
            piece2: Piece = load(f)
    except Exception as e:
        raise ValueError(f"Failed to load piece from {piece2_path}: {e}") from e
    
    # Find match
    match = find_match(piece1, piece2, params, verbose=False)
    
    # Save match if output path provided
    if path_output is not None:
        path_output = Path(path_output)
        path_output.mkdir(parents=True, exist_ok=True)
        out_path = path_output / f"match_{piece1.piece_id:04d}_{piece2.piece_id:04d}.pkl"
        with open(out_path, "wb") as f:
            dump(match, f)
    
    return match


def main():
    """Main function to run the complete pipeline."""
    ###########################################################################
    # Set Paths - img, pieces, log ############################################
    ###########################################################################
    # -- Paths --
    _puzzle_img_dir = Path("./data")
    puzzle_img_paths = sorted(_puzzle_img_dir.glob("*.JPG"))
    
    if not puzzle_img_paths:
        print(f"Warning: No JPG images found in {_puzzle_img_dir}")
        return
    
    print(f"Found {len(puzzle_img_paths)} images to process")
    
    # -- Pieces --
    puzzle_pieces_pkl_out = Path("./data/pieces")
    puzzle_pieces_pkl_out.mkdir(parents=True, exist_ok=True)
    
    # -- Matching --
    puzzle_matches_pkl_out = Path("./data/matches")
    puzzle_matches_pkl_out.mkdir(parents=True, exist_ok=True)
    
    # -- Log --
    log_file = Path("./data/processing_errors.jsonl")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    if log_file.exists():
        log_file.unlink()  # Clear previous log
    
    ###########################################################################
    # Piece extraction ########################################################
    ###########################################################################
    force = False  # Set to True to force overwrite existing files
    
    _extract_func = partial(
        extract_piece_from_img,
        path_output=puzzle_pieces_pkl_out,
        force=force,
        log_file=log_file,
    )
    
    print("Processing images to extract puzzle pieces...")
    with tqdm(total=len(puzzle_img_paths), desc="Processing images", unit="img") as pbar:
        with mp.Pool(mp.cpu_count()) as pool:
            for _ in pool.imap(_extract_func, puzzle_img_paths):
                pbar.update(1)
    
    ###########################################################################
    # Match pieces #############################################################
    ###########################################################################
    from itertools import combinations
    
    pkl_pieces_files = sorted(puzzle_pieces_pkl_out.glob("*.pkl"))
    
    if not pkl_pieces_files:
        print("Warning: No piece files found. Skipping matching step.")
        return
    
    print(f"Found {len(pkl_pieces_files)} pieces to match")
    combs = list(combinations(pkl_pieces_files, 2))
    print(f"Computing {len(combs)} pairwise matches...")
    
    _params = PipelineParamsMatching()
    _match_function = partial(
        match_pieces,
        params=_params,
        path_output=puzzle_matches_pkl_out,
    )
    
    print("Matching puzzle pieces...")
    with tqdm(total=len(combs), desc="Matching pieces", unit="pair") as pbar:
        with mp.Pool(mp.cpu_count()) as pool:
            for _ in pool.starmap(_match_function, combs):
                pbar.update(1)
    
    # Summary
    match_files = list(puzzle_matches_pkl_out.glob("*.pkl"))
    print(f"\nPipeline completed successfully!")
    print(f"  - Processed {len(puzzle_img_paths)} images")
    print(f"  - Extracted {len(pkl_pieces_files)} pieces")
    print(f"  - Computed {len(match_files)} matches")
    
    if log_file.exists() and log_file.stat().st_size > 0:
        print(f"  - Errors logged to {log_file}")


if __name__ == "__main__":
    main()
