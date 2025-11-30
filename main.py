from nb_pieces import PipelineProcessImg, PipelineProcessImgParams, Piece, EdgeType, EdgeColor
from glob import glob
from tqdm import tqdm
import multiprocessing as mp
from pickle import dump
import os


def extract_piece_from_img(str_img: str, path_output:str = "./data/pieces", force:bool=False) -> None:
    name_str_img = str_img.split("/")[-1].split(".")[0]
    path_output_img = os.path.join(path_output, f"{name_str_img}.pkl")

    #if os.path.exists(path_output_img) and not force:
    #    return

    params = PipelineProcessImgParams()
    piece = PipelineProcessImg(str_img, params, PLOT=False)

    with open(path_output_img, "wb") as f:
        dump(piece, f)

def main():
    str_imgs = glob("./data/*.JPG")
    # multicore process all str_imgs with bar progression
    with tqdm(total=len(str_imgs), desc="Processing images") as pbar:
        with mp.Pool(mp.cpu_count()) as pool:
            pool.map(extract_piece_from_img, str_imgs)
            pbar.update(len(str_imgs))

if __name__ == "__main__":
    main()