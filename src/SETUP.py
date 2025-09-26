from pathlib import Path

path_data_puzzle = Path("data/puzzle")
path_data_tests = Path("data/tests")

path_res_pieces = Path("results/pieces")
path_res_edges = Path("results/edges")
path_res_matches = Path("results/matches")

path_res_img_debug = Path("debug")

for path in [path_data_puzzle, path_data_tests, path_res_pieces, path_res_edges, path_res_matches, path_res_img_debug]:
    path.mkdir(parents=True, exist_ok=True)






