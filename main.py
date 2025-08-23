#!/usr/bin/env python3
import argparse
import os
from re import A
import sys
import logging
from pathlib import Path
import cv2
import numpy as np
from scipy.signal import find_peaks
import itertools
import math
from matplotlib import pyplot as plt

# TYPE_OF_IMG = 

class LoopingList(list):
    def __getitem__(self, i):
        if isinstance(i, int):
            return super().__getitem__(i % len(self))
        else:
            return super().__getitem__(i)

class Edge():
    def __init__(self, contour: np.ndarray, sign: int):
        self.contour = contour
        self.sign = sign
        self.color = "blue" if sign == 1 else "red" if sign == -1 else "green"
        
class Image():
    def __init__(self, 
                 img_id: int,
                 img: np.ndarray, 
                 contour: np.ndarray,
                 peaks_idx: list[int],
                 peaks: np.ndarray,
                 edges: list[np.ndarray],
                 edges_norm: list[Edge]
                 ):
        super().__init__()  # Initialize the parent list class
        self.img_id = img_id
        self.name = f"{img_id:04d}"
        self.img = img
        self.contour = contour
        self.peaks_idx = peaks_idx
        self.peaks = peaks
        self.edges = edges
        self.edges_norm = edges_norm
    
    def __str__(self):
        return f"Image(name={self.name}, shape={self.img.shape}, peaks={len(self.peaks_idx)}, edges={len(self.edges)})"
    
    def __repr__(self):
        return self.__str__()
        

def trim_image(img: np.ndarray, radius: int = 100) -> np.ndarray:
    h, w = img.shape[:2]
    center_x, center_y = w // 2, h // 2

    perc = radius / 100
    sz = int(perc * min(h, w))
    half_sz = sz // 2
    x1 = max(0, center_x - half_sz)
    y1 = max(0, center_y - half_sz)
    x2 = min(w, center_x + half_sz)
    y2 = min(h, center_y + half_sz)

    img = img[y1:y2, x1:x2]

    return img


# normalize the edge

## Other things plotting

def color_string_to_tuple(color_string: str) -> tuple[int, int, int]:
    match color_string:
        case "black":
            return (0, 0, 0)
        case "white":
            return (255, 255, 255)
        case "red":
            return (0, 0, 255)
        case "green":
            return (0, 255, 0)
        case "blue":
            return (255, 0, 0)
        case "yellow":
            return (0, 255, 255)
        case _:
            raise ValueError(f"Invalid color string: {color_string}")

COLOR_CONTOUR = color_string_to_tuple("green")
COLOR_EDGES = color_string_to_tuple("blue")
COLOR_LINES = color_string_to_tuple("blue")
COLOR_PEAKS = color_string_to_tuple("red")

THICKNESS_CONTOUR = 3
THICKNESS_LINES = 8
THICKNESS_PEAKS = 8

RADIUS_PEAKS = 15


################################################################################
# Plotting functions
################################################################################
def plot_edges(edges_norm: list[Edge], id_image: int):  
    plt.figure(figsize=(10, 8), dpi=400, tight_layout=True)
    plt.title(f"Normalized edges {id_image}")
    plt.ylim(-400,400)
    for edge in edges_norm:
        plt.scatter(edge.contour[:, 0, 0], edge.contour[:, 0, 1], color=edge.color, s=1)
    plt.savefig(f"data/results/plt_edge_{id_image}.png", bbox_inches='tight')
    plt.close()

def plot_subplots_images(list_images: list[np.ndarray], list_contours: list[np.ndarray], list_peaks: list[np.ndarray], list_edges: list[list[np.ndarray]], n_cols: int, only_contour = False) -> None:
    n_rows = len(list_images) // n_cols
    
    # Create a combined image for display
    max_height = max(img.shape[0] for img in list_images)
    max_width = max(img.shape[1] for img in list_images)
    
    # Create a grid layout
    combined_img = np.zeros((max_height * n_rows, max_width * n_cols, 3), dtype=np.uint8)
    
    for i, (img, contour, peaks, edges) in enumerate(zip(list_images, list_contours, list_peaks, list_edges)):
        row = i // n_cols
        col = i % n_cols
        
        # Background imagine
        if only_contour:
            # Create a black background for contour only
            display_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        else:
            # Convert grayscale to BGR if needed
            if len(img.shape) == 2:
                display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                display_img = img.copy()
        
        # Draw contour
        if len(contour) > 0:
            cv2.drawContours(
                display_img,
                contour,
                -1,
                COLOR_CONTOUR,
                THICKNESS_CONTOUR
            )

        # Draw peaks
        if len(peaks) > 0:
            for peak in peaks:
                peak_coords = (int(peak[0]), int(peak[1]))
                cv2.circle(display_img,center=peak_coords, radius=RADIUS_PEAKS, color=COLOR_PEAKS, thickness=THICKNESS_PEAKS)

        # Draw edge lines
        if len(edges) > 0:
            for edge in edges:
                if len(edge) == 0:
                    continue
                cv2.line(display_img, (edge[0][0], edge[0][1]), (edge[-1][0], edge[-1][1]), COLOR_LINES, THICKNESS_LINES)
                cv2.drawContours(display_img, [edge], -1, COLOR_EDGES, THICKNESS_LINES)
                
        
        # Place the image in the grid
        y_start = row * max_height
        y_end = y_start + img.shape[0]
        x_start = col * max_width
        x_end = x_start + img.shape[1]
        
        combined_img[y_start:y_end, x_start:x_end] = display_img
    
    # Save the combined image
    cv2.imwrite(f"data/results/00_subplots{"_cont" if only_contour else ""}.png", combined_img)
    
    # Display the combined image
    # cv2.imshow("Subplots", combined_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def plot_images(list_images: list[Image], n_cols: int, only_contour: bool = False) -> None:
    #plot_subplots_images(
    #    [i.img for i in list_images],
    #    [i.contour for i in list_images],
    #    [i.peaks for i in list_images],
    #    [i.edges for i in list_images],
    #    n_cols=len(list_images),
    #    only_contour=only_contour
    #)p
    n_rows = len(list_images) // n_cols
    
    # Create a combined image for display
    max_height = max(img.img.shape[0] for img in list_images)
    max_width = max(img.img.shape[1] for img in list_images)
    
    # Create a grid layout
    combined_img = np.zeros((max_height * n_rows, max_width * n_cols, 3), dtype=np.uint8)
    
    for i, this_img in enumerate(list_images):
        img = this_img.img
        contour = this_img.contour
        peaks = this_img.peaks
        edges = this_img.edges
        edges_norm = this_img.edges_norm

        row = i // n_cols
        col = i % n_cols

        # Background imagine
        if only_contour:
            # Create a black background for contour only
            display_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        else:
            # Convert grayscale to BGR if needed
            if len(img.shape) == 2:
                display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                display_img = img.copy()
        
        # Draw contour
        if len(contour) > 0:
            cv2.drawContours(
                display_img,
                contour,
                -1,
                COLOR_CONTOUR,
                THICKNESS_CONTOUR
            )

        # Draw peaks
        if len(peaks) > 0:
            for peak in peaks:
                peak_coords = (int(peak[0]), int(peak[1]))
                cv2.circle(display_img,center=peak_coords, radius=RADIUS_PEAKS, color=COLOR_PEAKS, thickness=THICKNESS_PEAKS)

        # Draw edge lines
        if len(edges) > 0:
            for edge in edges:
                if len(edge) == 0:
                    continue
                cv2.line(display_img, (edge[0][0], edge[0][1]), (edge[-1][0], edge[-1][1]), COLOR_LINES, THICKNESS_LINES)
                cv2.drawContours(display_img, [edge], -1, COLOR_EDGES, THICKNESS_LINES)
                
        
        # Place the image in the grid
        y_start = row * max_height
        y_end = y_start + img.shape[0]
        x_start = col * max_width
        x_end = x_start + img.shape[1]
        
        combined_img[x_start:x_end, y_start:y_end] = display_img
    
    # Save the combined image
    cv2.imwrite(f"data/results/00_subplots{"_cont" if only_contour else ""}.png", combined_img)
    
    # Display the combined image
    # cv2.imshow("Subplots", combined_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

################################################################################
# Preprocessing functions
################################################################################
def preprocess_images():
    out_path = Path("data/results")
    out_path.mkdir(parents=True, exist_ok=True)
    list_files = [Path(f) for f in Path("data").glob("*.JPG")]
    list_files.sort()
    
    list_Images: list[Image] = []
    list_photos = []
    list_contours = []
    list_peaks = []
    list_edges = []
    list_edges_norm = []
    for count_image, file in enumerate(list_files):
        img_name = file.stem
        img_id = int(img_name)
        img: np.ndarray = cv2.imread(str(file)) # 3000,4000,3
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 3000,4000
        img = trim_image(img,60)
        #list_photos.append(img)
        #list_contours.append([])
        #list_peaks.append([])

        #contour, _= cv2.findContours(img,cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        #contour = [c for c in contour if 100 < cv2.contourArea(c) < 1e6]
        #list_contours.append(contour)
        #list_photos.append(img)

        # # OTSU
        # _,th3 = cv2.threshold(img,110,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # contour, _= cv2.findContours(th3,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # contour = [c for c in contour if 100 < cv2.contourArea(c) < 1e6]
        # list_contours.append(contour)
        # list_photos.append(th3)
        
        # TODO: check in the array for actual values
        _,img= cv2.threshold(img,110,255,cv2.THRESH_BINARY)# + cv2.THRESH_OTSU)
        contours, _= cv2.findContours(img,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = [c for c in contours if 100 < cv2.contourArea(c) < 1e6]
        assert len(contours) == 1
        contour: np.ndarray = contours[0]

        list_photos.append(img)
        list_contours.append(contour)
        list_peaks.append([])
        list_edges.append([])
        list_edges_norm.append([])

        
        (cx, cy), cr = cv2.minEnclosingCircle(contour)
        centered_contour = contour - np.array([cx, cy])

        # ensure peaks are not at start or end of the distances array
        distances = np.sum(centered_contour**2, axis=2)[:, 0]
        distance_offset = np.argmin(distances)
        shifted_distances = np.concatenate([distances[distance_offset:], distances[:distance_offset]])

        # find peak distances
        PROMINENCE = 1500
        prominence = PROMINENCE
        counter = 0
        while True and counter < 10:
            counter += 1
            peak_indices = [(distance_idx + distance_offset) % len(distances) for distance_idx in find_peaks(shifted_distances, prominence=prominence)[0]]
            #print(f"Prominence: {prominence}, number of peaks: {len(peak_indices)}")
            if len(peak_indices) < 8:
                prominence = prominence *0.9
            elif len(peak_indices) > 20:
                prominence = prominence * 1.1
            else:
                break

        peak_indices.sort()
        list_photos.append(img)
        list_contours.append(contour)
        list_peaks.append(contour[peak_indices,0,:])
        list_edges.append([])
        list_edges_norm.append([])


    

        # TODO plot the peaks and fix prominance to reduce number of peaks

        # select the one that better approximate a rectangle
        def compute_rectangle_error(indices):
            # get coordinates of corners
            corners = LoopingList(np.take(contour, sorted(list(indices)), axis=0)[:, 0, :])
            # compute the side lengths and diagonal lengths
            lengths = [math.sqrt(np.sum((corners[i0] - corners[i1])**2)) for i0, i1 in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]]
            def f_error(a, b):
                return abs(b - a) / (a + b)
            return sum([f_error(lengths[i], lengths[j]) for i, j in [(0, 2), (1, 3), (4, 5), (0, 1)]])

        # form a good rectangle with peak indices
        rectangles = []  # list of (score, [indices])
        for indices in itertools.combinations(peak_indices, 4):
            error = compute_rectangle_error(indices)
            rectangles.append((error, indices))

        error, peak_indices = sorted(rectangles)[0]
        assert len(peak_indices) == 4

        list_photos.append(img)
        list_contours.append(contour)
        list_peaks.append(contour[peak_indices,0,:])
        list_edges.append([])
        list_edges_norm.append([])
        
        # Edges
        def extract_edges(contour, indices):
            e_idx = [
                range(indices[0], indices[1]),
                range(indices[1], indices[2]),
                range(indices[2], indices[3]),
                list(range(indices[3], len(contour))) + list(range(0, indices[0])),
            ]

            edges = [
                contour[e_idx[0],0,:],
                contour[e_idx[1],0,:],
                contour[e_idx[2],0,:],
                contour[e_idx[3],0,:],
            ]

            return edges

        edges = extract_edges(contour, peak_indices)
        
        list_photos.append(img)
        list_contours.append(contour)
        list_peaks.append(contour[peak_indices,0,:])
        list_edges.append(edges)
        list_edges_norm.append([])


        # Normalized edges
        idx = [
            [peak_indices[0], peak_indices[1]],
            [peak_indices[1], peak_indices[2]],
            [peak_indices[2], peak_indices[3]],
            [peak_indices[3], peak_indices[0]],
        ]

        edges_norm:list[Edge] = []
        for c, (idx0, idx1) in enumerate(idx):
            p0 = contour[idx0,0,:]
            p1 = contour[idx1,0,:]
            
            # Calculate the direction vector and straight length
            dx, dy = p1 - p0
            straight_length = math.sqrt(dx**2 + dy**2)
            
            # Calculate the angle to rotate the edge to horizontal
            angle_degrees = math.degrees(math.atan2(dy, dx))
            
            # Create transform to normalize: first point at (0, 0), last point at (X, 0)
            center = p0 
            matrix = cv2.getRotationMatrix2D(center.astype(np.float32), angle_degrees, 1)
            translate = (0, 0) - center
            transform = (matrix, translate)
            
            # Apply transform to the entire piece contour
            matrix, translate = transform
            normalized_piece_contour = cv2.transform(contour, matrix) + translate
            
            # Extract just the edge contour
            if idx1 + 1 > idx0:
                normalized_edge_contour = normalized_piece_contour[idx0:idx1 + 1]
            else:
                normalized_edge_contour = np.concatenate([normalized_piece_contour[idx0:], normalized_piece_contour[:idx1 + 1]])
            
            # Transform the piece center
            matrix, translate = transform
            normalized_piece_center = (cv2.transform(np.array([[center]]), matrix) + translate)[0, 0]
            
            # Compute the sign of the edge (male/female/flat)
            heights = normalized_edge_contour[:, 0, 1]
            if np.max(np.abs(heights)) < 10:
                sign = 0
            #elif np.max(np.abs(heights)) > 10:
            else:
                sign = -1 if np.max(heights) > -np.min(heights) else 1
            
            # For male contours (sign == 1), rotate by 180° for easier matching with female contours
            #if sign == 1:
            #    angle_degrees += 180
            #    # Replace get_contour_transform with actual implementation
            #    center_point = contour[idx1,0,:]
            #    matrix = cv2.getRotationMatrix2D(center_point.astype(np.float32), angle_degrees, 1)
            #    translate = (0, 0) - center_point
            #    transform = (matrix, translate)
            #    
            #    # Replace transform_contour with actual implementation
            #    matrix, translate = transform
            #    normalized_piece_contour = cv2.transform(contour, matrix) + translate
            #    
            #    # Replace transform_point with actual implementation
            #    normalized_piece_center = (cv2.transform(np.array([[center_point]]), matrix) + translate)[0, 0]
            if sign == 1:
                normalized_edge_contour[:,:,1] = -normalized_edge_contour[:,:,1]

            edges_norm.append(Edge(normalized_edge_contour, sign))

        plot_edges(edges_norm = edges_norm, id_image=count_image)

        # END PREPROCESSING
        new_image = Image(img_id ,img, contour, peak_indices, contour[peak_indices,0,:], edges, edges_norm)
        print(new_image)
        list_Images.append(new_image)

    plot_subplots_images(list_photos,list_contours,list_peaks,list_edges,n_cols=len(list_photos)//len(list_files), only_contour=True)
    plot_subplots_images(list_photos,list_contours,list_peaks,list_edges,n_cols=len(list_photos)//len(list_files), only_contour=False)

    list_Images.sort(key=lambda x: x.img_id)
    return list_Images

################################################################################
# Matching functions
################################################################################
def plot_scores(scores: np.ndarray):
    scores[scores == 0] = np.nan
    scores = np.log10(scores)
    scores[scores > 1.8] = np.nan
    
    plt.figure(figsize=(10, 8), dpi=400, tight_layout=True)
    plt.title("Scores")
    plt.xlabel("Piece 0")
    plt.ylabel("Piece 1")
    plt.xticks(range(len(scores)))
    plt.yticks(range(len(scores)))

    # plot the score matrix
    plt.imshow(scores, cmap="viridis")
    plt.colorbar(label="Score")

    plt.savefig(f"data/results/08_scores.png", bbox_inches='tight')
    plt.close()




def score_match(i: int, piece0: Image, j: int, piece1: Image) -> tuple[int, int, float]:
    """
    Improved edge matching function for puzzle pieces.
    Returns the best matching score between any edge of piece0 and piece1.
    """
    
    if i >= j:
        return (i, j, 0)

    best_score = float('inf')
    
    for e0_idx, e0 in enumerate(piece0.edges_norm):
        if e0.sign == 0:  # Skip flat edges
            continue
            
        for e1_idx, e1 in enumerate(piece1.edges_norm):
            if e1.sign == 0:  # Skip flat edges
                continue
            #if e0.sign == e1.sign:  # Same sign edges don't match
            #    continue
            
            # Make copies to avoid modifying original data
            contour0 = e0.contour.copy()[:, 0, :]  # Shape: (N, 2)
            contour1 = e1.contour.copy()[:, 0, :]  # Shape: (M, 2)

            
            # Remove edge points to avoid boundary artifacts
            trim_points = min(10, len(contour0)//4, len(contour1)//4)
            if len(contour0) > 2 * trim_points:
                contour0 = contour0[trim_points:-trim_points]
            if len(contour1) > 2 * trim_points:
                contour1 = contour1[trim_points:-trim_points]
            
            # remove first and last 20 points
            contour0 = contour0[30:-30]
            contour1 = contour1[30:-30]

            # Try both orientations of the second contour
            for flip in [False, True]:
                c1 = contour1[::-1] if flip else contour1
                
                # Normalize both contours to same length using interpolation
                target_length = min(len(contour0), len(c1), 100)  # Cap at 100 points
                
                # Resample contours to same length
                def resample_contour(contour, target_len):
                    if len(contour) <= 1:
                        return contour
                    
                    # Create parameter array for original contour
                    t_orig = np.linspace(0, 1, len(contour))
                    t_new = np.linspace(0, 1, target_len)
                    
                    # Interpolate x and y coordinates separately
                    x_new = np.interp(t_new, t_orig, contour[:, 0])
                    y_new = np.interp(t_new, t_orig, contour[:, 1])
                    
                    return np.column_stack([x_new, y_new])
                
                c0_resampled = resample_contour(contour0, target_length)
                c1_resampled = resample_contour(c1, target_length)
                
                if len(c0_resampled) < 2 or len(c1_resampled) < 2:
                    continue
                
                # Normalize position: align start points
                c0_normalized = c0_resampled - c0_resampled[0]
                c1_normalized = c1_resampled - c1_resampled[0]
                
                # Normalize scale: make end-to-end distance the same
                c0_length = np.linalg.norm(c0_normalized[-1] - c0_normalized[0])
                c1_length = np.linalg.norm(c1_normalized[-1] - c1_normalized[0])
                
                if c0_length > 0 and c1_length > 0:
                    c1_normalized = c1_normalized * (c0_length / c1_length)
                
                # Align end points by rotating c1
                if np.linalg.norm(c0_normalized[-1]) > 0:
                    # Calculate rotation angle to align end points
                    v0 = c0_normalized[-1]
                    v1 = c1_normalized[-1]
                    
                    angle0 = np.arctan2(v0[1], v0[0])
                    angle1 = np.arctan2(v1[1], v1[0])
                    rotation_angle = angle0 - angle1
                    
                    # Apply rotation to c1
                    cos_a, sin_a = np.cos(rotation_angle), np.sin(rotation_angle)
                    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                    c1_normalized = np.dot(c1_normalized, rotation_matrix.T)
                
                # For male edges (positive sign), flip vertically to match with female
                if e0.sign == 1:
                    c0_normalized[:, 1] = -c0_normalized[:, 1]
                if e1.sign == 1:
                    c1_normalized[:, 1] = -c1_normalized[:, 1]
                
                # Calculate the matching score
                # Use perpendicular distance as the primary metric
                diff = c0_normalized - c1_normalized
                
                # Weight the middle points more heavily (they're more characteristic)
                weights = np.exp(-((np.arange(len(diff)) - len(diff)/2) / (len(diff)/4))**2)
                weights = weights / np.sum(weights)
                
                # Calculate weighted RMS distance
                distances = np.linalg.norm(diff, axis=1)
                score = np.sqrt(np.sum(weights * distances**2))
                
                # Penalize large endpoint misalignment
                endpoint_penalty = np.linalg.norm(c0_normalized[-1] - c1_normalized[-1]) * 2
                score += endpoint_penalty
                
                best_score = min(best_score, score)
                
                # Debug visualization for specific pair
                if ((i == 4 and j == 5) or (i == 5 and j == 6)) and score == best_score:
                    plt.figure(figsize=(12, 8))
                    plt.subplot(1, 2, 1)
                    plt.plot(c0_normalized[:, 0], c0_normalized[:, 1], 'b-', label=f'Piece {i} edge {e0_idx}', linewidth=2)
                    plt.plot(c1_normalized[:, 0], c1_normalized[:, 1], 'r-', label=f'Piece {j} edge {e1_idx}', linewidth=2)
                    plt.scatter(c0_normalized[0, 0], c0_normalized[0, 1], color='blue', s=100, marker='o', label='Start')
                    plt.scatter(c0_normalized[-1, 0], c0_normalized[-1, 1], color='blue', s=100, marker='s', label='End')
                    plt.scatter(c1_normalized[0, 0], c1_normalized[0, 1], color='red', s=100, marker='o')
                    plt.scatter(c1_normalized[-1, 0], c1_normalized[-1, 1], color='red', s=100, marker='s')
                    plt.axis('equal')
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.title(f'Normalized Edges - Score: {score:.2f}')
                    
                    plt.subplot(1, 2, 2)
                    plt.plot(distances, 'g-', linewidth=2, label='Point distances')
                    plt.plot(weights * np.max(distances), 'k--', alpha=0.7, label='Weights (scaled)')
                    plt.xlabel('Point index')
                    plt.ylabel('Distance')
                    plt.legend()
                    plt.title('Distance Profile')
                    plt.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(f"data/results/debug_match_{i}_{j}.png", dpi=200)
                    plt.close()
    
    return (i, j, best_score if best_score != float('inf') else 1e6)


def score_pieces(list_Images: list[Image], multiprocessing: bool = False) -> np.ndarray:
    from multiprocessing import Pool, cpu_count

    # Calculate parallely score between all pieces
    if multiprocessing:
        with Pool(cpu_count()) as pool:
            scores = pool.starmap(score_match, [(i, piece0, j, piece1) for i,piece0 in enumerate(list_Images) for j,piece1 in enumerate(list_Images) ])
    else:
        scores = [score_match(i, piece0, j, piece1) for i,piece0 in enumerate(list_Images) for j,piece1 in enumerate(list_Images) ]

    scores_matrix = np.zeros((len(list_Images), len(list_Images)))
    for i,j,score in scores:
        scores_matrix[i,j] = score

    #scores_matrix = scores_matrix + scores_matrix.T

    return scores_matrix
        



if __name__ == "__main__":
    list_Images = preprocess_images()

    scores = score_pieces(list_Images)    
    plot_scores(scores)


