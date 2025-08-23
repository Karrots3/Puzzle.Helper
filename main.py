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



def compute_curvature(contour):
    """Compute curvature at each point of the contour."""
    if len(contour) < 3:
        return np.zeros(len(contour))
    
    # Smooth the contour slightly to reduce noise
    from scipy import ndimage
    x = ndimage.gaussian_filter1d(contour[:, 0], sigma=1.0)
    y = ndimage.gaussian_filter1d(contour[:, 1], sigma=1.0)
    
    # Compute first and second derivatives
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # Curvature formula: (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
    numerator = dx * ddy - dy * ddx
    denominator = (dx**2 + dy**2)**1.5
    
    # Avoid division by zero
    denominator[denominator < 1e-6] = 1e-6
    curvature = numerator / denominator
    
    return curvature

def normalize_edge_for_matching(edge_contour, target_length=50):
    """Normalize an edge contour for comparison."""
    if len(edge_contour) < 2:
        return None
    
    contour = edge_contour[:, 0, :] if len(edge_contour.shape) == 3 else edge_contour
    
    # Remove a few points from ends to avoid corner artifacts
    trim = min(5, len(contour) // 10)
    if len(contour) > 2 * trim:
        contour = contour[trim:-trim]
    
    # Resample to fixed number of points using arc-length parameterization
    distances = np.cumsum(np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1)))
    distances = np.concatenate([[0], distances])
    
    # Normalize arc length to [0, 1]
    if distances[-1] > 0:
        distances = distances / distances[-1]
    
    # Resample at uniform intervals
    uniform_params = np.linspace(0, 1, target_length)
    x_resampled = np.interp(uniform_params, distances, contour[:, 0])
    y_resampled = np.interp(uniform_params, distances, contour[:, 1])
    
    resampled = np.column_stack([x_resampled, y_resampled])
    
    # Center the contour (align first point to origin)
    resampled = resampled - resampled[0]
    
    # Normalize scale (make end-to-end distance = 1)
    end_to_end = np.linalg.norm(resampled[-1] - resampled[0])
    if end_to_end > 1e-6:
        resampled = resampled / end_to_end
    
    # Align end point to x-axis
    if np.linalg.norm(resampled[-1]) > 1e-6:
        angle = np.arctan2(resampled[-1][1], resampled[-1][0])
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        resampled = np.dot(resampled, rotation.T)
    
    return resampled

def sliding_window_match(c1, c2, window_size=0.8):
    """
    Compare two contours using sliding window to handle misalignment.
    Returns the best score from all possible alignments.
    """
    best_score = float('inf')
    best_offset = 0
    
    # Calculate how many points to slide (percentage of contour length)
    max_offset = int(len(c2) * (1 - window_size) / 2)
    
    for offset in range(-max_offset, max_offset + 1):
        # Create sliding window of c2
        start_idx = max(0, offset)
        end_idx = min(len(c2), offset + int(len(c2) * window_size))
        
        if end_idx <= start_idx:
            continue
            
        c2_window = c2[start_idx:end_idx]
        
        # Adjust c1 to match window length
        target_len = len(c2_window)
        if target_len < len(c1):
            # Trim c1 from both ends equally
            trim_start = (len(c1) - target_len) // 2
            trim_end = trim_start + target_len
            c1_adjusted = c1[trim_start:trim_end]
        else:
            c1_adjusted = c1
        
        # Resample both to same length for comparison
        min_len = min(len(c1_adjusted), len(c2_window))
        if min_len < 5:  # Skip if too short
            continue
            
        c1_resampled = c1_adjusted[np.linspace(0, len(c1_adjusted)-1, min_len, dtype=int)]
        c2_resampled = c2_window[np.linspace(0, len(c2_window)-1, min_len, dtype=int)]
        
        # Align starting points
        c1_aligned = c1_resampled - c1_resampled[0]
        c2_aligned = c2_resampled - c2_resampled[0]
        
        # Scale to same end-to-end distance
        c1_length = np.linalg.norm(c1_aligned[-1] - c1_aligned[0])
        c2_length = np.linalg.norm(c2_aligned[-1] - c2_aligned[0])
        
        if c1_length > 1e-6 and c2_length > 1e-6:
            c2_aligned = c2_aligned * (c1_length / c2_length)
        
        # Compute score
        diff = c1_aligned - c2_aligned
        score = np.mean(np.linalg.norm(diff, axis=1))
        
        if score < best_score:
            best_score = score
            best_offset = offset
    
    return best_score, best_offset

def score_edge_match(contour1, contour2, sign1, sign2, do_complementary: bool = False) -> tuple[float, dict]:
    """Score how well two edge contours match (lower is better)."""
    
    # Only opposite signs can match
    if sign1 == sign2 or sign1 == 0 or sign2 == 0:
        return float('inf'), {}
    
    # Normalize both contours
    norm1 = normalize_edge_for_matching(contour1)
    norm2 = normalize_edge_for_matching(contour2)
    
    if norm1 is None or norm2 is None:
        return float('inf'), {}
    
    best_score = float('inf')
    best_match_details = None
    
    # Try all four orientations
    for flip1 in [False, True]:
        for flip2 in [False, True]:
            c1 = norm1[::-1] if flip1 else norm1
            c2 = norm2[::-1] if flip2 else norm2
            
            # # For complementarity: if one is male (+1), flip it vertically
            # if sign1 == 1:  # c1 is male
            #     c1_test = c1.copy()
            #     c1_test[:, 1] = -c1_test[:, 1]
            #     c2_test = c2.copy()
            # else:  # c1 is female, so c2 must be male
            #     c1_test = c1.copy()
            #     c2_test = c2.copy()
            #     c2_test[:, 1] = -c2_test[:, 1]
            c1_test = c1.copy()
            c2_test = c2.copy()

            
            # Method 1: Direct alignment (original approach)
            diff_direct = c1_test - c2_test
            geometric_score_direct = np.mean(np.linalg.norm(diff_direct, axis=1))
            
            # Method 2: Sliding window match
            sliding_score, best_offset = sliding_window_match(c1_test, c2_test)
            
            # Method 3: Curvature complementarity
            curv1 = compute_curvature(c1_test)
            curv2 = compute_curvature(c2_test)
            
            if len(curv1) == len(curv2) and len(curv1) > 1:
                correlation = np.corrcoef(curv1, curv2)[0, 1]
                curvature_score = 1 + correlation  # Lower is better, perfect match = 0
            else:
                curvature_score = 1.0
            
            # Method 4: Endpoint alignment penalty
            endpoint_penalty = np.linalg.norm(c1_test[-1] - c2_test[-1])
            
            # Combined score with weights - use sliding match as primary geometric score
            geometric_score = min(geometric_score_direct, sliding_score)
            
            # calculate integrals
            #integral1 
            #area_score = abs(area1 - area2) / max(area1,area2)
            #
            #if area_score > 0.2:
            #    area_score = float('inf')
            #else:
            total_score = (geometric_score * 0.5 +           # Primary: best geometric match
                          sliding_score * 0.2 +              # Secondary: sliding window robustness  
                          curvature_score * 0.2 +            # Shape complementarity
                          endpoint_penalty * 0.1)            # Endpoint consistency
            
            #print(f"flip1: {flip1}, flip2: {flip2}, total_score: {total_score}, correlation: {curvature_score}, endpoint_penalty: {endpoint_penalty}")
            if total_score < best_score:
                best_score = total_score
                best_match_details = {
                    'flip1': flip1, 'flip2': flip2,
                    'geometric_direct': geometric_score_direct,
                    'sliding_score': sliding_score,
                    'sliding_offset': best_offset,
                    'curvature_score': curvature_score,
                    'endpoint_penalty': endpoint_penalty
                }
    
    return (best_score, best_match_details)

def score_match(i: int, piece0: Image, j: int, piece1: Image) -> tuple[int, int, float]:
    """
    Score matching between two puzzle pieces using shape complementarity and curvature.
    """
    print(f"Scoring match between piece {i} and piece {j}")
    
    if i >= j:
        print(f"Skipping match between piece {i} and piece {j}")
        return (i, j, 0)
    
    best_score = float('inf')
    best_match_info = None
    
    # Compare all edge combinations
    for e0_idx, e0 in enumerate(piece0.edges_norm):
        for e1_idx, e1 in enumerate(piece1.edges_norm):
            score, info = score_edge_match(e0.contour, e1.contour, e0.sign, e1.sign)
            
            if score < best_score:
                best_score = score
                best_match_info = (e0_idx, e1_idx, e0.sign, e1.sign, info)
    
    # Debug output
    if best_match_info:
        e0_idx, e1_idx, sign0, sign1, info = best_match_info
        print(f"  Best match: piece {i} edge {e0_idx} (sign {sign0}) with piece {j} edge {e1_idx} (sign {sign1})")
        print(f"  Score: {best_score:.4f}")
        
        # Create detailed visualization for specific pairs
        visualize_match(piece0.edges_norm[e0_idx], piece1.edges_norm[e1_idx], i, j, e0_idx, e1_idx, best_score)
    else:
        print(f"No match found between piece {i} and piece {j}")
    
    return (i, j, best_score if best_score != float('inf') else 1e6)

def visualize_match(edge0, edge1, piece0_id, piece1_id, edge0_id, edge1_id, score):
    """Create a detailed visualization of edge matching."""
    try:
        norm0 = normalize_edge_for_matching(edge0.contour)
        norm1 = normalize_edge_for_matching(edge1.contour)
        
        if norm0 is None or norm1 is None:
            return
            
        ## Apply the same transformations used in scoring
        #if edge0.sign == 1:
        #    norm0[:, 1] = -norm0[:, 1]
        #if edge1.sign == 1:
        #    norm1[:, 1] = -norm1[:, 1]
        
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Original normalized edges
        plt.subplot(1, 3, 1)
        plt.plot(norm0[:, 0], norm0[:, 1], 'b-', linewidth=3, label=f'Piece {piece0_id} edge {edge0_id} ({edge0.color})')
        plt.plot(norm1[:, 0], norm1[:, 1], 'r-', linewidth=3, label=f'Piece {piece1_id} edge {edge1_id} ({edge1.color})')
        plt.scatter([norm0[0, 0], norm0[-1, 0]], [norm0[0, 1], norm0[-1, 1]], c='blue', s=100, marker='o')
        plt.scatter([norm1[0, 0], norm1[-1, 0]], [norm1[0, 1], norm1[-1, 1]], c='red', s=100, marker='s')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.title(f'Normalized Edges\nScore: {score:.4f}')
        
        # Plot 2: Curvature profiles
        plt.subplot(1, 3, 2)
        curv0 = compute_curvature(norm0)
        curv1 = compute_curvature(norm1)
        plt.plot(curv0, 'b-', linewidth=2, label=f'Piece {piece0_id} curvature')
        plt.plot(curv1, 'r-', linewidth=2, label=f'Piece {piece1_id} curvature')
        plt.xlabel('Point index')
        plt.ylabel('Curvature')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        correlation = np.corrcoef(curv0, curv1)[0, 1] if len(curv0) == len(curv1) and len(curv0) > 1 else 0
        plt.title(f'Curvature Profiles\nCorrelation: {correlation:.3f}')
        
        # Plot 3: Distance profile
        plt.subplot(1, 3, 3)
        diff = norm0 - norm1
        distances = np.linalg.norm(diff, axis=1)
        plt.plot(distances, 'g-', linewidth=2, label='Point-to-point distance')
        plt.axhline(y=np.mean(distances), color='orange', linestyle='--', label=f'Mean: {np.mean(distances):.3f}')
        plt.xlabel('Point index')
        plt.ylabel('Distance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title('Distance Profile')
        
        plt.tight_layout()
        plt.savefig(f"data/results/match_debug_{piece0_id}_{piece1_id}_{edge0_id}_{edge1_id}.png", dpi=200, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Visualization failed: {e}")
        plt.close('all')


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
    print(scores)
    plot_scores(scores)


