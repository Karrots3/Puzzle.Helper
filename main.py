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
        
class Image(list):
    def __init__(self, img: np.ndarray, contour: np.ndarray | None = None, peaks: np.ndarray | None = None, edges: list[np.ndarray] | None = None, edges_norm: list[np.ndarray] | None = None):
        self.img = img
        self.contour = contour if contour is not None else np.array([])
        self.peaks = peaks if peaks is not None else np.array([])
        self.edges = edges if edges is not None else np.array([])
        self.edges_norm = edges_norm if edges_norm is not None else np.array([])
        assert len(self.edges)==0 or len(self.edges_norm)==0 or len(self.edges) == len(self.edges_norm)

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
            #print("len edges", len(edges))
            for edge in edges:
                if len(edge) == 0:
                    continue
                #print("len edge", len(edge))
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
            #print("len edges", len(edges))
            for edge in edges:
                if len(edge) == 0:
                    continue
                #print("len edge", len(edge))
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

def plot_edges(list_edges: list[list[np.ndarray]]) -> None:
    list_edges = [l for l in list_edges if len(l) > 0]
    if not list_edges:
        return
        
    n_cols = 4
    n_rows = math.ceil(len(list_edges) / n_cols)
    
    # Calculate bounds for all edges
    min_x = min(min(edge[:,0]) for edges in list_edges for edge in edges)
    min_y = min(min(edge[:,1]) for edges in list_edges for edge in edges)
    max_x = max(max(edge[:,0]) for edges in list_edges for edge in edges)
    max_y = max(max(edge[:,1]) for edges in list_edges for edge in edges)

    height = int(max_y - min_y + 100)  # Add padding
    width = int(max_x - min_x + 100)   # Add padding
    combined_img = np.zeros((n_rows * height, n_cols * width, 3), dtype=np.uint8)

    colors = [
        color_string_to_tuple("red"),
        color_string_to_tuple("green"),
        color_string_to_tuple("blue"),
        color_string_to_tuple("yellow")
    ]
    for i, edges in enumerate(list_edges):
        row = i // n_cols
        col = i % n_cols
        single_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        for edge,c  in zip(edges, colors):
            # Convert edge to integer coordinates and shift to center of single_img
            edge_coords = edge.astype(np.int32)
            edge_coords[:, 0] = edge_coords[:, 0] - min_x + 50  # Shift to center with padding
            edge_coords[:, 1] = edge_coords[:, 1] - min_y + 50  # Shift to center with padding
            
            # Reshape for cv2.drawContours (needs to be 3D array)
            edge_contour = edge_coords.reshape(-1, 1, 2)
            cv2.drawContours(single_img, [edge_contour], -1, c, THICKNESS_LINES)

        combined_img[row*height:(row+1)*height, col*width:(col+1)*width] = single_img

    cv2.imwrite(f"data/results/edges.png", combined_img)

def main():
    out_path = Path("data/results")
    out_path.mkdir(parents=True, exist_ok=True)
    list_files = [Path(f) for f in Path("data").glob("*.JPG")]
    
    list_Images = []
    list_photos = []
    list_contours = []
    list_peaks = []
    list_edges = []
    list_edges_norm = []
    for count_image, file in enumerate(list_files):
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
        list_Images.append(Image(img, contour, [], [], []))

        
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
        list_Images.append(Image(img, contour, contour[peak_indices,0,:], [], []))


    

        # TODO plot the peaks and fix prominance to reduce number of peaks
        #print(len(peak_indices))

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
        list_Images.append(Image(img, contour, contour[peak_indices,0,:], [], []))
        
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
        list_Images.append(Image(img, contour, contour[peak_indices,0,:], edges, []))


        # Normalized edges
        idx = [
            [peak_indices[0], peak_indices[1]],
            [peak_indices[1], peak_indices[2]],
            [peak_indices[2], peak_indices[3]],
            [peak_indices[3], peak_indices[0]],
        ]

        edges_norm = []
        plt.figure(figsize=(10, 8), dpi=400, tight_layout=True)
        plt.title(f"Normalized edges {count_image}")
        plt.ylim(-400,400)
        for c, (idx0, idx1) in enumerate(idx):
            p0 = contour[idx0,0,:]
            p1 = contour[idx1,0,:]
            #center_points = (p0 + p1) / 2
            
            # Calculate the direction vector and straight length
            dx, dy = p1 - p0
            straight_length = math.sqrt(dx**2 + dy**2)
            
            # Calculate the angle to rotate the edge to horizontal
            angle_degrees = math.degrees(math.atan2(dy, dx))
            
            # Create transform to normalize: first point at (0, 0), last point at (X, 0)
            # Replace get_contour_transform with actual implementation
            center = p0 # -> becomes (0,0)
            matrix = cv2.getRotationMatrix2D(center.astype(np.float32), angle_degrees, 1)
            translate = (0, 0) - center
            transform = (matrix, translate)
            
            # Apply transform to the entire piece contour
            # Replace transform_contour with actual implementation
            matrix, translate = transform
            normalized_piece_contour = cv2.transform(contour, matrix) + translate
            
            # Extract just the edge contour
            # Replace sub_contour with actual implementation
            if idx1 + 1 > idx0:
                normalized_edge_contour = normalized_piece_contour[idx0:idx1 + 1]
            else:
                normalized_edge_contour = np.concatenate([normalized_piece_contour[idx0:], normalized_piece_contour[:idx1 + 1]])
            
            # Transform the piece center
            # Replace transform_point with actual implementation
            matrix, translate = transform
            normalized_piece_center = (cv2.transform(np.array([[center]]), matrix) + translate)[0, 0]
            
            # Compute the sign of the edge (male/female/flat)
            heights = normalized_edge_contour[:, 0, 1]
            if np.max(np.abs(heights)) < 10:
                sign = 0
            elif np.max(np.abs(heights)) > 10:
                sign = 1 if np.max(heights) > -np.min(heights) else -1
            else:
                sign = -1
            
            # For male contours (sign == 1), rotate by 180° for easier matching with female contours
            if sign == 1:
                angle_degrees += 180
                # Replace get_contour_transform with actual implementation
                center_point = contour[idx1,0,:]
                matrix = cv2.getRotationMatrix2D(center_point.astype(np.float32), angle_degrees, 1)
                translate = (0, 0) - center_point
                transform = (matrix, translate)
                
                # Replace transform_contour with actual implementation
                matrix, translate = transform
                normalized_piece_contour = cv2.transform(contour, matrix) + translate
                
                # Replace transform_point with actual implementation
                normalized_piece_center = (cv2.transform(np.array([[center_point]]), matrix) + translate)[0, 0]

            color = "blue" if sign == 1 else "red" if sign == -1 else "green"
            plt.scatter(normalized_edge_contour[:, 0, 0], normalized_edge_contour[:, 0, 1], color=color, s=1)

            edges_norm.append(normalized_edge_contour)
 
        plt.savefig(f"data/results/plt_edge_{count_image}.png", bbox_inches='tight')
        plt.close()

    plot_subplots_images(list_photos,list_contours,list_peaks,list_edges,n_cols=len(list_photos)//len(list_files), only_contour=True)
    plot_subplots_images(list_photos,list_contours,list_peaks,list_edges,n_cols=len(list_photos)//len(list_files), only_contour=False)

    #plot_images(list_Images, n_cols=len(list_photos)//len(list_files), only_contour=True)

    plot_edges(list_edges_norm)


if __name__ == "__main__":
    main()
