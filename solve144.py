import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import itertools
import heapq
import statistics
from scipy import stats
from scipy.signal import find_peaks
from collections import Counter

"""# Add utilities"""

def imshow(img, title=None):
    plt.title(title)
    plt.axis('equal')
    plt.imshow(img)
    plt.show()

class Item():
    def __init__(self, **kwargs):
        self.update(**kwargs)

    def update(self, **kwargs):
        self.__dict__.update(kwargs)

class LoopingList(list):
    def __getitem__(self, i):
        if isinstance(i, int):
            return super().__getitem__(i % len(self))
        else:
            return super().__getitem__(i)

def plot_contour(contour, **kwargs):
    plt.axis('equal')
    plt.plot(contour[:, :, 0], -contour[:, :, 1], **kwargs)

def plot_point(point, **kwargs):
    plot_contour(np.array([[point]]), **kwargs)

def fill_contour(contour, **kwargs):
    plt.fill(contour[:, :, 0], -contour[:, :, 1], **kwargs)

def get_transform(center, x, y, degrees):
    matrix = cv2.getRotationMatrix2D(center, degrees, 1)
    translate = (x, y) - center
    return (matrix, translate)

def get_contour_transform(contour, idx, x, y, degrees):
    return get_transform(contour[idx][0], x, y, degrees)

def transform_contour(contour, transform):
    matrix, translate = transform
    return cv2.transform(contour, matrix) + translate

def transform_point(point, transform):
    matrix, translate = transform
    return (cv2.transform(np.array([[point]]), matrix) + translate)[0, 0]

def sub_contour(c, idx0, idx1):
    if idx1 > idx0:
        return c[idx0:idx1]
    else:
        return np.concatenate([c[idx0:], c[:idx1]])

"""# Detect pieces"""

def segment_colored_puzzle(img_rgb):
    """
    Segment puzzle pieces from a colored image using multiple methods
    """
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Adaptive thresholding
    img_binary_adaptive = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Method 2: Otsu's thresholding (automatic threshold detection)
    _, img_binary_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Method 3: Color-based segmentation
    # Try different color ranges for better detection
    color_masks = []
    
    # Range 1: Bright colors (high value)
    lower1 = np.array([0, 0, 100])
    upper1 = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower1, upper1)
    color_masks.append(mask1)
    
    # Range 2: Saturated colors
    lower2 = np.array([0, 50, 50])
    upper2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(img_hsv, lower2, upper2)
    color_masks.append(mask2)
    
    # Range 3: Non-black colors
    lower3 = np.array([0, 0, 30])
    upper3 = np.array([180, 255, 255])
    mask3 = cv2.inRange(img_hsv, lower3, upper3)
    color_masks.append(mask3)
    
    # Combine color masks
    color_mask = np.zeros_like(img_gray)
    for mask in color_masks:
        color_mask = cv2.bitwise_or(color_mask, mask)
    
    # Method 4: Edge detection
    edges = cv2.Canny(img_gray, 30, 100)
    kernel = np.ones((3,3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel)
    
    # Combine all methods
    img_binary = cv2.bitwise_or(img_binary_adaptive, img_binary_otsu)
    img_binary = cv2.bitwise_or(img_binary, color_mask)
    img_binary = cv2.bitwise_or(img_binary, edges_closed)
    
    # Clean up with morphological operations
    kernel_clean = np.ones((5,5), np.uint8)
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel_clean)
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel_clean)
    
    return img_binary, img_binary_adaptive, color_mask, edges_closed

img_rgb = cv2.imread("data/img.jpg")
h, w = img_rgb.shape[:2]
img_rgb = cv2.resize(img_rgb, (4*w, 4*h))

# Segment the colored image
img_binary, img_binary_adaptive, color_mask, edges_closed = segment_colored_puzzle(img_rgb)

# Show the segmentation results for debugging
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(img_binary_adaptive, cmap='gray')
plt.title('Adaptive Threshold')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(color_mask, cmap='gray')
plt.title('Color Mask')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(img_binary, cmap='gray')
plt.title('Combined Binary')
plt.axis('off')
plt.tight_layout()
plt.show()

# def extract_edges(self, image: np.ndarray) -> np.ndarray:
#     """
#     Extract edges from the image using Canny edge detection.
#     
#     Args:
#         image: Input image
#         
#     Returns:
#         Edge map as binary image
#     """
#     # Convert to grayscale if needed
#     if len(image.shape) == 3:
#         gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     else:
#         gray = image
#     
#     # Apply Gaussian blur
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     
#     # Canny edge detection
#     edges = cv2.Canny(blurred, 50, 150)
#     
#     return edges
# 

from typing import List
def find_contours(image: np.ndarray) -> List[np.ndarray]:
    """
    Find contours in the image.
        
    Args:
        image: Input image (preferably binary/edge image)
            
    Returns:
        List of contours
    """
    # Convert to binary if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
        
    # Threshold to create binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    return contours
# Find contours


# contours = cv2.findContours(img_binary_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
contours = find_contours(img_binary_adaptive)


pieces = []
counter = 0
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    if area < 10000:
        continue
    # compute piece name from its position
    # col = int((x - w/2) * 13 / 4540)
    # row = int(1 + (y - h/2) * 13 / 4450)
    # name = chr(ord('A') + col) + str(row)
    name = f"piece_{counter}"

    piece = Item(
        name=name,
        int_contour=contour,
        contour=contour.astype(np.float64),  # convert to float for rotation
        area=area,
    )
    pieces.append(piece)

piece_by_name = dict([(piece.name, piece) for piece in pieces])
print(f"{len(pieces)} detected pieces.")

imshow(img_rgb)

# show all piece contours
for piece in pieces:
    plot_contour(piece.contour)

# show the smallest and biggest pieces by area
pieces.sort(key= lambda piece: piece.area)

for piece in pieces[:1] + pieces[-1:]:
    plt.title(f"{piece.name}, area={int(piece.area)}")
    plot_contour(piece.contour)
    plt.show()

"""# Detect piece corners

## Find corners via peak distance from center
"""

for piece in pieces:
    (cx, cy), cr = cv2.minEnclosingCircle(piece.int_contour)
    centered_contour = piece.contour - np.array([cx, cy])

    # ensure peaks are not at start or end of the distances array
    distances = np.sum(centered_contour**2, axis=2)[:, 0]
    distance_offset = np.argmin(distances)
    shifted_distances = np.concatenate([distances[distance_offset:], distances[:distance_offset]])

    # find peak distances
    peak_indices = [(distance_idx + distance_offset) % len(distances) for distance_idx in find_peaks(shifted_distances, prominence=1000)[0]]
    peak_indices.sort()
    piece.update(center=np.array([cx, cy]),
                 peak_indices=LoopingList(peak_indices),
                 )

# Show the pieces having the smallest / highest number of peak indices
pieces.sort(key= lambda piece: len(piece.peak_indices))

for piece in pieces[:1] + pieces[-1:]:
    plt.title(f"{piece.name}, Number of peaks={len(piece.peak_indices)}")
    plot_contour(piece.contour)
    plot_contour(piece.contour[piece.peak_indices], marker='o', ls='', color='red')
    plt.show()

"""## Filter corners by rectangle geometry"""

for piece in pieces:
    def compute_rectangle_error(indices):
            # get coordinates of corners
            corners = LoopingList(np.take(piece.contour, sorted(list(indices)), axis=0)[:, 0, :])
            # compute the side lengths and diagonal lengths
            lengths = [math.sqrt(np.sum((corners[i0] - corners[i1])**2)) for i0, i1 in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]]
            def f_error(a, b):
                return abs(b - a) / (a + b)
            return sum([f_error(lengths[i], lengths[j]) for i, j in [(0, 2), (1, 3), (4, 5), (0, 1)]])

    # form a good rectangle with peak indices
    rectangles = []  # list of (score, [indices])
    for indices in itertools.combinations(piece.peak_indices, 4):
        error = compute_rectangle_error(indices)
        rectangles.append((error, indices))

    error, indices = sorted(rectangles)[0]
    piece.update(rectangle_error=error)
    piece.update(corner_indices=LoopingList(indices))

# Show the pieces having the best / worst rectangle
pieces.sort(key= lambda piece: piece.rectangle_error)

for piece in pieces[:1] + pieces[-1:]:
    plt.title(f"{piece.name}, Rectangle error={piece.rectangle_error}")
    plot_contour(piece.contour)
    plot_contour(piece.contour[piece.peak_indices], marker='o', ls='', color='red')
    plot_contour(piece.contour[piece.corner_indices], marker='', ls='-', color='red')
    plt.show()

# Show all rectangles
for piece in pieces:
    plot_contour(piece.contour[piece.corner_indices], marker='', ls='-', color='red')

"""# Compute edges

## Extract edges
"""

for piece in pieces:
    edges = LoopingList()
    for quarter in range(4):
        idx0 = piece.corner_indices[quarter]
        idx1 = piece.corner_indices[quarter+1]
        p0 = piece.contour[idx0][0]
        p1 = piece.contour[idx1][0]
        # normalize the contour: first point at (0, 0), last point at (X, 0)
        dx, dy = p1 - p0
        straight_length=math.sqrt(dx**2 + dy**2)
        angle_degrees = math.degrees(math.atan2(dy, dx))

        transform = get_contour_transform(piece.contour, idx0, 0, 0, angle_degrees)
        normalized_piece_contour = transform_contour(piece.contour, transform)
        normalized_edge_contour = sub_contour(normalized_piece_contour, idx0, idx1 + 1)
        normalized_piece_center = transform_point(piece.center, transform)

        # compute the sign of the edge
        heights = normalized_edge_contour[:, 0, 1]
        if np.max(np.abs(heights)) > 10:
            sign = 1 if np.max(heights) > - np.min(heights) else -1
        else:
            sign = 0

        # rotate male contours by 180° for easy match with female contours
        if sign == 1:
            angle_degrees += 180
            transform = get_contour_transform(piece.contour, idx1, 0, 0, angle_degrees)
            normalized_piece_contour = transform_contour(piece.contour, transform)
            normalized_piece_center = transform_point(piece.center, transform)

        edge = Item(
            idx0=idx0,
            idx1=idx1,
            normalized_piece_contour=normalized_piece_contour,
            normalized_piece_center=normalized_piece_center,
            angle_degrees=angle_degrees,
            sign=sign,
            straight_length=straight_length,
        )
        edges.append(edge)

    for idx, edge in enumerate(edges):
        edge.update(
            prev=edges[idx-1],
            next=edges[idx+1]
        )

    piece.update(
        edges=edges,
        nb_flats=len([edge for edge in edges if edge.sign == 0])
    )

print("edge sign:", Counter([edge.sign for piece in pieces for edge in piece.edges]))
print("nb of flats:", Counter([piece.nb_flats for piece in pieces]))

flat_pieces = [piece for piece in pieces if piece.nb_flats > 0]

for piece in flat_pieces:
    for edge in piece.edges:
        if edge.sign == 0 and edge.prev.sign != 0:
            first_flat = edge
        if edge.sign == 0 and edge.next.sign != 0:
            last_flat = edge
    piece.update(
        first_flat = first_flat,
        last_flat = last_flat,
        before_flat = first_flat.prev,
        after_flat = last_flat.next,
    )

# Show the pieces having the smallest / highest number of flats
pieces.sort(key= lambda piece: piece.nb_flats)

sign2color = {-1: "red", 0: "green", 1: "blue"}

for piece in pieces[:1] + pieces[-1:]:
    plt.title(f"{piece.name}, nb of flats={piece.nb_flats}")
    for edge in piece.edges:
        plot_contour(sub_contour(piece.contour, edge.idx0, edge.idx1), c=sign2color[edge.sign])
    plt.show()

# Show the pieces having the min/max edge straight length
edge_pieces = [(edge, piece) for piece in pieces for edge in piece.edges]
edge_pieces.sort(key= lambda ep: ep[0].straight_length)

for edge, piece in edge_pieces[:1] + edge_pieces[-1:]:
    plt.title(f"{piece.name}, edge straight length={edge.straight_length}")
    plot_contour(piece.contour)
    plot_contour(sub_contour(piece.contour, edge.idx0, edge.idx1), c='red')
    plt.show()

# Show some normalized edges
import random

for piece in random.sample(pieces, 1):
    for edge in piece.edges[:2]:
        plt.title(f"{piece.name} normalized edge & center")
        plot_contour(edge.normalized_piece_contour)
        plot_point(edge.normalized_piece_center, marker='o', c='red')
        plt.axhline(0, c='gray', ls=':')
        plt.axvline(0, c='gray', ls=':')
        plt.show()

"""# Compute puzzle size"""

def compute_size(area, perimeter):
    # perimeter = 2 * (H+W)
    # area = H*W
    # H**2 - perimeter/2 * H + area = 0
    a = 1
    b = -perimeter/2
    c = area
    delta = b**2 - 4*a*c
    h = int((-b - math.sqrt(delta)) / (2*a))
    w = int((-b + math.sqrt(delta)) / (2*a))
    return (min(h, w), max(h, w))

solution = Item()

nb_flats = Counter([piece.nb_flats for piece in pieces])
assert nb_flats[2] == 4
area = len(pieces)
perimeter = nb_flats[1] + 2*nb_flats[2]
w, h = compute_size(area, perimeter)
print(f"Size of puzzle grid: {w} x {h}")
assert w * h == area
assert 2 * (w + h) == perimeter

solution.update(grid_size = (w, h))

"""# Sample edges"""

NB_SAMPLES = 9

for piece in pieces:
    for edge in piece.edges:
        # compute the distance from the first point, this is not exactly edge.arc_length
        edge_contour = sub_contour(edge.normalized_piece_contour, edge.idx0, edge.idx1 + 1)
        deltas = edge_contour[1:] - edge_contour[:-1]
        distances = np.cumsum(np.sqrt(np.sum(deltas**2, axis=2)))
        arc_length = distances[-1]
        distance = arc_length / (NB_SAMPLES - 1)  # distance between 2 sample points
        # get N equidistant points
        sample_indices = (np.array([np.argmax(distances >= i*distance - 0.0001) for i in range(NB_SAMPLES)]) + edge.idx0) % len(piece.contour)

        edge.update(
            sample_indices=sample_indices,
            arc_length=arc_length,
        )

# Show the pieces having the min/max edge arc length
edge_pieces = [(edge, piece) for piece in pieces for edge in piece.edges]
edge_pieces.sort(key= lambda ep: ep[0].arc_length)

for edge, piece in edge_pieces[:1] + edge_pieces[-1:]:
    plt.title(f"{piece.name}, edge arc length={edge.arc_length}")
    plot_contour(piece.contour)
    plot_contour(sub_contour(piece.contour, edge.idx0, edge.idx1), c='red')
    plot_contour(piece.contour[edge.sample_indices], marker='o', ls='', color='red')
    plt.show()

"""# Match border pieces"""

points_before_flat = {}  # key=piece, value=sample points
points_after_flat = {}
for piece in flat_pieces:
    points_before_flat[piece] = piece.first_flat.normalized_piece_contour[piece.before_flat.sample_indices][::-1]
    points_after_flat[piece] = piece.last_flat.normalized_piece_contour[piece.after_flat.sample_indices]

matches_after_flat = {}  # key=piece0, value=[(score1, piece1)]
for piece0 in flat_pieces:
    points0 = points_after_flat[piece0]
    matches = []
    for piece1, points1 in points_before_flat.items():
        diff = points1 - points0
        offset = np.mean(diff, axis=0)
        score = np.sum((diff - offset)**2)
        if score < 2000:
            matches.append((score, piece1))
    matches.sort()
    matches_after_flat[piece0] = matches

sum_score = sum([matches[0][0] for matches in matches_after_flat.values()])
first_piece = piece_by_name['A1']
paths = [(sum_score, [first_piece])]
heapq.heapify(paths)
for _ in range(100):
    score, ordered_border = heapq.heappop(paths)
    if len(ordered_border) == len(flat_pieces) + 1:
        print("Minimum score:", sum_score)
        print("Score of the border:", score)
        break
    last_piece = ordered_border[-1]
    matches = matches_after_flat[last_piece]
    for match_score, next_piece in matches:
        if next_piece not in ordered_border[1:]:
            heapq.heappush(paths, (score + match_score - matches[0][0], ordered_border.copy() + [next_piece]))

print("Computed border pieces: ", ' '.join([piece.name for piece in ordered_border]))

assert ordered_border[-1] == ordered_border[0]  # loop on the first piece
ordered_border = ordered_border[:-1]  # remove the repeated first piece
h, w = solution.grid_size
if ordered_border[h-1].nb_flats == 1:
    h, w = w, h
    solution.grid_size = w, h
assert [idx for idx, piece in enumerate(ordered_border) if piece.nb_flats == 2] == [0, h-1, h+w-2, 2*h+w-3]

"""# Place the border"""

PAD = 30

solution.update(
    grid={} # key=(i, j), value=Cell
)

border_positions = []

def place_border():
    i, j = 0, 0
    x, y = 0, 0
    it_pieces = iter(ordered_border)
    for quarter, (di, dj) in zip([2, 3, 0, 1, 2], [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]):
        for piece in it_pieces:
            i += di
            j += dj
            flat_edge = piece.first_flat
            top_edge_idx = (piece.edges.index(flat_edge) - quarter) % 4
            if (i, j) == (0, 0):
                idx = flat_edge.idx1  # reference (xy) point is the corner
            else:
                idx = flat_edge.idx0

            transform = get_contour_transform(flat_edge.normalized_piece_contour, idx, x, y, 90 * quarter)
            cell = Item(
                piece = piece,
                top_edge_idx = top_edge_idx,
                contour = transform_contour(flat_edge.normalized_piece_contour, transform),
            )
            solution.grid[i, j] = cell
            border_positions.append((i, j))

            x, y = cell.contour[piece.last_flat.idx1][0]
            if piece.nb_flats == 2:
                break

place_border()

plt.title("Border")
for cell in solution.grid.values():
   plot_contour(cell.contour, c='blue', ls='-')
plt.show()

plt.title("Top edge")
for cell in solution.grid.values():
   top_edge = cell.piece.edges[cell.top_edge_idx]
   plot_contour(sub_contour(cell.contour, top_edge.idx0, top_edge.idx1), c='red')

"""# Match inner pieces"""

piece_edge_points = {}  # key=(piece, edge), value=sample points
for piece in pieces:
    for edge in piece.edges:
        if edge.sign != 0 and edge.prev.sign != 0 and edge.next.sign != 0:
            piece_edge_points[(piece, edge)] = edge.normalized_piece_contour[edge.sample_indices]

piece_edge_matches = {}  # key=(piece0, edge0), value=[(score1, piece1, edge1)]
for (piece0, edge0), points0 in piece_edge_points.items():
    points0 = points0[::-1]
    matches = []
    for (piece1, edge1), points1 in piece_edge_points.items():
        if edge0.sign != edge1.sign:
             diff = points1 - points0
             offset = np.mean(diff, axis=0)
             score = np.sum((diff - offset)**2)
             if score < 2000:
                 matches.append((score, piece1, edge1))

    matches.sort()
    piece_edge_matches[piece0, edge0] = matches

edge_pair_score = {}  # key=(edge0, edge1), value=matching score
edge_min_score = {}  # key=edge, value=min matching score
for piece in pieces:
    for edge in piece.edges:
        matches = piece_edge_matches.get((piece, edge), None)
        if matches:
            edge_min_score[edge] = matches[0][0]
            for match_score, match_piece, match_edge in matches:
                edge_pair_score[edge, match_edge] = match_score

min_inner_score = sum(edge_min_score.values()) / 2
print(min_inner_score)

# we will fill puzzle pieces in a spiral for improved piece placement accuracy

w, h = solution.grid_size
all_positions = border_positions.copy()

i, j = 1, 0
for (di, dj) in itertools.cycle([(0, 1), (1, 0), (0, -1), (-1, 0)]):
    while (i+di, j+dj) not in all_positions:
        i += di
        j += dj
        all_positions.append((i, j))
    if len(all_positions) == w*h:
        break

inner_positions = all_positions[len(border_positions):]

def inner_matches(i, j, grid):
    scores = {}  # key=(piece, top_edge_idx), value=score
    directions = []
    for direction, dpos in zip([2, 3, 0, 1], [(0, 1), (1, 0), (0, -1), (-1, 0)]):
        di, dj = dpos
        if (i+di, j+dj) in grid:
            directions.append(direction)
            neighbour = grid[i+di, j+dj]
            neighbour_edge = neighbour.piece.edges[neighbour.top_edge_idx + direction]

            for match_score, match_piece, match_edge in piece_edge_matches[neighbour.piece, neighbour_edge]:
                match_top_edge_idx = (match_piece.edges.index(match_edge) + 2 - direction) % 4
                scores.setdefault((match_piece, match_top_edge_idx), []).append(2 * match_score - edge_min_score[match_edge] - edge_min_score[neighbour_edge])

    matches = [(sum(match_scores), candidate, candidate_top_edge_idx) for (candidate, candidate_top_edge_idx), match_scores in scores.items() if len(match_scores) == len(directions)]
    matches.sort(key=lambda m: m[0])
    return matches

def place_inner_piece(i, j, cell):
    translations = []  # translations from piece.center
    rotations = []  # angle in degrees
    for direction, dpos in zip([2, 3, 0, 1], [(0, 1), (1, 0), (0, -1), (-1, 0)]):
        di, dj = dpos
        if (i+di, j+dj) in grid:
            neighbour = grid[i+di, j+dj]
            neighbour_edge = neighbour.piece.edges[neighbour.top_edge_idx + direction]
            x0, y0 = neighbour.contour[neighbour_edge.idx0][0]
            x1, y1 = neighbour.contour[neighbour_edge.idx1][0]
            dx, dy = x1-x0, y1-y0

            match_edge = cell.piece.edges[cell.top_edge_idx + direction + 2]
            angle_degrees = 90 * (3 + match_edge.sign) - math.degrees(math.atan2(dy, dx))
            transform = get_contour_transform(match_edge.normalized_piece_contour, match_edge.idx1, x0, y0, angle_degrees)
            translations.append(transform_point(match_edge.normalized_piece_center, transform))
            rotations.append((match_edge.angle_degrees + angle_degrees) % 360)

    angle_degrees = stats.circmean(rotations, high=360)
    x = statistics.mean([translation[0] for translation in translations])
    y = statistics.mean([translation[1] for translation in translations])
    transform = get_transform(cell.piece.center, x, y, angle_degrees)
    cell.contour = transform_contour(cell.piece.contour, transform)


w, h = solution.grid_size

nodes = [(min_inner_score, solution.grid.copy(), set())]  # (score, grid, used_inner_pieces)
heapq.heapify(nodes)
while len(nodes) > 0:
    grid_score, grid, used_inner_pieces = heapq.heappop(nodes)

    if len(used_inner_pieces) == 0:
        pass
    else:
        i, j = inner_positions[len(used_inner_pieces) - 1]
        place_inner_piece(i, j, grid[i, j])  # properly place the last piece

    if len(used_inner_pieces) == len(inner_positions):
        break

    i, j = inner_positions[len(used_inner_pieces)]  # next position to fill
    for match_score, match_piece, match_top_edge_idx in inner_matches(i, j, grid):
        if match_piece not in used_inner_pieces:
            cell = Item(
                piece = match_piece,
                top_edge_idx = match_top_edge_idx,
            )
            grid2 = grid.copy()
            grid2[i, j] = cell
            used_inner_pieces2 = used_inner_pieces.copy()
            used_inner_pieces2.add(match_piece)
            heapq.heappush(nodes, (grid_score + match_score, grid2, used_inner_pieces2))

print(min_inner_score)
print(grid_score)
for j in range(h):
    print(' '.join([f"{grid[i, j].piece.name if (i, j) in grid else ''}\t" for i in range(w)]))

for ij, cell in grid.items():
    i, j = ij
    plot_contour(cell.contour)

assert len(used_inner_pieces) == len(inner_positions)