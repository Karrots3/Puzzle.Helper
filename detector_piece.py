import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path
import argparse

DEBUG=False
DEBUG=True

list_pieces = {}
list_images_pieces = {}
class Piece:
    def __init__(self, piece_id, img_thresh, contour, area, perimeter, bounding_rect, min_area_rect, box_points, hull, solidity, aspect_ratio, extent):
        self.piece_id = piece_id
        self.img_thresh = img_thresh
        self.contour = contour
        self.area = area
        self.perimeter = perimeter
        self.bounding_rect = bounding_rect
        self.min_area_rect = min_area_rect
        self.box_points = box_points
        self.hull = hull
        self.solidity = solidity
        self.aspect_ratio = aspect_ratio
        self.extent = extent

def i2grid(index, n_cols=10, start=0):
    """
    Convert a linear index to a grid position (row, col).
    """
    row = (index - start) // n_cols
    col = (index - start) % n_cols
    return row, col

def plot_list_images_pipeline(list_images, suptitle="Image Processing Pipeline"):
    """
    Plot a list of images in a grid layout with at most 4 columns.
    """
    num_images = len(list_images)
    cols = 3
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    fig.suptitle(suptitle, fontsize=16, fontweight='bold')
    axes = axes.flatten()

    for i, (title, img) in enumerate(list_images.items()):
        if i < len(axes):
            if len(img.shape) == 2:  # Grayscale
                axes[i].imshow(img, cmap='gray')
            else:  # Color
                axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[i].set_title(title, fontsize=12, fontweight='bold')
            axes[i].axis('off')

    # Hide unused subplots
    for i in range(len(list_images), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig("data/results/pipeline.png", dpi=300, bbox_inches='tight')

def extract_pieces_from_file(img_path, min_area=1000, plot=False)->Piece:
    """
    Detect puzzle pieces in an image and return analysis results.
    """
    list_images_pipeline = {}
    img_rgb = cv2.imread(str(img_path))
    if img_rgb is None:
        print(f"Could not read image: {img_path}")
        return None

    list_images_pipeline["00-original"] = img_rgb

    # --- Preprocessing with simple threshold ---
    def _trim_thresh(img_rgb):
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        h, w = img_gray.shape[:2]
        center_x, center_y = w // 2, h // 2
        radius = 60
        sz = int((radius / 100) * min(h, w))
        half_sz = sz // 2
        x1, y1, x2, y2 = max(0, center_x - half_sz), max(0, center_y - half_sz), min(w, center_x + half_sz), min(h, center_y + half_sz)
        img_trimmed = img_gray[y1:y2, x1:x2]
        list_images_pipeline["01-grey-trimmed"] = img_trimmed

        _, img_thresh_fixed = cv2.threshold(img_trimmed, 110, 255, cv2.THRESH_BINARY)
        _, img_thresh_otsu = cv2.threshold(img_trimmed, 110, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return img_thresh_fixed, img_thresh_otsu

    img_thresh_fixed, img_thresh_otsu = _trim_thresh(img_rgb)
    
    list_images_pipeline["02-thresh-fixed"] = img_thresh_fixed
    list_images_pipeline["02-thresh-otsu"] = img_thresh_otsu

    # --- Contour extraction ---
    def get_valid_contours(img_thresh):
        contours, _ = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return contours
    
    def contour2piece(contour) -> Piece:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        rect = cv2.minAreaRect(contour)
        box = np.int8(cv2.boxPoints(rect))
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        return Piece(
            piece_id=i2grid(10),
            img_thresh = img_thresh_fixed,
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
        )

    contours_otsu = get_valid_contours(img_thresh_otsu)
    contours_fixed = get_valid_contours(img_thresh_fixed)


    def filter_pieces(pieces, min_perimeter=200, max_perimeter=7500, min_area=150000, max_area=700000):
        #return [p for p in pieces if p.area > min_area and p.area < max_area]
        return [p for p in pieces if p.aspect_ratio != 1 and p.perimeter > min_perimeter and p.perimeter < max_perimeter]

    pieces_otsu = [contour2piece(c) for c in contours_otsu]
    pieces_fixed = [contour2piece(c) for c in contours_fixed]
    pieces_otsu = filter_pieces(
        pieces_otsu,
        min_perimeter = 100,
        max_perimeter=7000,
        min_area=20,
        max_area=8000
    )
    pieces_fixed = filter_pieces(
        pieces_fixed,
        min_perimeter = 100,
        max_perimeter=8000,
        min_area=20,
        max_area=10000
    )

    img_otsu_full_contours = cv2.cvtColor(img_thresh_otsu.copy(), cv2.COLOR_GRAY2BGR)
    img_fixed_full_contours = cv2.cvtColor(img_thresh_fixed.copy(), cv2.COLOR_GRAY2BGR)

    print("#####################################################################")
    print(f"num pieces otsu: {len(pieces_otsu)}")
    print(f"num pieces fixed: {len(pieces_fixed)}")
    full_pieces_fixed = pieces_fixed.copy()
    full_pieces_otsu = pieces_otsu.copy()
    # pieces_fixed = filter_pieces(pieces_fixed)
    # pieces_otsu = filter_pieces(pieces_otsu)
    print(f"num pieces otsu: {len(pieces_otsu)}")
    print(f"num pieces fixed: {len(pieces_fixed)}")
    try:
        assert len(pieces_fixed) > 0 or len(pieces_otsu) > 0
    except:
        print("No pieces found!")
        for p in full_pieces_fixed+full_pieces_otsu:
            print(f"Piece {p.piece_id}: Area={p.area}, Perimeter={p.perimeter}, BoundingRect={p.bounding_rect}, Solidity={p.solidity:.2f}, AspectRatio={p.aspect_ratio:.2f}, Extent={p.extent:.2f}")
        list_images_pipeline["02b-full-contours-otsu"] = img_otsu_full_contours
        list_images_pipeline["02b-full-contours-fixed"] = img_fixed_full_contours
        print(f"Image: {img_path}")
        return None #TODO remove line
        plot_list_images_pipeline(list_images_pipeline)
        if input("Continue? (y/n)") != "y":
            return None
        
    for p in pieces_fixed+pieces_otsu:
        print(f"Piece {p.piece_id}: Area={p.area}, Perimeter={p.perimeter}, BoundingRect={p.bounding_rect}, Solidity={p.solidity:.2f}, AspectRatio={p.aspect_ratio:.2f}, Extent={p.extent:.2f}")


    img_otsu_contours = cv2.cvtColor(img_thresh_otsu.copy(), cv2.COLOR_GRAY2BGR)
    img_fixed_contours = cv2.cvtColor(img_thresh_fixed.copy(), cv2.COLOR_GRAY2BGR)
    
    for i,p in enumerate(pieces_fixed):
        cv2.drawContours(img_fixed_contours, [p.contour], -1, (0,255,0), 2)

    for i,p in enumerate(pieces_otsu):
        cv2.drawContours(img_otsu_contours, [p.contour], -1, (0,255,0), 2)

    list_images_pipeline[f"03-contours-otsu--{len(contours_otsu)}"] = img_otsu_contours
    list_images_pipeline[f"04-contours-fixed--{len(contours_fixed)}"] = img_fixed_contours
    #plot_list_images_pipeline(list_images_pipeline)
    
    list_images_pieces[img_path.name] = img_otsu_contours
    return [pieces_fixed+pieces_otsu][0]



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
    return None
    

def main():
    parser = argparse.ArgumentParser(description='Detect puzzle pieces in images')
    parser.add_argument('--data-folder', default='data/puzzle', help='Folder containing images')
    parser.add_argument('--image', help='Specific image to process (optional)')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization')
    # Method is now fixed to simple_threshold only
    
    args = parser.parse_args()
    
    #img_path = Path(args.image)
    img_path = "data/puzzle"
    

    list_pieces = {}
    for img_path in Path(args.data_folder).glob("*.JPG"):
        piece = extract_pieces_from_file(img_path)
        list_pieces[img_path.name] = piece
        

    # save list pieces
    import pickle
    with open('data/results/list_pieces.pkl', 'wb') as f:
        pickle.dump(list_pieces, f)
    
    #plot_list_images_pipeline(list_images_pieces)


if __name__ == "__main__":
    main()
