#!/usr/bin/env python3
import argparse
import os
import sys
import logging
from pathlib import Path
import cv2
import numpy as np


def image_trim(img: np.ndarray, radius: int = 100) -> np.ndarray:
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


def image_plot_subplots(list_images: list[np.ndarray], n_cols: int = 4, contours = None, only_contours = False) -> None:
    n_rows = len(list_images) // n_cols
    
    # Create a combined image for display
    max_height = max(img.shape[0] for img in list_images)
    max_width = max(img.shape[1] for img in list_images)
    
    # Create a grid layout
    combined_img = np.zeros((max_height * n_rows, max_width * n_cols, 3), dtype=np.uint8)
    
    for i, img in enumerate(list_images):
        row = i // n_cols
        col = i % n_cols
        
        if only_contours:
            # Create a black background for contours only
            display_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        else:
            # Convert grayscale to BGR if needed
            if len(img.shape) == 2:
                display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                display_img = img.copy()
        
        # Draw contours if provided
        if contours and i < len(contours):
            cs = contours[i]
            print(f"Imagine {i} has got {len(cs)} contours")
            # Draw contours on the display image
            cv2.drawContours(display_img, cs, -1, (0, 255, 0), 2)
        
        # Place the image in the grid
        y_start = row * max_height
        y_end = y_start + img.shape[0]
        x_start = col * max_width
        x_end = x_start + img.shape[1]
        
        combined_img[y_start:y_end, x_start:x_end] = display_img
    
    # Save the combined image
    cv2.imwrite(f"data/results/00_subplots{"_cont" if only_contours else ""}.png", combined_img)
    
    # Display the combined image
    cv2.imshow("Subplots", combined_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    list_files = [Path(f) for f in Path("data").glob("*.JPG")]

    list_photos = []
    list_contours = []
    for file in list_files:
        img = cv2.imread(str(file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = image_trim(img,60)
        contour, _= cv2.findContours(img,cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        list_contours.append(contour)
        list_photos.append(img)

        _,th3 = cv2.threshold(img,110,255,cv2.THRESH_BINARY)# + cv2.THRESH_OTSU)
        contour, _= cv2.findContours(th3,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        list_contours.append(contour)
        list_photos.append(th3)
        
        _,th3 = cv2.threshold(img,110,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contour, _= cv2.findContours(th3,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        list_contours.append(contour)
        list_photos.append(th3)



    image_plot_subplots(list_photos,n_cols=3, contours=list_contours, only_contours=True)
    image_plot_subplots(list_photos,n_cols=3, contours=list_contours)


if __name__ == "__main__":
    main()
