import cv2
import numpy as np
from pathlib import Path
from src.utils import plot_images

def ingest_image(img_file: Path, plot: bool=False) -> tuple[np.ndarray, np.ndarray]:
    img_bgr= cv2.imread(img_file, cv2.IMREAD_UNCHANGED)

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

    if plot:
        plot_images([
            (img_bgr, "Original"),
            (img_gray, "Gray"),
            (img_blurred, "Blurred"),
        ])
    return img_bgr, img_gray

def _get_image_hsv(img_bgr: np.ndarray):
    list_hsv = []
    # Read image
    hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Mouse callback function
    def pick_color(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # left mouse click
            pixel = hsv_img[y, x]  # HSV values
            h, s, v = pixel
            print(f"HSV: ({h}, {s}, {v})")
            list_hsv.append((h, s, v))
            # Optionally, show the color as a swatch
            swatch = np.zeros((100, 100, 3), np.uint8)
            swatch[:] = (h, s, v)
            swatch_bgr = cv2.cvtColor(swatch, cv2.COLOR_HSV2BGR)
            cv2.imshow("Picked Color", swatch_bgr)

    # Show image and set callback
    cv2.imshow("Image", img_bgr)
    cv2.setMouseCallback("Image", pick_color)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Convert to numpy array for easier min/max
    list_hsv = np.array(list_hsv)

    # Find per-channel min and max
    h_min, s_min, v_min = list_hsv.min(axis=0)
    h_max, s_max, v_max = list_hsv.max(axis=0)

    # Create lower and upper bounds
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    print("Lower HSV bound:", lower)
    print("Upper HSV bound:", upper)

    return lower, upper


def preprocessing_hsv(img_bgr: np.ndarray, lower: np.ndarray|None=None, upper: np.ndarray|None=None, plot: bool=False) -> tuple[np.ndarray, np.ndarray]:
    if lower is None:
        lower, upper = _get_image_hsv(img_bgr)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask_inv = cv2.bitwise_not(mask)

    img_bgr_masked = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    img_bgr_masked_inv = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_inv)

    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=1)

    mask_clean_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_clean_inv = cv2.morphologyEx(mask_clean_inv, cv2.MORPH_OPEN, kernel, iterations=1)

    img_bgr_masked_clean = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_clean)
    img_bgr_masked_clean_inv = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_clean_inv)

    if plot:
        plot_images([
            (mask, "Mask"),
            (mask_clean, "Mask Clean"),
            (mask_inv, "Mask Inverse"),
            (mask_clean_inv, "Mask Clean Inverse"),
        ])

        plot_images([
            (img_bgr, "Original"),
            (img_bgr_masked, "Masked"),
            (img_bgr_masked_inv, "Masked Inverse"),
            (img_bgr_masked_clean, "Masked Clean"),
            (img_bgr_masked_clean_inv, "Masked Clean Inverse"),
        ])

    return img_bgr_masked 