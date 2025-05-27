import matplotlib.pyplot as plt
from IPython.display import display
import ipywidgets as widgets
from IPython.display import clear_output
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from skimage import filters
from skimage.measure import shannon_entropy


def navigate_images(image_files, jump_to=0, pred_bboxes=[], gt_bboxes=[]):
    """
    Displays images with forward and backward navigation buttons.

    Parameters:
        image_files (list): List of image file paths.
    """
    if not image_files:
        print("No images to display.")
        return

    idx = {"value": jump_to}

    def show_image(i):
        clear_output(wait=True)
        try:
            img = visualize_bboxes(image_files[i], pred_bboxes[i], gt_bboxes[i])
            img_id = image_files[i].split("/")[1]
            width, height, _ = img.shape

            dpi = 100  # Can tweak this
            figsize = (width / dpi, height / dpi)

            plt.figure(figsize=figsize, dpi=dpi)
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"Image {i+1} of {len(image_files)}: {img_id}", fontsize=24)
            plt.show()
            display(nav_box)
        except FileNotFoundError:
            print(f"Error: File '{image_files[i]}' not found.")
            print("Exiting image viewer.")
        # except Exception as e:
        #     print(f"An unexpected error occurred: {e}")
        #     print("Exiting image viewer.")
            

    def on_prev_clicked(b):
        if idx["value"] > 0:
            idx["value"] -= 1
            show_image(idx["value"])

    def on_next_clicked(b):
        if idx["value"] < len(image_files) - 1:
            idx["value"] += 1
            show_image(idx["value"])

    prev_button = widgets.Button(description="⬅️ Previous")
    next_button = widgets.Button(description="Next ➡️")
    prev_button.on_click(on_prev_clicked)
    next_button.on_click(on_next_clicked)

    nav_box = widgets.HBox([prev_button, next_button])

    show_image(idx["value"])


def visualize_bboxes(image_path, pred_bboxes, gt_bboxes):
    """
    Visualize prediction and ground truth bounding boxes on an image.

    Args:
        image_path (str): Path to the image file.
        pred_bboxes (list of tuples): List of predicted bounding boxes [(x1, y1, x2, y2), ...].
        gt_bboxes (list of tuples): List of ground truth bounding boxes [(x1, y1, x2, y2), ...].
    """
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw prediction boxes in blue
    for x1, y1, x2, y2 in pred_bboxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Blue for predictions
        cv2.putText(
            image, "Pred", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )

    # Draw ground truth boxes in green
    for x1, y1, x2, y2 in gt_bboxes:
        cv2.rectangle(
            image, (x1, y1), (x2, y2), (0, 255, 0), 2
        )  # Green for ground truth
        cv2.putText(
            image, "GT", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

    return image


def compute_image_features(
    image_path,
    use_features=[
        "luminance",
        "contrast",
        "saturation",
        "sharpness",
        "temperature",
        "edge_density",
        "entropy",
    ],
):
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not found at the specified path.")

    # Convert to RGB (Pillow and OpenCV use different color formats)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Use Laplacian variance to determine sharpness
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Store results in a dictionary
    results = {}

    # Feature 1: Luminance
    if "luminance" in use_features:
        # Convert image to grayscale (luminance approximation)
        luminance = np.mean(gray_image) / 255
        results["luminance"] = luminance

    # Feature 2: Contrast
    if "contrast" in use_features:
        # Calculate contrast using standard deviation of the grayscale image
        contrast = np.std(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        results["contrast"] = contrast

    # Feature 3: Saturation
    if "saturation" in use_features:
        saturation = np.mean(hsv_image[:, :, 1])  # Mean of Saturation channel
        results["saturation"] = saturation

    # Feature 4: Sharpness/Bluriness
    if "sharpness" in use_features:
        laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        sharpness = laplacian_var
        results["sharpness"] = sharpness

    # Feature 5: White Balance
    if "temperature" in use_features:
        hue = hsv_image[:, :, 0]
        warm_pixels = np.sum((hue < 50) | (hue > 330))
        cool_pixels = np.sum((hue > 150) & (hue < 260))
        ratio = warm_pixels / (cool_pixels + 1e-5)
        results["temperature"] = ratio

    # Feature 6: Edge Density
    if "edge_density" in use_features:
        # Use Canny edge detection to find edges
        edges = cv2.Canny(image, 100, 200)
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
        results["edge_density"] = edge_density

    # Feature 7: Image Entropy
    if "entropy" in use_features:
        # Convert to grayscale for entropy calculation
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        entropy_value = shannon_entropy(gray_image)
        results["entropy"] = entropy_value

    return results
