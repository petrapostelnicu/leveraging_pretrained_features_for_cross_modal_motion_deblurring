import os
import sys

import numpy as np
import cv2

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root)

# Root directory
root_dir = "data/GOPRO_converted"
num_bins = 10  # Number of time bins
image_size = (1280, 720)  # Event resolution

def process_txt_to_event_images(txt_path, output_dir, base_filename):
    """Convert event data into multiple images (one per time bin)."""
    # Load event data
    data = np.loadtxt(txt_path)  # Assuming space or tab-separated values

    if data.size == 0:
        return  # Skip empty files

    # Extract columns
    x, y, t, p = data[:, 0].astype(int), data[:, 1].astype(int), data[:, 2], data[:, 3].astype(int)

    # Find min and max timestamps
    t_min, t_max = t.min(), t.max()
    t_bins = np.linspace(t_min, t_max, num_bins + 1)  # Create time bin edges

    for i in range(num_bins):
        # Initialize a 2-channel image for each bin
        event_image = np.zeros((image_size[1], image_size[0], 2), dtype=np.float32)

        # Select events in the current time bin
        mask = (t >= t_bins[i]) & (t < t_bins[i + 1])
        x_masked, y_masked, p_masked = x[mask], y[mask], p[mask]

        # Accumulate event counts
        for j in range(len(x_masked)):
            if 0 <= x_masked[j] < image_size[0] and 0 <= y_masked[j] < image_size[1]:
                event_image[y_masked[j], x_masked[j], p_masked[j]] += 1

        # Normalize to 0-255
        max_value = event_image.max()
        if max_value > 0:
            event_image = (event_image / max_value) * 255

        # Convert to uint8
        event_image = event_image.astype(np.uint8)

        # Convert to RGB format (Red = positive polarity, Green = negative polarity, Blue = empty)
        event_image_rgb = np.stack([
            event_image[:, :, 1],  # Positive polarity → Red
            event_image[:, :, 0],  # Negative polarity → Green
            np.zeros_like(event_image[:, :, 0])  # Blue = 0
        ], axis=-1)

        # Save image with bin index (_0 to _9)
        img_filename = f"{base_filename}_{i}.png"
        img_path = os.path.join(output_dir, img_filename)
        # NOTE: OpenCV saves in BGR format, not RGB
        cv2.imwrite(img_path, event_image_rgb)
        print(f"Saved: {img_path}")

def process_files():
    """Traverse folders, process events.txt, and save each bin as an individual image."""
    for subdir, _, files in os.walk(root_dir):
        if subdir.endswith("/events") or subdir.endswith("\\events"):
            event_img_dir = os.path.join(os.path.dirname(subdir), "events_img")
            os.makedirs(event_img_dir, exist_ok=True)

            for file in files:
                if file.endswith(".txt"):
                    txt_path = os.path.join(subdir, file)
                    base_filename = file.replace(".txt", "")  # Remove .txt for naming
                    process_txt_to_event_images(txt_path, event_img_dir, base_filename)
