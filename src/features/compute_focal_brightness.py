import cv2
import pandas as pd
import numpy as np
import argparse
import os

def compute_focal_brightness(video_path, csv_path, output_path, grid_size=11):
    """
    Compute focal brightness from a video and append it to the CSV file.

    Parameters:
    - video_path: Path to the video file (e.g., world.mp4).
    - csv_path: Path to the CSV file containing pupil data.
    - output_path: Path to save the updated CSV file with focal brightness.
    - grid_size: Size of the grid (default: 11x11).
    """
    # Load CSV file
    data = pd.read_csv(csv_path)
    if "norm_pos_x" not in data.columns or "norm_pos_y" not in data.columns:
        raise ValueError("[ERROR] Missing required columns: 'norm_pos_x' or 'norm_pos_y'.")

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"[ERROR] Cannot open video file: {video_path}")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare to append focal brightness
    focal_brightness = []

    # Process each row
    for _, row in data.iterrows():
        gaze_x = int(row['norm_pos_x'] * frame_width)
        gaze_y = int(row['norm_pos_y'] * frame_height)

        # Move to the corresponding video frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, row['world_index'])
        ret, frame = cap.read()
        if not ret:
            focal_brightness.append(np.nan)  # Append NaN if frame is missing
            continue

        # Extract grid of pixels
        y_min = max(0, gaze_y - grid_size // 2)
        y_max = min(frame_height, gaze_y + grid_size // 2 + 1)
        x_min = max(0, gaze_x - grid_size // 2)
        x_max = min(frame_width, gaze_x + grid_size // 2 + 1)
        grid = frame[y_min:y_max, x_min:x_max]

        # Compute brightness
        gray_grid = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray_grid)
        focal_brightness.append(brightness)

    # Close video file
    cap.release()

    # Append focal brightness to DataFrame
    data['focal_brightness'] = focal_brightness

    # Save the updated CSV file
    data.to_csv(output_path, index=False)
    print(f"[INFO] Updated CSV saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute focal brightness and append to CSV file.")
    parser.add_argument("--video_path", required=True, type=str, help="Path to the video file.")
    parser.add_argument("--csv_path", required=True, type=str, help="Path to the input CSV file.")
    parser.add_argument("--output_path", required=True, type=str, help="Path to save the updated CSV file.")
    parser.add_argument("--grid_size", default=11, type=int, help="Size of the grid (default: 11).")
    args = parser.parse_args()

    compute_focal_brightness(args.video_path, args.csv_path, args.output_path, args.grid_size)