#!/usr/bin/env python3
import argparse
import cv2
import pandas as pd
import numpy as np
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_brightness(video_file_path):
    """
    Extracts brightness values from the video.
    
    Parameters:
    - video_file_path: str, path to the video file
    
    Returns:
    - brightness_values: list, average brightness values per frame
    """
    logging.info(f"Reading video file: {video_file_path}")
    brightness_values = []
    cap = cv2.VideoCapture(video_file_path)
    
    if not cap.isOpened():
        logging.error(f"Cannot open video file: {video_file_path}")
        return brightness_values
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"Total frames in video: {frame_count}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to grayscale to calculate brightness
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray_frame)
        brightness_values.append(avg_brightness)
    
    cap.release()
    logging.info(f"Extracted {len(brightness_values)} brightness values.")
    return brightness_values


def append_brightness_to_csv(csv_file_path, brightness_values, output_csv_path):
    """
    Appends brightness values to the pupil positions CSV file.
    
    Parameters:
    - csv_file_path: str, path to the input CSV file
    - brightness_values: list, brightness values to append
    - output_csv_path: str, path to save the updated CSV
    """
    logging.info(f"Reading CSV file: {csv_file_path}")
    if not os.path.exists(csv_file_path):
        logging.error(f"CSV file not found: {csv_file_path}")
        return
    
    df = pd.read_csv(csv_file_path)
    logging.info(f"CSV contains {len(df)} rows.")
    
    if len(brightness_values) < len(df):
        logging.warning("Brightness values are fewer than CSV rows. Padding with NaNs.")
        brightness_values.extend([np.nan] * (len(df) - len(brightness_values)))
    
    df['ambient_brightness'] = brightness_values[:len(df)]
    logging.info(f"Appended 'ambient_brightness' column to the DataFrame.")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df.to_csv(output_csv_path, index=False)
    logging.info(f"Updated CSV saved to: {output_csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Append brightness values to a pupil positions CSV.")
    parser.add_argument("--csv_file_path", required=True, type=str, help="Path to the input CSV file.")
    parser.add_argument("--video_file_path", required=True, type=str, help="Path to the video file.")
    parser.add_argument("--output_csv_path", required=True, type=str, help="Path to save the updated CSV file.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logging.info("Starting script...")
    
    # Validate paths
    if not os.path.exists(args.csv_file_path):
        logging.error(f"CSV file not found: {args.csv_file_path}")
        return
    
    if not os.path.exists(args.video_file_path):
        logging.error(f"Video file not found: {args.video_file_path}")
        return

    # Extract brightness and update CSV
    brightness_values = extract_brightness(args.video_file_path)
    append_brightness_to_csv(args.csv_file_path, brightness_values, args.output_csv_path)

    logging.info("Script completed successfully.")


if __name__ == "__main__":
    main()