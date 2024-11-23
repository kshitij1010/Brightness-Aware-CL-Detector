#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from scipy import signal, ndimage

__author__ = "Paola Ruiz Puentes, Regine Bueter"

def mean_filter_signal(pupil_diam, filter_type, kernel_size):
    """
    Filters the pupil signal. First a Butterworth filter, and then a mean or median filter.

    Parameters
    ----------
    pupil_diam: list
        Pupil diameter or area signal
    filter_type: {'mean', 'median'}
        Specifies the filter type
    kernel_size: int
        The window size for the filter.

    Returns
    -------
    np.array
        The filtered pupil diameter
    """
    # Frequency filter
    sampling_rate = 120  # Sampling rate (Hz)
    butterworth_filter = signal.butter(2, (1 / 10) * sampling_rate, "low", output="sos", fs=sampling_rate)

    d = np.array(pupil_diam)
    d = signal.sosfilt(butterworth_filter, d)
    d[0:5] = [d[6]]  # Fix for edge case initialization
    
    # Median filter
    if filter_type == "median":
        pupil_diam = ndimage.median_filter(d, kernel_size)

    # Mean filter
    elif filter_type == "mean":
        pupil_diam = ndimage.uniform_filter(d, kernel_size)

    return pupil_diam

def mean_filter_fb(brightness_signal, filter_type, kernel_size):
    """
    Filters the brightness signal using either a mean or median filter.

    Parameters
    ----------
    brightness_signal: list
        Brightness signal data.
    filter_type: {'mean', 'median'}
        Specifies the filter type.
    kernel_size: int
        The window size for the filter.

    Returns
    -------
    np.array
        The filtered brightness signal.
    """
    if filter_type == "median":
        brightness = ndimage.median_filter(brightness_signal, kernel_size)
    elif filter_type == "mean":
        brightness = ndimage.uniform_filter(brightness_signal, kernel_size)
    else:
        raise ValueError("[ERROR] Invalid filter type. Choose 'mean' or 'median'.")
    return brightness


def validate_and_load_csv(input_file):
    """
    Validates and loads the CSV file. Checks for required columns.

    Parameters
    ----------
    input_file: str
        Path to the input CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.
    """
    try:
        data = pd.read_csv(input_file)
        if 'diameter' not in data.columns:
            raise ValueError("[ERROR] Column 'diameter' not found in input file.")
        return data
    except Exception as e:
        print(f"[ERROR] Failed to load CSV: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Filter pupil signals.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input CSV file (e.g., pupil_positions.csv).")
    parser.add_argument("--filter_type", type=str, choices=["mean", "median"], required=True, help="Type of filter to apply.")
    parser.add_argument("--kernel_size", type=int, required=True, help="Kernel size for the filter.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the filtered data.")
    args = parser.parse_args()

    # Load and validate data
    input_file = args.input_path
    try:
        data = validate_and_load_csv(input_file)
    except ValueError as e:
        print(e)
        return

    # Apply filtering
    print("[INFO] Filtering pupil diameter...")
    pupil_diam = data['diameter'].to_numpy()  # Ensure correct column name
    data['filtered_diameter'] = mean_filter_signal(pupil_diam, args.filter_type, args.kernel_size)

    # Save the filtered data
    try:
        data.to_csv(args.output_path, index=False)
        print(f"[INFO] Filtered data saved to {args.output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save CSV: {e}")


if __name__ == "__main__":
    main()