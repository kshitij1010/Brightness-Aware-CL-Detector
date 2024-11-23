#!/usr/bin/env python3
import numpy as np
import json
import pandas as pd
from src.features.process_data import mean_filter_fb, mean_filter_signal
from pathlib import Path
import os

__author__ = "Paola Ruiz Puentes, Regine Bueter"

def load_json_signal(path_to_file):
    """
    Loads pupillometry information from JSON files.

    Parameters
    ----------
    path_to_file: str
        Path to the pupillometry file.

    Returns
    -------
    d: array_like
        Pupil diameter
    d_t: array_like
        Timestamps
    d_cl: array_like
        Cognitive load labels
    d_fb: array_like
        Focal brightness
    d_ab: array_like
        Ambient brightness
    pd_d: pd.DataFrame
        Pandas dataframe with all the information
    """
    with open(path_to_file, 'r') as f:
        pupil_signal = json.load(f)

    if "PUPILTIME" in path_to_file:
        d = pupil_signal["pupil_diameter"]
        d_t = pupil_signal["base_time"]
    elif "WORLDTIME" in path_to_file:
        d = pupil_signal["pupil_diameter"]
        d_t = pupil_signal["world_time"]
    else:
        raise FileNotFoundError("Invalid file type. Expected 'PUPILTIME' or 'WORLDTIME'.")
    
    d_cl = np.asarray(pupil_signal.get("cognitive_load", []))
    d_cl[d_cl > 0] = d_cl[d_cl > 0] - 1
    d_cl = d_cl.tolist()

    d_fb = pupil_signal.get("focal_brightness", [])
    d_ab = pupil_signal.get("ambient_brightness_1d", [])

    pd_diameter = pd.DataFrame.from_dict(pupil_signal)

    return d, d_t, d_cl, d_fb, d_ab, pd_diameter


def load_csv_data(path):
    """
    Loads pupil diameter, timestamps, ambient brightness, and focal brightness from CSV files.

    Parameters
    ----------
    path: str
        Path to CSV file

    Returns
    -------
    pupil_diam: array_like
        Pupil diameter (filtered_diameter)
    time_stamps: array_like
        Timestamps (pupil_timestamp)
    ambient_brightness: array_like
        Ambient brightness
    focal_brightness: array_like
        Focal brightness
    """
    df = pd.read_csv(path)

    # Ensure all required columns are present
    required_columns = ['pupil_timestamp', 'filtered_diameter', 'ambient_brightness', 'focal_brightness']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"[ERROR] Required column '{col}' not found in {path}.")

    # Extract relevant columns
    pupil_diam = df['filtered_diameter'].to_numpy().astype(float)
    time_stamps = df['pupil_timestamp'].to_numpy().astype(float)
    ambient_brightness = df['ambient_brightness'].to_numpy().astype(float)
    focal_brightness = df['focal_brightness'].to_numpy().astype(float)

    return pupil_diam, time_stamps, ambient_brightness, focal_brightness

def append_light(signal, light, mode, step=None):
    """
    Updates the ambient light array for the signal.

    Parameters
    ----------
    signal: array_like
        Signal for the light calculation
    light: array_like
        Existing light data
    mode: str
        Mode for brightness calculation
    step: str
        Optional step information
    """
    if mode == "bw-image":
        light = np.concatenate((light, np.full(len(signal), 0.0 if "black" in step else 255)))
    elif mode == "slope":
        light = np.linspace(0, 255, len(signal))
    else:
        light = np.concatenate((light, np.array(signal)))
    return light


def load_concatenated_data_filtered(path: str, steps: list, filter_type='mean', kernel_size=1443):
    """
    Loads and filters concatenated data from multiple steps.

    Parameters
    ----------
    path: str
        Path to the user directory
    steps: list
        List of steps to process
    filter_type: str
        Filter type, either 'mean' or 'median'
    kernel_size: int
        Kernel size for filtering

    Returns
    -------
    pupil_diam, time_stamps, focal_brightness, ambient_light, cognit_load
    """
    focal_brightness = np.array([])
    ambient_light = np.array([])
    pupil_diam = np.array([])
    time_stamps = np.array([])
    cognit_load = np.array([])

    for step in steps:
        step_path = os.path.join(path, step)
        if step.endswith('.csv'):
            pupil_diam_local, time_stamps_local, ambient_light_local, focal_brightness_local = load_csv_data(step_path)
            focal_brightness = np.concatenate((focal_brightness, focal_brightness_local))
            ambient_light = np.concatenate((ambient_light, ambient_light_local))
        else:
            raise ValueError(f"Unsupported file format in step: {step}")

        pupil_diam = np.concatenate((pupil_diam, pupil_diam_local))
        time_stamps = np.concatenate((time_stamps, time_stamps_local))

    pupil_diam = mean_filter_signal(pupil_diam, filter_type, kernel_size)

    return pupil_diam, time_stamps, focal_brightness, ambient_light, cognit_load

    # Debugging logs (added after ambient_light and focal_brightness are defined)
    print(f"[DEBUG] Final ambient_light array: {ambient_light}, Length: {len(ambient_light)}")
    print(f"[DEBUG] Final focal_brightness array: {focal_brightness}, Length: {len(focal_brightness)}")

    if len(ambient_light) == 0:
        raise ValueError("[ERROR] Ambient light array is empty. Check the input data or processing pipeline.")

    return pupil_diam, time_stamps, focal_brightness, ambient_light


def load_steps_of_user_into_dict(path_user: str, steps: list, exp_id='000', max_n_back_task=3):
    """
    Loads the data from the given path to a user for the given steps into a dictionary, where they are sorted by the
    performed n-back task. If several sequences of a cognitive load value are present, they are appended in a
    separate list.

    Parameters
    ----------
    path_user: str
        The path to the folder of the user
    steps: list
        The list of steps to use
    exp_id:str, default='000'
        The exportID number
    max_n_back_task: int, default=2
        The maximum perfomed n-back task

    Returns
    -------
        dict: A dict of n-back tasks from the given data of a user with the given steps,
              one sequence contains the pupil diameter, the cognitive load labels and the timestamps
    """
    dict_sequences = {str(x): [] for x in range(0, max_n_back_task + 1)}

    steps = sort_steps(steps)
    # throw out the zero that was inserted for the sorting
    for step in steps:
        if '.DS_Store' == step:
            continue

        data_path = f"{path_user}/{step}/{exp_id}/{step}_{exp_id}_DATA_WORLDTIME.json"
        pupil_diam_local, time_stamps_local, cognit_load_local, focal_b, ambient_b, all_data_diameter = load_json_signal(
            data_path)
        cognit_load_local = np.asarray(cognit_load_local)

        prev_cl = -1
        bottom_idx = 0
        for idx, coag_load in enumerate(cognit_load_local):
            current_cl = coag_load
            if idx == 0:
                prev_cl = current_cl
                continue
            if idx == len(pupil_diam_local) - 1:
                d_t = time_stamps_local[bottom_idx:]
                d_cl = cognit_load_local[bottom_idx:]
                d = pupil_diam_local[bottom_idx:]

                information_segment = [d, d_cl, d_t]
                dict_sequences[str(int(prev_cl))].append(information_segment)

            if current_cl != prev_cl:
                top_idx = idx

                d_t = time_stamps_local[bottom_idx:top_idx]
                d_cl = cognit_load_local[bottom_idx:top_idx]
                d = pupil_diam_local[bottom_idx:top_idx]

                information_segment = [d, d_cl, d_t]
                dict_sequences[str(int(prev_cl))].append(information_segment)

                bottom_idx = top_idx
                prev_cl = current_cl

    return dict_sequences

def load_gaze(path_to_file):
    """
        Loads gaze information.
        PUPILTIME file has 30hz data
        WORLDTIME file has 120hz data

        Parameters
        ----------
        path_to_file: str
            Path to the pupillometry file.

        Returns
        -------
        g: array_like
            gaze_points [x,y]
        d_t: array_like
            Timestamps
        """
    f = open(path_to_file)
    pupil_signal = json.load(f)
    if "WORLDTIME" in path_to_file:

        g = pupil_signal['gaze_xy']
        d_t = pupil_signal["world_time"]

    elif "PUPILTIME" in path_to_file:

        d_t = pupil_signal["base_time"]
        g = pupil_signal['gaze_xy']

    else:
        raise FileNotFoundError("Path to file was given wrong, must be either \"PUPILTIME\" or \"WORLDTIME\" ")
    return g, d_t


def load_gaze_clipped(path_to_file, width, height):
    """
    Loads gaze information clipped to the video width and height
    PUPILTIME file has 30hz data
    WORLDTIME file has 120hz data

    Parameters
    ----------
    path_to_file: str
        Path to the pupillometry file.
    width: int
        Width of the video
    height: int
        Height of the video

    Returns
    -------
    g: array_like
        gaze_points [x,y]
    d_t: array_like
        Timestamps
    """
    g, d_t = load_gaze(path_to_file)

    gaze = np.asarray(g)
    gaze_x = np.clip(gaze[:, 0], 0, width)
    gaze_y = np.clip(gaze[:, 1], 0, height)

    return np.asarray([gaze_x, gaze_y]).T, d_t

