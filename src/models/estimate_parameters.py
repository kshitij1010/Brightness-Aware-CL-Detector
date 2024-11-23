#!/usr/bin/env python3
from pathlib import Path
import argparse
import os
import json
import math
from src.features.Pupil_Model import PupilModelWithEstParams
from src.features.load_data import load_concatenated_data_filtered
from sklearn.metrics import mean_squared_error

__author__ = "Regine Bueter"

def calculate_prediction_error(args, path_user):
    print(f"[DEBUG] Calculating prediction error for step: {args.prediction_error_step}")
    use_step = [args.prediction_error_step]
    
    try:
        pupil_diam, time_stamps, focal_brightness, ambient_light, cognit_load = load_concatenated_data_filtered(path_user, use_step)
    except FileNotFoundError as e:
        print(f"[ERROR] Missing data for prediction error calculation: {e}")
        return
    except ValueError as e:
        print(f"[ERROR] Unsupported file format: {e}")
        return

    pupil_model = PupilModelWithEstParams(user=args.user, load_parameters_from_file=True)
    est_diam = pupil_model.calculate_pupil_light_response(time_stamps, focal_brightness, ambient_light)
    mse = mean_squared_error(pupil_diam, est_diam)
    print(f"[RESULT] MSE of prediction: {mse}")
    print(f"[RESULT] RMSE of prediction: {math.sqrt(mse)}")

    save_path = Path(f"{args.save_path}/results/mse_prediction.json")
    if not save_path.parent.exists():
        print(f"[DEBUG] Creating results directory: {save_path.parent}")
        os.makedirs(save_path.parent, exist_ok=True)

    output = {}
    if save_path.exists():
        print(f"[DEBUG] Loading existing results file: {save_path}")
        with open(save_path, 'r') as f:
            output = json.load(f)

    output[args.user] = {"mse": mse}
    with open(save_path, 'w') as out_file:
        json.dump(output, out_file, indent=6)
    print(f"[DEBUG] Results saved to {save_path}")


def estimate_parameters(args):
    print(f"[DEBUG] Starting parameter estimation for user: {args.user}")
    pupil_model = PupilModelWithEstParams(user=args.user, load_parameters_from_file=False)
    
    path = str(Path(args.folder_path) / args.eye / args.user)
    print(f"[DEBUG] Loading data from path: {path}")
    pupil_diam, time_stamps, focal_brightness, ambient_light, cognit_load = load_concatenated_data_filtered(path, args.steps)
    
    print(f"[DEBUG] Estimating parameters using least square method...")
    pupil_model.least_square_with_search_delta(pupil_diam, time_stamps, focal_brightness, ambient_light)
    pupil_model.save_parameters()
    pupil_model.plot_estimation(pupil_diam, time_stamps, focal_brightness, ambient_light, save_name='least_square_training')

    print(f"[RESULT] Estimated parameters: "
          f"a: {pupil_model.a}, b: {pupil_model.b}, delta: {pupil_model.delta}, "
          f"w1: {pupil_model.w1}, w2: {pupil_model.w2}")
    print(f"[RESULT] MSE of fit: {pupil_model.mse}")
    print(f"[RESULT] RMSE of fit: {math.sqrt(pupil_model.mse)}")

    # Check if prediction error step is valid
    if args.prediction_error_step in args.steps:
        calculate_prediction_error(args, path)
    else:
        print(f"[DEBUG] Prediction error step '{args.prediction_error_step}' not in specified steps. Skipping.")


def main(args):
    print(f"[DEBUG] Trial file: {args.trial}")
    if not Path(args.trial).exists():
        print(f"[ERROR] Trial file not found: {args.trial}")
        raise FileNotFoundError(f"Trial file not found: {args.trial}")

    with open(args.trial) as file:
        use_data = [line.strip() for line in file if line.strip()]
    print(f"[DEBUG] Loaded trial data: {use_data}")

    all_users_path = os.path.join(args.folder_path, args.eye)
    print(f"[DEBUG] Checking user directory: {all_users_path}")
    if not os.path.exists(all_users_path):
        print(f"[ERROR] User directory does not exist: {all_users_path}")
        return

    all_users = sorted(os.listdir(all_users_path))
    print(f"[DEBUG] Found users: {all_users}")

    for user in all_users:
        if user.startswith(".") or "._" in user:
            continue
        args.user = user
        user_data = [line for line in use_data if line.startswith(f"{user}:")]
        if not user_data:
            print(f"[DEBUG] No data for user {user}. Skipping.")
            continue

        for data_entry in user_data:
            _, step = data_entry.split(":")
            print(f"[DEBUG] Processing step: {step} for user {user}")
            
            step_path = os.path.join(all_users_path, user, step)
            if not os.path.exists(step_path):
                print(f"[ERROR] File or directory not found: {step_path}")
                continue

            if os.path.isfile(step_path):
                # If it's a file, process it
                args.steps = [step]
                estimate_parameters(args)
            else:
                # If it's a directory, process all steps within
                all_steps = os.listdir(step_path)
                args.steps = [s for s in all_steps if f"{user}:{s}" in use_data]
                if args.steps:
                    estimate_parameters(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters for Probabilistic Model")
    parser.add_argument("--folder_path", type=str, default="data/processed",
                        help="Path to the data folder containing the processed data")
    parser.add_argument("--prediction_error_step", type=str, help="The step for which the prediction error should be calculated")
    parser.add_argument("--eye", type=str, default="right", choices=["right", "left"],
                        help="The eye to be used")
    parser.add_argument("--trial", type=str, required=True,
                        help='Path to a text file specifying user:step pairs')
    parser.add_argument("--save_path", type=str, default="reports",
                        help="Path to save the results and figures")
    args = parser.parse_args()

    main(args)