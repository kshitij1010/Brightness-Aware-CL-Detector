import json
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from pathlib import Path
import matplotlib.pyplot as plt
import os

__author__ = "Regine Bueter"

class PupilModelWithEstParams:
    """
    The Pupil Light Response (PLR) model estimates the pupil diameter based on the lighting situation. First, it estimates the
    model parameters via a least squares estimation.
    """

    def __init__(self, user, load_parameters_from_file, path_to_params="models/Parameters.json"):
        if load_parameters_from_file:
            data = json.load(open(path_to_params))[user]
            self.a = data["a"]
            self.b = data["b"]
            self.delta = data["delta"]
            self.w1 = data["w1"]
            self.w2 = data["w2"]
            self.mse = data["mse"]
            self.user = user
        else:
            self.user = user
            self.a = 0
            self.b = 0
            self.delta = 0
            self.w1 = 0.5
            self.w2 = 0.5
            self.mse = np.inf

    def calculate_pupil_light_response(self, t, focal_b, ambient_b):
        focal_b, ambient_b = self.__adapt_light(t, self.delta, focal_b, ambient_b)
        d = self.a * np.exp(self.b * (self.w1 * focal_b + self.w2 * ambient_b))
        return d

    def least_square_with_search_delta(self, pupil_diam, time_stamps, focal_brightness, ambient_light):
        # Filter out invalid data points
        valid_indices = ~(
            np.isnan(pupil_diam) | np.isnan(time_stamps) |
            np.isnan(focal_brightness) | np.isnan(ambient_light) |
            np.isinf(pupil_diam) | np.isinf(time_stamps) |
            np.isinf(focal_brightness) | np.isinf(ambient_light)
        )
        pupil_diam = pupil_diam[valid_indices]
        time_stamps = time_stamps[valid_indices]
        focal_brightness = focal_brightness[valid_indices]
        ambient_light = ambient_light[valid_indices]

        if len(pupil_diam) == 0:
            raise ValueError("No valid data points available after filtering.")

        mse_opt = np.inf
        for d in np.arange(-30, 0, 0.1):
            t = time_stamps
            i = self.__adapt_light(t, d, focal_brightness, ambient_light)

            # Skip if there is insufficient variability in adapted light values
            if np.std(i[0]) < 1e-5 or np.std(i[1]) < 1e-5:
                continue

            # Initial guesses and bounds for curve fitting
            a_0 = max(np.mean(pupil_diam), 1)
            b_0 = -1e-3
            w1_0 = 0.5
            w2_0 = 0.5

            try:
                popt, _ = curve_fit(self.__exp_model, xdata=i, ydata=pupil_diam,
                                    bounds=([1e-07, -1, 0, 0], [10, 0, 1, 1]),
                                    p0=(a_0, b_0, w1_0, w2_0))
                mse = mean_squared_error(pupil_diam, self.__exp_model(i, *popt))
            except Exception as e:
                continue

            if mse < mse_opt:
                mse_opt = mse
                self.a = popt[0]
                self.b = popt[1]
                self.w1 = popt[2]
                self.w2 = popt[3]
                self.delta = d

        self.mse = mse_opt
        print(f"Estimated parameters: a={self.a}, b={self.b}, delta={self.delta}, w1={self.w1}, w2={self.w2}")
        print(f"MSE of fit: {self.mse}")

    def save_parameters(self, save_path="models/Parameters.json"):
        if not Path(save_path).parent.exists():
            os.mkdir(Path(save_path).parent)
        if not Path(save_path).exists():
            f = open(save_path, 'x')
            output = {}
        else:
            f = open(save_path)
            output = json.load(f)

        output[self.user] = {"a": self.a, "b": self.b, "delta": self.delta, "w1": self.w1, "w2": self.w2,
                             "mse": self.mse}

        with open(save_path, "w") as out_file:
            json.dump(output, out_file, indent=6)

    def plot_estimation(self, pupil_diameter, time_stamps, focal_brightness, ambient_light, save_name,
                        save_path="reports/figures/", show_plot=False):
        i = self.__adapt_light(time_stamps, self.delta, focal_brightness, ambient_light)

        fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
        ax.plot(time_stamps, pupil_diameter, color='blue', label="Measured pupil diameter")
        ax.plot(time_stamps, self.__exp_model(i, self.a, self.b, self.w1, self.w2), color='orange',
                label=f'Fitted model: a={self.a:.3f}, b={self.b:.3f}, delta={self.delta:.3f}, w1={self.w1:.3f}, w2={self.w2:.3f}')
        ax.plot(time_stamps, focal_brightness / 255, color='red', label="Focal brightness (normalized)")
        ax.plot(time_stamps, ambient_light / 255, color='green', label="Ambient brightness (normalized)")
        ax.grid()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Diameter (mm)')
        ax.legend(loc='best')

        save_dir = Path(save_path) / self.user
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f"{save_name}.png", dpi=300)
        if show_plot:
            plt.show()
        else:
            plt.close()

    def __exp_model(self, i, a, b, w1, w2):
        fb, ab = i
        return a * np.exp(b * (w1 * fb + w2 * ab))

    def __adapt_light(self, t, d, focal_b, ambient_b):
        t = np.array(t).astype(float)
        adapted_t = t + d

        if adapted_t[0] >= 0:
            idx = (np.abs(t - adapted_t[0])).argmin()
            focal_b = np.concatenate((focal_b[idx:], np.full(idx, focal_b[-1])))
            ambient_b = np.concatenate((ambient_b[idx:], np.full(idx, ambient_b[-1])))
        else:
            num_idx_adding_neg = (np.abs(t - np.abs(adapted_t[0]))).argmin()
            focal_b = np.concatenate((np.full(num_idx_adding_neg, focal_b[0]), focal_b[:-num_idx_adding_neg]))
            ambient_b = np.concatenate((np.full(num_idx_adding_neg, ambient_b[0]), ambient_b[:-num_idx_adding_neg]))
        return focal_b, ambient_b