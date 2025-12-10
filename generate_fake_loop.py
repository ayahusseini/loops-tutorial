"""A script that generates a fake hysterisis loop for testing"""
import csv
import numpy as np
import matplotlib.pyplot as plt

NPOINTS = 1000
THETA_MIN = 0.0
THETA_MAX = 2 * np.pi
NOISE = 0.01
X_MISSINGNESS = 0.01
Y_MISSINGNESS = 0.02


def ellipse(theta: np.array):
    """Parametric equation in the shape of an ellipse"""
    x = np.cos(theta) + np.sin(theta)
    y = np.sin(theta)
    return x, y


def generate_fake_data(theta_min: float, theta_max: float, npoints: int, noise: float, equation):
    """Generates npoints fake points between (theta_min, theta_max)
    according to some parametric equation, allowing for some random noise
    """
    theta = np.linspace(theta_min, theta_max, npoints)

    x, y = equation(theta)

    # Add noise proportional to the amplitude
    x_amplitude = np.max(np.abs(x))
    y_amplitude = np.max(np.abs(y))

    x_noisy = x + x_amplitude * np.random.normal(0, noise, npoints)
    y_noisy = y + y_amplitude * np.random.normal(0, noise, npoints)

    return theta, x_noisy, y_noisy


def remove_random_points(data: np.ndarray, missing_fraction: float, random_seed=None):
    """Randomly remove a fraction of data points to simulate missing data
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n_total = len(data)
    n_keep = int(n_total * (1 - missing_fraction))

    keep_indices = np.random.choice(n_total, size=n_keep, replace=False)
    keep_indices = np.sort(keep_indices)

    # Create a boolean mask and set missing points to NaN
    mask = np.zeros(n_total, dtype=bool)
    mask[keep_indices] = True
    data[~mask] = np.nan

    return data


if __name__ == "__main__":

    # Generate data points

    theta, x, y = generate_fake_data(
        THETA_MIN, THETA_MAX, NPOINTS, NOISE, ellipse)

    x = remove_random_points(x, X_MISSINGNESS, random_seed=42)
    y = remove_random_points(y, Y_MISSINGNESS, random_seed=50)

    # Plot the loop
    plt.figure(figsize=(10, 8))
    plt.plot(x, y, 'r-', linewidth=1.5, marker='x',
             markersize=5, label='Loop with missing data')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plt.savefig('plots/elliptical_loop.png', dpi=150)

    data_to_save = np.column_stack([x, y])
    np.savetxt('data/loop_data.csv', data_to_save, delimiter=',',
               header='x,y', comments='', fmt='%.6f')
