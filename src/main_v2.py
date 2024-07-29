"""
The general idea is to create a metrics for the dose distribution, which will be used to assess the quality of the dose
distribution in the reactor. The metrics will be based on the empirical moments of the dose distribution, which are
calculated from the dose values and their corresponding probabilities. The objective function for the least squares
optimization is defined, which takes into account the empirical moments and a weighting factor for the right tail of the
distribution. The optimization process finds the optimal parameters (alpha, beta, and loc) that best fit the empirical
moments while considering the weighting for the right tail. The final output will be the efficiency metrics, which
include DSL, CV, and TCV, along with the overall efficiency.

What can be considered as a good reactor with proper DDF?
- DSL (notmalized distance between the average and the minumum) is close to 0
- CV (coefficient of variation) is close to 0
- TCV (tail coefficient of variation) is close to 0
Tail Coefficient of Variation is a measure of the spread of the distribution,
indicating how long the tails are. Lower value indicates shorter tails, characteristic of an efficient reactor.
In this specific case, left tail shall have more influence on TCV than the right tail.
- IGF (Inverse Gama Factor) is close to 0
Inverse Gama Factor is a parameter to provide an additional measure of efficiency,
with higher values indicating better performance.
The rationale is that higher alpha values (less skewness) combined with lower beta values (less spread)
suggest a more concentrated and efficient dose distribution.
- Overall efficiency close to 100%
Overall Efficiency is a comprehensive indicator that takes into account DSL, CV, and TCV.
The closer to 100%, the more efficient the reactor.

What can be considered as a bad reactor with improper DDF?
- DSL (normalized distance between the average and the minumum) is far from 0, closer to 1
- CV (coefficient of variation) is far from 0 - meaning the distribution is wider than normal
- IGF (Inverse Gama Factor) is far from 0 - meaning the distribution is not Inverse-Gamma shape
- Overall efficiency close to 0% - meaning the reactor is not efficient
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tabulate import tabulate

# Ignore runtime warnings
warnings.filterwarnings("ignore")

def load_data(file_path):
    """Load data from CSV file."""
    return pd.read_csv(file_path)

def calculate_empirical_moments(dose_values, distribution_values):
    """Calculate empirical moments of the dose distribution."""
    empirical_mean = np.sum(dose_values * distribution_values) / np.sum(distribution_values)
    empirical_var = np.sum(distribution_values * (dose_values - empirical_mean) ** 2) / np.sum(distribution_values)
    empirical_std = np.sqrt(empirical_var)
    empirical_skew = np.sum(distribution_values * (dose_values - empirical_mean) ** 3) / (
            np.sum(distribution_values) * empirical_std ** 3)
    empirical_kurt = np.sum(distribution_values * (dose_values - empirical_mean) ** 4) / (
            np.sum(distribution_values) * empirical_var ** 2) - 3
    empirical_median = np.median(dose_values)
    empirical_min = np.min(dose_values)

    return empirical_mean, empirical_var, empirical_std, empirical_skew, empirical_kurt, empirical_median, empirical_min

def calculate_efficiency(dose_values, distribution_values):
    """Calculate dimensionless factors and overall efficiency."""
    empirical_mean, empirical_var, empirical_std, empirical_skew, empirical_kurt, empirical_median, empirical_min = calculate_empirical_moments(
        dose_values, distribution_values)

    epsilon = 0.01

    # Calculate the penalties
    # DSL penalty: normalized distance between the average and the minumum dose
    # DSL should be close to 0
    min_penalty = max(epsilon, 1 - abs(empirical_min / empirical_mean))

    # Adjusted penalty for skewness: heavier penalty on left skew (negative) than on right skew (positive):
    skewness_penalty = max(epsilon, np.exp(-abs(empirical_skew / 5))) if empirical_skew < 0 else max(epsilon, np.exp(-abs(empirical_skew / 20)))

    # Adjusted penalty for kurtosis: smaller overall effect, prefer positive kurtosis
    kurtosis_penalty = max(epsilon, np.exp(-abs(empirical_kurt / 20)))

    cv_penalty = max(epsilon, 1 - (empirical_std / empirical_mean))

    # Ensure penalties are normalized to [0, 1]
    min_penalty = min(1, min_penalty)
    skewness_penalty = min(1, skewness_penalty)
    kurtosis_penalty = min(1, kurtosis_penalty)
    cv_penalty = min(1, cv_penalty)

    # Weights for penalties
    weight_min = 0.80
    weight_skewness = 0.05
    weight_kurtosis = 0.05
    weight_cv = 0.1

    # Calculate weighted total efficiency
    efficiency = (min_penalty * weight_min + skewness_penalty * weight_skewness + kurtosis_penalty * weight_kurtosis + cv_penalty * weight_cv)

    return efficiency, min_penalty, skewness_penalty, kurtosis_penalty, cv_penalty

def plot_distribution(dose_values, distribution_values, empirical_mean, min_position_empirical, max_position_empirical):
    """Plot the dose distribution and empirical statistics."""
    # Calculate bar width
    width = (dose_values.max() - dose_values.min()) / len(dose_values)

    plt.figure(figsize=(10, 6))
    plt.bar(dose_values, distribution_values, width=width, alpha=0.5, label='Data')
    plt.axvline(x=empirical_mean, color='orange', linestyle='--', label='Empirical Mean')
    plt.axvline(x=min_position_empirical, color='g', linestyle='--', label='Minimum Value')
    plt.axvline(x=max_position_empirical, color='purple', linestyle='--', label='Empirical Max Position')
    plt.xlim(left=0)
    plt.xlabel('Dose [mJ/cm^2]')
    plt.ylabel('u(D)')
    plt.title('Dose Distribution and Empirical Statistics')
    plt.legend()
    plt.grid(True)
    plt.show()

def create_report(empirical_mean, empirical_var, empirical_skew, empirical_kurt, min_position_empirical,
                  max_position_empirical, min_penalty, skew_penalty, kurtosis_penalty, cv_penalty,
                  efficiency):
    """Create a report DataFrame with calculated metrics."""
    report = pd.DataFrame([
        ['Mean Dose', empirical_mean],
        ['Dose Variance', empirical_var],
        ['Dose Skewness', empirical_skew],
        ['Dose Kurtosis', empirical_kurt],
        ['Minimum Dose', min_position_empirical],
        ['Most Probable Dose', max_position_empirical],
        ['Penalty: DSL', min_penalty],
        ['Penalty: Skewness', skew_penalty],
        ['Penalty: Kurtosis', kurtosis_penalty],
        ['Penalty: Coefficient of Variation', cv_penalty],
        ['Overall Efficiency', efficiency]
    ], columns=['Metric', 'Value'])
    report = report.round(2)
    report = report.set_index('Metric')
    print(tabulate(report, headers='keys', tablefmt='heavy_outline'))

def main():
    # Load the data
    data = load_data("../resources/DDF_good.csv")  # Adjust the file path as needed
    dose_values = data["range[mJ/cm^2]"]
    distribution_values = data["u(D)"]

    # Calculate empirical moments
    empirical_mean, empirical_var, empirical_std, empirical_skew, empirical_kurt, empirical_median, empirical_min = calculate_empirical_moments(
        dose_values, distribution_values)
    max_position_empirical = dose_values[np.argmax(distribution_values)]
    min_position_empirical = dose_values[distribution_values.ne(0).idxmax()]

    # Calculate efficiency
    efficiency, min_penalty, skew_penalty, kurtosis_penalty, cv_penalty = calculate_efficiency(
        dose_values, distribution_values)

    # Plot the histogram and the empirical statistics
    plot_distribution(dose_values, distribution_values, empirical_mean, min_position_empirical, max_position_empirical)

    # Create and print the report
    create_report(empirical_mean, empirical_var, empirical_skew, empirical_kurt, min_position_empirical,
                  max_position_empirical, min_penalty, skew_penalty, kurtosis_penalty, cv_penalty,
                  efficiency)

if __name__ == "__main__":
    main()
