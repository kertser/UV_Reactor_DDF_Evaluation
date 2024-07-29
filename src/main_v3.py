import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.stats import skew, kurtosis, norm, invgamma, kstest, cramervonmises
from tabulate import tabulate

# Ignore runtime warnings
warnings.filterwarnings("ignore")


def load_data(file_path):
    """Load data from CSV file."""
    return pd.read_csv(file_path)


def clean_data(dose_values, distribution_values, threshold=0.01):
    """Clean data by removing outliers with u(D) < x%."""

    threshold = threshold * np.max(distribution_values)
    mask = distribution_values >= threshold
    return dose_values[mask], distribution_values[mask]


def calculate_empirical_moments(dose_values, distribution_values):
    """Calculate empirical moments of the dose distribution."""
    total_weight = np.sum(distribution_values)
    empirical_mean = np.sum(dose_values * distribution_values) / total_weight

    # Debug: print intermediate values for mean
    print(f"Empirical Mean: {empirical_mean}")

    empirical_var = np.sum(distribution_values * (dose_values - empirical_mean) ** 2) / total_weight

    # Debug: print intermediate values for variance
    print(f"Empirical Variance: {empirical_var}")

    empirical_std = np.sqrt(empirical_var)

    # Debug: print intermediate values for standard deviation
    print(f"Empirical Std Dev: {empirical_std}")

    empirical_skew = np.sum(distribution_values * (dose_values - empirical_mean) ** 3) / (
                total_weight * empirical_std ** 3)

    # Debug: print intermediate values for skewness
    print(f"Empirical Skewness: {empirical_skew}")

    empirical_kurt = np.sum(distribution_values * (dose_values - empirical_mean) ** 4) / (
                total_weight * empirical_var ** 2) - 3

    # Debug: print intermediate values for kurtosis
    print(f"Empirical Kurtosis: {empirical_kurt}")

    empirical_median = np.median(dose_values)
    empirical_min = np.min(dose_values)

    return empirical_mean, empirical_var, empirical_std, empirical_skew, empirical_kurt, empirical_median, empirical_min


def calculate_efficiency(dose_values, distribution_values):
    """Calculate dimensionless factors and overall efficiency."""
    empirical_mean, empirical_var, empirical_std, empirical_skew, empirical_kurt, empirical_median, empirical_min = calculate_empirical_moments(
        dose_values, distribution_values)

    epsilon = 0.01

    # Calculate the penalties
    # DSL penalty: normalized distance between the average and the minimum dose
    # DSL should be close to 0
    dsl = abs(empirical_min / empirical_mean)

    # CV penalty: coefficient of variation
    cv = empirical_std / empirical_mean

    # TCV penalty: tail coefficient of variation
    tcv = np.percentile(dose_values, 95) / np.percentile(dose_values, 5)

    # Normalize TCV: Let's assume an expected range for TCV between 1 and 10 for typical efficiency.
    tcv_normalized = min(1, max(epsilon, (tcv - 1) / 9))

    # Normalize penalties to [0, 1]
    dsl_penalty = min(1, max(epsilon, 1 - dsl))
    cv_penalty = min(1, max(epsilon, 1 - cv))
    tcv_penalty = min(1, tcv_normalized)

    # Weights for penalties
    weight_dsl = 0.33
    weight_cv = 0.33
    weight_tcv = 0.34

    # Calculate weighted total efficiency
    efficiency = (dsl_penalty * weight_dsl + cv_penalty * weight_cv + tcv_penalty * weight_tcv)

    return efficiency * 100, dsl_penalty, cv_penalty, tcv_penalty


def compare_distributions(dose_values):
    """Compare the empirical distribution with Gaussian and Inverse Gamma distributions."""
    # Fit Gaussian distribution
    mu, std = norm.fit(dose_values)

    # Fit Inverse Gamma distribution
    alpha, loc, beta = invgamma.fit(dose_values)

    # Kolmogorov-Smirnov test for Gaussian
    ks_gaussian = kstest(dose_values, 'norm', args=(mu, std))

    # Kolmogorov-Smirnov test for Inverse Gamma
    ks_invgamma = kstest(dose_values, 'invgamma', args=(alpha, loc, beta))

    # Cramér-von Mises test for Gaussian
    cvm_gaussian = cramervonmises(dose_values, 'norm', args=(mu, std))

    # Cramér-von Mises test for Inverse Gamma
    cvm_invgamma = cramervonmises(dose_values, 'invgamma', args=(alpha, loc, beta))

    # Combine p-values from KS and CVM tests to evaluate the fit
    gaussian_fit = (ks_gaussian.pvalue + cvm_gaussian.pvalue) / 2 * 100
    invgamma_fit = (ks_invgamma.pvalue + cvm_invgamma.pvalue) / 2 * 100

    return gaussian_fit, invgamma_fit


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
                  max_position_empirical, dsl_penalty, cv_penalty, tcv_penalty,
                  efficiency, gaussian_fit, invgamma_fit):
    """Create a report DataFrame with calculated metrics."""
    report = pd.DataFrame([
        ['Mean Dose', round(empirical_mean, 2), 'Average dose delivered.'],
        ['Dose Variance', round(empirical_var, 2), 'Spread of dose values around the mean.'],
        ['Dose Skewness', round(empirical_skew, 2), 'Asymmetry of the dose distribution.'],
        ['Dose Kurtosis', round(empirical_kurt, 2), 'Peakedness of the dose distribution.'],
        ['Minimum Dose', round(min_position_empirical, 2), 'Minimum dose value.'],
        ['Most Probable Dose', round(max_position_empirical, 2), 'Dose value with the highest probability.'],
        ['Penalty: DSL', f"{dsl_penalty * 100:.2f}%",
         'Normalized distance between the mean and minimum dose. Should be close to 0%.'],
        ['Penalty: CV', f"{cv_penalty * 100:.2f}%", 'Coefficient of variation. Should be close to 0%.'],
        ['Penalty: TCV', f"{tcv_penalty * 100:.2f}%", 'Tail coefficient of variation. Should be close to 0%.'],
        ['Overall Efficiency', f"{efficiency:.2f}%",
         'Weighted efficiency considering all penalties. Close to 100% is good.'],
        ['Gaussian Fit', f"{gaussian_fit:.2f}%", 'Goodness-of-fit to Gaussian distribution (higher is better).'],
        ['Inverse Gamma Fit', f"{invgamma_fit:.2f}%",
         'Goodness-of-fit to Inverse Gamma distribution (higher is better).']
    ], columns=['Metric', 'Value', 'Explanation'])
    report = report.round(2)
    report = report.set_index('Metric')
    print(tabulate(report, headers='keys', tablefmt='heavy_outline'))


def main():
    # Load the data
    data = load_data("../resources/DDF_good.csv")  # Adjust the file path as needed
    dose_values = data["range[mJ/cm^2]"]
    distribution_values = data["u(D)"]

    # Preliminary data cleaning: remove outliers with u(D) < 3%
    dose_values, distribution_values = clean_data(dose_values, distribution_values, threshold=.03)

    # Calculate empirical moments
    empirical_mean, empirical_var, empirical_std, empirical_skew, empirical_kurt, empirical_median, empirical_min = calculate_empirical_moments(
        dose_values, distribution_values)
    max_position_empirical = dose_values[np.argmax(distribution_values)]
    min_position_empirical = dose_values[distribution_values.ne(0).idxmax()]

    # Calculate efficiency
    efficiency, dsl_penalty, cv_penalty, tcv_penalty = calculate_efficiency(
        dose_values, distribution_values)

    # Compare with Gaussian and Inverse Gamma distributions
    gaussian_fit, invgamma_fit = compare_distributions(dose_values)

    # Plot the histogram and the empirical statistics
    plot_distribution(dose_values, distribution_values, empirical_mean, min_position_empirical, max_position_empirical)

    # Create and print the report
    create_report(empirical_mean, empirical_var, empirical_skew, empirical_kurt, min_position_empirical,
                  max_position_empirical, dsl_penalty, cv_penalty, tcv_penalty,
                  efficiency, gaussian_fit, invgamma_fit)


if __name__ == "__main__":
    main()
