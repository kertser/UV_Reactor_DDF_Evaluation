import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.stats import norm, invgamma
from scipy.optimize import minimize
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
    empirical_var = np.sum(distribution_values * (dose_values - empirical_mean) ** 2) / total_weight
    empirical_std = np.sqrt(empirical_var)
    empirical_skew = np.sum(distribution_values * (dose_values - empirical_mean) ** 3) / (
        total_weight * empirical_std ** 3)
    empirical_kurt = np.sum(distribution_values * (dose_values - empirical_mean) ** 4) / (
        total_weight * empirical_var ** 2) - 3
    empirical_min = np.min(dose_values)
    dsl = abs(empirical_min / empirical_mean)  # DSL penalty: normalized distance between the average and the minimum dose
    cv = empirical_std / empirical_mean  # CV penalty: coefficient of variation
    return empirical_mean, empirical_var, empirical_std, empirical_skew, empirical_kurt, empirical_min, dsl, cv


def weighted_fit_norm(dose_values, distribution_values):
    """Fit Gaussian distribution to dose values weighted by distribution values."""
    mean = np.average(dose_values, weights=distribution_values)
    variance = np.average((dose_values - mean) ** 2, weights=distribution_values)
    std = np.sqrt(variance)
    return mean, std


def weighted_fit_invgamma(dose_values, distribution_values):
    """Fit Inverse Gamma distribution to dose values weighted by distribution values."""
    def neg_log_likelihood(params):
        alpha, loc, beta = params
        return -np.sum(distribution_values * invgamma.logpdf(dose_values, alpha, loc=loc, scale=beta))
    result = minimize(neg_log_likelihood, [2, 0, 1], bounds=((0.01, None), (None, None), (0.01, None)))
    alpha, loc, beta = result.x
    return alpha, loc, beta


def compare_distributions(dose_values, distribution_values):
    """Compare the empirical distribution with Gaussian and Inverse Gamma distributions."""
    # Fit Gaussian distribution with weights
    mu, std = weighted_fit_norm(dose_values, distribution_values)

    # Fit Inverse Gamma distribution with weights
    alpha, loc, beta = weighted_fit_invgamma(dose_values, distribution_values)

    # Empirical CDF
    ecdf = np.cumsum(distribution_values / np.sum(distribution_values))

    # Theoretical CDF for Gaussian
    cdf_gaussian = norm.cdf(dose_values, mu, std)

    # Theoretical CDF for Inverse Gamma
    cdf_invgamma = invgamma.cdf(dose_values, alpha, loc, beta)

    # Goodness-of-fit: maximum difference (KS statistic)
    ks_gaussian = np.max(np.abs(ecdf - cdf_gaussian))
    ks_invgamma = np.max(np.abs(ecdf - cdf_invgamma))

    # Goodness-of-fit in percentage
    gof_gaussian = (1 - ks_gaussian) * 100
    gof_invgamma = (1 - ks_invgamma) * 100

    return mu, std, alpha, loc, beta, gof_gaussian, gof_invgamma


def plot_distribution(dose_values, distribution_values, empirical_mean, min_position_empirical, max_position_empirical, mu, std, alpha, loc, beta):
    """Plot the dose distribution and empirical statistics."""
    # Calculate bar width
    width = (dose_values.max() - dose_values.min()) / len(dose_values)

    plt.figure(figsize=(10, 6))
    plt.bar(dose_values, distribution_values, width=width, alpha=0.5, label='Data')
    plt.axvline(x=empirical_mean, color='orange', linestyle='--', label='Empirical Mean')
    plt.axvline(x=min_position_empirical, color='g', linestyle='--', label='Minimum Value')
    plt.axvline(x=max_position_empirical, color='purple', linestyle='--', label='Empirical Max Position')

    # Plot Gaussian fit
    x = np.linspace(dose_values.min(), dose_values.max(), 1000)
    gaussian_fit = norm.pdf(x, mu, std)
    gaussian_fit *= np.trapz(distribution_values, dose_values) / np.trapz(gaussian_fit, x)
    plt.plot(x, gaussian_fit, 'r-', lw=2, label='Gaussian Fit')

    # Plot Inverse Gamma fit
    invgamma_fit = invgamma.pdf(x, alpha, loc, beta)
    invgamma_fit *= np.trapz(distribution_values, dose_values) / np.trapz(invgamma_fit, x)
    plt.plot(x, invgamma_fit, 'b-', lw=2, label='Inverse Gamma Fit')

    plt.xlim(left=0)
    plt.xlabel('Dose [mJ/cm^2]')
    plt.ylabel('u(D)')
    plt.title('Dose Distribution and Empirical Statistics')
    plt.legend()
    plt.grid(True)
    plt.show()


def create_report(empirical_mean, empirical_var, empirical_skew, empirical_kurt, min_position_empirical,
                  max_position_empirical, dsl_penalty, cv_penalty, gof_gaussian, gof_invgamma):
    """Create a report DataFrame with calculated metrics."""
    report = pd.DataFrame([
        ['Mean Dose', round(empirical_mean, 2), 'Average dose delivered.'],
        ['Dose Variance', round(empirical_var, 2), 'Spread of dose values around the mean.'],
        ['Dose Skewness', round(empirical_skew, 2), 'Asymmetry of the dose distribution.'],
        ['Dose Kurtosis', round(empirical_kurt, 2), 'Peakedness of the dose distribution.'],
        ['Minimum Dose', round(min_position_empirical, 2), 'Minimum dose value.'],
        ['Most Probable Dose', round(max_position_empirical, 2), 'Dose value with the highest probability.'],
        ['Penalty: DSL', f"{dsl_penalty:.2f}", 'Normalized distance between the mean and minimum dose. Should be close to 0%.'],
        ['Penalty: CV', f"{cv_penalty:.2f}", 'Coefficient of variation. Should be close to 0%.'],
        ['Goodness of fit: Gaussian', f"{gof_gaussian:.2f}%", 'Goodness of fit for Gaussian distribution.'],
        ['Goodness of fit: Inverse Gamma', f"{gof_invgamma:.2f}%", 'Goodness of fit for Inverse Gamma distribution.'],

    ], columns=['Metric', 'Value', 'Explanation'])
    report = report.round(2)
    report = report.set_index('Metric')
    print(tabulate(report, headers='keys', tablefmt='heavy_outline'))


def main():
    # Load the data
    file_path = '../resources/DDF_good.csv'
    data = load_data(file_path)
    dose_values = data["range[mJ/cm^2]"]
    distribution_values = data["u(D)"]

    # Preliminary data cleaning: remove outliers with u(D) < 3%
    dose_values, distribution_values = clean_data(dose_values, distribution_values, threshold=.03)

    # Calculate empirical moments
    empirical_mean, empirical_var, empirical_std, empirical_skew, empirical_kurt, empirical_min, dsl_penalty, cv_penalty = calculate_empirical_moments(
        dose_values, distribution_values)
    max_position_empirical = dose_values[np.argmax(distribution_values)]
    min_position_empirical = dose_values[distribution_values.ne(0).idxmax()]

    # Compare with Gaussian and Inverse Gamma distributions
    mu, std, alpha, loc, beta, gof_gaussian, gof_invgamma = compare_distributions(dose_values, distribution_values)

    # Plot the histogram and the empirical statistics
    plot_distribution(dose_values, distribution_values, empirical_mean, min_position_empirical, max_position_empirical, mu, std, alpha, loc, beta)

    # Create and print the report
    create_report(empirical_mean, empirical_var, empirical_skew, empirical_kurt, min_position_empirical,
                  max_position_empirical, dsl_penalty, cv_penalty, gof_gaussian, gof_invgamma)


if __name__ == "__main__":
    main()
