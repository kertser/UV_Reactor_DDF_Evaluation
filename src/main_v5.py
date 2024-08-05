import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, invgamma
from scipy.optimize import minimize
import warnings
from tabulate import tabulate

# Ignore runtime warnings
warnings.filterwarnings("ignore")

# Load the data
file_path = '../resources/DDF_good.csv'
data = pd.read_csv(file_path)
dose_values = data["range[mJ/cm^2]"]
distribution_values = data["u(D)"]

# Preliminary data cleaning: remove outliers with u(D) < 3%
threshold = 0.03 * np.max(distribution_values)
mask = distribution_values >= threshold
dose_values, distribution_values = dose_values[mask], distribution_values[mask]

# Calculate empirical moments
def calculate_empirical_moments(dose_values, distribution_values):
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

empirical_mean, empirical_var, empirical_std, empirical_skew, empirical_kurt, empirical_min, dsl_penalty, cv_penalty = calculate_empirical_moments(dose_values, distribution_values)
max_position_empirical = dose_values[np.argmax(distribution_values)]
min_position_empirical = dose_values[distribution_values.ne(0).idxmax()]

# Compare with Gaussian and Inverse Gamma distributions
def weighted_fit_norm(dose_values, distribution_values):
    def neg_log_likelihood(params):
        mu, sigma = params
        return -np.sum(distribution_values * norm.logpdf(dose_values, mu, sigma))

    def constraint(params):
        mu, sigma = params
        return mu - max_position_empirical

    initial_params = [empirical_mean, empirical_std]
    bounds = [(None, None), (0.01, None)]

    result = minimize(neg_log_likelihood, initial_params, method='SLSQP', constraints={'type': 'eq', 'fun': constraint}, bounds=bounds, options={'ftol': 1e-12})
    mu, sigma = result.x
    return mu, sigma

def weighted_fit_invgamma(dose_values, distribution_values):
    def neg_log_likelihood(params):
        alpha, loc, beta, k = params
        return -np.sum(distribution_values * invgamma.logpdf(dose_values, alpha, loc=loc, scale=beta))

    def constraint(params):
        alpha, loc, beta, k = params
        mode = beta / (alpha + k) + loc
        return mode - max_position_empirical

    initial_params = [2, 0, 1, 2]
    bounds = [(0.01, None), (None, None), (0.01, None), (0.01, 10)]

    result = minimize(neg_log_likelihood, initial_params, method='SLSQP', constraints={'type': 'eq', 'fun': constraint}, bounds=bounds, options={'ftol': 1e-12})
    alpha, loc, beta, k = result.x
    return alpha, loc, beta, k

mu, std = weighted_fit_norm(dose_values, distribution_values)
alpha, loc, beta, k = weighted_fit_invgamma(dose_values, distribution_values)

# Plot the histogram and the empirical statistics
def plot_distribution(dose_values, distribution_values, empirical_mean, min_position_empirical, max_position_empirical, mu, std, alpha, loc, beta):
    width = (dose_values.max() - dose_values.min()) / len(dose_values)
    plt.figure(figsize=(10, 6))
    plt.bar(dose_values, distribution_values, width=width, alpha=0.5, label='Data')
    plt.axvline(x=empirical_mean, color='orange', linestyle='--', label='Empirical Mean')
    plt.axvline(x=min_position_empirical, color='g', linestyle='--', label='Minimum Value')
    plt.axvline(x=max_position_empirical, color='purple', linestyle='--', label='Empirical Max Position')

    x = np.linspace(dose_values.min(), dose_values.max(), 1000)
    gaussian_fit = norm.pdf(x, mu, std)
    gaussian_fit *= np.trapz(distribution_values, dose_values) / np.trapz(gaussian_fit, x)
    plt.plot(x, gaussian_fit, 'r-', lw=2, label='Gaussian Fit')

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

plot_distribution(dose_values, distribution_values, empirical_mean, min_position_empirical, max_position_empirical, mu, std, alpha, loc, beta)

# Create and print the report
def create_report(empirical_mean, empirical_var, empirical_skew, empirical_kurt, min_position_empirical, max_position_empirical, dsl_penalty, cv_penalty, gof_gaussian, gof_invgamma):
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

# Calculate goodness-of-fit
ecdf = np.cumsum(distribution_values / np.sum(distribution_values))
cdf_gaussian = norm.cdf(dose_values, mu, std)
cdf_invgamma = invgamma.cdf(dose_values, alpha, loc, beta)
ks_gaussian = np.max(np.abs(ecdf - cdf_gaussian))
ks_invgamma = np.max(np.abs(ecdf - cdf_invgamma))
gof_gaussian = (1 - ks_gaussian) * 100
gof_invgamma = (1 - ks_invgamma) * 100

# Generate the report
report = create_report(empirical_mean, empirical_var, empirical_skew, empirical_kurt, min_position_empirical,
                       max_position_empirical, dsl_penalty, cv_penalty, gof_gaussian, gof_invgamma)
