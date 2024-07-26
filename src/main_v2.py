import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invgamma, skew, kurtosis
from scipy.optimize import minimize
from scipy.integrate import simpson
import warnings
from config import resource_path as path
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
    return empirical_mean, empirical_var, empirical_skew, empirical_kurt


def objective(params, x, data, weight_factor, empirical_mean, empirical_var, empirical_skew, empirical_kurt,
              min_position_empirical, max_position_empirical):
    """Objective function for least squares optimization with weighting for the right tail."""
    alpha, beta, loc = params
    fitted_pdf = invgamma.pdf(x, alpha, loc=loc, scale=beta)
    weights = np.linspace(1, weight_factor, len(data))  # Increase the weight for the right tail
    weighted_loss = np.sum(weights * (data - fitted_pdf) ** 2)

    # Calculate numerical moments
    mean_fitted = simpson(y=x * fitted_pdf, x=x)
    var_fitted = simpson(y=(x - mean_fitted) ** 2 * fitted_pdf, x=x)
    std_fitted = np.sqrt(var_fitted)
    skew_fitted = simpson(y=(x - mean_fitted) ** 3 * fitted_pdf, x=x) / std_fitted ** 3
    kurt_fitted = simpson(y=(x - mean_fitted) ** 4 * fitted_pdf, x=x) / var_fitted ** 2 - 3

    # Add penalty for not matching the max, min positions and moments
    max_position_fitted = x[np.argmax(fitted_pdf)]
    min_position_fitted = loc
    penalty = (5000 * (min_position_fitted - min_position_empirical) ** 2 +
               100 * (max_position_fitted - max_position_empirical) ** 2 +
               50 * (mean_fitted - empirical_mean) ** 2 +
               20 * (var_fitted - empirical_var) ** 2 +
               10 * (skew_fitted - empirical_skew) ** 2 +
               10 * (kurt_fitted - empirical_kurt) ** 2)
    return weighted_loss + penalty


def optimize_parameters(dose_values, distribution_values, empirical_mean, empirical_var, empirical_skew, empirical_kurt,
                        min_position_empirical, max_position_empirical):
    """Perform grid search and optimization to find the best parameters."""
    alpha_range = np.linspace(1.0, 2.0, 10)  # More dense range for alpha
    beta_range = np.linspace(150, 250, 10)  # More dense range for beta
    loc_range = [min_position_empirical]
    weight_factor_range = [10, 15, 20, 25]

    best_initial_guess = None
    best_weight_factor = None
    best_loss = float('inf')

    for alpha in alpha_range:
        for beta in beta_range:
            for loc in loc_range:
                for weight_factor in weight_factor_range:
                    initial_guess = [alpha, beta, loc]
                    result = minimize(objective, initial_guess, args=(
                    dose_values, distribution_values, weight_factor, empirical_mean, empirical_var, empirical_skew,
                    empirical_kurt, min_position_empirical, max_position_empirical),
                                      bounds=((1e-5, None), (1e-5, None), (None, None)),
                                      options={'maxiter': 10000, 'disp': False})
                    if result.fun < best_loss:
                        best_loss = result.fun
                        best_initial_guess = initial_guess
                        best_weight_factor = weight_factor

    # Perform the final optimization with the best initial guess
    result = minimize(objective, best_initial_guess, args=(
    dose_values, distribution_values, best_weight_factor, empirical_mean, empirical_var, empirical_skew, empirical_kurt,
    min_position_empirical, max_position_empirical),
                      bounds=((1e-5, None), (1e-5, None), (None, None)),
                      options={'maxiter': 10000, 'disp': True})
    return result.x


def calculate_fitted_moments(x, alpha_opt, beta_opt, loc_opt):
    """Calculate the moments of the fitted inverse gamma distribution."""
    pdf_fitted = invgamma.pdf(x, alpha_opt, loc=loc_opt, scale=beta_opt)
    mean_value = simpson(y=x * pdf_fitted, x=x)
    var_value = simpson(y=(x - mean_value) ** 2 * pdf_fitted, x=x)
    std_value = np.sqrt(var_value)
    skew_value = simpson(y=(x - mean_value) ** 3 * pdf_fitted, x=x) / std_value ** 3
    kurt_value = simpson(y=(x - mean_value) ** 4 * pdf_fitted, x=x) / var_value ** 2 - 3
    return mean_value, var_value, skew_value, kurt_value, pdf_fitted


def calculate_efficiency(doses, mean_value, var_value, skew_value, kurt_value, min_position_empirical,
                         max_position_empirical):
    """Calculate dimensionless factors and overall efficiency."""
    D_min = np.min(doses)
    D_avg = np.mean(doses)
    D_median = np.median(doses)
    sigma = np.std(doses)
    skewness = abs(skew(doses))  # Use absolute skewness
    kurt = abs(kurtosis(doses)) + 3  # Adjust kurtosis to ensure positive contribution

    min_penalty = (D_min / D_avg) * (1 - np.exp(-10 * D_min / D_avg))
    asymmetry_penalty = 1 - abs((D_avg - D_median) / sigma)

    # Normalize and ensure positive contributions
    norm_skewness = 1 - skewness
    norm_kurtosis = 1 - abs((kurt / 3) - 1)
    norm_cv = 1 - (sigma / D_avg)

    # Ensure all normalization values are within [0, 1]
    norm_skewness = max(0, norm_skewness)
    norm_kurtosis = max(0, norm_kurtosis)
    norm_cv = max(0, norm_cv)

    # Calculate efficiency
    efficiency = min_penalty * asymmetry_penalty * norm_skewness * norm_kurtosis * norm_cv

    return efficiency, min_penalty, asymmetry_penalty, norm_skewness, norm_kurtosis, norm_cv


def main():
    # Load the data
    # data = load_data(path + "DDF_bad.csv") # Bad distribution
    data = load_data(path + "DDF.csv") # Good distribution
    dose_values = data["range[mJ/cm^2]"]
    distribution_values = data["u(D)"]

    # Calculate empirical moments
    empirical_mean, empirical_var, empirical_skew, empirical_kurt = calculate_empirical_moments(dose_values,
                                                                                                distribution_values)
    max_position_empirical = dose_values[np.argmax(distribution_values)]
    min_position_empirical = dose_values[distribution_values.ne(0).idxmax()]

    # Optimize parameters
    alpha_opt, beta_opt, loc_opt = optimize_parameters(dose_values, distribution_values, empirical_mean, empirical_var,
                                                       empirical_skew, empirical_kurt, min_position_empirical,
                                                       max_position_empirical)

    # Generate x values for plotting the fitted distribution
    x = np.linspace(0, max(dose_values), 10000)

    # Calculate fitted moments and PDF
    mean_value, var_value, skew_value, kurt_value, pdf_fitted = calculate_fitted_moments(x, alpha_opt, beta_opt,
                                                                                         loc_opt)

    # Normalize the fitted PDF to match the height of the original data
    scaling_factor = max(distribution_values) / max(pdf_fitted)
    pdf_fitted_scaled = pdf_fitted * scaling_factor

    # Calculate efficiency
    efficiency, min_penalty, asymmetry_penalty, norm_skewness, norm_kurtosis, norm_cv = calculate_efficiency(
        dose_values, mean_value, var_value, skew_value, kurt_value, min_position_empirical, max_position_empirical)

    # Plot the histogram and the fitted distribution
    plt.figure(figsize=(10, 6))
    plt.bar(dose_values, distribution_values, width=20, alpha=0.5, label='Data')
    plt.plot(x, pdf_fitted_scaled, 'r-', lw=2, label='Optimized & Scaled Inverse Gamma Distribution')
    plt.axvline(x=mean_value, color='b', linestyle='--', label='Mean Value')
    plt.axvline(x=empirical_mean, color='orange', linestyle='--', label='Empirical Mean')
    plt.axvline(x=min_position_empirical, color='g', linestyle='--', label='Minimum Value')
    plt.axvline(x=max_position_empirical, color='purple', linestyle='--', label='Empirical Max Position')
    plt.xlim(left=0)
    plt.xlabel('Dose [mJ/cm^2]')
    plt.ylabel('u(D)')
    plt.title('Dose Distribution and Optimized & Scaled Inverse Gamma Distribution')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print results
    print(f"Optimized Alpha (Shape): {alpha_opt}")
    print(f"Optimized Beta (Scale): {beta_opt}")
    print(f"Optimized Loc (Location): {loc_opt}")
    print(f"Mean Value: {mean_value}")
    print(f"Variance: {var_value}")
    print(f"Skewness: {skew_value}")
    print(f"Kurtosis: {kurt_value}")
    print(f"Minimum Value (Empirical): {min_position_empirical}")
    print(f"Empirical Max Position: {max_position_empirical}")
    print(f"Empirical Mean Value: {empirical_mean}")
    print(f"Empirical Variance: {empirical_var}")
    print(f"Empirical Skewness: {empirical_skew}")
    print(f"Empirical Kurtosis: {empirical_kurt}")

    print(f"Min Penalty: {min_penalty}, Asymmetry Penalty: {asymmetry_penalty}")
    print(f"Norm Skewness: {norm_skewness}, Norm Kurtosis: {norm_kurtosis}, Norm CV: {norm_cv}")
    print(f"Efficiency: {efficiency:.2f}")

    # Create a DataFrame with the report
    report = pd.DataFrame([
        'Alpha', 'Beta', 'Loc', 'Mean', 'Var', 'Skew', 'Kurt', 'Min', 'Max',
        'Empirical Mean', 'Empirical Var', 'Empirical Skew', 'Empirical Kurt',
        'Min Penalty', 'Asymmetry Penalty', 'Norm Skewness', 'Norm Kurtosis', 'Norm CV',
        'Overall Efficiency'
    ], columns=['Description'])
    report['Value'] = [
        alpha_opt, beta_opt, loc_opt, mean_value, var_value, skew_value, kurt_value,
        min_position_empirical, max_position_empirical, empirical_mean, empirical_var,
        empirical_skew, empirical_kurt, min_penalty, asymmetry_penalty, norm_skewness,
        norm_kurtosis, norm_cv, efficiency
    ]
    report = report.round(2)
    report = report.set_index('Description')
    print(tabulate(report, headers='keys', tablefmt='heavy_outline'))


if __name__ == "__main__":
    main()
