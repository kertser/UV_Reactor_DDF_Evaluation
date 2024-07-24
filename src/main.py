import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invgamma
from scipy.optimize import minimize
from scipy.integrate import simpson
import warnings
from config import resource_path as path

# Ignore runtime warnings
warnings.filterwarnings("ignore")

# Load the data from the uploaded CSV file
data = pd.read_csv(path + "DDF.csv")

# Extract the dose values and distribution values
dose_values = data["range[mJ/cm^2]"]
distribution_values = data["u(D)"]

# Find the position of the maximum and minimum in the empirical data
max_position_empirical = dose_values[np.argmax(distribution_values)]
min_position_empirical = dose_values[distribution_values.ne(0).idxmax()]

# Calculate empirical moments
empirical_mean = np.sum(dose_values * distribution_values) / np.sum(distribution_values)
empirical_var = np.sum(distribution_values * (dose_values - empirical_mean)**2) / np.sum(distribution_values)
empirical_std = np.sqrt(empirical_var)
empirical_skew = np.sum(distribution_values * (dose_values - empirical_mean)**3) / (np.sum(distribution_values) * empirical_std**3)
empirical_kurt = np.sum(distribution_values * (dose_values - empirical_mean)**4) / (np.sum(distribution_values) * empirical_var**2) - 3

# Define the objective function for least squares optimization with weighting for the right tail
def objective(params, x, data, weight_factor, empirical_mean, empirical_var, empirical_skew, empirical_kurt):
    alpha, beta, loc = params
    fitted_pdf = invgamma.pdf(x, alpha, loc=loc, scale=beta)
    weights = np.linspace(1, weight_factor, len(data))  # Increase the weight for the right tail
    weighted_loss = np.sum(weights * (data - fitted_pdf) ** 2)
    # Calculate numerical moments
    mean_fitted = simpson(y=x * fitted_pdf, x=x)
    var_fitted = simpson(y=(x - mean_fitted)**2 * fitted_pdf, x=x)
    std_fitted = np.sqrt(var_fitted)
    skew_fitted = simpson(y=(x - mean_fitted)**3 * fitted_pdf, x=x) / std_fitted**3
    kurt_fitted = simpson(y=(x - mean_fitted)**4 * fitted_pdf, x=x) / var_fitted**2 - 3
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

# Define the grid search space
alpha_range = np.linspace(1.0, 2.0, 10)  # More dense range for alpha
beta_range = np.linspace(150, 250, 10)   # More dense range for beta
loc_range = [min_position_empirical]
weight_factor_range = [10, 15, 20, 25]

best_initial_guess = None
best_weight_factor = None
best_loss = float('inf')

# Perform grid search
for alpha in alpha_range:
    for beta in beta_range:
        for loc in loc_range:
            for weight_factor in weight_factor_range:
                initial_guess = [alpha, beta, loc]
                result = minimize(objective, initial_guess, args=(dose_values, distribution_values, weight_factor, empirical_mean, empirical_var, empirical_skew, empirical_kurt),
                                  bounds=((1e-5, None), (1e-5, None), (None, None)),
                                  options={'maxiter': 10000, 'disp': False})
                if result.fun < best_loss:
                    best_loss = result.fun
                    best_initial_guess = initial_guess
                    best_weight_factor = weight_factor

print(f"Best initial guess: {best_initial_guess}")
print(f"Best weight factor: {best_weight_factor}")
print(f"Best loss: {best_loss}")

# Perform the final optimization with the best initial guess
result = minimize(objective, best_initial_guess, args=(dose_values, distribution_values, best_weight_factor, empirical_mean, empirical_var, empirical_skew, empirical_kurt),
                  bounds=((1e-5, None), (1e-5, None), (None, None)),
                  options={'maxiter': 10000, 'disp': True})

# Extract optimized parameters
alpha_opt, beta_opt, loc_opt = result.x

# Generate x values for plotting the fitted distribution, starting from zero
x = np.linspace(0, max(dose_values), 1000)

# Calculate the inverse gamma PDF values using the optimized parameters
pdf_fitted_optimized = invgamma.pdf(x, alpha_opt, loc=loc_opt, scale=beta_opt)

# Normalizing the fitted PDF to match the height of the original data
scaling_factor = max(distribution_values) / max(pdf_fitted_optimized)
pdf_fitted_optimized_scaled = pdf_fitted_optimized * scaling_factor

# Calculate the moments of the fitted distribution numerically
mean_value = simpson(y=x * pdf_fitted_optimized, x=x)
var_value = simpson(y=(x - mean_value)**2 * pdf_fitted_optimized, x=x)
std_value = np.sqrt(var_value)
skew_value = simpson(y=(x - mean_value)**3 * pdf_fitted_optimized, x=x) / std_value**3
kurt_value = simpson(y=(x - mean_value)**4 * pdf_fitted_optimized, x=x) / var_value**2 - 3

# Calculate dimensionless factors
tuf = (mean_value - min_position_empirical) / mean_value
cv = std_value / mean_value

# Calculate TCV with a consistent threshold
tail_threshold = 1e-6  # Probability threshold to consider for tail
tail_indices_real = pdf_fitted_optimized > tail_threshold
tail_values_real = x[tail_indices_real]
tail_pdf_real = pdf_fitted_optimized[tail_indices_real]

mean_tail_real = simpson(y=tail_values_real * tail_pdf_real, x=tail_values_real)
var_tail_real = simpson(y=(tail_values_real - mean_tail_real)**2 * tail_pdf_real, x=tail_values_real)
std_tail_real = np.sqrt(var_tail_real)
tcv_real = std_tail_real / mean_tail_real if mean_tail_real != 0 else np.nan

# Plotting the histogram and the fitted distribution with optimized parameters and scaling
plt.figure(figsize=(10, 6))
plt.bar(dose_values, distribution_values, width=20, alpha=0.5, label='Data')
plt.plot(x, pdf_fitted_optimized_scaled, 'r-', lw=2, label='Optimized & Scaled Inverse Gamma Distribution')

# Add vertical lines for mean and minimum values
plt.axvline(x=mean_value, color='b', linestyle='--', label='Mean Value')
plt.axvline(x=empirical_mean, color='orange', linestyle='--', label='Empirical Mean')
plt.axvline(x=min_position_empirical, color='g', linestyle='--', label='Minimum Value')
plt.axvline(x=max_position_empirical, color='purple', linestyle='--', label='Empirical Max Position')

plt.xlim(left=0)  # Set x-axis to start at zero
plt.xlabel('Dose [mJ/cm^2]')
plt.ylabel('u(D)')
plt.title('Dose Distribution and Optimized & Scaled Inverse Gamma Distribution')
plt.legend()
plt.grid(True)
plt.show()

# Print the optimized parameters and calculated values
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

# Print the dimensionless factors with interpretations
print(f"TUF (Trajectory Uniformity Factor): {tuf:.2f}")
print("TUF: Indicator of how close the dose distribution is to the ideal distribution. The closer to 1, the more efficient the reactor.")
print(f"CV (Coefficient of Variation): {cv:.2f}")
print("CV: Coefficient of variation, shows the relative variability of doses. Lower value indicates a narrower distribution, characteristic of an efficient reactor.")
print(f"TCV (Tail Coefficient of Variation): {tcv_real:.2f}")
print("TCV: Coefficient of variation for the tails of the distribution, shows how long the tails are. Lower value indicates shorter tails, characteristic of an efficient reactor.")

# Calculate the overall efficiency based on the dimensionless factors
# Assuming Gaussian distribution has 0% efficiency
# Here we are calculating a simple weighted sum for demonstration, the weights can be adjusted as needed
weights = {'TUF': 0.4, 'CV': 0.4, 'TCV': 0.2}
efficiency = (weights['TUF'] * (1 - abs(tuf - 1)) +
              weights['CV'] * (1 - cv) +
              weights['TCV'] * (1 - tcv_real)) * 100

print(f"Overall Efficiency: {efficiency:.2f}%")
print("Overall Efficiency: A comprehensive indicator that takes into account TUF, CV, and TCV. The closer to 100%, the more efficient the reactor.")
