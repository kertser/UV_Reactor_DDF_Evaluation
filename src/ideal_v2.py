import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.stats import skew, kurtosis
from tabulate import tabulate

# Define parameters for the ideal delta function
ideal_loc = 320.0  # Minimum value
ideal_width = 1.0  # Very narrow width to simulate a delta function

# Increase the number of points for more precise analysis
x = np.linspace(0, 1000, 10000)  # Increased from 1000 to 10000 points

# Create the ideal distribution: a very narrow Gaussian for visualization purposes
ideal_pdf = np.exp(-0.5 * ((x - ideal_loc) / ideal_width) ** 2)
ideal_pdf /= trapezoid(ideal_pdf, x)  # Normalize to ensure area under curve is 1

# Calculate the moments of the ideal distribution numerically
mean_value_ideal = trapezoid(x * ideal_pdf, x)
var_value_ideal = trapezoid((x - mean_value_ideal) ** 2 * ideal_pdf, x)
std_value_ideal = np.sqrt(var_value_ideal)
skew_value_ideal = trapezoid((x - mean_value_ideal) ** 3 * ideal_pdf, x) / std_value_ideal ** 3
kurt_value_ideal = trapezoid((x - mean_value_ideal) ** 4 * ideal_pdf, x) / var_value_ideal ** 2 - 3

# Round-up to zero
if skew_value_ideal < 0: skew_value_ideal = 0
if kurt_value_ideal < 0: kurt_value_ideal = 0

# Calculate dimensionless factors for the ideal distribution
dsl_ideal = (mean_value_ideal - ideal_loc) / mean_value_ideal
cv_ideal = std_value_ideal / mean_value_ideal

# Calculate TCV for the ideal distribution
tail_threshold = 1e-6  # Probability threshold to consider for tail
tail_indices = ideal_pdf > tail_threshold
tail_values_ideal = x[tail_indices]
tail_pdf_ideal = ideal_pdf[tail_indices]

# Debug: Print tail values and corresponding PDF values
print(f"Tail Values (Ideal): {tail_values_ideal}")
print(f"Tail PDF (Ideal): {tail_pdf_ideal}")

if len(tail_values_ideal) > 0:
    mean_tail_ideal = trapezoid(tail_values_ideal * tail_pdf_ideal, tail_values_ideal)
    var_tail_ideal = trapezoid((tail_values_ideal - mean_tail_ideal) ** 2 * tail_pdf_ideal, tail_values_ideal)
    std_tail_ideal = np.sqrt(var_tail_ideal)
    tcv_ideal = std_tail_ideal / mean_tail_ideal if mean_tail_ideal != 0 else 0
else:
    tcv_ideal = 0

# Plotting the ideal distribution
plt.figure(figsize=(10, 6))
plt.plot(x, ideal_pdf, 'r-', lw=2, label='Ideal Distribution (Delta Function)')
plt.axvline(x=ideal_loc, color='g', linestyle='--', label='Minimum Value')
plt.xlabel('Dose [mJ/cm^2]')
plt.ylabel('Probability Density')
plt.title('Ideal Distribution Representing a 100% Efficient Reactor')
plt.legend()
plt.grid(True)
plt.show()

# Print the calculated values
print(f"Ideal Mean Value: {mean_value_ideal}")
print(f"Ideal Variance: {var_value_ideal}")
print(f"Ideal Skewness: {skew_value_ideal}")
print(f"Ideal Kurtosis: {kurt_value_ideal}")
print(f"DSL (Trajectory Uniformity Factor): {dsl_ideal:.2f}")
print(
    "DSL: Indicator of how close the dose distribution of the reactor is to the ideal. The closer to 0, the more efficient the reactor.")
print(f"CV (Coefficient of Variation): {cv_ideal:.2f}")
print(
    "CV: Coefficient of variation, shows the relative variability of doses. A lower value indicates a narrower distribution, characteristic of an efficient reactor.")
print(f"TCV (Tail Coefficient of Variation): {tcv_ideal:.2f}")
print(
    "TCV: Tail coefficient of variation, shows how long the tails of the distribution are. A lower value indicates shorter tails, characteristic of an efficient reactor.")

# Calculate the overall efficiency based on the dimensionless factors
# Adjust weights to ensure ideal reactor shows close to 99% efficiency
weights = {'DSL': 0.5, 'CV': 0.25, 'TCV': 0.25}
efficiency_ideal = (weights['DSL'] * (1 - dsl_ideal) +
                    weights['CV'] * (1 - cv_ideal) +
                    weights['TCV'] * (1 - tcv_ideal)) * 100

print(f"Overall Efficiency: {efficiency_ideal:.2f}%")
print(
    "Overall Efficiency: A composite measure considering DSL, CV, and TCV. The closer to 100%, the more efficient the reactor.")


# Additional components for new criteria:
def reactor_efficiency(doses, k=10):
    D_min = np.min(doses)
    D_avg = np.mean(doses)
    D_median = np.median(doses)
    sigma = np.std(doses)
    skewness = abs(skew(doses))  # Use absolute skewness
    kurt = abs(kurtosis(doses)) + 3  # Adjust kurtosis to ensure positive contribution

    min_penalty = (D_min / D_avg) * (1 - np.exp(-k * D_min / D_avg))
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

    print(
        f"D_min: {D_min}, D_avg: {D_avg}, D_median: {D_median}, Sigma: {sigma}, Skewness: {skewness}, Kurtosis: {kurt}")
    print(f"Min Penalty: {min_penalty}, Asymmetry Penalty: {asymmetry_penalty}")
    print(f"Norm Skewness: {norm_skewness}, Norm Kurtosis: {norm_kurtosis}, Norm CV: {norm_cv}")
    print(f"Efficiency: {efficiency}")

    return {
        'D_min': D_min,
        'D_avg': D_avg,
        'D_median': D_median,
        'Sigma': sigma,
        'Skewness': skewness,
        'Kurtosis': kurt,
        'Min_Penalty': min_penalty,
        'Asymmetry_Penalty': asymmetry_penalty,
        'Efficiency': efficiency
    }


# Calculate efficiency using the new criteria
doses = np.random.normal(ideal_loc, ideal_width, 10000)  # Increased from 1000 to 10000 points
efficiency_data = reactor_efficiency(doses)

# Prepare the report
report = pd.DataFrame([
    ['Alpha', None],
    ['Beta', None],
    ['Loc', ideal_loc],
    ['Mean', mean_value_ideal],
    ['Var', var_value_ideal],
    ['Skew', skew_value_ideal],
    ['Kurt', kurt_value_ideal],
    ['Min', np.min(doses)],
    ['Max', np.max(doses)],
    ['Empirical Mean', np.mean(doses)],
    ['Empirical Var', np.var(doses)],
    ['Empirical Skew', abs(skew(doses))],  # Use absolute value
    ['Empirical Kurt', abs(kurtosis(doses))],  # Use absolute value
    ['DSL', dsl_ideal],
    ['CV', cv_ideal],
    ['TCV', tcv_ideal],
    ['IGF', efficiency_ideal],  # Assuming IGF is the overall efficiency
    ['Overall Efficiency', efficiency_data['Efficiency']]
], columns=['Description', 'Value'])

report = report.round(2)
report = report.set_index('Description')

# Print the report using tabulate
print(tabulate(report, headers='keys', tablefmt='heavy_outline'))
