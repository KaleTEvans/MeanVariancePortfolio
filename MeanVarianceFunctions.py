import math
import numpy as np

def calculate_tangency_portfolio_weights(mean_vector, cov_matrix, rf_rate):
    n = cov_matrix.shape[0]
    ones_vector = np.ones(n)

    inv_cov_matrix = np.linalg.inv(cov_matrix)

    mean_excess = mean_vector - (rf_rate * ones_vector)

    numerator = inv_cov_matrix @ mean_excess
    denominator = ones_vector.T @ inv_cov_matrix @ mean_excess
    return numerator / denominator


mean_vector = np.array([0.2, 0.13])
stdev1 = 0.38
stdev2 = 0.12
var1 = 0.38 ** 2
var2 = 0.12 ** 2
cov = 0.7 * stdev1 * stdev2
cov_matrix = np.array([[var1, cov], [cov, var2]])

print(cov_matrix)
print()

weights = calculate_tangency_portfolio_weights(mean_vector, cov_matrix, 0.02)
print(weights)

print(weights.T @ cov_matrix @ weights)
print(0.1 + weights.T @ (mean_vector - (0.1 * np.array([1, 1]))))
print()
print(np.linalg.inv(cov_matrix))
print()
print(np.linalg.inv(cov_matrix) @ (mean_vector - (0.1 * np.array([1, 1]))))

risk_tolerance = 0.473815
term1 = ((10/17) * 0.2)
term2 = ((3/17) * 1.25)
multiplier = (17/4)

solution = multiplier * (risk_tolerance - term1 - term2)
print(solution)

print()
term3 = ((4/17) * 0.5762)
print(term1 + term2 + term3)

mean_e = np.linalg.inv(cov_matrix) @ (mean_vector - (0.1 * np.array([1, 1])))
w_1 = 0.2 * mean_e
print(1 - (w_1 @ np.array([1, 1])))
w_2 = 1.25 * mean_e
print(1 - (w_2 @ np.array([1, 1])))
w_3 = solution * mean_e
print(1 - (w_3 @ np.array([1, 1])))