import csv
from datetime import datetime
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

stock_data = {}

# Open the CSV file
with open('HW7Data.csv', newline='') as csvfile:
    # Create a CSV reader object
    csvreader = csv.reader(csvfile)
    
    # Get headers for each column
    headers = next(csvreader)
    stock_names = headers[1:]  

    # Loop over each row in the CSV
    for row in csvreader:
        # Read the date from the first column and convert it to a date object
        date = datetime.strptime(row[0], '%m/%d/%Y').date()
        
        # Create a dictionary for returns and factors
        values = {column_name: float(price) for column_name, price in zip(stock_names, row[1:])}
        
        stock_data[date] = values

# Gather returns and factor matrices
returns = []
excess_returns = []
factors = []

for date in stock_data:
    return_1 = stock_data[date]['gm']
    return_2 = stock_data[date]['adbe']
    return_3 = stock_data[date]['ora']
    return_4 = stock_data[date]['flo']

    r1 = stock_data[date]['gm'] - stock_data[date]['rf']
    r2 = stock_data[date]['adbe'] - stock_data[date]['rf']
    r3 = stock_data[date]['ora'] - stock_data[date]['rf']
    r4 = stock_data[date]['flo'] - stock_data[date]['rf']

    mkt_rf = stock_data[date]['mkt_rf']
    smb = stock_data[date]['smb']
    hml = stock_data[date]['hml']

    return_data = np.array([return_1, return_2, return_3, return_4])
    returns.append(return_data)

    excess_return_data = np.array([r1, r2, r3, r4])
    excess_returns.append(excess_return_data)

    factor = np.array([1, mkt_rf, smb, hml])
    factors.append(factor)

# Regular returns matrix
returns_matrix = np.asarray(returns)

# R Matrix
excess_returns_matrix = np.asarray(excess_returns)
# F Matrix
factor_matrix = np.asarray(factors)
factor_matrix_transpose = factor_matrix.T

inverse_factor_matrix = np.linalg.inv(factor_matrix_transpose @ factor_matrix)
ols_regression = inverse_factor_matrix @ factor_matrix_transpose @ excess_returns_matrix
print("Regression Results")
print(ols_regression)

residuals_matrix = excess_returns_matrix - (factor_matrix @ ols_regression)
# print(residuals_matrix)

def get_covariance_matrix(matrix):
    column_means = np.mean(matrix, axis=0)
    column_variance = np.var(matrix, axis=0)

    n = len(column_means)
    l = len(matrix)
    cov_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            if i == j: # Variance on the diagonal
                cov_matrix[i, j] = column_variance[i]
            else:
                # Find the covariance
                covariance_sum = 0.0
                for k in range(matrix.shape[0]):
                    val1 = matrix[k, i]
                    val1_mean = column_means[i]
                    val2 = matrix[k, j]
                    val2_mean = column_means[j]

                    covariance_sum += (val1 - val1_mean) * (val2 - val2_mean)
                
                covariance = covariance_sum / l
                cov_matrix[i, j] = cov_matrix[j, i] = covariance

    return cov_matrix

# This function will be used to find the correlation matrices
def get_correlation_matrix(matrix):
    # First, we need to get the mean and variance of each column
    column_means = np.mean(matrix, axis=0)
    column_variance = np.var(matrix, axis=0)
    
    # Get cov matrix
    cov_matrix = get_covariance_matrix(matrix)

    n = len(cov_matrix)
    correlation_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            cov = cov_matrix[i, j]
            var1 = column_variance[i]
            var2 = column_variance[j]
            correlation = cov / math.sqrt(var1 * var2)
            correlation_matrix[i, j] = correlation_matrix[j, i] = correlation

    return correlation_matrix

returns_correlation = get_correlation_matrix(excess_returns_matrix)
residuals_correlation = get_correlation_matrix(residuals_matrix)

print("=======================")
print("Returns Correlation")
print(returns_correlation)
print("=======================")
print("Residuals Correlation")
print(residuals_correlation)
print("=======================")

# For principal components analysis, we need to look at just the returns and not the excess returns
# First get the vector for the means
mean_vector_of_returns = np.mean(returns_matrix, axis=0)
# Then get the covariance matrix
returns_cov_matrix = get_covariance_matrix(returns_matrix)

print("Covariance Matrix of Returns")
print(returns_cov_matrix)
print("=======================")

# Lambda is a diagonal matrix containing the eigenvalues of the covariance matrix
# Find eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(returns_cov_matrix)

# We need to sort the eigenvalues from largest to smallest
# Get the index values from the sort
idx = eigenvalues.argsort()[::-1]
sorted_eigenvalues = eigenvalues[idx]
sorted_eigenvectors = eigenvectors[:, idx]

# Our gamma matrix is the sorted eigenvectors matrix
gamma_matrix = sorted_eigenvectors
print("Gamma Matrix")
print(gamma_matrix)
print("======================")
# Create the lambda matrix
lambda_matrix = np.array([[sorted_eigenvalues[0], 0, 0, 0],
                          [0, sorted_eigenvalues[1], 0, 0],
                          [0, 0, sorted_eigenvalues[2], 0],
                          [0, 0, 0, sorted_eigenvalues[3]]])

gamma_transpose = gamma_matrix.T

# To get the principal components, we need to subtract the mean vector from each row of returns
mean_subtracted_returns = []
n = len(returns_matrix)
for i in range(n):
    r1 = returns_matrix[i, 0] - mean_vector_of_returns[0]
    r2 = returns_matrix[i, 1] - mean_vector_of_returns[1]
    r3 = returns_matrix[i, 2] - mean_vector_of_returns[2]
    r4 = returns_matrix[i, 3] - mean_vector_of_returns[3]

    mean_subtracted_returns.append([r1, r2, r3, r4])

# Note the transpose, as each r should be represented as a row
mean_subtracted_returns = np.asarray(mean_subtracted_returns).T

# # Now we can find Y
principal_components = gamma_transpose @ mean_subtracted_returns

# We take the first two rows for Y1 and Y2
# Now run a regression, using Y1 in Y2 as the F matrix, and the returns as the R
# Create the F matrix with a preceding column of 1s
PCA_F_Matrix = []
Y1_Vector = []
for i in range(principal_components.shape[1]):
    value = principal_components[:, i]

    F_data = np.array([1, value[0], value[1]])
    PCA_F_Matrix.append(F_data)
    Y1_Vector.append(value[0])

PCA_F_Matrix = np.asarray(PCA_F_Matrix)
PCA_F_Matrix_Transpose = PCA_F_Matrix.T
Y1_Vector = np.asarray(Y1_Vector)

# With the returns matrix as R, compute the regression
PCA_F_Inverse = np.linalg.inv(PCA_F_Matrix_Transpose @ PCA_F_Matrix)
PCA_B_Matrix = PCA_F_Inverse @ PCA_F_Matrix_Transpose @ returns_matrix

print("B Matrix Obtained From PCA")
print(PCA_B_Matrix)
print("=======================")

# Now we compute the residuals matrix
PCA_Residuals_Matrix = returns_matrix - (PCA_F_Matrix @ PCA_B_Matrix)

# Send to correlation function to retrieve correlation matrix
PCA_Residuals_Correlation = get_correlation_matrix(PCA_Residuals_Matrix)
PCA_Residuals_Covariance = get_covariance_matrix(PCA_Residuals_Matrix)

print("PCA Factor Model Residuals Correlation Matrix")
print(PCA_Residuals_Correlation)
print("=====================")
print("PCA Factor Model Residuals Covariance Matrix")
print(PCA_Residuals_Covariance)

# Now we wish to plot Y1 with the market returns
market_returns = []
for date in stock_data:
    rm = stock_data[date]['mkt_rf'] + stock_data[date]['rf']
    market_returns.append(rm)

market_returns = np.asarray(market_returns)

# Calculate correlation between Y1 and Market Returns
print(np.corrcoef(Y1_Vector, market_returns))

plt.scatter(Y1_Vector, market_returns)
plt.xlabel('First Principal Component (Y1)')
plt.ylabel('Market Return (rm)')
plt.title('First Principal Component Vs Stock Market Return')
plt.show()