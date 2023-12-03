import csv
from datetime import datetime
import pandas as pd
import numpy as np
import math

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
# Lambda is a diagonal matrix containing the eigenvalues of the covariance matrix
# Find eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(returns_cov_matrix)
print(np.sort(eigenvalues))
print(eigenvectors)


    