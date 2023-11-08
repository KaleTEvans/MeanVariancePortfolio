import csv
from datetime import datetime
import numpy as np

from StockStatistics import StockStatistics

stock_data = {}

# Open the CSV file
with open('StockData.csv', newline='') as csvfile:
    # Create a CSV reader object
    csvreader = csv.reader(csvfile)
    
    # Read the header row to get stock names
    headers = next(csvreader)
    stock_names = headers[1:]  

    # Loop over each row in the CSV
    for row in csvreader:
        # Read the date from the first column and convert it to a date object
        date = datetime.strptime(row[0], '%m/%d/%Y').date()
        
        # Create a dictionary for stock prices on that date
        prices = {stock_name: float(price) for stock_name, price in zip(stock_names, row[1:])}
        
        # Add the dictionary of prices to our stock_data under the key of that date
        stock_data[date] = prices

# Now stock_data contains all the data organized by date
# print(stock_data)

stock_stats = StockStatistics(stock_data)

mean_variance = stock_stats.stock_statistics

# Print the results
for stock, stats in mean_variance.items():
    print(f"Stock: {stock}")
    print(f"  Mean Return: {stats['mean']:.8f}")
    print(f"  Variance of Return: {stats['variance']:.8f}")

mean_vector = stock_stats.get_mean_vector()

# print(mean_vector)

# cov_matrix = stock_stats.get_cov_matrix()
# print(cov_matrix)

annual_cov_matrix = stock_stats.get_annualized_cov_matrix()
print(annual_cov_matrix)

gm_values = stock_stats.get_global_min_variance_values(annual_cov_matrix, True)
print("Weight Vector: ", gm_values['weights'])
print("Expected Return: ", gm_values['return'])
print("Variance: ", gm_values['variance'])
print("Volaitlity: ", gm_values['volatility'])