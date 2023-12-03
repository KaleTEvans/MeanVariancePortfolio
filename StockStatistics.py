import numpy as np
import matplotlib.pyplot as plt
import math

class StockStatistics:
    def __init__(self, stock_data):
        self.stock_data = stock_data
        self.stock_returns = {}
        self.stock_statistics = {}
        self.annual_statistics = {}
        self.get_arithmetic_returns()
        self.get_mean_variance()

    def get_arithmetic_returns(self):
        # Sort dates to ensure organization
        sorted_dates = sorted(self.stock_data.keys())

        for i in range(1, len(sorted_dates)):
            cur_date = sorted_dates[i]
            prev_date = sorted_dates[i-1]

            daily_returns = {}

            for stock in self.stock_data[cur_date]:
                cur_price = self.stock_data[cur_date][stock]
                prev_price = self.stock_data[prev_date][stock]

                daily_returns[stock] = self.calculate_arithmetic_returns(cur_price, prev_price)

            # Add daily returns to the dictionary
            self.stock_returns[cur_date] = daily_returns

    def calculate_arithmetic_returns(self, cur_price, prev_price):
        return ((cur_price / prev_price) - 1)
    
    def get_mean_variance(self):

        # Collect all stock names
        stock_names = list(self.stock_returns[next(iter(self.stock_returns))].keys())

        # Calculate mean and variance for each stock's returns
        for stock in stock_names:
            # Extract returns for current stock
            all_returns = [day_returns[stock] for date, day_returns in self.stock_returns.items() if stock in day_returns]

            # Calculate mean and variance using numpy
            mean_return = np.mean(all_returns)
            variance = np.var(all_returns)

            annual_mean = ((mean_return + 1) ** 252) - 1
            annual_variance = ((variance + ((mean_return + 1) ** 2)) ** 252) - ((mean_return + 1) ** (252 * 2))

            # Add values to the dictionary
            self.stock_statistics[stock] = {
                'mean' : mean_return,
                'variance' : variance,
                'annual_mean' : annual_mean,
                'annual_variance' : annual_variance
            }
    
    def get_mean_vector(self):
        mean_values = []
        
        for stats in self.stock_statistics.values():
            mean_values.append(stats['mean'])

        mean_vector = np.array(mean_values)
        return mean_vector
    
    def get_annual_mean_vector(self):
        mean_values = []
        
        for stats in self.stock_statistics.values():
            mean_values.append(stats['annual_mean'])

        mean_vector = np.array(mean_values)
        return mean_vector
    
    def get_cov_matrix(self):
        stock_names = list(self.stock_returns[next(iter(self.stock_returns))].keys())
        n = len(stock_names)
        cov_matrix = np.zeros((n, n)) # Initialize a matrix of zeroes

        for i, stock1 in enumerate(stock_names):
            for j in range(i, n):
                stock2 = stock_names[j]
                if i == j: # Variance on the diagonal
                    cov_matrix[i, j] = self.stock_statistics[stock1]['variance']
                else:
                    covariance = self.calculate_covariance(stock1, stock2)
                    cov_matrix[i, j] = cov_matrix[j, i] = covariance # Since this is a symmetrical matrix
                    
        return cov_matrix

    def calculate_covariance(self, stock1, stock2):
        covariance_sum = 0.0

        # Number of observations
        n = len(self.stock_returns)

        for date in self.stock_returns:
            stock1_returns = self.stock_returns[date][stock1]
            stock2_returns = self.stock_returns[date][stock2]
            covariance_sum += (stock1_returns - self.stock_statistics[stock1]['mean']) * (stock2_returns - self.stock_statistics[stock2]['mean'])

        covariance = covariance_sum / (n - 1)

        return covariance

    def get_annualized_cov_matrix(self):
        stock_names = list(self.stock_returns[next(iter(self.stock_returns))].keys())
        n = len(stock_names)
        annual_cov_matrix = np.zeros((n, n)) # Initialize a matrix of zeroes

        for i, stock1 in enumerate(stock_names):
            for j in range(i, n):
                stock2 = stock_names[j]
                if i == j:
                    annual_cov_matrix[i, j] = self.stock_statistics[stock1]['annual_variance']
                else :
                    mean1 = self.stock_statistics[stock1]['mean']
                    mean2 = self.stock_statistics[stock2]['mean']
                    var1 = self.stock_statistics[stock1]['variance']
                    var2 = self.stock_statistics[stock2]['variance']

                    # Now the formula for annual covariance is a little trickier
                    daily_covariance = self.calculate_covariance(stock1, stock2)

                    annual_covariance = ((daily_covariance + ((mean1 + 1) * (mean2 + 1))) ** 252) - (((mean1 + 1) ** 252) * ((mean2 + 1) ** 252))
                    annual_cov_matrix[i, j] = annual_cov_matrix[j, i] = annual_covariance

        return annual_cov_matrix
    
    def get_global_min_variance_values(self, cov_matrix, annual=True):
        # First find the weights vector
        n = cov_matrix.shape[0]
        ones_vector = np.ones(n)

        inversed_cov_matrix = np.linalg.inv(cov_matrix)

        numerator = inversed_cov_matrix @ ones_vector
        denominator = ones_vector.T @ inversed_cov_matrix @ ones_vector

        gm_weights = numerator / denominator

        # Then get the expected return from the annual mean
        if annual is True:
            gm_mean = self.get_annual_mean_vector()
        else:
            gm_mean = self.get_mean_vector()

        # Now find gm returns and variance
        expected_return = gm_weights.T @ gm_mean
        gm_variance = gm_weights.T @ cov_matrix @ gm_weights

        # Take square root to get stdev ie volatility
        gm_volatility = math.sqrt(gm_variance)

        # Determine psi for target mean and variance
        term1 = ones_vector.T @ inversed_cov_matrix @ ones_vector # Labeled PsiA
        term2 = gm_mean.T @ inversed_cov_matrix @ gm_mean # Labeled PsiC
        term3 = ones_vector.T @ inversed_cov_matrix @ gm_mean # Labeled PsiB
        psi =  (term1 * term2) - (term3 ** 2)

        gm_values = {
            'weights' : gm_weights,
            'return' : expected_return,
            'variance' : gm_variance,
            'volatility' : gm_volatility,
            'psi' : psi,
            'psi_A' : term1,
            'psi_B' : term3,
            'psi_C' : term2
        }

        return gm_values
    
    def find_weights_from_target_return(self, cov_matrix, gm_values, target):
        # The formula to determine the weighting for a target return is the soluition to the Markowitz problem
        mean_vector = self.get_annual_mean_vector()
        psi = gm_values['psi']

        psi_A = gm_values['psi_A']
        psi_B = gm_values['psi_B']
        psi_C = gm_values['psi_C']

        # Solve for the lagrange multipliers below
        lambda_one = (psi_B * target - psi_C) / psi
        lambda_two = (psi_B - (psi_A * target)) / psi

        # get the inversed cov matrix
        inversed_cov_matrix = np.linalg.inv(cov_matrix)

        n = cov_matrix.shape[0]
        ones_vector = np.ones(n)

        term1 = lambda_one * (inversed_cov_matrix @ ones_vector)
        term2 = lambda_two * (inversed_cov_matrix @ mean_vector)

        return (-term1 - term2)

    
    def get_target_stdev(self, gm_values, target_return):
        target_variance = (gm_values['variance']) + (((target_return - gm_values['return']) ** 2) / (gm_values['psi'] * gm_values['variance']))
        target_stdev = math.sqrt(target_variance)
        return target_stdev
    
    def plot_mean_variance_curve(self, gm_values):
        # Define the range of target returns
        target_returns = np.linspace(-1.1, 1.1, 100)

        # Calculate the target stdev
        target_stdev = np.array([self.get_target_stdev(gm_values, tr) for tr in target_returns])

        # Plot the efficient frontier
        plt.figure(figsize=(10, 6))
        plt.plot(target_stdev, target_returns, label='Efficient Frontier')

        # Plot the individual stocks' volatility and return
        for stock, stats in self.stock_statistics.items():
            plt.scatter(np.sqrt(stats['annual_variance']), stats['annual_mean'], label=f"{stock} (Stock)")

        # Additional plot settings
        plt.title('Mean-Variance Efficient Frontier')
        plt.xlabel('Volatility (Standard Deviation)')
        plt.ylabel('Expected Return')
        plt.legend()
        plt.grid(True)
        plt.show()