import numpy as np

class StockStatistics:
    def __init__(self, stock_data):
        self.stock_data = stock_data
        self.stock_returns = {}
        self.stock_statistics = {}
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

            # Add values to the dictionary
            self.stock_statistics[stock] = {
                'mean' : mean_return,
                'variance' : variance
            }
    
    def get_mean_vector(self):
        mean_values = []
        
        for stats in self.stock_statistics.values():
            mean_values.append(stats['mean'])

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
