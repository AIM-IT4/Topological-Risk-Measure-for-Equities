import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
from scipy.spatial.distance import euclidean


# Function to fetch stock data
def fetch_stock_data(ticker, period="1y"):
    stock_data = yf.download(ticker, period=period)
    return stock_data['Close']

# Your existing imports and functions remain the same.

# Add the new functions for normalization and VaR, CVaR calculation
def normalize_time_series(time_series):
    min_val = np.min(time_series)
    max_val = np.max(time_series)
    normalized_series = (time_series - min_val) / (max_val - min_val)
    return normalized_series

def calculate_returns(time_series):
    returns = np.diff(time_series)[1:] / time_series[1:-1]
    return returns

def calculate_var_cvar(returns, confidence_level=0.95):
    sorted_returns = np.sort(returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    var = sorted_returns[index]
    if index == 0:
        cvar = None
    else:
        cvar = sorted_returns[:index].mean()
    return var, cvar

# Main Program
if __name__ == "__main__":
    # Tickers for the 3 stocks
    tickers = ["AAPL", "MSFT", "GOOGL"]
    
    for ticker in tickers:
        time_series = fetch_stock_data(ticker)
        
        # Normalize the time series
        normalized_series = normalize_time_series(time_series)
        
        # Calculate daily returns
        daily_returns = calculate_returns(normalized_series)
        
        # Calculate VaR and CVaR
        var, cvar = calculate_var_cvar(daily_returns)
        print(f"VaR for {ticker} at 95% CI: {var}")
        print(f"CVaR for {ticker} at 95% CI: {cvar}")

        # Your existing TDA code can be placed here.

# Function to compute persistent homology with GUDHI
def compute_persistent_homology(time_series, window_size=10):
    # Create a higher-dimensional representation using a sliding window
    points = [time_series[i:i+window_size] for i in range(len(time_series) - window_size)]
    points = np.array(points)
    
    # Compute Rips complex and persistence diagram
    rips_complex = gd.RipsComplex(points=points)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
    diag = simplex_tree.persistence()
    return diag

# Function to plot persistence diagrams
def plot_persistence_diagrams(diagrams):
    gd.plot_persistence_diagram(diagrams)
    plt.show()

# Main Program
if __name__ == "__main__":
    # Tickers for the 3 stocks
    tickers = ["AAPL", "MSFT", "GOOGL"]
    
    # Fetch stock data and compute baseline persistent homology
    baseline_diagrams = {}
    for ticker in tickers:
        #time_series = fetch_stock_data(ticker)
        diagrams = compute_persistent_homology(daily_returns)
        baseline_diagrams[ticker] = diagrams
        
        # Plot baseline persistence diagrams
        print(f"Baseline persistence diagram for {ticker}:")
        plot_persistence_diagrams(diagrams)
    
    # Simulate stress scenarios (Here, we simply take a random subset for demonstration)
    stress_diagrams = {}
    for ticker in tickers:
        time_series = fetch_stock_data(ticker)
        stress_series = time_series.sample(frac=0.5)  # Randomly taking 50% data as a stress scenario
        diagrams = compute_persistent_homology(stress_series)
        stress_diagrams[ticker] = diagrams

        # Plot stress persistence diagrams
        print(f"Stress persistence diagram for {ticker}:")
        plot_persistence_diagrams(diagrams)

    # Remove NaN values and replace inf with a large number
def clean_array(arr):
    arr = arr[np.isfinite(arr)]  # Remove inf and NaN
    arr[np.isinf(arr)] = 1e9  # Replace inf with a large number, if any left
    return arr

# Compute Euclidean distance to quantify density change
for ticker in tickers:
    baseline_dgm = np.array([p[1][1] for p in baseline_diagrams[ticker] if p[0] == 0])  # Taking only death times from 0-dim diagram
    stress_dgm = np.array([p[1][1] for p in stress_diagrams[ticker] if p[0] == 0])  # Taking only death times from 0-dim diagram
    
    # Clean arrays
    baseline_dgm = clean_array(baseline_dgm)
    stress_dgm = clean_array(stress_dgm)

    if baseline_dgm.shape[0] == 0 or stress_dgm.shape[0] == 0:
        print(f"One of the diagrams for {ticker} is empty. Skipping distance computation.")
    else:
        if len(baseline_dgm) < len(stress_dgm):
            baseline_dgm = np.pad(baseline_dgm, (0, len(stress_dgm) - len(baseline_dgm)), 'constant')
        elif len(stress_dgm) < len(baseline_dgm):
            stress_dgm = np.pad(stress_dgm, (0, len(baseline_dgm) - len(stress_dgm)), 'constant')
        
        distance = euclidean(baseline_dgm, stress_dgm)
        print(f"Euclidean distance for {ticker}: {distance}")

import matplotlib.pyplot as plt

# Assuming the code provided for fetching and processing data has been run and we have 'normalized_series' and 'daily_returns' dictionaries
# Unfortunately, I can't run the yfinance-based code, but I can guide you on how to plot the subplots.

# Code to plot subplots for each stock's time series and normalized daily returns
fig, axs = plt.subplots(len(tickers), 2, figsize=(15, 10))

for i, ticker in enumerate(tickers):
    # Assuming time_series and normalized_series are dictionaries containing the stock data and normalized data respectively
    # Plotting the stock price time series
    axs[i, 0].plot(normalized_series[ticker].index, normalized_series[ticker].values, label=f'{ticker} Price')
    axs[i, 0].set_title(f'{ticker} Stock Price Time Series')
    axs[i, 0].set_xlabel('Date')
    axs[i, 0].set_ylabel('Normalized Price')
    axs[i, 0].legend()
    
    # Plotting the normalized daily returns
    axs[i, 1].plot(daily_returns[ticker].index[1:], daily_returns[ticker].values[1:], label=f'{ticker} Normalized Return', color='r')
    axs[i, 1].set_title(f'{ticker} Normalized Daily Returns')
    axs[i, 1].set_xlabel('Date')
    axs[i, 1].set_ylabel('Normalized Daily Return')
    axs[i, 1].legend()

plt.tight_layout()
plt.show()
# Corrected Python code snippet to fetch stock data, normalize it, calculate daily returns, and then get descriptive statistics for all three stocks.

# Importing required libraries
import yfinance as yf
import numpy as np
import pandas as pd

# Function to fetch stock data
def fetch_stock_data(ticker, period="1y"):
    stock_data = yf.download(ticker, period=period)
    return stock_data['Close']

# Function to normalize time series
def normalize_time_series(time_series):
    min_val = np.min(time_series)
    max_val = np.max(time_series)
    normalized_series = (time_series - min_val) / (max_val - min_val)
    return normalized_series

# Function to calculate daily returns
def calculate_returns(time_series):
    # Prevent division by zero by adding a small constant to the denominator
    epsilon = 1e1
    returns = np.diff(time_series) / (time_series[:-1] + epsilon)
    return returns


# Fetch and process data for the three stocks
tickers = ["AAPL", "MSFT", "GOOGL"]
normalized_series = {}
daily_returns = {}

for ticker in tickers:
    # Fetch stock data
    time_series = fetch_stock_data(ticker)
    
    # Normalize the time series
    normalized_series[ticker] = normalize_time_series(time_series)
    
    # Calculate daily returns
    daily_returns[ticker] = calculate_returns(normalized_series[ticker])

# Create DataFrame and calculate descriptive statistics
df_daily_returns = pd.DataFrame(daily_returns)
descriptive_stats = df_daily_returns.describe()

# Uncomment the next line to print descriptive statistics
# print(descriptive_stats)

# Note: This code won't run in this environment due to lack of internet access.
# You can run this code snippet in your local environment to calculate the descriptive statistics.
descriptive_stats