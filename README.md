
# Topological Risk Analysis

# Description

This research aims to fetch stock data for specific tickers and perform various analyses including normalization, calculation of Value-at-Risk (VaR), Conditional Value-at-Risk (CVaR), and Topological Data Analysis (TDA) using persistent homology. The program also plots the stock data and persistence diagrams for better visualization.

## Dependencies

- yfinance
- numpy
- matplotlib
- gudhi
- scipy

You can install these dependencies using pip:

```
pip install yfinance numpy matplotlib gudhi scipy
```

## Functions

### `fetch_stock_data(ticker, period="1y")`

Fetches the stock data for a given `ticker` and `period`.

### `normalize_time_series(time_series)`

Normalizes the time-series data between 0 and 1.

### `calculate_returns(time_series)`

Calculates the daily returns of a stock based on its time series.

### `calculate_var_cvar(returns, confidence_level=0.95)`

Calculates VaR and CVaR based on the given returns and confidence level.

### `compute_persistent_homology(time_series, window_size=10)`

Computes persistent homology for the time series.

### `plot_persistence_diagrams(diagrams)`

Plots the persistence diagrams.

### `clean_array(arr)`

Cleans the array by removing NaN and infinite values.

## How to Run

1. Clone this repository.
2. Run `pip install -r requirements.txt` to install dependencies.
3. Execute the script using `python <script_name>.py`.

## Output

- Prints VaR and CVaR values for each stock at a 95% confidence interval.
- Plots the persistence diagrams for each stock.
- Computes and prints the Euclidean distance between the baseline and stress scenarios for each stock.

## License

This project is open-source and available under the MIT License.


