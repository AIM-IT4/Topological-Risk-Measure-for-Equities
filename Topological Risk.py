#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Step 1: Generate simulated historical pricing data
np.random.seed(0)
n_observations = 252
n_assets = 3

# Simulate daily returns
mean_daily_returns = [0.0001, 0.0002, -0.0001]
cov_matrix = [[0.00001, 0.000005, 0],
              [0.000005, 0.00002, 0],
              [0, 0, 0.000015]]

daily_returns = np.random.multivariate_normal(mean_daily_returns, cov_matrix, n_observations)

# Create DataFrame
columns = ['Options', 'Futures', 'Swaps']
df_daily_returns = pd.DataFrame(daily_returns, columns=columns)

# Convert daily returns to price
initial_prices = [100, 100, 100]  # initial price for each asset
df_prices = (df_daily_returns + 1).cumprod() * initial_prices

# Step 2: Preprocess the data (Normalization)
df_prices_normalized = (df_prices - df_prices.min()) / (df_prices.max() - df_prices.min())

# Show first few rows of the normalized prices
df_prices_normalized.head()


# In[ ]:


# Step 3: Calculate traditional risk measures (VaR and CVaR)

# Calculate VaR at 95% confidence level
alpha = 0.95
var_95 = df_daily_returns.quantile(1-alpha)

# Calculate CVaR at 95% confidence level
cvar_95 = df_daily_returns[df_daily_returns.lt(var_95, axis=1)].mean()

# Create DataFrame for VaR and CVaR
risk_measures_df = pd.DataFrame({
    'VaR_95': var_95,
    'CVaR_95': cvar_95
})

risk_measures_df


# In[ ]:


from scipy.signal import find_peaks

# Step 7: Simplified Topological Data Analysis (TDA) to demonstrate the discussed concepts

# 1. Vietoris-Rips-like complex: Since we're dealing with 1D data, the complex is essentially the data itself.

# 2. Define two "radii" (epsilon values) to mimic scales.
# In our simplified example, we'll use the mean and mean + 1 standard deviation of normalized prices as epsilon values.
epsilon_1 = df_prices_normalized.mean()
epsilon_2 = df_prices_normalized.mean() + df_prices_normalized.std()

# 3. Calculate "persistent homology-like" features (peaks in the time series as stand-ins for topological features)
# We'll find peaks that are above the epsilon_1 and epsilon_2 thresholds for each asset type.
peaks_epsilon_1 = {}
peaks_epsilon_2 = {}

for col in df_prices_normalized.columns:
    peaks_epsilon_1[col] = find_peaks(df_prices_normalized[col], height=epsilon_1[col])[0]
    peaks_epsilon_2[col] = find_peaks(df_prices_normalized[col], height=epsilon_2[col])[0]

# 4. Calculate DCUS (Density Change Under Stress) as described.
# DCUS = (Number of features at epsilon_2 - Number of features at epsilon_1) / (epsilon_2 - epsilon_1)
dcus_tda = {}

for col in df_prices_normalized.columns:
    dcus_tda[col] = (len(peaks_epsilon_2[col]) - len(peaks_epsilon_1[col])) / (epsilon_2[col] - epsilon_1[col])

# Create DataFrame for DCUS
dcus_tda_df = pd.DataFrame({
    'DCUS_TDA': dcus_tda
})

dcus_tda_df


# In[ ]:


# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Generate synthetic financial data
# Note: In a real-world application, this data would come from a reliable financial database
np.random.seed(0)  # for reproducibility

# Generate synthetic daily returns for three types of financial derivatives: options, futures, and swaps
n_days = 252  # Typical number of trading days in a year
n_assets = 3  # Number of different types of financial derivatives

# Mean and standard deviation for each asset's returns
mean_returns = [0.001, 0.002, 0.0015]  # Options, Futures, Swaps
std_dev_returns = [0.02, 0.025, 0.015]  # Options, Futures, Swaps

# Generate synthetic daily returns
daily_returns = np.random.normal(loc=mean_returns, scale=std_dev_returns, size=(n_days, n_assets))

# Convert to a DataFrame for easier handling
daily_returns_df = pd.DataFrame(daily_returns, columns=['Options', 'Futures', 'Swaps'])

# Step 3: Normalize the data
# Normalization is a common preprocessing step in many data-driven applications. 
# Here, we'll use Min-Max normalization.
normalized_data = (daily_returns_df - daily_returns_df.min()) / (daily_returns_df.max() - daily_returns_df.min())

# Generate descriptive statistics for the recreated and normalized dataset
desc_stats = normalized_data.describe()

# Convert to LaTeX table format with superscript for the reference
latex_desc_stats = desc_stats.to_latex(float_format="%.2f", caption="Descriptive Statistics of Normalized Data", label="tab:desc_stats").replace('toprule', 'toprule[1.5pt]').replace('bottomrule', 'bottomrule[1.5pt]').replace('midrule', 'midrule[1pt]')

# Adding a superscript for reference
latex_desc_stats = latex_desc_stats.replace("Descriptive Statistics of Normalized Data", "Descriptive Statistics of Normalized Data\\textsuperscript{34}")

latex_desc_stats


# In[ ]:


# Use the previously calculated VaR and CVaR values for the comparative charts
actual_var_cvar = pd.DataFrame({
    'VaR_95': [-0.004969, -0.007611, -0.006676],
    'CVaR_95': [-0.006593, -0.008594, -0.009270]
}, index=['Options', 'Futures', 'Swaps'])

# Merge with DCUS_TDA DataFrame for comparison
actual_comparison_df = pd.merge(actual_var_cvar, dcus_tda_df, left_index=True, right_index=True)

# Create new comparative plots
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Traditional Measures (VaR and CVaR)
axs[0].bar(actual_comparison_df.index, actual_comparison_df['VaR_95'], color='blue', width=0.4, label='VaR 95')
axs[0].bar(actual_comparison_df.index, actual_comparison_df['CVaR_95'], color='green', width=0.4, label='CVaR 95', bottom=actual_comparison_df['VaR_95'])
axs[0].set_title('Traditional Risk Measures')
axs[0].set_xlabel('Asset Type')
axs[0].set_ylabel('Risk Measure Value')
axs[0].legend()

# Plot 2: Topological Measure (DCUS_TDA)
axs[1].bar(actual_comparison_df.index, actual_comparison_df['DCUS_TDA'], color='orange', width=0.4, label='DCUS_TDA')
axs[1].set_title('Topological Risk Measure (DCUS)')
axs[1].set_xlabel('Asset Type')
axs[1].set_ylabel('DCUS Value')
axs[1].legend()

plt.tight_layout()
plt.show()


