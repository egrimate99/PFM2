import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import altair as alt


# Model parameters
S0 = 100  # initial stock price
r = 0.05  # risk-free rate
T = 1.0   # time horizon in years
N = 252   # number of time steps (trading days in a year)

# Functions to simulate stock price paths using fixed Wiener increments
def simulate_stock_path_fixed_wiener(k, m, sigma, S0, wiener_increments, T, N):
    dt = T/N
    t = np.linspace(0, T, N+1)
    S = np.zeros(N+1)
    S[0] = S0

    for i in range(1, N+1):
        dW = wiener_increments[i-1]  # Use the precomputed Wiener increment
        S[i] = S[i-1] + k * (m - S[i-1]) * dt + sigma * S[i-1] * dW

    return t, S

def simulate_stock_path_risk_neutral_fixed_wiener(k, m, sigma, r, S0, wiener_increments, T, N):
    dt = T/N
    t = np.linspace(0, T, N+1)
    S = np.zeros(N+1)
    S[0] = S0

    for i in range(1, N+1):
        dW = wiener_increments[i-1]  # Use the precomputed Wiener increment
        S[i] = S[i-1] + r * S[i-1] * dt + sigma * S[i-1] * dW

    return t, S

# Function to generate multiple stock price paths for a given parameter set
def generate_multiple_paths(k, m, sigma, S0, r, T, N, num_paths, wiener_increments_all_paths):
    paths_a = []
#paths_b = []

    for path_idx in range(num_paths):
        # Simulate path for (a) part
        t, S = simulate_stock_path_fixed_wiener(k, m, sigma, S0, wiener_increments_all_paths[path_idx], T, N)
        paths_a.append(S)

        # Simulate path for (b) part
        #t, S_rn = simulate_stock_path_risk_neutral_fixed_wiener(k, m, sigma, r, S0, wiener_increments_all_paths[path_idx], T, N)
       #paths_b.append(S_rn)

    return t, paths_a, #paths_b

# Set random seed for reproducibility
np.random.seed(42)

# Streamlit app
st.title('Stock Price Simulation')

st.markdown("""

            
$$ c^2 = a^2 + b^2 $$
""")

# Slider for the mean-reversion parameter k
k = st.slider('Select the mean-reversion parameter k', min_value=0.01, max_value=10.0, value=0.1, step=0.01)
m = st.slider('Select the mean-reversion parameter m', min_value=0.01, max_value=200.0, value=100.0, step=0.01)
sigma = st.slider('Select the mean-reversion parameter sigma', min_value=0.01, max_value=1.0, value=0.3, step=0.01)

# m and sigma are kept constant for this example
# m = 100
# sigma = 0.3

# Number of paths per parameter set
num_paths = 10

# Precompute Wiener increments for all paths
wiener_increments_all_paths = np.random.normal(0, np.sqrt(T/N), (num_paths, N))

# Generate paths
# t, paths_a, paths_b = generate_multiple_paths(k, m, sigma, S0, r, T, N, num_paths, wiener_increments_all_paths)
t, paths_a = generate_multiple_paths(k, m, sigma, S0, r, T, N, num_paths, wiener_increments_all_paths)
# Convert paths to DataFrame for Streamlit's line_chart
df_paths_a = pd.DataFrame(paths_a).T  # Transpose to match the index t
# df_paths_b = pd.DataFrame(paths_b).T

# Set the index to be the time points
df_paths_a.index = t
# df_paths_b.index = t

# Convert DataFrame to long format for Altair
df_paths_a_long = df_paths_a.reset_index().melt('index', var_name='path', value_name='price')
# df_paths_b_long = df_paths_b.reset_index().melt('index', var_name='path', value_name='price')

# Create Altair line chart for (a) part
chart_a = alt.Chart(df_paths_a_long).mark_line().encode(
    x=alt.X('index:T', title='Time (Years)'),
    y=alt.Y('price:Q', title='Stock Price', scale=alt.Scale(domain=[50, 300])),
    color='path:N',
    tooltip=['index', 'price']
).interactive(bind_y =True).properties(width=600, height=500)  # Increased width for better visibility




# Use container width to make charts larger and more readable
st.altair_chart(chart_a, use_container_width=True)
st.markdown("""

            
$$ c^2 = a^2 + b^2 $$
""")

k_ = st.slider('Select the mean-reversion parameter k_', min_value=0.01, max_value=10.0, value=0.1, step=0.01)
m_ = st.slider('Select the mean-reversion parameter m_', min_value=0.01, max_value=200.0, value=100.0, step=0.01)
sigma_ = st.slider('Select the mean-reversion parameter sigma_', min_value=0.01, max_value=1.0, value=0.3, step=0.01)
t, paths_b = generate_multiple_paths(k_, m_, sigma_, S0, r, T, N, num_paths, wiener_increments_all_paths)
df_paths_b = pd.DataFrame(paths_b).T
df_paths_b.index = t
df_paths_b_long = df_paths_b.reset_index().melt('index', var_name='path', value_name='price')

# Create Altair line chart for (b) part
chart_b = alt.Chart(df_paths_b_long).mark_line().encode(
    x=alt.X('index:T', title='Time (Years)'),
    y=alt.Y('price:Q', title='Stock Price', scale=alt.Scale(domain=[50, 300])),
    color='path:N',
    tooltip=['index', 'price']
).interactive(bind_y =True).properties(width=600, height=500)  # Increased width for better visibility
st.altair_chart(chart_b, use_container_width=True)