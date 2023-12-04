import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import altair as alt
from scipy.stats import norm

T = 1.0   # time horizon in years   # number of time steps (trading days in a year)

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
def generate_multiple_paths(k, m, sigma, S0, r, T, N, num_paths, wiener_increments_all_paths, risk_neutral=False):
    paths = []
#paths_b = []

    for path_idx in range(num_paths):
        # Simulate path for (a) part
        if not risk_neutral:
            t, S = simulate_stock_path_fixed_wiener(k, m, sigma, S0, wiener_increments_all_paths[path_idx], T, N)
            paths.append(S)

        # Simulate path for (b) part
        if risk_neutral:
            t, S_rn = simulate_stock_path_risk_neutral_fixed_wiener(k, m, sigma, r, S0, wiener_increments_all_paths[path_idx], T, N)
            paths.append(S_rn)

    return t, paths #paths_b


# Set random seed for reproducibility
np.random.seed(42)

# Streamlit app
st.title('Mean-Reversion Simulation')

S0 = st.number_input('Kezdeti részvényár: S(0)', min_value=60, max_value=200, value=100, step=1)
r = st.number_input('Kockázatmentes kamatláb: r', min_value=0.0, max_value=0.2, value=0.05, step=0.01)
N = st.number_input('Időfelbontás: N', min_value=10, max_value=1000, value=250, step=1)

st.markdown("""

Mean-Reversion folyamat:
                        
$$ \\boxed{dS(t) = k(m-S(t))dt+\\sigma S(t)dW(t)} $$
""")

# Slider for the mean-reversion parameter k
k = st.slider('Mean-reversion intenzitás: k', min_value=0.01, max_value=10.0, value=1.5, step=0.01)
m = st.slider('Mean-reversion középérték: m', min_value=0.01, max_value=200.0, value=100.0, step=0.01)
sigma = st.slider('Volatilitás: sigma   ', min_value=0.01, max_value=0.8, value=0.3, step=0.01)


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

chart_a = alt.Chart(df_paths_a_long).mark_line().encode(
    x=alt.X('index:T', title='Time'),
    y=alt.Y('price:Q', title='Stock Price', scale=alt.Scale(domain=[50, 300])),
    color='path:N',
    tooltip=['index', 'price']
).properties(
    width=600,
    height=500,
    title='Mean-reversion trajektóriák'
)



# Use container width to make charts larger and more readable
st.altair_chart(chart_a, use_container_width=True)


num_paths_density=3000

# Generate Wiener increments
wiener_increments_all_paths_density = np.random.normal(0, np.sqrt(T/N), (num_paths_density, N))

# Generate multiple stock price paths
t, paths_a_density = generate_multiple_paths(k, m, sigma, S0, r, T, N, num_paths_density, wiener_increments_all_paths_density)

# Collect the stock prices at T=1 from each path
final_prices_a = [path[-1] for path in paths_a_density]

# Create a DataFrame
df = pd.DataFrame({'Final Price': final_prices_a})

# Create the density plot using Altair
chart_a_density = alt.Chart(df, ).transform_density(
    'Final Price',
    as_=['Final Price', 'Density'],
    
).mark_area().encode(
    x=alt.X('Final Price:Q', scale=alt.Scale(domain=[0, 500])),
    y=alt.Y('Density:Q', scale=alt.Scale(domain=[0, 0.05])),
    # y='Density:Q',
).properties(width=600, 
             height=400,
             title='Sűrűség t=1-ben'
)

# Display the chart
st.altair_chart(chart_a_density, use_container_width=True)



st.markdown("""
Kockázatsemleges trajektóriák:
            
$dS(t) = \\underbrace{k(m-S(t))}_{\\alpha(t)}dt+\\underbrace{\\sigma S(t)}_{\\sigma(t)}dW(t)$


$dD(t)=-rD(t)dt$


$X(t)=D(t)S(t)$


$dX(t)=D(t)dS(t)+S(t)dD(t)=D(t\\left[k(m-S(t))dt+\\sigma S(t)dW(t)\\right]-rS(t)D(t)dt$


$=\\sigma D(t)S(t)\\underbrace{\\left[dW(t)+\\frac{k(m-S(t))-rS(t)}{\\sigma S(t)}dt\\right]}_{\\tilde{W}(t)} $


$dS(t) = k(m-S(t))dt+\\sigma S(t)dW(t)=$


$k(m-S(t))dt+\\sigma S(t)d\\tilde{W}(t)-\\left[k(m-S(t))-rS(t)\\right]dt \\implies$


$\\boxed{dS(t) = rS(t)dt+\\sigma S(t)d\\tilde{W}(t)}$
""")

k_ = st.slider('Mean-reversion intenzitás: k (nem függ tőle)', min_value=0.01, max_value=10.0, value=1.5, step=0.01)
m_ = st.slider('Mean-reversion középérték: m (nem függ tőle)', min_value=0.01, max_value=200.0, value=100.0, step=0.01)
sigma_ = st.slider('Volatilitás: sigma', min_value=0.01, max_value=0.8, value=0.3, step=0.01)
t, paths_b = generate_multiple_paths(k_, m_, sigma_, S0, r, T, N, num_paths, wiener_increments_all_paths,True)
df_paths_b = pd.DataFrame(paths_b).T
df_paths_b.index = t
df_paths_b_long = df_paths_b.reset_index().melt('index', var_name='path', value_name='price')

# Create Altair line chart for (b) part
chart_b = alt.Chart(df_paths_b_long).mark_line().encode(
    x=alt.X('index:T', title='Time'),
    y=alt.Y('price:Q', title='Stock Price', scale=alt.Scale(domain=[50, 300])),
    color='path:N',
    tooltip=['index', 'price']
).properties(
    width=600,
    height=500,
    title='Kockázatsemleges trajektóriák'
)  # Increased width for better visibility
st.altair_chart(chart_b, use_container_width=True)

# Generate multiple stock price paths
t, paths_b_density = generate_multiple_paths(k_, m_, sigma_, S0, r, T, N, num_paths_density, wiener_increments_all_paths_density,True)

# Collect the stock prices at T=1 from each path
final_prices_b = [path[-1] for path in paths_b_density]

# Create a DataFrame
df_b = pd.DataFrame({'Final Price': final_prices_b})

# Create the density plot using Altair
chart_b_density = alt.Chart(df_b).transform_density(
    'Final Price',
    as_=['Final Price', 'Density'],
).mark_area().encode(
    x=alt.X('Final Price:Q', scale=alt.Scale(domain=[0, 500])),
    y=alt.Y('Density:Q', scale=alt.Scale(domain=[0, 0.05])),
).properties(width=600, 
             height=400,
             title='Sűrűség t=1-ben'
)
st.altair_chart(chart_b_density, use_container_width=True)

# Display the chart
# chart_b_density

def black_scholes(S, K, T, r, sigma, option_type="c"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "c":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price
# from py_vollib_vectorized import vectorized_implied_volatility as implied_vol
def implied_volatility(price, S, K, T, r, option_type):
    precision = 0.00001
    max_iterations = 100
    sigma = 0.5  # initial guess
    for i in range(max_iterations):
        price_estimate = black_scholes(S, K, T, r, sigma, option_type)
        vega = (1 / np.sqrt(2 * np.pi)) * S * np.sqrt(T) * np.exp(-0.5 * (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) ** 2 / (sigma ** 2 * T))
        diff = price_estimate - price

        if abs(diff) < precision:
            return sigma

        sigma -= diff / vega  # Newton-Raphson step

        if sigma <= 0:  # ensure sigma stays positive
            return np.nan

    return np.nan  # return NaN if not converged
def calculate_option_prices_and_ivs(paths, S0, K_values, T, r):
    final_prices = np.array([path[-1] for path in paths])
    puts = np.array([np.exp(-r*T) * np.mean(np.maximum(k - final_prices, 0)) for k in K_values])
    calls = np.array([np.exp(-r*T) * np.mean(np.maximum(final_prices - k, 0)) for k in K_values])

    put_ivs = np.array([implied_volatility(put_price, S0, k, T, r, 'p') for put_price, k in zip(puts, K_values)])
    call_ivs = np.array([implied_volatility(call_price, S0, k, T, r, 'c') for call_price, k in zip(calls, K_values)])

    return puts, calls, put_ivs, call_ivs

K_values = np.linspace(S0 - 50, S0 + 100, 10)  # Ensure this is an array

# Now calling the function with K_values
puts, calls, put_ivs, call_ivs = calculate_option_prices_and_ivs(paths_b_density, S0, K_values, T, r)


data = pd.DataFrame({
    'Strike Price': K_values,
    'Implied Volatility': call_ivs
})
c = alt.Chart(data).mark_line().encode(
    x=alt.X('Strike Price', scale=alt.Scale(domain=[S0-50, S0+100])),
    y=alt.Y('Implied Volatility', scale=alt.Scale(domain=[0.0, 1.0])),
).properties(width=600, 
             height=400,
             title='Visszaszámított volatilitás'
)
st.altair_chart(c, use_container_width=True)