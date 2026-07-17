import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
import yfinance as yf
import datetime

# --- Black-Scholes Formula ---
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == "call":
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

#Implied Volatility
def implied_vol(price, S, K, T, r, option_type="call"):
    if T <= 0 or price <= 0:
        return np.nan
    f = lambda sigma: black_scholes(S,K,T,r,sigma,option_type) - price
    try:
        return brentq(f, 1e-6, 10) 
    except:
        return np.nan

#Data Collections--META Options
meta = yf.Ticker("META")
S = meta.history(period="1d")["Close"].iloc[-1]
r = 0.04

surface_data = []
for expiry in meta.options:   
    T_days = (pd.to_datetime(expiry) - datetime.datetime.today()).days
    T=T_days/365
    if T_days<1:
        continue
    chain = meta.option_chain(expiry)
    for df, otype in [(chain.calls,"call"), (chain.puts,"put")]:
        for _, row in df.iterrows():
            K = row["strike"]
            if not np.isnan(row["bid"]) and row["bid"] > 0 and not np.isnan(row["ask"]) and row["ask"] > 0:
                mid = (row["bid"] + row["ask"]) / 2
            elif not np.isnan(row["bid"]) and row["bid"] > 0:
                mid = row["bid"]
            elif not np.isnan(row["ask"]) and row["ask"] > 0:
                mid = row["ask"]
            else:
                continue  # skip if neither is valid
            iv = implied_vol(mid, S, K, T, r, otype)
            if np.isnan(iv) or iv <= 0 or iv > 5: continue
            surface_data.append([K, T*365, iv, otype])

df_surface = pd.DataFrame(surface_data, columns=["Strike","MaturityDays","ImpliedVol","Type"])
print("ALL IV data:")
print(df_surface.to_string(index=False))

#Volatility Surface
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(df_surface["Strike"], df_surface["MaturityDays"], df_surface["ImpliedVol"],
           c=df_surface["ImpliedVol"], cmap="viridis")
ax.set_xlabel("Strike")
ax.set_ylabel("Maturity (days)")
ax.set_zlabel("Implied Volatility")
plt.title("META Implied Volatility Surface (Calls & Puts)")
plt.show()

#Sensitivity Analysis
# Perturb spot price and risk-free rate slightly
spot_shifts = [-0.02, 0, 0.02]   # -2%, base, +2%
rate_shifts = [-0.01, 0, 0.01]   # -1%, base, +1%

# Choose one expiry and a range of strikes for sensitivity
expiry = meta.options[0]
T = (pd.to_datetime(expiry) - datetime.datetime.today()).days / 365
chain = meta.option_chain(expiry)
strikes = chain.calls["strike"].values[:10]  # first 10 strikes

print(f"Strike {K}, Mid {mid:.2f}, IV {iv}")

plt.figure(figsize=(10,6))
for ds in spot_shifts:
    ivs = []
    for K in strikes:
        mid = (chain.calls.loc[chain.calls["strike"]==K]["bid"].values[0] +
               chain.calls.loc[chain.calls["strike"]==K]["ask"].values[0]) / 2
        iv = implied_vol(mid, S*(1+ds), K, T, r, "call")
        ivs.append(iv)
    plt.plot(strikes, ivs, marker="o", label=f"Spot shift {ds*100:.0f}%")

plt.xlabel("Strike")
plt.ylabel("Implied Volatility")
plt.title("Sensitivity of IV Curve to Spot Price (META, nearest expiry)")
plt.legend()
plt.grid(True)
plt.show()
ivs = [iv for iv in ivs if not np.isnan(iv)]

plt.figure(figsize=(10,6))
for dr in rate_shifts:
    ivs = []
    strikes=[]
    for K in strikes:
        mid = (chain.calls.loc[chain.calls["strike"]==K]["bid"].values[0] +
               chain.calls.loc[chain.calls["strike"]==K]["ask"].values[0]) / 2
        iv = implied_vol(mid, S*(1+ds), K, T, r+dr, "call")
        if not np.isnan(iv):
            strikes.append(K)
            ivs.append(iv)
    if ivs:
        plt.plot(strikes, ivs, marker="o", label=f"Rate shift {dr*100:.0f}bp")

plt.xlabel("Strike")
plt.ylabel("Implied Volatility")
plt.title("Sensitivity of IV Curve to Risk-Free Rate (META, nearest expiry)")
plt.legend()
plt.grid(True)
plt.show()
