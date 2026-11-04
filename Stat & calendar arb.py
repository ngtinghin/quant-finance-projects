import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
nvda=yf.Ticker("NVDA")
s=nvda.history(period='1d')['Close'].iloc[-1]

expiration=nvda.options[0]
option_chain=nvda.option_chain(expiration)

calls=option_chain.calls
calls=calls[(calls['strike']>=100)]
T=(np.datetime64(expiration)-np.datetime64('today'))/np.timedelta64(365,'D')
calls['k']=np.log(calls['strike']/s)
calls['w']=(calls['impliedVolatility']**2)*T
calls=calls.sort_values('k')
w=calls['w'].values
k=calls['k'].values
secondd=np.gradient(np.gradient(w,k),k)
arbitrage=np.any(secondd<0)
print("Butterfly arbitrage opportunities=", arbitrage)

surface=[]
for exp in nvda.options:
    T=(np.datetime64(exp)-np.datetime64('today'))/np.timedelta64(365,'D')
    option_chain=nvda.option_chain(exp)
    calls=option_chain.calls
    calls['k']=np.log(calls['strike']/s)
    calls['w']=(calls['impliedVolatility']**2)*T
    calls['T']=T
    surface.append(calls[['strike','k','w','T']])
df=pd.concat(surface)
arbitrage=[]
for k,group in df.groupby('k'):
    group=group.sort_values('T')
    if any(group['w'].diff().dropna()<0):
        arbitrage.append(k)
opportunity=True if arbitrage else False
print("Calendar arbitrage opportunities=", opportunity)

strikes=[210,220]
fig,axes=plt.subplots(1,2,figsize=(12,6))
for k in strikes:
    group=df.loc[df['strike']==k].sort_values('T')
    if not group.empty:
        axes[0].plot(group['T'],group['w'],marker='o',label=f"Strike{k}")
axes[0].set_xlabel("Maturity")
axes[0].set_ylabel("Variance")
axes[0].set_title("Variance vs Maturity")
axes[0].legend()


calls=option_chain.calls
calls=calls[(calls['strike']>=s) & (calls['strike']<=317.23)]
puts=option_chain.puts
puts=puts[(puts['strike']<=s) & (puts['strike']>=117.23)]

axes[1].plot(calls['strike'],calls['impliedVolatility'],label='Calls',marker='*')
axes[1].plot(puts['strike'],puts['impliedVolatility'],label='Puts',marker='^')
axes[1].set_xlabel("Strike Price")
axes[1].set_ylabel("Implied Volatility")
axes[1].axvline(s,color='black',linestyle='--',label=f'Underlying Price {s:.2f}')
axes[1].set_title(f"NVDA Options Volatility Smile ({expiration})")
axes[1].legend()

plt.tight_layout()
plt.show()
