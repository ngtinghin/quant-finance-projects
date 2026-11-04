import yfinance as yf
from arch import arch_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ticker="SMCI"
df=yf.download(ticker,start="2025-06-01",end="2026-11-03")
df["returns"]=np.log(df["Close"]/df["Close"].shift(1))
df=df.dropna()

#model=arch_model(df["returns"]*100,mean="Zero",vol="GARCH",p=1,q=1)
#result=model.fit()
#forecast=result.forecast(horizon=1)

window=252
forecast_vol=[]
dates=[]
for i in range(window,len(df)):
    
    model=arch_model(df["returns"].iloc[i-window:i]*100,mean="Zero",vol="GARCH",p=1,q=1)
    result=model.fit(disp="off")
    forecast=result.forecast(horizon=1)
    vol=np.sqrt(forecast.variance.iloc[-1,0])
    forecast_vol.append(vol)
    dates.append(df.index[i])
forecast_df=pd.DataFrame({"ForecastVol":forecast_vol},index=dates)
forecast_df["ForecastVol"]/=100
df=df.loc[forecast_df.index]
df["ForecastVol"]=forecast_df["ForecastVol"]

#Regime strategy
low=df["ForecastVol"].quantile(0.25)
high=df["ForecastVol"].quantile(0.65)
df["Weight_Regime"]=0.0
df.loc[df["ForecastVol"]<low,"Weight_Regime"]=1.0
df.loc[(df["ForecastVol"]>=low) &(df["ForecastVol"]<=high),"Weight_Regime"]=0.5
df.loc[df["ForecastVol"]>high,"Weight_Regime"]=0.1
df["Strategy_Regime"]=(df["Weight_Regime"].shift(1)*df["returns"])
df["Strategy_Regime"]=(df["Strategy_Regime"].fillna(0))

#Breakout strategy
df["Vol_Jump"]=(df["ForecastVol"]/df["ForecastVol"].shift(1))
df["Weight_Breakout"]=np.where(df["Vol_Jump"]>1.3,0,1)
df["Strategy_Breakout"]=(df["Weight_Breakout"].shift(1)*df["returns"])
df["Strategy_Breakout"]=(df["Strategy_Breakout"].fillna(0))

#Buy&Hold strategy
df["Buy&Hold"]=df["returns"]

def performance(result):
    annual_return=(np.exp(result.sum()))**(252/len(result))-1
    annual_vol=(result.std()*np.sqrt(252))
    sharpe=(annual_return/annual_vol)
    cumulative=np.exp(result.cumsum())
    drawdown=(cumulative/cumulative.cummax()-1)
    max_dd=drawdown.min()
    return [annual_return,annual_vol,sharpe,max_dd]
results=pd.DataFrame({"Buy&Hold":performance(df["Buy&Hold"]),
                      "Regime":performance(df["Strategy_Regime"]),
                      "Breakout":performance(df["Strategy_Breakout"])},
                     index=["Annual Return","Annual Vol","Sharpe","Max Drawdown"])
print(results)
buyhold_curve = np.exp(df["Buy&Hold"].cumsum())
regime_curve = np.exp(df["Strategy_Regime"].cumsum())
breakout_curve = np.exp(df["Strategy_Breakout"].cumsum())
plt.figure(figsize=(7,4))
plt.plot(buyhold_curve,label="Buy & Hold")
plt.plot(regime_curve,label="Regime")
plt.plot(breakout_curve,label="Breakout")
plt.legend()
plt.title("GARCH Trading Strategies")
plt.show()
