import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Manual code for ewma volatility

#def ewma_series(r,lam=0.94):
#    r=r.dropna()
#    vols=[r[:20].var()]
#    for x in r.iloc[20:]:
#        v=lam*vols[-1]+(1-lam)*x**2
#        vols.append(v)
#    return pd.Series(vols,index=r.index[19:])


df=pd.read_csv("qqq data.csv")
df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"],dayfirst=True)
df=df.sort_values(["Ticker","DlyCalDt"])
df["return"]=(df.groupby("Ticker")["DlyPrc"]
    .transform(lambda x: np.log(x/x.shift(1))))

#Using the ewm function to find ewma vol with lambda=0.94
df["ewma_var"]=(df.groupby("Ticker")["return"]
    .transform(lambda x: x.pow(2).ewm(alpha=1-0.94,adjust=False).mean()))
df["ewma_vol"]=np.sqrt(df["ewma_var"]*252)

#Finding the correlation of vol between these securities
vol_pivot=df.pivot(index="DlyCalDt",columns="Ticker",values="ewma_vol")
corr=vol_pivot.corr()

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5))

for ticker in ["QQQ","SPY","TQQQ","XLF","XLK"]:
    tmp=df[df["Ticker"]==ticker]
    ax1.plot(tmp["DlyCalDt"],tmp["ewma_vol"],label=ticker)
ax1.set_title("EWMA Volatility")
ax1.set_xlabel("Date")
ax1.set_ylabel("Annualized Vol")
ax1.grid(True)
ax1.legend()

sns.heatmap(corr,annot=True,fmt=".2f",cmap="coolwarm",ax=ax2)
ax2.set_title("Volatility Correlation")

plt.tight_layout()
plt.show()

