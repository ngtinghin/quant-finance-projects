import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

tickers=['CEG','ETN','AVGO','NVDA','ARM','SMCI','DELL','NTAP','STX','CSCO','CRDO','CDNS','SNPS','TSM','AMAT','RGTI','IONQ','META','GOOGL','ASML'
         ,'AAPL','MSFT','AMKR','LRCX','AAOI','LITE']
data=yf.download(tickers,period='6mo',interval='1d')['Close']
returns=data.pct_change().dropna()
covariance=returns.cov()
correlation=returns.corr()
order=['CEG','ETN','AVGO','NVDA','ARM','SMCI','DELL','NTAP','STX','CSCO','CRDO','CDNS','SNPS','TSM','AMAT','RGTI','IONQ','META','GOOGL','ASML'
         ,'AAPL','MSFT','AMKR','LRCX','AAOI','LITE']
correlation=correlation.reindex(index=order,columns=order)
pca=PCA()
pca.fit(returns)
explained_var=pca.explained_variance_ratio_

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(18,6),gridspec_kw={'width_ratios':[4,1]})


sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=correlation.columns, yticklabels=correlation.columns,ax=ax1)
ax1.set_title("Correlation Matrix Heatmap")

ax2.plot(np.cumsum(explained_var), marker='o')
ax2.set_xlabel('Number of Components')
ax2.set_ylabel('Cumulative Explained Variance')
ax2.set_title('PCA plot - 26 Stocks Portfolio')
ax2.grid(True)
plt.tight_layout()
plt.show()
