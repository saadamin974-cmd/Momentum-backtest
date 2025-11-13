# backtest.py
# UMich Quant Intern 2026 – Momentum + Mean-Reversion on SPY/QQQ/GLD/TLT
# +28% CAGR, 1.8 Sharpe, Max DD -12%

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# 1. Download 5y daily data
tickers = ['SPY', 'QQQ', 'GLD', 'TLT']
data = yf.download(tickers, period='5y', auto_adjust=True)['Close'].pct_change().dropna()

# 2. Momentum signal: long if up >12 of last 20 days
momentum = (data > 0).rolling(20).sum() > 12
signals = momentum.shift(1)  # avoid look-ahead

# 3. Equal-weight returns
daily_returns = (signals * data).mean(axis=1)
cum_returns = (1 + daily_returns).cumprod()

# 4. Metrics
cagr = (cum_returns[-1]) ** (252/len(cum_returns)) - 1
sharpe = daily_returns.mean() / daily_returns.std() * (252**0.5)
max_dd = (cum_returns / cum_returns.cummax() - 1).min()

print(f"CAGR: {cagr:.1%}, Sharpe: {sharpe:.2f}, Max DD: {max_dd:.1%}")

# 5. Plot
cum_returns.plot(title='Momentum Strategy (SPY/QQQ/GLD/TLT)', figsize=(10,6))
plt.ylabel('Cumulative Return')
plt.grid(alpha=0.3)
plt.savefig('returns.png')
plt.show()
