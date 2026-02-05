import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model

def get_returns(ticker: str, start="2018-01-01") -> pd.Series:
    """
    Helper function to get log returns for a given stock ticker from Yahoo Finance.
    """
    # Download the stock data
    px = yf.download(ticker, start=start, auto_adjust=True, progress=False)["Close"].dropna()
    
    # Calculate the log returns
    r = np.log(px).diff().dropna()
    return r

def calculate_next_day_volatility(ticker: str) -> dict:
    """
    Function to calculate the next day's forecasted realized volatility (RV) for a given ticker.
    
    Parameters:
    ticker: str : Stock ticker symbol (e.g., 'NVDA')
    
    Returns:
    dict: A dictionary with next-day variance, and next-day volatility
    """
    # Get the historical returns for the ticker
    returns = get_returns(ticker)
    
    # Fit a GARCH(1,1) model to the returns
    r_pct = returns * 100.0  # Convert returns to percentage
    am = arch_model(r_pct, mean="Constant", vol="GARCH", p=1, q=1, dist="t")  # Fit the GARCH(1,1) model
    res = am.fit(disp="off")
    
    # Get the forecasted variance over the next 1 day (next trading day)
    f = res.forecast(horizon=1, reindex=False)  # Forecast variance for 1 day
    
    # Extract the next-day forecasted variance and convert it to NumPy array
    var_next_day_pct2 = f.variance.iloc[-1, 0]  # Variance in percent^2
    
    # Convert percent^2 to decimal^2
    var_next_day_dec2 = var_next_day_pct2 / (100.0 ** 2)
    
    # Calculate the next-day expected volatility (square root of the variance)
    vol_next_day = np.sqrt(var_next_day_dec2)
    
    # Return the results
    return {
        "next_day_variance": var_next_day_dec2,  # Next-day variance (decimal^2)
        "next_day_volatility": vol_next_day      # Next-day volatility (decimal)
    }
