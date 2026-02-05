import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model

TRADING_DAYS = 252

def get_returns(ticker: str, start="2025-01-01") -> pd.Series:
    px = yf.download(ticker, start=start, auto_adjust=True, progress=False)["Close"].dropna()
    r = np.log(px).diff().dropna()  # log returns
    return r

def fit_garch11(returns: pd.Series):
    # arch works best if returns are scaled (e.g., percent)
    r_pct = returns * 100.0
    am = arch_model(r_pct, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
    res = am.fit(disp="off")
    return res

def forecast_realized_vol(res, T: int) -> dict:
    """
    Returns:
      var_1: next-day variance (in decimal^2, not percent^2)
      rv_T: expected realized variance over T days (decimal^2)
      vol_T: expected realized volatility over T days (decimal)
      ann_vol: annualized volatility implied by T-day vol (decimal)
    """
    f = res.forecast(horizon=T, reindex=False)

    # f.variance is in the same units as the model's input (percent^2)
    var_path_pct2 = f.variance.iloc[-1].to_numpy()  # length T, percent^2

    # convert percent^2 -> decimal^2:
    # if r_pct = 100*r_dec, then var_pct2 = 100^2 * var_dec2
    var_path_dec2 = var_path_pct2 / (100.0 ** 2)

    var_1 = float(var_path_dec2[0])
    rv_T = float(var_path_dec2.sum())
    vol_T = float(np.sqrt(rv_T))

    # convert T-day vol to annualized volatility
    # vol_T is over T days; annual vol approx = vol_T * sqrt(252/T)
    ann_vol = float(vol_T * np.sqrt(TRADING_DAYS / T))

    return {"var_1": var_1, "rv_T": rv_T, "vol_T": vol_T, "ann_vol": ann_vol}

if __name__ == "__main__":
    ticker = "NVDA"
    T = 21  # number of trading days to option expiry

    returns = get_returns(ticker)
    res = fit_garch11(returns)

    out = forecast_realized_vol(res, T)
    print(f"{ticker} | T={T} trading days")
    print(f"Next-day variance: {out['var_1']:.8f}")
    print(f"Expected RV(T):    {out['rv_T']:.8f}")
    print(f"Expected vol(T):   {out['vol_T']:.4%}")
    print(f"Annualized vol:    {out['ann_vol']:.2%}")
