# Import required libraries for data collection, web scraping, and analysis
import pytz
import yfinance
import requests
import threading
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from utils import timeme
from utils import save_pickle, load_pickle
from alpha1 import Alpha1
from alpha2 import Alpha2
from alpha3 import Alpha3
from alpha4 import Alpha4 
from utils import Portfolio

def get_sp500_tickers():
    """
    Scrape S&P 500 ticker symbols from Wikipedia.
    
    Returns:
        list: List of ticker symbols (e.g., ['AAPL', 'MSFT', 'GOOGL', ...])
    """
    # Fetch Wikipedia page containing S&P 500 companies
    res = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = BeautifulSoup(res.content,'html')
    
    # Extract the first table (contains current S&P 500 companies)
    table = soup.find_all('table')[0] 
    df = pd.read_html(str(table))
    
    # Extract ticker symbols from the 'Symbol' column
    tickers = list(df[0].Symbol)
    return tickers

def get_history(ticker, period_start, period_end, granularity="1d", tries=0):
    """
    Download historical price data for a single ticker using yfinance.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        period_start: Start date for data collection
        period_end: End date for data collection  
        granularity: Data frequency ('1d' for daily, '1h' for hourly, etc.)
        tries: Current retry attempt (for internal recursion)
        
    Returns:
        DataFrame: Cleaned price data with columns [open, high, low, close, volume]
                  Indexed by datetime with UTC timezone
    """
    try:
        # Download data from Yahoo Finance API
        df = yfinance.Ticker(ticker).history(
            start=period_start,
            end=period_end,
            interval=granularity,
            auto_adjust=True  # Adjusts for splits and dividends
        ).reset_index()
    except Exception as err:
        # Retry up to 5 times on API failures
        if tries < 5:
            return get_history(ticker, period_start, period_end, granularity, tries+1)
        return pd.DataFrame()
    
    # Standardize column names to lowercase
    df = df.rename(columns={
        "Date":"datetime",
        "Open":"open",
        "High":"high", 
        "Low":"low",
        "Close":"close",
        "Volume":"volume"
    })
    
    # Handle empty datasets
    if df.empty:
        return pd.DataFrame()
    
    # Standardize datetime format and timezone
    df["datetime"] = df["datetime"].dt.tz_localize(pytz.utc)
    
    # Remove unnecessary columns (already adjusted in auto_adjust=True)
    df = df.drop(columns=["Dividends", "Stock Splits"])
    
    # Set datetime as index for easy time-series operations
    df = df.set_index("datetime",drop=True)
    return df

def get_histories(tickers, period_starts,period_ends, granularity="1d"):
    """
    Download historical data for multiple tickers using parallel processing.
    
    Args:
        tickers: List of ticker symbols to download
        period_starts: List of start dates (one per ticker)
        period_ends: List of end dates (one per ticker)
        granularity: Data frequency for all tickers
        
    Returns:
        tuple: (filtered_tickers, dataframes)
            - filtered_tickers: List of successful ticker downloads
            - dataframes: List of corresponding DataFrames (same order as tickers)
    """
    # Initialize results list with None placeholders
    dfs = [None]*len(tickers)
    
    def _helper(i):
        """Helper function for threading - downloads single ticker."""
        print(tickers[i])  # Progress indicator
        df = get_history(
            tickers[i],
            period_starts[i], 
            period_ends[i], 
            granularity=granularity
        )
        dfs[i] = df
    
    # Create thread for each ticker download
    threads = [threading.Thread(target=_helper,args=(i,)) for i in range(len(tickers))]
    
    # Start all threads (parallel execution)
    [thread.start() for thread in threads]
    
    # Wait for all threads to complete
    [thread.join() for thread in threads]
    
    # Filter out failed downloads (empty DataFrames)
    tickers = [tickers[i] for i in range(len(tickers)) if not dfs[i].empty]
    dfs = [df for df in dfs if not df.empty]
    
    return tickers, dfs

def get_ticker_dfs(start,end):
    """
    Get ticker data with caching to avoid repeated downloads.
    
    Args:
        start: Start date for data collection
        end: End date for data collection
        
    Returns:
        tuple: (tickers, ticker_dfs)
            - tickers: List of successfully downloaded ticker symbols
            - ticker_dfs: Dictionary mapping ticker -> DataFrame
    """
    
    try:
        # Attempt to load cached data
        tickers, ticker_dfs = load_pickle("dataset.obj")
        print(f"Loaded cached data for {len(tickers)} tickers")
    except Exception as err:
        print("No cached data found. Downloading fresh data...")
        
        # Download fresh data from Yahoo Finance
        tickers = get_sp500_tickers()
        
        # Create identical start/end dates for all tickers
        starts=[start]*len(tickers)
        ends=[end]*len(tickers)
        
        # Download data using parallel processing
        tickers,dfs = get_histories(tickers,starts,ends,granularity="1d")
        
        # Convert to dictionary format for easy access
        ticker_dfs = {ticker:df for ticker,df in zip(tickers,dfs)}
        
        # Save for future use
        save_pickle("dataset.obj", (tickers,ticker_dfs))
        print(f"Downloaded and cached data for {len(tickers)} tickers")
        
    return tickers, ticker_dfs 

def main():
    """
    Main execution function for quantitative trading strategy backtesting.
    
    The system tests multiple quantitative strategies:
    - Alpha1: Volume-price momentum (contrarian)
    - Alpha2: Gap fade (mean reversion)  
    - Alpha3: Multi-timeframe trend following
    """
    # Define backtesting period
    # Start: 2010 provides sufficient history for long-term strategies
    # End: Current time for most recent performance
    period_start = datetime(2010, 1, 1, tzinfo=pytz.utc)
    period_end = datetime.now(pytz.utc)
    
    # Load market data with caching
    tickers, ticker_dfs = get_ticker_dfs(start=period_start, end=period_end)
    
    # Limit to subset for faster testing/development
    # Note: Full S&P 500 (~500 tickers) takes significantly longer
    testfor = 200
    print(f"testing {testfor} out of {len(tickers)} tickers")
    tickers = tickers[:testfor]

    # Initialize multiple Alpha strategies
    # Each strategy uses the same universe and period for fair comparison
    alpha1 = Alpha1(insts=tickers, dfs=ticker_dfs, start=period_start, end=period_end)
    alpha2 = Alpha2(insts=tickers, dfs=ticker_dfs, start=period_start, end=period_end)
    alpha3 = Alpha3(insts=tickers, dfs=ticker_dfs, start=period_start, end=period_end)
    alpha4 = Alpha4(insts=tickers, dfs=ticker_dfs, start=period_start, end=period_end)

    # Run strategy simulations with performance timing
    # @timeme decorator measures execution time for optimization
    print("Running Alpha1 (Volume-Price Momentum)...")
    df1 = alpha1.run_simulation()
    print(f"Alpha1 final capital: ${list(df1.capital)[-1]:,.2f}")
    
    print("Running Alpha2 (Gap Fade)...")
    df2 = alpha2.run_simulation()
    print(f"Alpha2 final capital: ${list(df2.capital)[-1]:,.2f}")
    
    print("Running Alpha3 (Multi-Timeframe Trend)...")
    df3 = alpha3.run_simulation()
    print(f"Alpha3 final capital: ${list(df3.capital)[-1]:,.2f}")

    df4 = alpha4.run_simulation()
    print(f"Alpha4 final capital: ${list(df4.capital)[-1]:,.2f}")

    print("Running Portfolio (Combined Strategies)...")
    portfolio = Portfolio(insts=tickers, dfs=ticker_dfs, start=period_start, end=period_end, strategies=[alpha1, alpha2, alpha3])
    df = portfolio.run_simulation()
    print(f"Portfolio final capital: ${list(df.capital)[-1]:,.2f}")
    port_ret = df['capital'].pct_change().dropna()
    import numpy as np
    # 2) annualize
    rets     = df["capital_ret"].dropna()
    ann_ret  = rets.mean() * 253
    ann_vol  = rets.std() * np.sqrt(253)
    sharpe   = ann_ret / ann_vol

    print(f"Ann. return: {ann_ret:.2%}")
    print(f"Ann. volatility: {ann_vol:.2%}")
    print(f"Sharpe ratio: {sharpe:.2f}")

    import matplotlib.pyplot as plt

    plt.plot(df1.capital, label="Alpha1")
    plt.plot(df2.capital, label="Alpha2")
    plt.plot(df3.capital, label="Alpha3")
    plt.plot(df4.capital, label="Alpha4")
    plt.plot(df.capital, label="Portfolio", linewidth=2, color="black")
    plt.title("Strategy Equity Curves")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

"""
PERFORMANCE BENCHMARKING RESULTS

=== Historical Performance (Before Optimizations) ===
Testing 200 out of 501 tickers:
- Alpha1: 199.6 seconds → Final Capital: $68,985.96
- Alpha2: 224.5 seconds → Final Capital: $31,529.18  
- Alpha3: 211.1 seconds → Final Capital: $167,449.46

=== Optimized Performance (After .loc → .at + Vectorization) ===
Testing 200 out of 501 tickers:
- Alpha1: 65.7 seconds → Final Capital: $68,985.96 (3.0x speedup)
- Alpha2: 79.1 seconds → Final Capital: $31,529.18 (2.8x speedup)
- Alpha3: 67.9 seconds → Final Capital: $167,449.46 (3.1x speedup)

=== Key Optimizations Applied ===
1. DataFrame Access: .loc → .at (single value access)
2. Eligibility Calculation: rolling.apply(lambda) → np.convolve 
3. DataFrame Structure: pd.concat for column initialization
4. Consolidated DataFrames: Single arrays vs individual instrument lookups
5. Vectorized P&L: Numpy operations vs iterative calculations

=== Strategy Performance Analysis ===
- Alpha3 (Trend Following): Best performer with $167K (+1,574% return)
- Alpha1 (Volume-Price): Solid performance with $69K (+589% return)  
- Alpha2 (Gap Fade): Modest performance with $32K (+215% return)

All strategies started with $10,000 capital over 2010-present period.
Performance differences reflect strategy alpha, not implementation efficiency.

post vector
@timeme: run_simulation took 3.60949969291687 seconds
Alpha1 final capital: $68,985.96

post zscore raw=True and replacing .at with arrays
@timeme: run_simulation took 2.137002468109131 seconds
Alpha1 final capital: $69,025.16

@timeme: run_simulation took 1.8349988460540771 seconds
Alpha2 final capital: $31,529.18

@timeme: run_simulation took 1.9094998836517334 seconds
Alpha3 final capital: $167,449.46
"""

