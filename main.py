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
from utils import Portfolio

def get_sp500_tickers():
    res = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = BeautifulSoup(res.content,'html')
    table = soup.find_all('table')[0] 
    df = pd.read_html(str(table))
    tickers = list(df[0].Symbol)
    return tickers

def get_history(ticker, period_start, period_end, granularity="1d", tries=0):
    try:
        df = yfinance.Ticker(ticker).history(
            start=period_start,
            end=period_end,
            interval=granularity,
            auto_adjust=True
        ).reset_index()
    except Exception as err:
        if tries < 5:
            return get_history(ticker, period_start, period_end, granularity, tries+1)
        return pd.DataFrame()
    
    df = df.rename(columns={
        "Date":"datetime",
        "Open":"open",
        "High":"high",
        "Low":"low",
        "Close":"close",
        "Volume":"volume"
    })
    if df.empty:
        return pd.DataFrame()
    
    df["datetime"] = df["datetime"].dt.tz_localize(pytz.utc)
    df = df.drop(columns=["Dividends", "Stock Splits"])
    df = df.set_index("datetime",drop=True)
    return df

def get_histories(tickers, period_starts,period_ends, granularity="1d"):
    dfs = [None]*len(tickers)
    def _helper(i):
        print(tickers[i])
        df = get_history(
            tickers[i],
            period_starts[i], 
            period_ends[i], 
            granularity=granularity
        )
        dfs[i] = df
    threads = [threading.Thread(target=_helper,args=(i,)) for i in range(len(tickers))]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]
    tickers = [tickers[i] for i in range(len(tickers)) if not dfs[i].empty]
    dfs = [df for df in dfs if not df.empty]
    return tickers, dfs

def get_ticker_dfs(start,end):
    from utils import load_pickle,save_pickle
    try:
        tickers, ticker_dfs = load_pickle("dataset.obj")
    except Exception as err:
        tickers = get_sp500_tickers()
        starts=[start]*len(tickers)
        ends=[end]*len(tickers)
        tickers,dfs = get_histories(tickers,starts,ends,granularity="1d")
        ticker_dfs = {ticker:df for ticker,df in zip(tickers,dfs)}
        save_pickle("dataset.obj", (tickers,ticker_dfs))
    return tickers, ticker_dfs 

def main():
    period_start = datetime(2010, 1, 1, tzinfo=pytz.utc)
    period_end = datetime.now(pytz.utc)
    tickers, ticker_dfs = get_ticker_dfs(start=period_start, end=period_end)
    testfor = 20
    print(f"testing {testfor} out of {len(tickers)} tickers")
    tickers = tickers[:testfor]

    alpha1 = Alpha1(insts=tickers, dfs=ticker_dfs, start=period_start, end=period_end)
    alpha2 = Alpha2(insts=tickers, dfs=ticker_dfs, start=period_start, end=period_end)
    alpha3 = Alpha3(insts=tickers, dfs=ticker_dfs, start=period_start, end=period_end)

    df1 = alpha1.run_simulation()
    print(list(df1.capital)[-1])
    # df2 = alpha2.run_simulation()
    # print(list(df2.capital)[-1])
    # df3 = alpha3.run_simulation()
    # print(list(df3.capital)[-1])

if __name__ == "__main__":
    main()


"""
testing 200 out of 501 tickers
@timeme: run_simulation took 199.55849957466125 seconds
68985.96086326586
@timeme: run_simulation took 224.54899740219116 seconds
31529.17547384153
@timeme: run_simulation took 211.13549733161926 seconds
167449.46414392962


ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    780/1    0.004    0.000   70.900   70.900 {built-in method builtins.exec}
        1    0.024    0.024   70.900   70.900 main.py:1(<module>)
        1    0.000    0.000   70.161   70.161 main.py:83(main)
        1    0.000    0.000   68.230   68.230 utils.py:7(timediff)
        1    0.916    0.916   68.229   68.229 utils.py:114(run_simulation)
   260912    0.959    0.000   37.385    0.000 indexing.py:882(__setitem__)
   260912    0.920    0.000   31.331    0.000 indexing.py:1785(_setitem_with_indexer)
   260912    0.564    0.000   28.981    0.000 indexing.py:1946(_setitem_with_indexer_split_path)
  1469433    2.437    0.000   27.957    0.000 indexing.py:1176(__getitem__)
   260912    1.070    0.000   27.621    0.000 indexing.py:2111(_setitem_single_column)
  1095161    1.369    0.000   12.688    0.000 frame.py:4191(_get_value)
   260913    0.348    0.000   12.553    0.000 generic.py:6432(dtypes)
272740/272719    1.520    0.000   11.206    0.000 series.py:389(__init__)
   260912    0.559    0.000   10.708    0.000 managers.py:1298(column_setitem)
     5671    0.242    0.000   10.490    0.002 utils.py:25(get_pnl_stats)
        1    0.002    0.002    6.923    6.923 utils.py:92(compute_meta_info)
  1095583    0.905    0.000    5.766    0.000 frame.py:4626(_get_item_cache)
   260972    3.060    0.000    5.277    0.000 managers.py:1066(iset)
       60    0.000    0.000    5.256    0.088 rolling.py:562(_apply)
       60    0.000    0.000    5.256    0.088 rolling.py:460(_apply_columnwise)

testing 20 out of 501 tickers
@timeme: run_simulation took 68.22950029373169 seconds
31563.732732798675
>>.loc > .at
testing 20 out of 501 tickers
@timeme: run_simulation took 22.14350175857544 seconds
31563.732732798675

  ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    780/1    0.003    0.000   27.519   27.519 {built-in method builtins.exec}
        1    0.032    0.032   27.518   27.518 main.py:1(<module>)
        1    0.000    0.000   26.852   26.852 main.py:83(main)
        1    0.000    0.000   24.993   24.993 utils.py:7(timediff)
        1    0.647    0.647   24.992   24.992 utils.py:114(run_simulation)
  1095161    0.751    0.000   12.997    0.000 indexing.py:2568(__getitem__)
  1095161    0.698    0.000   11.666    0.000 indexing.py:2518(__getitem__)
  1095161    1.049    0.000   10.796    0.000 frame.py:4191(_get_value)
        1    0.002    0.002    6.289    6.289 utils.py:92(compute_meta_info)
     5671    0.218    0.000    5.341    0.001 utils.py:25(get_pnl_stats)
  1095583    0.754    0.000    5.046    0.000 frame.py:4626(_get_item_cache)
       60    0.000    0.000    4.745    0.079 rolling.py:562(_apply)
       60    0.000    0.000    4.744    0.079 rolling.py:460(_apply_columnwise)
       60    0.000    0.000    4.744    0.079 rolling.py:440(_apply_series)
       60    0.000    0.000    4.740    0.079 rolling.py:595(homogeneous_func)
       60    0.002    0.000    4.739    0.079 rolling.py:601(calc)
       20    0.000    0.000    4.737    0.237 rolling.py:2016(apply)
       20    0.000    0.000    4.737    0.237 rolling.py:1471(apply)
       20    0.119    0.006    4.734    0.237 rolling.py:1531(apply_func)
   260912    0.204    0.000    4.284    0.000 indexing.py:2577(__setitem__)


   Wrote profile results to main.py.lprof
Timer unit: 1e-06 s

Total time: 9.61358 s
File: C:/Users/trist/Documents/HangukQuant/Q101/utils.py
Function: compute_meta_info at line 92

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    92                                               @profile
    93                                               def compute_meta_info(self,trade_range):
    94         1      14989.3  14989.3      0.2          self.pre_compute(trade_range=trade_range)
    95
    96        21         13.3      0.6      0.0          for inst in self.insts:
    97        20       5236.8    261.8      0.1              df=pd.DataFrame(index=trade_range)
    98        20      11069.7    553.5      0.1              inst_vol = (-1 + self.dfs[inst]["close"]/self.dfs[inst]["close"].shift(1)).rolling(30).std()
    99        20      26547.8   1327.4      0.3              self.dfs[inst] = df.join(self.dfs[inst]).ffill().bfill()
   100        20      11555.1    577.8      0.1              self.dfs[inst]["ret"] = -1 + self.dfs[inst]["close"]/self.dfs[inst]["close"].shift(1)
   101        20      11891.3    594.6      0.1              self.dfs[inst]["vol"] = inst_vol
   102        20       5459.6    273.0      0.1              self.dfs[inst]["vol"] = self.dfs[inst]["vol"].ffill().fillna(0)
   103        20       5244.4    262.2      0.1              self.dfs[inst]["vol"] = np.where(self.dfs[inst]["vol"] < 0.005, 0.005, self.dfs[inst]["vol"])
   104        20       5873.2    293.7      0.1              sampled = self.dfs[inst]["close"] != self.dfs[inst]["close"].shift(1).bfill()
   105        20    7430619.9 371531.0     77.3              eligible = sampled.rolling(5).apply(lambda x: int(np.any(x))).fillna(0)
   106        20      14710.8    735.5      0.2              self.dfs[inst]["eligible"] = eligible.astype(int) & (self.dfs[inst]["close"] > 0).astype(int)
   107
   108         1    2070370.1    2e+06     21.5          self.post_compute(trade_range=trade_range)
   109         1          0.3      0.3      0.0          return

>> .apply(raw=False) > .apply(raw=True) 
testing 20 out of 501 tickers
@timeme: run_simulation took 6.906998157501221 seconds
31563.732732798675

>> .apply to using convolve
testing 20 out of 501 tickers
@timeme: run_simulation took 6.251500129699707 seconds
31563.732732798675
"""

