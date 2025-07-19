import lzma
import dill as pickle
import numpy as np
import pandas as pd
from copy import deepcopy
import time
from functools import wraps

def timeme(func):
    """
    Decorator to measure and print execution time of functions.
    Useful for performance profiling of trading simulations.
    """
    @wraps(func)
    def timediff(*arggs,**kwargs):
        a = time.time()
        result = func(*arggs,**kwargs)
        b= time.time()
        print(f'@timeme: {func.__name__} took {b - a} seconds')
        return result
    return timediff

def load_pickle(path):
    """Load compressed pickle file using lzma compression."""
    with lzma.open(path,"rb") as fp:
        file = pickle.load(fp)
    return file

def save_pickle(path,obj):
    """Save object to compressed pickle file using lzma compression."""
    with lzma.open(path,"wb") as fp:
        pickle.dump(obj,fp)

def get_pnl_stats(last_weights, last_units, prev_close, portfolio_i, ret_row, portfolio_df):
    """
    Vectorized P&L calculation for EfficientAlpha class.
    
    Args:
        last_weights: Array of position weights from previous day
        last_units: Array of units held from previous day  
        prev_close: Array of previous day's closing prices
        portfolio_i: Current portfolio index
        ret_row: Array of daily returns for all instruments
        portfolio_df: Portfolio DataFrame to update
        
    Returns:
        tuple: (day_pnl, capital_ret) - daily P&L and capital return
    """
    ret_row = np.nan_to_num(ret_row, nan=0, posinf=0, neginf=0)
    # Vectorized P&L calculation: units * previous_price * return
    day_pnl = np.sum(last_units * prev_close * ret_row)
    
    # Vectorized nominal return: dot product of weights and returns  
    nominal_ret = np.dot(last_weights, ret_row)
    
    # Apply leverage to get capital return
    capital_ret = nominal_ret * portfolio_df.at[portfolio_i - 1, "leverage"]
    
    # Update portfolio statistics
    portfolio_df.at[portfolio_i,"capital"] = portfolio_df.at[portfolio_i - 1,"capital"] + day_pnl
    portfolio_df.at[portfolio_i,"day_pnl"] = day_pnl
    portfolio_df.at[portfolio_i,"nominal_ret"] = nominal_ret
    portfolio_df.at[portfolio_i,"capital_ret"] = capital_ret
    return day_pnl, capital_ret

class AbstractImplementationException(Exception):
    """Custom exception for abstract methods that must be implemented by subclasses."""
    pass

class Alpha():
    """
    Highly optimized version of Alpha class using vectorized operations.
    
    This class provides significant performance improvements over the base Alpha class by:
    - Using consolidated DataFrames for all instruments
    - Implementing vectorized operations instead of loops
    - Minimizing DataFrame indexing operations
    - Using numpy arrays for mathematical computations
    
    Designed for high-frequency backtesting and production trading systems.
    """
    
    def __init__(self, insts, dfs, start, end, portfolio_vol=0.20):
        """
        Initialize EfficientAlpha with same parameters as Alpha.
        
        Args:
            insts: List of instrument/ticker symbols
            dfs: Dictionary of DataFrames containing price/volume data
            start: Start date for simulation
            end: End date for simulation  
            portfolio_vol: Target portfolio volatility (default 20%)
        """
        self.insts = insts
        self.dfs = deepcopy(dfs)
        self.start = start 
        self.end = end
        self.portfolio_vol = portfolio_vol

    def init_portfolio_settings(self, trade_range):
        """
        Initialize portfolio DataFrame - identical to Alpha class.
        
        Args:
            trade_range: Date range for simulation
            
        Returns:
            DataFrame: Portfolio DataFrame with pre-allocated columns
        """
        # Create base DataFrame with datetime column
        portfolio_df = pd.DataFrame(index=trade_range)\
            .reset_index()\
            .rename(columns={"index":"datetime"})
        
        # Prepare all columns that will be used during simulation
        columns_to_init = ["capital", "day_pnl", "capital_ret", "nominal_ret", "nominal", "leverage"]
        for inst in self.insts:
            columns_to_init.extend([f"{inst} w", f"{inst} units"])
        
        # Initialize all columns at once using pd.concat to avoid fragmentation
        init_data = pd.DataFrame(
            data=np.nan, 
            index=portfolio_df.index, 
            columns=columns_to_init
        )
        portfolio_df = pd.concat([portfolio_df, init_data], axis=1)
            
        portfolio_df.at[0,"capital"] = 10000
        portfolio_df.at[0,"day_pnl"] = 0.0
        portfolio_df.at[0,"capital_ret"] = 0.0
        portfolio_df.at[0,"nominal_ret"] = 0.0
        return portfolio_df
    
    def pre_compute(self,trade_range):
        """Hook for subclasses to perform pre-computation steps."""
        pass

    def post_compute(self,trade_range):
        """Hook for subclasses to perform post-computation steps."""
        pass

    def compute_signal_distribution(self, eligibles, date):
        """
        Abstract method for signal generation - must be implemented by subclasses.
        
        Args:
            eligibles: Array of eligibility flags for instruments
            date: Current date index
            
        Returns:
            tuple: (forecasts_array, total_forecast_magnitude)
        """
        raise AbstractImplementationException("no concrete implementation for signal generation")

    def get_strat_scaler(self, target_vol, ewmas, ewstrats):
        """
        Calculate strategy scaling factor - identical to Alpha class.
        
        Args:
            target_vol: Target portfolio volatility
            ewmas: List of exponentially weighted moving averages
            ewstrats: List of exponentially weighted strategy scalers
            
        Returns:
            float: Strategy scaling factor
        """
        ann_realized_vol = np.sqrt(ewmas[-1] * 253)
        return target_vol / ann_realized_vol * ewstrats[-1]

    def compute_meta_info(self,trade_range):
        """
        Compute metadata for all instruments - similar to Alpha but with optimizations.
        
        Args:
            trade_range: Date range for computation
        """
        self.pre_compute(trade_range=trade_range)
        
        def is_any_one(x):
            """Helper function for eligibility checking."""
            return int(np.any(x))

        # Collect data for all instruments
        closes, eligibles, vols, rets = [], [], [], []
        
        for inst in self.insts:
            df=pd.DataFrame(index=trade_range)
            
            # Calculate rolling volatility
            inst_vol = (-1 + self.dfs[inst]["close"]/self.dfs[inst]["close"].shift(1)).rolling(30).std()
            
            # Align data with trade range
            self.dfs[inst] = df.join(self.dfs[inst]).ffill().bfill()
            
            # Compute derived metrics
            self.dfs[inst]["ret"] = -1 + self.dfs[inst]["close"]/self.dfs[inst]["close"].shift(1)
            self.dfs[inst]["vol"] = inst_vol
            self.dfs[inst]["vol"] = self.dfs[inst]["vol"].ffill().fillna(0)       
            self.dfs[inst]["vol"] = np.where(self.dfs[inst]["vol"] < 0.005, 0.005, self.dfs[inst]["vol"])
            
            # Fast eligibility calculation using convolution
            sampled = self.dfs[inst]["close"] != self.dfs[inst]["close"].shift(1).bfill()
            convolved = np.convolve(sampled, np.ones(5, dtype=int), mode='valid')
            eligible_vectorized = (convolved >= 1).astype(int)
            eligible = pd.Series(np.concatenate([np.zeros(4, dtype=int), eligible_vectorized]), index=sampled.index)
            close_condition = (self.dfs[inst]['close'] > 0).astype(int)
            eligibles.append(eligible & close_condition)
            closes.append(self.dfs[inst]["close"])
            vols.append(self.dfs[inst]["vol"])
            rets.append(self.dfs[inst]["ret"])

        # Create consolidated DataFrames for vectorized operations
        self.eligiblesdf = pd.concat(eligibles, axis=1)
        self.eligiblesdf.columns = self.insts
        self.closedf = pd.concat(closes, axis=1)
        self.closedf.columns = self.insts
        self.voldf = pd.concat(vols, axis=1)
        self.voldf.columns = self.insts
        self.retdf = pd.concat(rets, axis=1)
        self.retdf.columns = self.insts


        self.post_compute(trade_range=trade_range)
        return

    @timeme      
    def run_simulation(self):
        """
        Execute optimized trading simulation using vectorized operations.
        
        This method provides significant performance improvements over Alpha.run_simulation()
        by using:
        - Generator-based data iteration to minimize memory usage
        - Vectorized P&L calculations
        - Reduced DataFrame indexing operations
        - Numpy arrays for mathematical operations
        
        Returns:
            DataFrame: Portfolio performance with datetime index
        """
        # Setup simulation
        date_range = pd.date_range(start=self.start,end=self.end, freq="D")
        self.compute_meta_info(trade_range=date_range)
        self.portfolio_df = self.init_portfolio_settings(trade_range=date_range)

        # Initialize tracking variables
        units_held, weights_held = [], []  # Store position arrays for vectorized P&L
        close_prev = None  # Previous day's closing prices
        
        # Risk management state
        ewmas, ewstrats = [0.01], [1]
        strat_scalars = []

        portfolio_df = self.portfolio_df

        # Main simulation loop using data generator
        for data in self.zip_data_generator():
            # Extract data for current iteration
            portfolio_i = data['portfolio_i']
            portfolio_row = data['portfolio_row']
            ret_i = data['ret_i']
            ret_row = data['ret_row']        # Array of returns for all instruments
            close_row = data['close_row']    # Array of closing prices for all instruments  
            eligibles_row = data['eligibles_row']  # Array of eligibility flags
            vol_row = data['vol_row']        # Array of volatilities for all instruments
            
            strat_scalar = 2  # Default scaling factor

            # Calculate P&L and update risk metrics (skip first day)
            if portfolio_i != 0:
                # Dynamic strategy scaling based on recent performance
                strat_scalar = self.get_strat_scaler(
                    target_vol=self.portfolio_vol,
                    ewmas=ewmas,
                    ewstrats=ewstrats
                )
                
                # Vectorized P&L calculation using previous positions
                day_pnl, capital_ret = get_pnl_stats(
                    last_weights=weights_held[-1], 
                    last_units=units_held[-1], 
                    prev_close=close_prev, 
                    portfolio_i=portfolio_i, 
                    ret_row=ret_row, 
                    portfolio_df=portfolio_df
                    )

                ewmas.append(0.06 * (capital_ret**2) + 0.94 * ewmas[-1] if capital_ret != 0 else ewmas[-1])
                ewstrats.append(0.06 * strat_scalar + 0.94 * ewstrats[-1] if capital_ret != 0 else ewstrats[-1])
            strat_scalars.append(strat_scalar)
            # Generate trading signals for current day
            forecasts = self.compute_signal_distribution(
                eligibles_row,
                ret_i
            )
            if type(forecasts) == pd.Series: forecasts = forecasts.values
            forecasts = forecasts / eligibles_row     

            forecasts = np.nan_to_num(forecasts, nan=0, posinf=0, neginf=0)
            forecast_chips = np.sum(np.abs(forecasts))
            # Calculate position sizing target
            vol_target = (self.portfolio_vol / np.sqrt(253)) \
                * portfolio_df.at[portfolio_i,"capital"]

            # Vectorized position sizing calculation
            positions = strat_scalar * \
                    forecasts / forecast_chips  \
                    * vol_target \
                    / (vol_row * close_row) if forecast_chips != 0 else np.zeros_like(len(self.insts))
            positions = np.nan_to_num(positions, nan=0, posinf=0, neginf=0) 
               
            # Calculate total nominal exposure
            nominal_tot = np.linalg.norm(positions * close_row, ord=1)

            # Store positions for next iteration's P&L calculation
            units_held.append(positions)
            weights = positions * close_row / nominal_tot
            weights = np.nan_to_num(weights, nan=0, posinf=0, neginf=0)
            weights_held.append(weights)


            # Update portfolio statistics
            portfolio_df.at[portfolio_i, "nominal"] = nominal_tot
            portfolio_df.at[portfolio_i, "leverage"] \
                = nominal_tot / portfolio_df.at[portfolio_i, "capital"]
            
            # Store current prices for next iteration
            close_prev = close_row
        return portfolio_df.set_index("datetime",drop=True)

    def zip_data_generator(self):
        """
        Generator that yields aligned data for each simulation day.
        """
        for (portfolio_i, portfolio_row), \
            (ret_i, ret_row), \
            (close_i, close_row), \
            (eligibles_i, eligibles_row), \
            (vol_i, vol_row) in zip(
                self.portfolio_df.iterrows(),    # Portfolio tracking data
                self.retdf.iterrows(),           # Daily returns for all instruments
                self.closedf.iterrows(),         # Closing prices for all instruments
                self.eligiblesdf.iterrows(),     # Eligibility flags for all instruments
                self.voldf.iterrows()            # Volatilities for all instruments
            ):
            yield {
                'portfolio_i': portfolio_i,
                'portfolio_row': portfolio_row.values,
                'ret_i': ret_i,
                'ret_row': ret_row.values,
                'close_row': close_row.values,
                'eligibles_row': eligibles_row.values,
                'vol_row': vol_row.values
            }
            
from collections import defaultdict

class Portfolio(Alpha):
    """
    Portfolio class for combining multiple trading strategies.
    """

    def __init__(self, insts, dfs, start, end, strategies):
        """
        Initialize Portfolio with multiple strategy DataFrames.
        
        Args:
            insts: List of instrument symbols
            dfs: Dictionary of price/volume DataFrames
            start: Start date for simulation
            end: End date for simulation
            stratdfs: List of strategy DataFrames from individual Alpha models
        """
        super().__init__(insts, dfs, start, end)
        self.strategies = strategies 
    
    def pre_compute(self, trade_range):
        """Prepare individual strategies for simulation."""
        for strat in self.strategies:
            strat.compute_meta_info(trade_range=trade_range)

    def post_compute(self, trade_range):
        """Portfolio doesn't need post-computation - signals are combined dynamically."""
        pass

    def compute_signal_distribution(self, eligibles, date):
        """
        Combine signals from all underlying strategies dynamically.
        
        Args:
            eligibles: Array of eligible instruments
            date: Current date for signal computation
            
        Returns:
            numpy.array: Combined forecasts array matching instrument order
        """
        # Get signals from each strategy for the current date
        strategy_signals = []
        
        for strat in self.strategies:
            # Generate signal for current date using each strategy
            signal = strat.compute_signal_distribution(eligibles, date)
            
            # Clean the signal
            signal = np.nan_to_num(signal, nan=0, posinf=0, neginf=0)
            
            # Normalize each strategy's signal to have unit L1 norm
            # This ensures all strategies contribute equally regardless of their natural scale
            signal = signal / np.sum(np.abs(signal)) if np.sum(np.abs(signal)) != 0 else signal 
            
            strategy_signals.append(signal)
        
        # Take the mean of normalized signals
        # All strategies now have equal weight in the final portfolio
        combined = np.mean(strategy_signals, axis=0)
            
        return combined