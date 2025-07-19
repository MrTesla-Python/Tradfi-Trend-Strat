import numpy as np
import pandas as pd
from utils import Alpha

class Alpha4(Alpha):
    """
    Multi-Timeframe Trend Following Strategy (Alpha3)
    """

    def __init__(self, lookbacks=[(16,64),(32,128),(64,256),(128,512)], **kwargs):
        """
        Initialize Alpha3 strategy.
        
        Args:
            insts: List of instrument symbols
            dfs: Dictionary of price/volume DataFrames
            start: Strategy start date
            end: Strategy end date
        """
        super().__init__(**kwargs)
        self.lookbacks = lookbacks
    
    def pre_compute(self,trade_range):
        """
        Calculate multi-timeframe trend signals for all instruments.
        
        Each indicator outputs 1 (bullish) or 0 (bearish).
        Combined alpha ranges from 0-3 based on trend alignment.
        
        Args:
            trade_range: Date range for computation
        """
        for inst in self.insts:
            inst_df = self.dfs[inst]
            
            trending = pd.Series(0.0, index=inst_df.index)
            
            # Calculate crossover signal for each lookback period
            for n1, n2 in self.lookbacks:
                # Calculate short and long exponential moving averages
                ema1 = inst_df.close.ewm(span=n1, adjust=False).mean()
                ema2 = inst_df.close.ewm(span=n2, adjust=False).mean()

                # Calculate percentage difference signal
                sig =  100 * (ema1 - ema2) / ema2
                # Clip signal to prevent extreme outliers
                sig = sig.clip(-20, 20)

                # Add to composite trending signal
                trending = trending.add(sig, fill_value=0)

            # Store the computed alpha signal
            self.dfs[inst]['alpha'] = trending.astype(np.float64)
        return
    
    def post_compute(self,trade_range):
        """
        Finalize trend signals and update instrument eligibility.
        
        
        Args:
            trade_range: Date range for computation
        """
        temp = []
        for inst in self.insts:
            temp.append(self.dfs[inst]["alpha"])
        alphadf = pd.concat(temp, axis=1)
        alphadf.columns = self.insts
        alphadf = alphadf.ffill()
        self.eligiblesdf = self.eligiblesdf & (~pd.isna(alphadf))
        self.alphadf = alphadf
        return

    def compute_signal_distribution(self, eligibles, date):
        """
        Generate trend-based signals proportional to trend strength.
        
        Args:
            eligibles: List of eligible instrument symbols
            date: Current date for signal generation
            
        Returns:
            tuple: (forecasts_dict, total_forecast_magnitude)
                - forecasts_dict: {instrument: trend_score} (values 0-3)
                - total_forecast_magnitude: Sum of all forecast values
        """
        return self.alphadf.loc[date].values
    