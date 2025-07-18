import numpy as np
import pandas as pd
from utils import Alpha

class Alpha3(Alpha):
    """
    Multi-Timeframe Trend Following Strategy (Alpha3)
    """

    def __init__(self,insts,dfs,start,end):
        """
        Initialize Alpha3 strategy.
        
        Args:
            insts: List of instrument symbols
            dfs: Dictionary of price/volume DataFrames
            start: Strategy start date
            end: Strategy end date
        """
        super().__init__(insts,dfs,start,end)
    
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
            
            # Fast trend signal: 10-day MA vs 50-day MA
            # Detects short-term momentum changes
            fast = np.where(inst_df.close.rolling(10).mean() > inst_df.close.rolling(50).mean(), 1, 0)
            
            # Medium trend signal: 20-day MA vs 100-day MA  
            # Provides medium-term trend confirmation
            medium = np.where(inst_df.close.rolling(20).mean() > inst_df.close.rolling(100).mean(), 1, 0)
            
            # Slow trend signal: 50-day MA vs 200-day MA
            # Classic long-term trend filter (Golden Cross / Death Cross)
            slow = np.where(inst_df.close.rolling(50).mean() > inst_df.close.rolling(200).mean(), 1, 0)
            
            # Combine all three trend signals
            # Alpha ranges from 0 (all bearish) to 3 (all bullish)
            # Higher values indicate stronger trend alignment across timeframes
            alpha = fast + medium + slow
            
            self.dfs[inst]["alpha"] = alpha
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
    