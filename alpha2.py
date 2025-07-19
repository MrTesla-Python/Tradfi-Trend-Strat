import numpy as np
import pandas as pd
from utils import Alpha

class Alpha2(Alpha):
    """
    Gap Fade Strategy (Alpha2)
    
    This strategy exploits mean reversion in overnight gaps between close and open prices.
    
    Strategy Logic:
    1. Calculate overnight gap ratio: (open - close) / close
    2. Apply 12-period smoothing to identify persistent gap patterns
    3. Generate contrarian signals (fade the gap direction)
    4. Use continuous signal strength rather than binary long/short
    """

    def __init__(self,insts,dfs,start,end):
        """
        Initialize Alpha2 strategy.
        
        Args:
            insts: List of instrument symbols
            dfs: Dictionary of price/volume DataFrames
            start: Strategy start date
            end: Strategy end date
        """
        super().__init__(insts,dfs,start,end)
    
    def pre_compute(self,trade_range):
        """
        Calculate gap fade alpha signal for all instruments.
        
        The alpha signal is based on overnight gaps between close and next open:
        - Gap up (open > close): Negative signal (expect reversion down)
        - Gap down (open < close): Positive signal (expect reversion up)
        
        Formula: alpha = -1 * (1 - open/close).rolling(12).mean()
        
        Breakdown:
        - (open/close): Ratio of open to previous close
        - (1 - open/close): Gap magnitude (negative for gap up, positive for gap down)
        - rolling(12).mean(): 12-period smoothing to reduce noise
        - -1 * (...): Inversion for contrarian positioning
        
        Args:
            trade_range: Date range for computation
        """
        self.alphas = {}
        
        for inst in self.insts:
            inst_df = self.dfs[inst]
            
            # Calculate smoothed gap fade signal
            # (1 - open/close) gives gap direction and magnitude
            # Rolling mean smooths out daily noise
            # Negative sign creates contrarian signal
            alpha = -1 * (1-(inst_df.open/inst_df.close)).rolling(12).mean()
            
            self.alphas[inst] = alpha
        return
    
    def post_compute(self,trade_range):
        """
        Finalize alpha signals and update instrument eligibility.
        
        Args:
            trade_range: Date range for computation
        """
        temp = []
        for inst in self.insts:
            # Store alpha signal in instrument DataFrame
            self.dfs[inst]["alpha"] = self.alphas[inst]
            temp.append(self.dfs[inst]["alpha"])
        alphadf = pd.concat(temp, axis=1)
        alphadf.columns = self.insts
        alphadf = alphadf.ffill()
        self.eligiblesdf = self.eligiblesdf & (~pd.isna(alphadf))
        self.alphadf = alphadf
        return

    def compute_signal_distribution(self, eligibles, date):
        """
        Generate continuous alpha signals for position sizing.
        
        Alpha2 uses continuous signal strength:
        - Proportional position sizing based on signal confidence
        - More nuanced risk allocation across instruments
        - Better utilization of signal information
        
        Args:
            eligibles: List of eligible instrument symbols
            date: Current date for signal generation
            
        Returns:
            tuple: (forecasts_dict, total_forecast_magnitude)
                - forecasts_dict: {instrument: alpha_signal} (continuous values)
                - total_forecast_magnitude: Sum of absolute forecast values
        """
        forecasts = self.alphadf.loc[date].values
        # Return forecasts and total magnitude for normalization
        return forecasts
    