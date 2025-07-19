import numpy as np
import pandas as pd
from utils import Alpha

class Alpha1(Alpha):
    """
    Volume-Price Momentum Strategy (Alpha1)
    
    Strategy Logic:
    1. Calculate volume-weighted price momentum indicator (op4)
    2. Cross-sectionally rank instruments using z-scores
    3. Apply 12-period smoothing to reduce noise
    4. Go long top quartile, short bottom quartile instruments
    """

    def __init__(self,insts,dfs,start,end):
        """
        Initialize Alpha1 strategy.
        
        Args:
            insts: List of instrument symbols
            dfs: Dictionary of price/volume DataFrames
            start: Strategy start date
            end: Strategy end date
        """
        super().__init__(insts,dfs,start,end)
    
    def pre_compute(self,trade_range):
        """
        Calculate the volume-weighted price momentum indicator (op4) for all instruments.
        
        Args:
            trade_range: Date range for computation
        """
        self.op4s = {}
        
        for inst in self.insts:
            inst_df = self.dfs[inst]
            
            # Volume component - measures market participation
            op1 = inst_df.volume
            
            # Price position within daily range
            # (close-low) = distance from low
            # (high-close) = distance from high  
            # Difference shows bias toward high or low
            op2 = (inst_df.close - inst_df.low ) - (inst_df.high - inst_df.close)
            
            # Daily price range (normalization factor)
            op3 = inst_df.high - inst_df.low
            
            # Volume-weighted directional pressure
            # High volume + close near high = strong buying
            # High volume + close near low = strong selling
            op4 = op1 * op2 / op3       
            
            self.op4s[inst] = op4
        return 


    def post_compute(self,trade_range):
        """
        Process raw op4 indicators into normalized alpha signals.
        
        Steps:
        1. Consolidate all op4 values into cross-sectional DataFrame
        2. Handle infinite values (replace with 0)
        3. Apply cross-sectional z-score normalization (rank-based)
        4. Apply 12-period smoothing to reduce noise
        5. Invert signal (negative alpha values for contrarian approach)
        6. Update instrument eligibility based on data availability
        
        Args:
            trade_range: Date range for computation
        """
        # Store op4 values in instrument DataFrames
        temp = []
        for inst in self.insts:
            self.dfs[inst]["op4"] = self.op4s[inst]
            temp.append(self.dfs[inst]["op4"])

        # Create cross-sectional DataFrame for ranking
        temp_df = pd.concat(temp,axis=1)
        temp_df.columns = self.insts
        
        # Clean infinite values that can occur when high-low = 0
        temp_df = temp_df.replace(np.inf, np.nan).replace(-np.inf, np.nan)
        
        # Cross-sectional z-score normalization
        # Converts raw values to relative ranks on each date
        zscore = lambda x: (x - np.nanmean(x))/np.nanstd(x)
        cszcre_df = temp_df.ffill().apply(zscore, axis=1, raw=True)

        alphas = []
        
        # Process normalized signals for each instrument
        for inst in self.insts:
            # Apply smoothing and invert signal (contrarian approach)
            # 12-period rolling mean reduces noise and lag
            # Multiplication by -1 creates contrarian signal
            self.dfs[inst]["alpha"] = cszcre_df[inst].rolling(12).mean() * -1
            alphas.append(self.dfs[inst]["alpha"])
        alphadf = pd.concat(alphas,axis=1)
        alphadf.columns = self.insts
        self.eligiblesdf = self.eligiblesdf & (~pd.isna(alphadf))
        self.alphadf = alphadf
        masked_df = self.alphadf/self.eligiblesdf
        masked_df = masked_df.replace([-np.inf, np.inf],np.nan)
        num_eligibles = self.eligiblesdf.sum(axis=1)
        rankdf= masked_df.rank(axis=1,method="average",na_option="keep",ascending=True)
        shortdf = rankdf.apply(lambda col: col <= num_eligibles.values/4, axis=0, raw=True)
        longdf = rankdf.apply(lambda col: col > np.ceil(num_eligibles - num_eligibles/4), axis=0, raw=True)

        forecast_df = -1*shortdf.astype(np.int32) + longdf.astype(np.int32)
        self.forecast_df = forecast_df
        return

    def compute_signal_distribution(self, eligibles, date):
        """
        Generate long/short signals based on alpha score rankings.
        
        Args:
            eligibles: List of eligible instrument symbols
            date: Current date for signal generation
            
        Returns:
            tuple: (forecasts_dict, total_forecast_magnitude)
                - forecasts_dict: {instrument: signal} where signal in {-1, 0, 1}
                - total_forecast_magnitude: Sum of absolute forecast values
        """
        forecasts = self.forecast_df.loc[date].values
        return forecasts
    