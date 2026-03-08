import numpy as np
import pandas as pd
import pywt
from scipy.stats import zscore
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class FractalWaveletManager:
    """
    Implements a Point-in-Time Tail Risk Warning System based on the Fractal Markets Hypothesis.
    Uses Continuous Wavelet Transform (CWT) to detect when short investment horizons 
    dominate the market, signaling potential tail-risk events.
    """
    
    def __init__(self, window_size=256, z_window=60, wavelet='cmor1.5-1.0'):
        """
        Initialize the manager.
        
        Args:
            window_size (int): Rolling window for CWT (default 256 trading days).
            z_window (int): Rolling window for Z-score calculation.
        """
        self.window_size = window_size
        self.z_window = z_window
        self.wavelet = 'cmor1.5-1.0'
        
        # Refined Scale Range: 1 to 512 days with spacing of 3
        self.scales = np.arange(1, 513, 3)
        
        # Mask for short investment horizons (e.g., <= 16 trading days)
        self.short_horizon_mask = self.scales <= 16

    def _calculate_log_returns(self, prices):
        """Calculate daily log-returns from price series."""
        return np.log(prices / prices.shift(1)).dropna()

    def _get_pit_power(self, returns_slice):
        """
        Calculate PIT power spectrum for the current day.
        
        Args:
            returns_slice (pd.Series): A slice of log-returns of length self.window_size.
            
        Returns:
            np.array: Power spectrum for the latest day (PIT).
        """
        # Compute CWT
        coefficients, frequencies = pywt.cwt(returns_slice.values, self.scales, self.wavelet)
        
        # Power is square of magnitude of complex coefficients
        power_matrix = np.abs(coefficients)**2
        
        # Extract ONLY the final column (PIT)
        return power_matrix[:, -1]

    def run_historical_backtest(self, prices):
        """
        Execute the Point-in-Time backtest logic.
        
        Args:
            prices (pd.Series): Daily closing prices.
            
        Returns:
            pd.DataFrame: PIT results including Ratio, Z-score, and Signals.
        """
        if not isinstance(prices, pd.Series):
            if isinstance(prices, pd.DataFrame):
                prices = prices.iloc[:, 0]
            else:
                raise ValueError("Input 'prices' must be a pandas Series.")

        returns = self._calculate_log_returns(prices)
        n = len(returns)
        
        if n < self.window_size + self.z_window:
            raise ValueError(f"Insufficient data ({n} days). Need at least {self.window_size + self.z_window} days.")

        ratios = np.full(n, np.nan)
        total_powers = np.full(n, np.nan)
        pit_power_history = [] 
        
        print(f"Running PIT Backtest for {n} return days...")
        
        # OPTIMIZATION: Convert returns to numpy once
        returns_np = returns.values
        
        # We can speed up by processing in larger strides if we don't need EVERY day,
        # but the user wants a full historical DataFrame.
        # Let's keep the daily loop but ensure it's as fast as possible.
        for t in range(self.window_size - 1, n):
            # Window slice
            window = returns_np[t - (self.window_size - 1) : t + 1]
            
            # Compute PIT power
            # pywt.cwt is already quite fast, but repeating it 6000 times is slow.
            coefficients, _ = pywt.cwt(window, self.scales, self.wavelet)
            pit_power = np.abs(coefficients[:, -1])**2
            
            pit_power_history.append(pit_power)
            
            # Calculate Dominance Ratio & Total Power
            short_power = np.sum(pit_power[self.short_horizon_mask])
            total_p = np.sum(pit_power)
            
            total_powers[t] = total_p
            if total_p > 0:
                ratios[t] = short_power / total_p
            else:
                ratios[t] = 0.0

            if t % 1000 == 0:
                print(f"  Progress: {t}/{n} days")

        # Construct DataFrame
        results = pd.DataFrame(index=returns.index)
        results['Price'] = prices.loc[returns.index]
        results['Returns'] = returns
        results['Dominance_Ratio'] = ratios
        results['Total_Power'] = total_powers
        
        # Calculate Rolling 60-day Z-score
        rolling_mean = results['Dominance_Ratio'].rolling(window=self.z_window).mean()
        rolling_std = results['Dominance_Ratio'].rolling(window=self.z_window).std()
        
        results['Z_Score'] = (results['Dominance_Ratio'] - rolling_mean) / rolling_std
        results['Z_Score'] = results['Z_Score'].fillna(0)
        
        # Signal Generation
        results['Signal'] = 'SAFE'
        results.loc[results['Z_Score'] > 2.0, 'Signal'] = 'WARNING'
        
        # Store PIT Power Spectrum Matrix
        valid_idx = results.index[self.window_size - 1:]
        self.pit_power_df = pd.DataFrame(
            np.array(pit_power_history), 
            index=valid_idx, 
            columns=self.scales
        )
        
        return results

    def plot_results(self, results, market_name="Market"):
        """Visualize results with a 4-panel system."""
        fig, (ax1, ax2, ax_tp, ax3) = plt.subplots(4, 1, figsize=(15, 22), gridspec_kw={'height_ratios': [1, 1, 1, 1.5]})
        
        # Formatting x-axis: Strict Date Formatting
        date_fmt = mdates.DateFormatter('%Y-%m')
        
        # 1. Price Chart
        ax1.plot(results.index, results['Price'], color='black', label='Price', linewidth=1.5)
        ax1.set_title(f"{market_name} - Price & Tail-Risk Warnings", fontsize=16, fontweight='bold')
        ax1.set_ylabel("Price")
        
        # Add Red Shades for WARNING signals
        warnings = results[results['Signal'] == 'WARNING']
        if not warnings.empty:
            for start, end in self._get_contiguous_periods(warnings.index):
                ax1.axvspan(start, end, color='red', alpha=0.3, label='_nolegend_')
        
        ax1.legend(['Price', 'Tail Risk Warning'], loc='upper left')
        ax1.grid(True, alpha=0.2)

        # 2. Dominance Ratio & Rolling Z-Score
        ax2.plot(results.index, results['Dominance_Ratio'], label='Dominance Ratio', color='blue', alpha=0.4)
        ax2.set_ylabel("Ratio")
        
        ax2_z = ax2.twinx()
        ax2_z.plot(results.index, results['Z_Score'], label='Z-Score', color='green', linewidth=1)
        ax2_z.axhline(2.0, color='red', linestyle='--', alpha=0.7, label='Threshold (2.0)')
        ax2_z.set_ylabel("Z-Score")
        
        ax2.set_title("Short-Horizon Dominance (Local Fragility)", fontsize=14)
        
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_z.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ax2.grid(True, alpha=0.2)

        # 3. THESIS VERIFICATION: Total Power (Panic Across All Horizons)
        ax_tp.fill_between(results.index, 0, results['Total_Power'], color='purple', alpha=0.3, label='Total Power')
        ax_tp.set_ylabel("Total Energy")
        ax_tp.set_title("Total Wavelet Power (Aggregate Market Panic)", fontsize=14)
        ax_tp.grid(True, alpha=0.2)
        ax_tp.set_yscale('log') # Log scale helps see volatility spikes clearly

        # 4. Wavelet Power Spectrum Heatmap (Investment Horizon Axis)
        if hasattr(self, 'pit_power_df') and not self.pit_power_df.empty:
            X = mdates.date2num(self.pit_power_df.index)
            Y = self.scales # Investment Horizon in Days
            
            Z = np.sqrt(self.pit_power_df.values.T)
            
            # Using shading='auto' with pcolormesh
            im = ax3.pcolormesh(X, Y, Z, cmap='jet', shading='auto')
            
            # Custom ticks for common investment horizons
            # 2d (ultra-short), 5d (weekly), 21d (monthly), 63d (quarterly), 126d (half-year), 252d (yearly), 504d (2-year)
            target_horizons = [2, 5, 21, 63, 126, 252, 504]
            ax3.set_yticks(target_horizons)
            ax3.set_yticklabels([str(h) for h in target_horizons])
            
            ax3.set_ylabel("Investment Horizon (Days)")
            ax3.set_title("PIT Wavelet Power Spectrum (Horizon Axis)", fontsize=14)
            # ax3.set_yscale('log') # Optional: log scale for Y if 512 range is too squashed
            
            cbar = fig.colorbar(im, ax=ax3, orientation='vertical', pad=0.01)
            cbar.set_label('Sqrt(Power)')

        for ax in [ax1, ax2, ax_tp, ax3]:
            ax.xaxis.set_major_formatter(date_fmt)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig

    def _get_contiguous_periods(self, dates):
        """Helper to find contiguous date blocks for background shading."""
        if dates.empty:
            return []
        
        periods = []
        start = dates[0]
        prev = dates[0]
        
        # We assume daily frequency; if gap > 5 days (weekend etc), we split
        for d in dates[1:]:
            if (d - prev).days <= 5: # Small gap tolerance for weekends/holidays
                prev = d
            else:
                periods.append((start, prev))
                start = d
                prev = d
        periods.append((start, prev))
        return periods
