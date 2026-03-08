import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wavelet_risk_engine import FractalWaveletManager
import os
import base64
from io import BytesIO

def get_base64_image(fig):
    """Convert matplotlib figure to base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def calculate_forward_metrics(results, horizons=[5, 10, 21, 63]):
    """
    Calculate future returns, volatility, and max drawdown for given horizons.
    """
    performance = {}
    
    for h in horizons:
        # Future Log Return (cumulative)
        fwd_returns = results['Returns'].shift(-h).rolling(window=h).sum().shift(1-h)
        
        # Future Volatility (annualized)
        fwd_vol = results['Returns'].shift(-h).rolling(window=h).std().shift(1-h) * np.sqrt(252)
        
        # Future Max Drawdown (relative to price at t)
        # We look at the min price over the next h days vs the current price
        fwd_min_p = results['Price'].shift(-h).rolling(window=h).min().shift(1-h)
        fwd_drawdown = (fwd_min_p / results['Price']) - 1

        performance[h] = {
            'Return': fwd_returns,
            'Vol': fwd_vol,
            'Drawdown': fwd_drawdown
        }
        
    return performance

def main():
    # Define Markets, Tickers, and Descriptions
    markets = {
        "US (S&P 500)": {
            "ticker": "^GSPC",
            "desc": "The benchmark for US large-cap equities, representing 500 of the largest companies listed on stock exchanges in the US."
        },
        "China (CSI 300)": {
            "ticker": "000300.SS",
            "desc": "A capitalization-weighted stock market index designed to replicate the performance of the top 300 stocks traded on the Shanghai and Shenzhen stock exchanges."
        },
        "Hong Kong (Hang Seng)": {
            "ticker": "^HSI",
            "desc": "A free float-adjusted market capitalization-weighted stock market index in Hong Kong, used to record and monitor daily changes of the largest companies of the Hong Kong stock market."
        },
        "Japan (Nikkei 225)": {
            "ticker": "^N225",
            "desc": "A stock market index for the Tokyo Stock Exchange, measuring the performance of 225 large, publicly owned companies in Japan."
        },
        "Taiwan (TAIEX)": {
            "ticker": "^TWII",
            "desc": "The Taiwan Capitalization Weighted Stock Index, which covers all listed stocks on the Taiwan Stock Exchange."
        },
        "South Korea (KOSPI)": {
            "ticker": "^KS11",
            "desc": "The Korea Composite Stock Price Index, the index of all common stocks traded on the Stock Market Division of the Korea Exchange."
        },
        "Europe (Euro Stoxx 50)": {
            "ticker": "^STOXX50E",
            "desc": "A stock index of Eurozone stocks designed by Stoxx, representing the 50 largest and most liquid stocks in the Eurozone."
        }
    }
    
    # Initialize Engine
    manager = FractalWaveletManager(window_size=256, z_window=60)
    
    # Storage for HTML Report
    html_sections = []
    summary_data = []
    
    # Create directory for plots if needed
    if not os.path.exists("plots"):
        os.makedirs("plots")

    print("Starting Global Multi-Market Backtest...")
    
    for market_name, info in markets.items():
        ticker = info['ticker']
        desc = info['desc']
        print(f"\nProcessing {market_name} ({ticker})...")
        
        try:
            # Fetch Data (Extended range from 2000)
            data = yf.download(ticker, start="2000-01-01", end="2026-03-01")
            
            if data.empty:
                print(f"Warning: No data found for {ticker}")
                continue
            
            # Use 'Adj Close' if available, otherwise 'Close'
            price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
            prices = data[price_col].squeeze()
            
            # Run PIT Backtest
            results = manager.run_historical_backtest(prices)
            
            # Calculate Forward Metrics
            horizons = [5, 10, 21, 63]
            fwd_metrics = calculate_forward_metrics(results, horizons)
            
            # Group Stats by Signal
            stats_list = []
            for h in horizons:
                for sig in ['WARNING', 'SAFE']:
                    mask = results['Signal'] == sig
                    avg_ret = fwd_metrics[h]['Return'][mask].mean() * 100
                    avg_vol = fwd_metrics[h]['Vol'][mask].mean() * 100
                    avg_dd = fwd_metrics[h]['Drawdown'][mask].mean() * 100
                    
                    stats_list.append({
                        "Horizon": f"{h}d",
                        "Signal": sig,
                        "Avg Return (%)": avg_ret,
                        "Avg Vol (An.) (%)": avg_vol,
                        "Avg Drawdown (%)": avg_dd
                    })
            
            stats_df = pd.DataFrame(stats_list)
            stats_html = stats_df.to_html(classes='stats-table', index=False, float_format='%.2f')

            # Calculate Overall Statistics
            total_days = len(results)
            warning_days = (results['Signal'] == 'WARNING').sum()
            warning_pct = (warning_days / total_days) * 100
            
            summary_data.append({
                "Market": market_name,
                "Ticker": ticker,
                "Total Days": total_days,
                "Warning Days": warning_days,
                "Warning %": f"{warning_pct:.2f}%"
            })
            
            # Visualize Results
            print(f"Generating plots and encoding for report...")
            fig = manager.plot_results(results, market_name=market_name)
            
            img_b64 = get_base64_image(fig)
            
            # Also save file
            safe_name = market_name.replace(" ", "_").replace("(", "").replace(")", "").replace("&", "and").lower()
            fig.savefig(f"plots/{safe_name}_risk_analysis.png")
            plt.close(fig)
            
            # Add to HTML
            html_sections.append(f"""
            <div class="market-section">
                <h2>{market_name} ({ticker})</h2>
                <p class="description">{desc}</p>
                
                <div class="stats-container">
                    <div class="stats-overview">
                        <strong>Total Backtest Days:</strong> {total_days} | 
                        <strong>Warning Days:</strong> {warning_days} | 
                        <strong>Warning Probability:</strong> {warning_pct:.2f}%
                    </div>
                    
                    <h3>Forward Performance Analysis (Signal vs. Safe)</h3>
                    {stats_html}
                </div>
                
                <img src="data:image/png;base64,{img_b64}" alt="{market_name} Analysis">
            </div>
            """)
            
        except Exception as e:
            print(f"Error processing {market_name}: {e}")

    # Generate Final HTML Report
    if html_sections:
        summary_df = pd.DataFrame(summary_data)
        summary_table = summary_df.to_html(classes='summary-table', index=False)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fractal Wavelet Risk Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f4f7f6; }}
                h1 {{ color: #2c3e50; text-align: center; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }}
                .summary-container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 30px; }}
                .summary-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                .summary-table th, .summary-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                .summary-table th {{ background-color: #2c3e50; color: white; }}
                .market-section {{ background: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 40px; }}
                .market-section h2 {{ color: #e67e22; margin-top: 0; }}
                .description {{ font-style: italic; color: #7f8c8d; margin-bottom: 20px; border-left: 4px solid #e67e22; padding-left: 15px; }}
                .stats-container {{ margin-bottom: 25px; }}
                .stats-overview {{ background: #ecf0f1; padding: 15px; border-radius: 4px; margin-bottom: 15px; font-weight: bold; }}
                .stats-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9em; }}
                .stats-table th, .stats-table td {{ border: 1px solid #eee; padding: 8px; text-align: right; }}
                .stats-table th {{ background: #f8f9fa; text-align: center; }}
                .stats-table td:nth-child(1), .stats-table td:nth-child(2) {{ text-align: center; font-weight: bold; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 8px; margin-top: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.05); }}
                .footer {{ text-align: center; margin-top: 50px; font-size: 0.8em; color: #bdc3c7; }}
            </style>
        </head>
        <body>
            <h1>Point-in-Time Fractal Wavelet Risk Analysis</h1>
            
            <div class="summary-container">
                <h2>Executive Summary</h2>
                <p>This report presents the results of the Point-in-Time (PIT) Tail Risk Warning System based on the Fractal Markets Hypothesis. 
                Warnings are triggered when short-horizon investment dominance (high-frequency energy) exceeds a 2.0 Z-score threshold.</p>
                {summary_table}
            </div>

            {''.join(html_sections)}

            <div class="footer">
                <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Confidential - Quantitative Risk Strategy</p>
            </div>
        </body>
        </html>
        """
        with open("risk_report.html", "w") as f:
            f.write(html_content)
        
        print("\n" + "="*50)
        print("GLOBAL MARKET TAIL-RISK SUMMARY")
        print("="*50)
        print(summary_df.to_string(index=False))
        print("="*50)
        print("\nHTML Report saved to 'risk_report.html'.")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
