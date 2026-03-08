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

def main():
    markets = {
        "US (S&P 500)": "^GSPC",
        "China (CSI 300)": "000300.SS",
        "Hong Kong (Hang Seng)": "^HSI",
        "Japan (Nikkei 225)": "^N225",
        "Taiwan (TAIEX)": "^TWII",
        "South Korea (KOSPI)": "^KS11",
        "Europe (Euro Stoxx 50)": "^STOXX50E"
    }
    
    manager = FractalWaveletManager(window_size=256, z_window=60)
    
    dashboard_sections = []
    summary_list = []
    
    if not os.path.exists("monitor_plots"):
        os.makedirs("monitor_plots")

    print("Starting Daily Risk Monitor...")
    
    for market_name, ticker in markets.items():
        print(f"Monitoring {market_name}...")
        try:
            # Fetch last 500 days (enough for 256 window + 60 z-score + some buffer)
            data = yf.download(ticker, period="2y") # 2 years is safe
            
            if data.empty:
                continue
                
            price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
            prices = data[price_col].squeeze()
            
            results = manager.run_historical_backtest(prices)
            
            # Get latest status
            latest = results.iloc[-1]
            status = latest['Signal']
            z_score = latest['Z_Score']
            tp = latest['Total_Power']
            
            summary_list.append({
                "Market": market_name,
                "Status": status,
                "Z-Score": f"{z_score:.2f}",
                "Date": results.index[-1].strftime('%Y-%m-%d')
            })
            
            # Focus plot on last 252 days (1 year)
            plot_results = results.tail(252)
            fig = manager.plot_results(plot_results, market_name=f"{market_name} (Last 1Y)")
            img_b64 = get_base64_image(fig)
            plt.close(fig)
            
            status_class = "status-warning" if status == "WARNING" else "status-safe"
            
            dashboard_sections.append(f"""
            <div class="market-card">
                <div class="card-header">
                    <h2>{market_name}</h2>
                    <span class="status-badge {status_class}">{status}</span>
                </div>
                <div class="card-stats">
                    <strong>Current Z-Score:</strong> {z_score:.2f} | 
                    <strong>Last Updated:</strong> {results.index[-1].strftime('%Y-%m-%d')}
                </div>
                <img src="data:image/png;base64,{img_b64}" alt="{market_name} Monitoring">
            </div>
            """)
            
        except Exception as e:
            print(f"Error monitoring {market_name}: {e}")

    # Generate Dashboard HTML
    summary_df = pd.DataFrame(summary_list)
    summary_table = summary_df.to_html(classes='summary-table', index=False)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Daily Tail-Risk Dashboard</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background: #f0f2f5; margin: 0; padding: 20px; color: #1c1e21; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            h1 {{ text-align: center; color: #1a73e8; }}
            .summary-box {{ background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 30px; }}
            .summary-table {{ width: 100%; border-collapse: collapse; }}
            .summary-table th, .summary-table td {{ padding: 12px; border-bottom: 1px solid #eee; text-align: left; }}
            .status-safe {{ color: #28a745; font-weight: bold; }}
            .status-warning {{ color: #dc3545; font-weight: bold; background: #fff3f3; padding: 2px 8px; border-radius: 4px; }}
            .market-card {{ background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 30px; }}
            .card-header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #eee; padding-bottom: 10px; margin-bottom: 15px; }}
            .status-badge {{ padding: 5px 15px; border-radius: 20px; font-size: 0.9em; }}
            img {{ width: 100%; height: auto; border-radius: 8px; margin-top: 15px; }}
            .footer {{ text-align: center; font-size: 0.8em; color: #65676b; margin-top: 40px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Daily Global Tail-Risk Dashboard</h1>
            
            <div class="summary-box">
                <h2>Quick Overview</h2>
                {summary_table}
            </div>
            
            {''.join(dashboard_sections)}
            
            <div class="footer">
                <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open("daily_risk_dashboard.html", "w") as f:
        f.write(html_content)
    print("Daily Dashboard generated: daily_risk_dashboard.html")

if __name__ == "__main__":
    main()
