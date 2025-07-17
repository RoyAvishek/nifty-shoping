#!/usr/bin/env python3
"""
Nifty 50 RSI Strategy - Quick Start Guide
=======================================

This guide helps you quickly understand and run the RSI-based trading strategy.

STRATEGY OVERVIEW:
-----------------
This is a mathematical approach to trading Nifty 50 stocks using RSI (Relative Strength Index)
with Pi-based thresholds for entry, exit, and risk management.

KEY RULES:
---------
1. ENTRY: Buy when RSI < 35 (priority to lowest RSI)
2. EXIT: Sell at 6.28% profit (2 × Pi)
3. AVERAGING: Add more when RSI drops to 30, 25, 20, 15
4. PRICE DROP: Must have 3.14% (Pi) drop from last purchase for averaging
5. CORONA RULE: If loss > 20%, start SIP (1/15th monthly for 15 months)

CAPITAL MANAGEMENT:
------------------
- Total Capital: ₹4,00,000
- Position Size: ₹10,000 per trade
- Tax Rate: 24.96% (20% STCG + 4% cess)
- Compounding: 50% reinvestment, 50% self-dividend

EXPECTED PERFORMANCE:
--------------------
Based on historical backtesting on 5 years of Nifty 50 data.
Results will vary based on market conditions.

HOW TO USE:
----------
1. Ensure you have the historical_data folder with Nifty 50 stock files
2. Run: python rsi_strategy.py
3. Check results folder for detailed analysis
4. Review trade log and performance charts

FILES GENERATED:
---------------
- results/rsi_strategy_trade_log.csv
- results/rsi_strategy_daily_results.csv  
- results/rsi_strategy_performance_report.txt
- results/visualizations/rsi_strategy_dashboard.png
- results/visualizations/rsi_vs_benchmark_comparison.png

RISK DISCLAIMER:
---------------
This is a backtesting strategy based on historical data. Past performance
does not guarantee future results. Use at your own risk and consider
consulting a financial advisor.

Author: RSI Strategy Implementation
Date: July 17, 2025
"""

import sys
import os

def check_data_availability():
    """Check if required data files are available."""
    data_dir = "../historical_data"
    
    required_files = [
        "nifty50_index_data.csv",
        "RELIANCE_historical.csv",  # Sample stock file
        "TCS_historical.csv"        # Sample stock file
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(data_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required data files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    return True

def print_strategy_summary():
    """Print a summary of the strategy."""
    print("""
🎯 RSI STRATEGY SUMMARY
======================

📊 ENTRY CONDITIONS:
   • RSI < 35 (14-period RSI)
   • Priority to stock with lowest RSI
   • Maximum 1 stock purchase per day
   • Position size: ₹10,000

🔄 AVERAGING CONDITIONS:  
   • RSI drops to 30, 25, 20, or 15
   • Price must drop 3.14% from last purchase
   • Maximum 7 averaging attempts per stock
   • Only if no new RSI < 35 candidates

💰 EXIT CONDITIONS:
   • Sell at 6.28% profit target
   • Higher profits (7%, 8%, 9%+) also captured
   • After Market Order (AMO) recommended

🦠 RISK MANAGEMENT:
   • "Corona" rule: >20% loss triggers SIP mode
   • SIP: 1/15th of invested amount monthly for 15 months
   • Quarantine stock from active trading

💎 CAPITAL MANAGEMENT:
   • 50% of profit as self-dividend
   • 50% reinvestment for compounding
   • Tax: 24.96% on profits
   • Position size grows with compounding

🎪 EXPECTED FEATURES:
   • Mathematical precision with Pi-based thresholds
   • Risk-controlled averaging down
   • Systematic profit withdrawal
   • Tax-efficient trading
   • Benchmark outperformance potential
""")

def main():
    """Main function for quick start guide."""
    print("=" * 60)
    print("🚀 NIFTY 50 RSI STRATEGY - QUICK START")
    print("=" * 60)
    
    # Check data availability
    print("\n📊 Checking data availability...")
    if not check_data_availability():
        print("\n💡 Please ensure you have the required data files in ../historical_data/")
        print("   You can download Nifty 50 historical data from NSE or financial data providers.")
        return
    
    print("✅ Required data files found!")
    
    # Print strategy summary
    print_strategy_summary()
    
    print("\n🚀 READY TO RUN STRATEGY!")
    print("=" * 60)
    print("Run the following command to start backtesting:")
    print("   python rsi_strategy.py")
    print("=" * 60)
    
    # Ask user if they want to run now
    try:
        response = input("\nDo you want to run the backtest now? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            print("\n🔄 Starting RSI strategy backtest...")
            
            # Import and run the main strategy
            try:
                from rsi_strategy import main as run_strategy
                run_strategy()
            except ImportError as e:
                print(f"❌ Error importing strategy: {e}")
                print("Please ensure rsi_strategy.py is in the same directory.")
        else:
            print("\n👍 You can run the strategy later using: python rsi_strategy.py")
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Run 'python rsi_strategy.py' when ready.")

if __name__ == "__main__":
    main()
