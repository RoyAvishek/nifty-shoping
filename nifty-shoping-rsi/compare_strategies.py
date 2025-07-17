#!/usr/bin/env python3
"""
RSI Strategy Comparison Tool
===========================

This script runs both the simple profit target RSI strategy and the 
trailing stop enhanced version to compare their performance.

Features:
- Side-by-side performance comparison
- Detailed metrics analysis
- Visual comparison charts
- Strategy recommendation

Author: RSI Strategy Comparison
Date: July 17, 2025
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rsi_strategy import NiftyRSIStrategy

def main():
    """Main function to run strategy comparison."""
    print("=" * 70)
    print("🔄 NIFTY 50 RSI STRATEGY COMPARISON")
    print("=" * 70)
    print("Comparing Simple Profit Target vs Trailing Stop Enhanced")
    print("=" * 70)
    
    # Initialize RSI strategy
    strategy = NiftyRSIStrategy()
    
    try:
        # Load data
        print("\n📊 Loading Nifty 50 historical data...")
        strategy.load_data()
        print("✅ Data loaded successfully!")
        print(f"📈 Loaded {len(strategy.stock_data)} stocks")
        
        # Run comparison
        print("\n🔄 Running strategy comparison...")
        print("This will run both strategies and may take several minutes...")
        comparison_results = strategy.compare_strategies()
        print("✅ Comparison completed successfully!")
        
        # Generate and display comparison report
        print("\n📋 Generating comparison report...")
        comparison_report = strategy.create_strategy_comparison_report(comparison_results)
        print(comparison_report)
        
        # Save comparison results
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        print("\n💾 Saving comparison results...")
        
        # Save simple strategy results
        simple_trade_log = os.path.join(results_dir, "simple_strategy_trade_log.csv")
        comparison_results['simple_strategy']['trade_log'].to_csv(simple_trade_log, index=False)
        print(f"📝 Simple strategy trade log saved to: {simple_trade_log}")
        
        # Save trailing strategy results
        trailing_trade_log = os.path.join(results_dir, "trailing_strategy_trade_log.csv")
        comparison_results['trailing_strategy']['trade_log'].to_csv(trailing_trade_log, index=False)
        print(f"📝 Trailing strategy trade log saved to: {trailing_trade_log}")
        
        # Save comparison report
        comparison_report_file = os.path.join(results_dir, "strategy_comparison_report.txt")
        with open(comparison_report_file, 'w') as f:
            f.write(comparison_report)
        print(f"📄 Comparison report saved to: {comparison_report_file}")
        
        # Save detailed results
        simple_daily = os.path.join(results_dir, "simple_strategy_daily_results.csv")
        comparison_results['simple_strategy']['daily_results'].to_csv(simple_daily, index=False)
        
        trailing_daily = os.path.join(results_dir, "trailing_strategy_daily_results.csv")
        comparison_results['trailing_strategy']['daily_results'].to_csv(trailing_daily, index=False)
        
        print(f"📊 Daily results saved for both strategies")
        
        print("\n" + "=" * 70)
        print("🎯 STRATEGY COMPARISON COMPLETE!")
        print("=" * 70)
        
        # Quick summary
        simple_return = comparison_results['simple_strategy']['total_return_pct']
        trailing_return = comparison_results['trailing_strategy']['total_return_pct']
        
        print(f"📈 Simple Strategy Return: {simple_return:.2%}")
        print(f"🏃 Trailing Strategy Return: {trailing_return:.2%}")
        print(f"📊 Performance Difference: {trailing_return - simple_return:+.2%}")
        
        if trailing_return > simple_return:
            print("🏆 Trailing Stop strategy wins!")
        elif simple_return > trailing_return:
            print("🏆 Simple strategy wins!")
        else:
            print("⚖️ Both strategies perform equally!")
        
        print("=" * 70)
        print("✅ COMPARISON ANALYSIS READY FOR REVIEW!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error running strategy comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
