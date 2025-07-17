# Nifty 50 RSI Trading Strategy

A mathematical approach to trading Nifty 50 stocks using RSI (Relative Strength Index) with Pi-based thresholds for precise entry, exit, and risk management.

## 🎯 Strategy Overview

This strategy implements a systematic approach to trading Nifty 50 stocks based on RSI levels, with built-in risk management and compounding mechanisms.

### Key Features
- **Mathematical Precision**: Uses Pi-based thresholds (3.14%, 6.28%)
- **Risk Management**: Corona quarantine system with SIP recovery
- **Compounding Growth**: 50% reinvestment, 50% self-dividend
- **Tax Efficiency**: Built-in tax calculations for Indian markets
- **Benchmark Comparison**: Performance tracking against Nifty 50

## 📊 Strategy Rules

### Entry Conditions
- **Primary Entry**: Buy when RSI < 35 (14-period RSI)
- **Priority**: Stock with lowest RSI gets first preference
- **Limit**: Only 1 stock purchase per day
- **Position Size**: ₹10,000 per trade

### Averaging Down
- **Triggers**: RSI drops to 30, 25, 20, or 15
- **Price Condition**: Must have 3.14% drop from last purchase price
- **Prerequisites**: No new stocks available with RSI < 35
- **Limit**: Maximum 7 averaging attempts per stock

### Exit Strategy
- **Profit Target**: 6.28% minimum profit
- **Bonus**: Higher profits (7%, 8%, 9%+) are also captured
- **Execution**: After Market Order (AMO) recommended

### Risk Management: "Corona" Rule
- **Trigger**: Stock loss > 20% from average purchase price
- **Action**: Move to SIP mode, stop active trading
- **SIP**: Invest 1/15th of total invested amount monthly for 15 months
- **Goal**: Recovery to 6.28% profit before re-entering active trading

## 💰 Capital Management

### Initial Setup
- **Total Capital**: ₹4,00,000
- **Position Size**: ₹10,000 per trade
- **Tax Rate**: 24.96% (20% STCG + 4% cess)

### Compounding System
- **Self-Dividend**: 50% of after-tax profit withdrawn
- **Reinvestment**: 50% of after-tax profit added to position size
- **Growth**: Position size increases with each profitable trade

## 🚀 Getting Started

### Prerequisites
1. Python 3.8+ installed
2. Required libraries: pandas, numpy, matplotlib, seaborn
3. Historical data files in `../historical_data/` directory

### Installation
```bash
# Navigate to the RSI strategy directory
cd nifty-shoping-rsi

# Install required packages
pip install pandas numpy matplotlib seaborn

# Run quick start guide
python quick_start_rsi.py

# Or run the strategy directly
python rsi_strategy.py
```

### Data Requirements
The strategy requires the following data files in `../historical_data/`:
- Individual Nifty 50 stock files: `STOCKNAME_historical.csv`
- Nifty 50 index data: `nifty50_index_data.csv`

Each stock file should contain columns: `date`, `symbol`, `open`, `high`, `low`, `close`, `volume`

## 📈 Output Files

After running the backtest, the following files are generated in the `results/` directory:

### Data Files
- `rsi_strategy_trade_log.csv` - Detailed log of all trades
- `rsi_strategy_daily_results.csv` - Daily portfolio performance
- `rsi_strategy_performance_report.txt` - Comprehensive performance analysis

### Visualizations
- `rsi_strategy_dashboard.png` - Complete performance dashboard
- `rsi_vs_benchmark_comparison.png` - Strategy vs Nifty 50 comparison

## 📊 Expected Performance Metrics

The strategy tracks the following key metrics:
- **Total Return**: Portfolio appreciation percentage
- **Excess Return**: Outperformance vs Nifty 50 benchmark
- **Win Rate**: Percentage of profitable trades
- **Self-Dividend**: Total amount withdrawn as personal income
- **Annualized Return**: Year-over-year growth rate

## ⚠️ Risk Considerations

### Strategy Risks
- **Market Risk**: Stock prices can fall beyond RSI oversold levels
- **Concentration Risk**: Limited to Nifty 50 stocks only
- **Timing Risk**: Daily execution timing affects results
- **Tax Impact**: Short-term capital gains tax reduces net returns

### Risk Mitigation
- **Corona Rule**: Automatic quarantine for severely declining stocks
- **SIP Recovery**: Systematic investment for recovery positions
- **Diversification**: Spread across multiple Nifty 50 stocks
- **Position Limits**: Maximum investment per stock controlled

## 🔧 Customization

You can modify the following parameters in `rsi_strategy.py`:

```python
# Strategy parameters
self.rsi_period = 14                    # RSI calculation period
self.entry_rsi_threshold = 35           # Entry RSI level
self.averaging_rsi_thresholds = [30, 25, 20, 15]  # Averaging levels
self.profit_target = 0.0628             # 6.28% profit target
self.price_drop_threshold = 0.0314      # 3.14% price drop for averaging
self.corona_threshold = 0.20            # 20% loss threshold
self.max_averaging_attempts = 7         # Maximum averaging attempts

# Capital management
self.total_capital = 400000             # ₹4 lakhs
self.position_size = 10000              # ₹10k per trade
self.tax_rate = 0.2496                  # 24.96% tax rate
self.reinvestment_rate = 0.5            # 50% reinvestment
```

## 📚 Mathematical Foundation

The strategy is based on mathematical principles:

### Pi-Based Thresholds
- **3.14%**: Minimum price drop required for averaging down
- **6.28%**: Profit target (2 × π)
- **Rationale**: Mathematical constants provide consistent, emotion-free decision points

### RSI Levels
- **35**: Primary entry threshold (oversold territory)
- **30, 25, 20, 15**: Progressive averaging levels (increasing oversold conditions)
- **14-period**: Standard RSI calculation period

### Risk Ratios
- **20%**: Corona threshold (significant loss requiring intervention)
- **50/50**: Profit split between reinvestment and withdrawal
- **1/15**: SIP ratio for recovery positions

## 🤝 Contributing

This strategy is designed for educational and research purposes. Feel free to:
- Modify parameters for your risk tolerance
- Add additional technical indicators
- Implement different exit strategies
- Test on different stock universes

## ⚖️ Disclaimer

**Important**: This strategy is based on historical backtesting and is for educational purposes only. 

- Past performance does not guarantee future results
- All investments carry risk of loss
- Consider consulting a qualified financial advisor
- Test thoroughly before implementing with real money
- Understand tax implications in your jurisdiction

## 📞 Support

For questions or issues:
1. Review the strategy documentation
2. Check the generated performance reports
3. Examine the trade logs for understanding trade decisions
4. Modify parameters to suit your risk profile

---

**Author**: RSI Strategy Implementation  
**Date**: July 17, 2025  
**Version**: 1.0  
**License**: Educational Use Only
