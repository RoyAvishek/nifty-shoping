# RSI Strategy Backtest Results Summary

## 🚀 Strategy Performance

**Backtest Period**: August 6, 2020 to July 15, 2025 (1,228 trading days)

### 📊 Key Performance Metrics

| Metric | Value | 
|--------|-------|
| **Initial Capital** | ₹4,00,000 |
| **Final Portfolio Value** | ₹20,94,994 |
| **Total Self-Dividend** | ₹1,25,229 |
| **Effective Final Value** | ₹22,20,223 |
| **Portfolio Return** | **423.75%** |
| **Effective Return (incl. dividend)** | **455.06%** |
| **Annualized Return** | **40.47%** |
| **Benchmark Return (Nifty 50)** | 124.96% |
| **Excess Return vs Benchmark** | **298.79%** |

### 🎯 Trading Statistics

| Metric | Value |
|--------|-------|
| **Total Trades** | 382 |
| **Profitable Trades** | 342 |
| **Win Rate** | **89.53%** |
| **RSI Entry Threshold** | < 35 |
| **Profit Target** | 6.28% |

### 💼 Risk Management

| Metric | Value |
|--------|-------|
| **Corona Stocks (Quarantined)** | 40 |
| **Active SIP Positions** | 40 |
| **Final Active Positions** | 19 |
| **Tax Rate Applied** | 24.96% |

## 🏆 Strategy Highlights

### ✅ Strengths
- **Exceptional Returns**: 423.75% portfolio return vs 124.96% benchmark
- **High Win Rate**: 89.53% profitable trades
- **Risk Management**: Corona rule prevented major losses
- **Systematic Approach**: Mathematical Pi-based thresholds
- **Tax Efficiency**: Built-in tax calculations
- **Compounding Effect**: 50% reinvestment strategy

### ⚠️ Risk Factors
- **High Corona Rate**: 40 stocks moved to SIP (quarantine)
- **Market Dependency**: Performance tied to overall market conditions
- **Complexity**: Requires systematic daily monitoring
- **Tax Impact**: 24.96% tax rate reduces net gains

## 📈 Strategy Rules Applied

### Entry Conditions
- **RSI < 35**: Primary entry signal using 14-period RSI
- **Priority System**: Lowest RSI stock gets preference
- **Daily Limit**: Maximum 1 stock purchase per day
- **Position Size**: ₹10,000 per trade

### Averaging Strategy
- **RSI Thresholds**: 30, 25, 20, 15 for additional purchases
- **Price Drop**: 3.14% minimum drop from last purchase
- **Limit**: Maximum 7 averaging attempts per stock

### Exit Strategy
- **Profit Target**: 6.28% minimum profit booking
- **Higher Profits**: 7%, 8%, 9%+ also captured when available

### Corona Rule (Risk Management)
- **Trigger**: >20% loss from average purchase price
- **Action**: Move to SIP mode (1/15th monthly investment for 15 months)
- **Purpose**: Systematic recovery for severely declining stocks

## 💰 Capital Management

### Compounding System
- **Profit Split**: 50% self-dividend, 50% reinvestment
- **Tax Treatment**: 20% STCG + 4% cess = 24.96%
- **Position Size Growth**: Increased with each profitable cycle

### Self-Dividend Withdrawals
- **Total Withdrawn**: ₹1,25,229
- **Purpose**: Regular income while growing capital
- **Benefit**: Liquidity without touching principal

## 📊 Files Generated

### Data Files
- `results/rsi_strategy_trade_log.csv` - Complete trade history
- `results/rsi_strategy_daily_results.csv` - Daily portfolio values
- `results/rsi_strategy_performance_report.txt` - Detailed analysis

### Visualizations
- `results/visualizations/rsi_strategy_dashboard.png` - Comprehensive dashboard
- `results/visualizations/rsi_vs_benchmark_comparison.png` - Performance comparison

## 🔄 Next Steps

### For Implementation
1. **Paper Trading**: Test with virtual money first
2. **Capital Scaling**: Adjust position sizes based on available capital
3. **Risk Assessment**: Consider personal risk tolerance
4. **Monitoring Setup**: Establish daily RSI monitoring system

### For Optimization
1. **Parameter Tuning**: Test different RSI thresholds
2. **Corona Threshold**: Adjust the 20% loss trigger
3. **Profit Targets**: Experiment with different exit levels
4. **Tax Planning**: Optimize for tax efficiency

## ⚖️ Disclaimer

This backtest is based on historical data and past performance does not guarantee future results. The strategy:

- Requires active daily monitoring
- Involves significant risk of loss
- Should be thoroughly tested before real implementation
- May not perform similarly in different market conditions

**Recommendation**: Start with a small portion of capital and scale up gradually after gaining confidence with the strategy.

---

**Generated**: July 17, 2025  
**Strategy**: RSI-Based Mathematical Trading  
**Backtest Engine**: Custom Python Implementation
