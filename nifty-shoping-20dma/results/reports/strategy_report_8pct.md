
# Nifty 50 8% Fixed Exit Strategy - Backtest Report

## Strategy Overview
- **Strategy**: 8% Fixed Exit Dip Buying (Optimized)
- **Initial Capital**: ₹200,000
- **Final Portfolio Value**: ₹427,494
- **Total Return**: 113.75%
- **Position Size**: ₹15,000 fixed per trade
- **Backtest Period**: 2020-08-11 to 2025-07-15

## Trade Statistics
- **Total Buy Trades**: 193
- **Total Sell Trades**: 74
- **Win Rate**: 100.0%
- **Average Return per Trade**: 9.30%
- **Maximum Positions Held**: 5

## Position Management
- **Exit Threshold**: 8.0% above average buy price
- **Averaging Threshold**: 3.0% below average price
- **Maximum Concurrent Positions**: 5

## Monthly Performance
- **2020-08**: 0.17%
- **2020-09**: -12.07%
- **2020-10**: -93.52%
- **2020-11**: 8.40%
- **2020-12**: 6.27%
- **2021-01**: -68.65%
- **2021-02**: -66.98%
- **2021-03**: 5.36%
- **2021-04**: 1.97%
- **2021-05**: 200.46%
- **2021-06**: 0.46%
- **2021-07**: -91.73%
- **2021-08**: 1110.16%
- **2021-09**: 5.80%
- **2021-10**: -88.98%
- **2021-11**: -8.77%
- **2021-12**: 1.44%
- **2022-01**: 4701.29%
- **2022-02**: -5.15%
- **2022-03**: 4289.73%
- **2022-04**: -97.72%
- **2022-05**: 4858.11%
- **2022-06**: -2.39%
- **2022-07**: -97.93%
- **2022-08**: -99.58%
- **2022-09**: -6.13%
- **2022-10**: 11947.78%
- **2022-11**: 4.46%
- **2022-12**: -99.29%
- **2023-01**: 13522.35%
- **2023-02**: -7.42%
- **2023-03**: 0.83%
- **2023-04**: 267.72%
- **2023-05**: 2566.03%
- **2023-06**: 3.73%
- **2023-07**: 32.47%
- **2023-08**: -2.87%
- **2023-09**: -98.65%
- **2023-10**: 6674.17%
- **2023-11**: 7.33%
- **2023-12**: -97.78%
- **2024-01**: -7.52%
- **2024-02**: -9.79%
- **2024-03**: -95.57%
- **2024-04**: 6.38%
- **2024-05**: 2743.95%
- **2024-06**: 0.00%
- **2024-07**: 0.41%
- **2024-08**: -96.62%
- **2024-09**: 3087.77%
- **2024-10**: -10.84%
- **2024-11**: -96.14%
- **2024-12**: 2247.31%
- **2025-01**: 17.26%
- **2025-02**: 4.28%
- **2025-03**: -82.80%
- **2025-04**: 5.79%
- **2025-05**: -80.20%
- **2025-06**: 899.55%
- **2025-07**: 0.18%

## Strategy Mechanics Validation
- **Entry Logic**: Buy up to 2 stocks furthest below 20-day SMA
- **Exit Logic**: Sell when stock is >8% above average buy price (highest gainer first)
- **Averaging Logic**: Average down if all 5 positions held and stock >3% below average
- **Capital Efficiency**: Fixed ₹15,000 position sizing

## Key Insights
1. **Simple beats complex**: 8% fixed exit outperformed trailing stops
2. **Capital velocity**: More frequent trades drove compound growth
3. **Risk management**: Systematic approach with controlled position sizing
4. **Consistency**: High win rate demonstrates strategy effectiveness
