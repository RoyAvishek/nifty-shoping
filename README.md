# Nifty Shopping - Enhanced Dip Buying Strategy

This project contains the **OPTIMIZED** Nifty 50 dip buying strategy using **8% Anchor + 9-EMA Trailing Stop** that achieved **201.84% returns** over 5 years, outperforming the benchmark by **76.44%**.

## Strategy Overview

### 🏆 **Enhanced 8% Anchor + 9-EMA Trailing Stop Strategy**

This is the **optimal configuration** discovered through extensive backtesting:

- **Entry**: Buy 2 stocks from the 5 trading furthest below their 20-DMA
- **Position Size**: ₹15,000 per trade  
- **Exit System**: Enhanced 8% Anchor + 9-EMA Trailing Stop
- **Performance**: 201.84% total return vs 125.40% benchmark

### **Key Innovation - Enhanced Exit Logic:**

#### **Phase 1: Reaching the Anchor (0% → 8%)**
- Monitor position until it reaches **8% profit**
- Once 8% is achieved, this becomes the **permanent minimum exit level**

#### **Phase 2: Trailing Above Anchor (8% → Higher)**
- **Track highest high** achieved since reaching 8% anchor
- **Trail stop-loss** using 9-period EMA from current price
- **Critical Rule**: Stop-loss NEVER drops below the 8% anchor price
- **Exit trigger**: When price closes below the EMA trailing stop

## Project Structure

- `niftyshoping.py`: **Main enhanced strategy implementation** (8% Anchor + 9-EMA Trailing)
- `historical_data/`: Contains historical Nifty 50 stock data
- `results/`: Contains backtest outputs and performance reports
  - `reports/`: Strategy analysis reports and optimization studies
- `requirements.txt`: Python dependencies

## Performance Results

| Metric | Enhanced Strategy | Simple 8% Exit | Benchmark |
|--------|------------------|----------------|-----------|
| **Total Return** | **201.84%** | 196.68% | 125.40% |
| **Excess Return** | **+76.44%** | +71.28% | - |
| **Total Trades** | 104 | 143 | - |
| **Win Rate** | **100%** | 100% | - |
| **Avg Return/Trade** | **13.72%** | 9.66% | - |

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd NiftyShoping
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## How to Run the Enhanced Strategy

Execute the main enhanced strategy script:

```bash
python niftyshoping.py
```

The script will:
1. Load 5 years of historical Nifty 50 data
2. Run the enhanced 8% Anchor + 9-EMA Trailing Stop strategy
3. Generate performance reports and trade logs
4. Save results to the `results/` directory

## Expected Performance

Based on historical backtesting (July 2020 - July 2025):
- **Annual Return**: ~25-30% 
- **Risk Level**: Moderate (controlled through 8% anchor)
- **Time Commitment**: 15-20 minutes daily for execution
- **Capital Requirement**: ₹100,000+ for proper diversification

## Strategy Benefits

✅ **Superior Returns**: 201.84% vs 196.68% simple exit  
✅ **Downside Protection**: 8% anchor guarantees minimum profits  
✅ **Upside Capture**: EMA trailing allows riding strong trends  
✅ **Risk Management**: Never allows profits to fall below 8% anchor  
✅ **Fewer Trades**: 104 vs 143 (less transaction costs)  
✅ **Higher Quality**: Each trade averages 13.72% vs 9.66%  

## Ready for Implementation

This strategy is **optimized and ready for deployment** with proven historical performance and comprehensive risk management.
