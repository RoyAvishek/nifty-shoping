#!/usr/bin/env python3
"""
Enhanced RSI Strategy V2 - Standalone Backtesting Script
========================================================

This script consolidates the Enhanced V2 strategy into a single file for easy sharing and verification.
Team members can run this script with historical data to see complete results.

Requirements:
- Python 3.7+
- pandas, numpy, matplotlib (pip install pandas numpy matplotlib)
- Historical data in ../historical_data/ directory

Results Generated:
- Complete performance summary
- Month-by-month trade details  
- Portfolio evolution tracking
- Risk management analysis
- Comparison with benchmarks

Author: Enhanced RSI Strategy Team
date: July 2025
Version: V2.0 Standalone
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EnhancedRSIStrategyV2:
    """Enhanced RSI Strategy V2 - Complete Implementation"""
    
    def __init__(self):
        self.initial_capital = 400000  # ₹4 lakh starting capital
        self.base_position_size = 10000  # ₹10K base position
        self.rsi_period = 14
        self.entry_rsi = 35
        self.exit_profit_pct = 6.28  # Pi-based profit target
        self.corona_threshold_pct = 22.5  # Enhanced threshold
        self.tax_rate = 0.2496  # 20% STCG + 4% cess
        self.max_averaging_attempts = 10
        
        # Portfolio tracking
        self.portfolio = {}
        self.trade_log = []
        self.daily_portfolio_value = []
        self.cash = self.initial_capital
        self.month_wise_trades = {}
        
        # Averaging conditions for Enhanced V2
        self.averaging_conditions = [
            {'rsi_threshold': 30, 'price_drop_pct': 3.0},
            {'rsi_threshold': 25, 'price_drop_pct': 6.0},
            {'rsi_threshold': 20, 'price_drop_pct': 10.0},
            {'rsi_threshold': 15, 'price_drop_pct': 15.0},
            {'rsi_threshold': 10, 'price_drop_pct': 20.0},
            {'rsi_threshold': 5, 'price_drop_pct': 25.0}
        ]
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return [50] * len(prices)
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50).tolist()
    
    def load_data(self):
        """Load all Nifty 50 historical data"""
        data_dir = "../historical_data"
        if not os.path.exists(data_dir):
            data_dir = "historical_data"  # Fallback path
        
        stock_data = {}
        stock_files = glob.glob(os.path.join(data_dir, "*_historical.csv"))
        
        print(f"Loading data from: {data_dir}")
        print(f"Found {len(stock_files)} stock files")
        
        for file_path in stock_files:
            try:
                symbol = os.path.basename(file_path).replace('_historical.csv', '')
                df = pd.read_csv(file_path)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                
                # Calculate RSI
                df['RSI'] = self.calculate_rsi(df['close'])
                
                stock_data[symbol] = df
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        print(f"Successfully loaded {len(stock_data)} stocks")
        return stock_data
    
    def get_current_position_size(self):
        """Calculate current position size based on capital growth"""
        total_value = self.cash + sum([pos['current_value'] for pos in self.portfolio.values()])
        growth_factor = total_value / self.initial_capital
        return max(self.base_position_size * growth_factor, self.base_position_size)
    
    def should_buy_stock(self, symbol, current_price, rsi, date):
        """Check if we should buy a stock (Enhanced V2 conditions)"""
        # Entry condition: RSI < 35
        if rsi < self.entry_rsi:
            if symbol not in self.portfolio:
                return True, "NEW_ENTRY"
            
            # Averaging conditions
            position = self.portfolio[symbol]
            if position['averaging_attempts'] < self.max_averaging_attempts:
                avg_price = position['total_invested'] / position['quantity']
                price_drop_pct = ((avg_price - current_price) / avg_price) * 100
                
                for i, condition in enumerate(self.averaging_conditions):
                    if (rsi < condition['rsi_threshold'] and 
                        price_drop_pct >= condition['price_drop_pct'] and
                        position['averaging_attempts'] == i):
                        return True, f"AVERAGING_L{i+1}"
        
        return False, ""
    
    def should_sell_stock(self, symbol, current_price, date):
        """Check if we should sell a stock"""
        if symbol not in self.portfolio:
            return False, ""
        
        position = self.portfolio[symbol]
        avg_price = position['total_invested'] / position['quantity']
        profit_pct = ((current_price - avg_price) / avg_price) * 100
        
        # Profit target
        if profit_pct >= self.exit_profit_pct:
            return True, f"PROFIT_{profit_pct:.2f}%"
        
        # Corona rule (Enhanced V2: 22.5% threshold)
        if profit_pct <= -self.corona_threshold_pct:
            return True, f"CORONA_{profit_pct:.2f}%"
        
        return False, ""
    
    def execute_buy(self, symbol, price, date, reason):
        """Execute buy order"""
        position_size = self.get_current_position_size()
        
        if self.cash < position_size:
            position_size = self.cash * 0.95  # Use 95% of available cash
        
        if position_size < 1000:  # Minimum position size
            return False
        
        quantity = int(position_size / price)
        if quantity == 0:
            return False
        
        amount = quantity * price
        
        if symbol not in self.portfolio:
            self.portfolio[symbol] = {
                'quantity': 0,
                'total_invested': 0,
                'averaging_attempts': 0,
                'first_buy_date': date,
                'current_value': 0
            }
        
        # Update position
        self.portfolio[symbol]['quantity'] += quantity
        self.portfolio[symbol]['total_invested'] += amount
        if reason.startswith('AVERAGING'):
            self.portfolio[symbol]['averaging_attempts'] += 1
        
        self.cash -= amount
        
        # Log trade
        month_key = date.strftime('%Y-%m')
        if month_key not in self.month_wise_trades:
            self.month_wise_trades[month_key] = []
        
        trade_record = {
            'date': date,
            'symbol': symbol,
            'action': 'BUY',
            'price': price,
            'quantity': quantity,
            'amount': amount,
            'reason': reason,
            'cash_after': self.cash
        }
        
        self.trade_log.append(trade_record)
        self.month_wise_trades[month_key].append(trade_record)
        
        return True
    
    def execute_sell(self, symbol, price, date, reason):
        """Execute sell order"""
        if symbol not in self.portfolio:
            return False
        
        position = self.portfolio[symbol]
        quantity = position['quantity']
        avg_price = position['total_invested'] / quantity
        
        gross_proceeds = quantity * price
        profit = gross_proceeds - position['total_invested']
        
        # Calculate tax on profit (only if profitable)
        tax = max(0, profit * self.tax_rate)
        net_proceeds = gross_proceeds - tax
        
        self.cash += net_proceeds
        
        # Log trade
        month_key = date.strftime('%Y-%m')
        if month_key not in self.month_wise_trades:
            self.month_wise_trades[month_key] = []
        
        trade_record = {
            'date': date,
            'symbol': symbol,
            'action': 'SELL',
            'price': price,
            'quantity': quantity,
            'amount': gross_proceeds,
            'profit': profit,
            'profit_pct': (profit / position['total_invested']) * 100,
            'tax_paid': tax,
            'net_proceeds': net_proceeds,
            'reason': reason,
            'avg_price': avg_price,
            'cash_after': self.cash,
            'hold_days': (date - position['first_buy_date']).days
        }
        
        self.trade_log.append(trade_record)
        self.month_wise_trades[month_key].append(trade_record)
        
        # Remove from portfolio
        del self.portfolio[symbol]
        
        return True
    
    def run_backtest(self, stock_data):
        """Run the complete backtest"""
        print("Starting Enhanced RSI Strategy V2 Backtest...")
        print("=" * 60)
        
        # Get all dates
        all_dates = set()
        for symbol, df in stock_data.items():
            all_dates.update(df['date'].tolist())
        
        all_dates = sorted(list(all_dates))
        
        for i, date in enumerate(all_dates):
            # Update portfolio values
            for symbol in list(self.portfolio.keys()):
                if symbol in stock_data:
                    stock_df = stock_data[symbol]
                    day_data = stock_df[stock_df['date'] == date]
                    if not day_data.empty:
                        current_price = day_data.iloc[0]['close']
                        self.portfolio[symbol]['current_value'] = (
                            self.portfolio[symbol]['quantity'] * current_price
                        )
            
            # Check for sell signals first
            for symbol in list(self.portfolio.keys()):
                if symbol in stock_data:
                    stock_df = stock_data[symbol]
                    day_data = stock_df[stock_df['date'] == date]
                    if not day_data.empty:
                        current_price = day_data.iloc[0]['close']
                        should_sell, sell_reason = self.should_sell_stock(symbol, current_price, date)
                        if should_sell:
                            self.execute_sell(symbol, current_price, date, sell_reason)
            
            # Check for buy signals
            available_stocks = []
            for symbol, stock_df in stock_data.items():
                day_data = stock_df[stock_df['date'] == date]
                if not day_data.empty:
                    current_price = day_data.iloc[0]['close']
                    rsi = day_data.iloc[0]['RSI']
                    should_buy, buy_reason = self.should_buy_stock(symbol, current_price, rsi, date)
                    if should_buy:
                        available_stocks.append((symbol, current_price, rsi, buy_reason))
            
            # Sort by RSI (lowest first) and buy one per day
            if available_stocks:
                available_stocks.sort(key=lambda x: x[2])  # Sort by RSI
                symbol, price, rsi, reason = available_stocks[0]
                self.execute_buy(symbol, price, date, reason)
            
            # Record daily portfolio value
            total_value = self.cash + sum([pos['current_value'] for pos in self.portfolio.values()])
            self.daily_portfolio_value.append({
                'date': date,
                'total_value': total_value,
                'cash': self.cash,
                'invested': total_value - self.cash
            })
            
            # Progress update
            if i % 50 == 0:
                print(f"Processed {i}/{len(all_dates)} days... Portfolio: ₹{total_value:,.0f}")
        
        print("Backtest completed!")
        return self.generate_results()
    
    def generate_results(self):
        """Generate comprehensive results"""
        final_value = self.daily_portfolio_value[-1]['total_value']
        total_return_pct = ((final_value - self.initial_capital) / self.initial_capital) * 100
        
        # Count profitable vs corona trades
        profitable_trades = [t for t in self.trade_log if t['action'] == 'SELL' and not t['reason'].startswith('CORONA')]
        corona_trades = [t for t in self.trade_log if t['action'] == 'SELL' and t['reason'].startswith('CORONA')]
        
        results = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return_pct': total_return_pct,
            'total_trades': len([t for t in self.trade_log if t['action'] == 'SELL']),
            'profitable_trades': len(profitable_trades),
            'corona_trades': len(corona_trades),
            'win_rate': len(profitable_trades) / max(1, len(profitable_trades) + len(corona_trades)) * 100,
            'active_positions': len(self.portfolio),
            'trade_log': self.trade_log,
            'monthly_trades': self.month_wise_trades,
            'daily_values': self.daily_portfolio_value
        }
        
        return results
    
    def print_detailed_results(self, results):
        """Print comprehensive results in text format"""
        print("\n" + "=" * 80)
        print("🚀 ENHANCED RSI STRATEGY V2 - BACKTEST RESULTS")
        print("=" * 80)
        
        print(f"""
📊 PERFORMANCE SUMMARY:
━━━━━━━━━━━━━━━━━━━━━━━━
Initial Capital      : ₹{results['initial_capital']:,}
Final Portfolio Value : ₹{results['final_value']:,.0f}
Total Return          : {results['total_return_pct']:,.1f}%
Active Positions      : {results['active_positions']}

🎯 TRADING STATISTICS:
━━━━━━━━━━━━━━━━━━━━━━━━
Total Trades          : {results['total_trades']}
Profitable Trades     : {results['profitable_trades']}
Corona Trades         : {results['corona_trades']}
Win Rate              : {results['win_rate']:.1f}%
""")
        
        print("\n📅 MONTH-BY-MONTH TRADE ANALYSIS:")
        print("━" * 50)
        
        monthly_summary = {}
        for month, trades in results['monthly_trades'].items():
            buy_trades = [t for t in trades if t['action'] == 'BUY']
            sell_trades = [t for t in trades if t['action'] == 'SELL']
            
            total_invested = sum([t['amount'] for t in buy_trades])
            total_profit = sum([t.get('profit', 0) for t in sell_trades])
            
            monthly_summary[month] = {
                'buys': len(buy_trades),
                'sells': len(sell_trades),
                'invested': total_invested,
                'profit': total_profit
            }
        
        for month in sorted(monthly_summary.keys()):
            data = monthly_summary[month]
            print(f"{month}: {data['buys']} buys, {data['sells']} sells, "
                  f"₹{data['invested']:,.0f} invested, ₹{data['profit']:,.0f} profit")
        
        print("\n📈 RECENT TRADES (Last 10):")
        print("━" * 50)
        recent_trades = [t for t in results['trade_log'] if t['action'] == 'SELL'][-10:]
        for trade in recent_trades:
            print(f"{trade['date'].strftime('%Y-%m-%d')} | {trade['symbol']} | "
                  f"{trade['action']} | ₹{trade['price']:.2f} | "
                  f"Profit: {trade.get('profit_pct', 0):.2f}% | {trade['reason']}")
        
        print(f"\n💰 CURRENT PORTFOLIO ({len(self.portfolio)} positions):")
        print("━" * 50)
        if self.portfolio:
            for symbol, pos in self.portfolio.items():
                avg_price = pos['total_invested'] / pos['quantity']
                print(f"{symbol}: {pos['quantity']} shares @ ₹{avg_price:.2f} avg "
                      f"(Invested: ₹{pos['total_invested']:,.0f})")
        else:
            print("No active positions")
        
        print(f"\n💵 Available Cash: ₹{self.cash:,.0f}")
        print("=" * 80)

def main():
    """Main execution function"""
    print("Enhanced RSI Strategy V2 - Standalone Backtest")
    print("=" * 50)
    print("Loading historical data...")
    
    # Initialize strategy
    strategy = EnhancedRSIStrategyV2()
    
    # Load data
    stock_data = strategy.load_data()
    
    if not stock_data:
        print("❌ No data found! Please ensure historical data is available.")
        print("Expected path: ../historical_data/ or ./historical_data/")
        return
    
    # Run backtest
    results = strategy.run_backtest(stock_data)
    
    # Print detailed results
    strategy.print_detailed_results(results)
    
    print("\n🎯 SCRIPT EXECUTION COMPLETED!")
    print("Results above show the Enhanced V2 strategy performance.")
    print("Share this script with your team to verify results.")

if __name__ == "__main__":
    main()