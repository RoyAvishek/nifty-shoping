#!/usr/bin/env python3
"""
Nifty 50 RSI-Based Trading Strategy
=================================

RSI-based trading strategy for Nifty 50 stocks with the following rules:

ENTRY CONDITIONS:
- Buy when RSI < 35 (14-period RSI)
- Priority to stock with lowest RSI if multiple candidates
- Only one stock purchase per day
- Maximum position size: ₹10,000 per trade

AVERAGING CONDITIONS:
- Additional purchases when RSI drops to 30, 25, 20, 15
- Must have price drop of at least 3.14% from last purchase price
- Maximum 7 averaging attempts per stock
- No new stocks available for RSI < 35

EXIT CONDITIONS:
- Sell at 6.28% profit target (minimum)
- Higher profits (7%, 8%, 9%+) are also captured

RISK MANAGEMENT:
- "Corona" Rule: Stop averaging if stock drops >20% from average price
- OR if stock is removed from Nifty 50
- Implement SIP: 1/15th of invested amount monthly for 15 months

CAPITAL MANAGEMENT:
- Total Capital: ₹4,00,000
- Position Size: ₹10,000 per trade
- Tax: 20% STCG + 4% cess on profits
- Compounding: 50% reinvestment, 50% self-dividend

Author: RSI Strategy Implementation
Date: July 17, 2025
"""

import os
import pandas as pd
import numpy as np
import logging
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set matplotlib style
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('default')

class NiftyRSIStrategy:
    """Nifty 50 RSI-based trading strategy implementation."""
    
    def __init__(self, data_dir: str = "../historical_data"):
        """
        Initialize the RSI strategy.
        
        Args:
            data_dir: Directory containing historical data files
        """
        self.data_dir = data_dir
        self.stock_data = {}
        self.nifty_index = None
        self.trade_log = []
        self.corona_stocks = set()  # Stocks under "corona" quarantine
        self.sip_positions = {}  # SIP tracking for corona stocks
        
        # Strategy parameters
        self.rsi_period = 14
        self.entry_rsi_threshold = 35
        self.averaging_rsi_thresholds = [30, 25, 20, 15]
        self.profit_target = 0.0628  # 6.28%
        self.price_drop_threshold = 0.0314  # 3.14%
        self.corona_threshold = 0.20  # 20% loss threshold
        self.max_averaging_attempts = 7
        
        # Capital management
        self.total_capital = 400000  # ₹4 lakhs
        self.position_size = 10000   # ₹10k per trade
        self.tax_rate = 0.2496  # 20% STCG + 4% cess = 24.96%
        self.reinvestment_rate = 0.5  # 50% reinvestment
        
    def load_data(self) -> None:
        """Load all Nifty 50 stock data and index data."""
        logger.info("Loading Nifty 50 stock data...")
        
        # Load individual stock files
        stock_files = glob.glob(os.path.join(self.data_dir, "*_historical.csv"))
        
        for file_path in stock_files:
            stock_name = os.path.basename(file_path).replace("_historical.csv", "")
            try:
                df = pd.read_csv(file_path)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.sort_index(inplace=True)
                
                # Calculate RSI
                df['rsi'] = self.calculate_rsi(df['close'])
                
                self.stock_data[stock_name] = df
                
            except Exception as e:
                logger.warning(f"Could not load {stock_name}: {e}")
        
        logger.info(f"Loaded {len(self.stock_data)} stocks")
        
        # Load Nifty 50 index data
        try:
            nifty_file = os.path.join(self.data_dir, "nifty50_index_data.csv")
            self.nifty_index = pd.read_csv(nifty_file)
            self.nifty_index['date'] = pd.to_datetime(self.nifty_index['date'])
            self.nifty_index.set_index('date', inplace=True)
            self.nifty_index.sort_index(inplace=True)
            logger.info(f"Loaded Nifty 50 index data: {len(self.nifty_index)} days")
            
        except Exception as e:
            logger.error(f"Could not load Nifty 50 index data: {e}")
            raise
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_trading_dates(self) -> pd.DatetimeIndex:
        """Get common trading dates across all stocks."""
        if not self.stock_data:
            return pd.DatetimeIndex([])
        
        # Find intersection of all stock dates
        common_dates = None
        for stock_name, df in self.stock_data.items():
            if common_dates is None:
                common_dates = df.index
            else:
                common_dates = common_dates.intersection(df.index)
        
        # Ensure we have enough data for RSI calculation
        return common_dates[self.rsi_period:]
    
    def find_rsi_candidates(self, date: pd.Timestamp) -> List[Tuple[str, float]]:
        """Find stocks with RSI < 35, sorted by lowest RSI."""
        candidates = []
        
        for stock_name, df in self.stock_data.items():
            if date in df.index and stock_name not in self.corona_stocks:
                try:
                    rsi_value = df.loc[date, 'rsi']
                    
                    # Handle potential Series return
                    if isinstance(rsi_value, pd.Series):
                        rsi_value = rsi_value.iloc[0] if len(rsi_value) > 0 else np.nan
                    
                    if pd.notna(rsi_value) and float(rsi_value) < self.entry_rsi_threshold:
                        candidates.append((stock_name, float(rsi_value)))
                except Exception as e:
                    logger.warning(f"Error processing RSI for {stock_name} on {date}: {e}")
                    continue
        
        # Sort by lowest RSI (highest priority)
        candidates.sort(key=lambda x: x[1])
        return candidates
    
    def can_average_down(self, stock: str, date: pd.Timestamp, holdings: Dict) -> bool:
        """Check if we can average down on a stock."""
        if stock not in self.stock_data or date not in self.stock_data[stock].index:
            return False
        
        try:
            stock_df = self.stock_data[stock]
            current_price = stock_df.loc[date, 'close']
            current_rsi = stock_df.loc[date, 'rsi']
            
            # Handle potential Series returns
            if isinstance(current_price, pd.Series):
                current_price = current_price.iloc[0] if len(current_price) > 0 else np.nan
            if isinstance(current_rsi, pd.Series):
                current_rsi = current_rsi.iloc[0] if len(current_rsi) > 0 else np.nan
            
            if pd.notna(current_price) and pd.notna(current_rsi):
                current_price = float(current_price)
                current_rsi = float(current_rsi)
            else:
                return False
            
            # Check if RSI qualifies for averaging
            rsi_qualifies = False
            for threshold in self.averaging_rsi_thresholds:
                if current_rsi < threshold:
                    rsi_qualifies = True
                    break
            
            if not rsi_qualifies:
                return False
            
            # Check price drop condition (3.14% from last purchase)
            last_purchase_price = holdings['last_purchase_price']
            price_drop = (last_purchase_price - current_price) / last_purchase_price
            
            if price_drop < self.price_drop_threshold:
                return False
            
            # Check averaging attempts limit
            if holdings['averaging_attempts'] >= self.max_averaging_attempts:
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error in can_average_down for {stock} on {date}: {e}")
            return False
    
    def check_corona_condition(self, stock: str, holdings: Dict, current_price: float) -> bool:
        """Check if stock should be marked as 'corona' (quarantined)."""
        avg_price = holdings['avg_price']
        loss_pct = (avg_price - current_price) / avg_price
        
        return loss_pct > self.corona_threshold
    
    def execute_sip(self, stock: str, date: pd.Timestamp, cash: float) -> Tuple[float, Dict]:
        """Execute SIP for corona stocks."""
        if stock not in self.sip_positions:
            return cash, {}
        
        sip_data = self.sip_positions[stock]
        
        # Check if it's time for SIP (monthly)
        last_sip_date = sip_data.get('last_sip_date')
        if last_sip_date and (date - last_sip_date).days < 30:
            return cash, {}
        
        # Calculate SIP amount (1/15th of total invested)
        sip_amount = sip_data['total_invested'] / 15
        
        if cash < sip_amount or sip_data['sips_completed'] >= 15:
            return cash, {}
        
        # Execute SIP
        current_price = self.stock_data[stock].loc[date, 'close']
        
        # Handle potential Series return
        if isinstance(current_price, pd.Series):
            current_price = current_price.iloc[0] if len(current_price) > 0 else np.nan
        
        if pd.notna(current_price):
            current_price = float(current_price)
        else:
            return cash, {}
        
        quantity = sip_amount / current_price
        
        # Update SIP position
        sip_data['quantity'] += quantity
        sip_data['total_invested'] += sip_amount
        sip_data['sips_completed'] += 1
        sip_data['last_sip_date'] = date
        
        # Update average price
        sip_data['avg_price'] = sip_data['total_invested'] / sip_data['quantity']
        
        # Log SIP trade
        trade_entry = {
            'date': date,
            'action': 'SIP',
            'stock': stock,
            'price': current_price,
            'quantity': quantity,
            'value': sip_amount,
            'avg_price': sip_data['avg_price'],
            'return_pct': 0,
            'strategy': 'RSI_SIP'
        }
        self.trade_log.append(trade_entry)
        
        return cash - sip_amount, {}
    
    def backtest_rsi_strategy(self) -> Dict:
        """Execute the complete RSI strategy backtest."""
        logger.info("Starting RSI strategy backtest...")
        
        if not self.stock_data:
            raise ValueError("No stock data loaded. Call load_data() first.")
        
        # Initialize tracking variables
        portfolio = {}  # {stock: holdings_info}
        cash = self.total_capital
        current_position_size = self.position_size
        total_self_dividend = 0
        daily_results = []
        total_trades = 0
        profitable_trades = 0
        
        trading_dates = self.get_trading_dates()
        
        if len(trading_dates) == 0:
            raise ValueError("No common trading dates found across stocks.")
        
        logger.info(f"Backtesting from {trading_dates[0]} to {trading_dates[-1]}")
        logger.info(f"Total trading days: {len(trading_dates)}")
        
        print(f"📊 Starting RSI strategy backtest")
        print(f"💰 Initial capital: ₹{self.total_capital:,.0f}")
        print(f"🎯 Position size: ₹{current_position_size:,.0f}")
        print(f"📈 RSI Entry threshold: < {self.entry_rsi_threshold}")
        print(f"💎 Profit target: {self.profit_target:.2%}")
        
        for i, date in enumerate(trading_dates):
            
            # STEP 1: Execute SIP for corona stocks
            for stock in list(self.sip_positions.keys()):
                cash, _ = self.execute_sip(stock, date, cash)
            
            # STEP 2: CHECK EXIT CONDITIONS (before entry)
            stocks_to_sell = []
            for stock, holdings in portfolio.items():
                if stock in self.stock_data and date in self.stock_data[stock].index:
                    try:
                        current_price = self.stock_data[stock].loc[date, 'close']
                        
                        # Handle potential Series return
                        if isinstance(current_price, pd.Series):
                            current_price = current_price.iloc[0] if len(current_price) > 0 else np.nan
                        
                        if pd.notna(current_price):
                            current_price = float(current_price)
                            
                            # Calculate current return
                            current_return = (current_price - holdings['avg_price']) / holdings['avg_price']
                            
                            # Check for profit target
                            if current_return >= self.profit_target:
                                stocks_to_sell.append((stock, current_return, 'PROFIT'))
                            
                            # Check for corona condition
                            elif self.check_corona_condition(stock, holdings, current_price):
                                if stock not in self.corona_stocks:
                                    self.corona_stocks.add(stock)
                                    # Initialize SIP for this stock
                                    self.sip_positions[stock] = {
                                        'quantity': holdings['quantity'],
                                        'avg_price': holdings['avg_price'],
                                        'total_invested': holdings['total_invested'],
                                        'sips_completed': 0,
                                        'last_sip_date': None
                                    }
                                    logger.info(f"🦠 CORONA: {stock} marked for SIP (>{self.corona_threshold:.0%} loss)")
                                    
                                    # Remove from active portfolio
                                    stocks_to_sell.append((stock, current_return, 'CORONA'))
                    except Exception as e:
                        logger.warning(f"Error checking exit conditions for {stock} on {date}: {e}")
                        continue
            
            # Execute sells
            for stock, return_pct, reason in stocks_to_sell:
                if stock in portfolio:
                    holdings = portfolio[stock]
                    current_price = self.stock_data[stock].loc[date, 'close']
                    
                    # Calculate sale proceeds
                    gross_proceeds = holdings['quantity'] * current_price
                    
                    if return_pct > 0:
                        # Calculate tax on profit
                        profit = gross_proceeds - holdings['total_invested']
                        tax = profit * self.tax_rate
                        net_proceeds = gross_proceeds - tax
                        
                        # Split profit for compounding
                        net_profit = net_proceeds - holdings['total_invested']
                        self_dividend = net_profit * (1 - self.reinvestment_rate)
                        reinvestment = net_profit * self.reinvestment_rate
                        
                        total_self_dividend += self_dividend
                        cash += holdings['total_invested'] + self_dividend + reinvestment
                        
                        # Increase position size for compounding
                        current_position_size += reinvestment / max(1, len(portfolio))
                        
                        total_trades += 1
                        profitable_trades += 1
                        
                    else:
                        cash += gross_proceeds
                        total_trades += 1
                    
                    # Log trade
                    trade_entry = {
                        'date': date,
                        'action': f'SELL_{reason}',
                        'stock': stock,
                        'price': current_price,
                        'quantity': holdings['quantity'],
                        'value': gross_proceeds,
                        'avg_price': holdings['avg_price'],
                        'return_pct': return_pct,
                        'strategy': 'RSI'
                    }
                    self.trade_log.append(trade_entry)
                    
                    print(f"💰 SELL {stock} ({reason}): {return_pct:.1%} at ₹{current_price:.2f}")
                    
                    if reason != 'CORONA':
                        del portfolio[stock]
            
            # STEP 3: ENTRY LOGIC
            
            # Find RSI candidates
            rsi_candidates = self.find_rsi_candidates(date)
            
            # Check for new stock entries (no existing position)
            new_stock_candidates = [
                (stock, rsi) for stock, rsi in rsi_candidates 
                if stock not in portfolio and stock not in self.corona_stocks
            ]
            
            # Buy one new stock if available and we have cash
            if new_stock_candidates and cash >= current_position_size:
                stock_to_buy, rsi_value = new_stock_candidates[0]  # Lowest RSI
                
                if date in self.stock_data[stock_to_buy].index:
                    entry_price = self.stock_data[stock_to_buy].loc[date, 'close']
                    
                    if pd.notna(entry_price) and entry_price > 0:
                        quantity = current_position_size / entry_price
                        
                        portfolio[stock_to_buy] = {
                            'quantity': quantity,
                            'avg_price': entry_price,
                            'total_invested': current_position_size,
                            'last_purchase_price': entry_price,
                            'averaging_attempts': 0
                        }
                        
                        cash -= current_position_size
                        
                        # Log trade
                        trade_entry = {
                            'date': date,
                            'action': 'BUY',
                            'stock': stock_to_buy,
                            'price': entry_price,
                            'quantity': quantity,
                            'value': current_position_size,
                            'avg_price': entry_price,
                            'return_pct': 0,
                            'strategy': 'RSI'
                        }
                        self.trade_log.append(trade_entry)
                        
                        print(f"📈 BUY {stock_to_buy} (RSI: {rsi_value:.1f}) at ₹{entry_price:.2f}")
            
            # STEP 4: AVERAGING DOWN LOGIC
            else:
                # Check for averaging opportunities
                averaging_candidates = []
                for stock, holdings in portfolio.items():
                    if (stock in rsi_candidates and 
                        self.can_average_down(stock, date, holdings)):
                        
                        stock_rsi = next(rsi for s, rsi in rsi_candidates if s == stock)
                        averaging_candidates.append((stock, stock_rsi))
                
                # Average down on stock with lowest RSI
                if averaging_candidates and cash >= current_position_size:
                    averaging_candidates.sort(key=lambda x: x[1])  # Sort by RSI
                    stock_to_average, rsi_value = averaging_candidates[0]
                    
                    if date in self.stock_data[stock_to_average].index:
                        entry_price = self.stock_data[stock_to_average].loc[date, 'close']
                        
                        if pd.notna(entry_price) and entry_price > 0:
                            new_quantity = current_position_size / entry_price
                            
                            # Update existing position
                            holdings = portfolio[stock_to_average]
                            total_quantity = holdings['quantity'] + new_quantity
                            total_invested = holdings['total_invested'] + current_position_size
                            new_avg_price = total_invested / total_quantity
                            
                            portfolio[stock_to_average] = {
                                'quantity': total_quantity,
                                'avg_price': new_avg_price,
                                'total_invested': total_invested,
                                'last_purchase_price': entry_price,
                                'averaging_attempts': holdings['averaging_attempts'] + 1
                            }
                            
                            cash -= current_position_size
                            
                            # Log trade
                            trade_entry = {
                                'date': date,
                                'action': 'AVERAGE',
                                'stock': stock_to_average,
                                'price': entry_price,
                                'quantity': new_quantity,
                                'value': current_position_size,
                                'avg_price': new_avg_price,
                                'return_pct': 0,
                                'strategy': 'RSI'
                            }
                            self.trade_log.append(trade_entry)
                            
                            attempts = holdings['averaging_attempts'] + 1
                            print(f"🔄 AVERAGE {stock_to_average} (RSI: {rsi_value:.1f}, #{attempts}) at ₹{entry_price:.2f}")
            
            # STEP 5: Calculate portfolio value
            position_value = 0
            
            # Active portfolio
            for stock, holdings in portfolio.items():
                if date in self.stock_data[stock].index:
                    current_price = self.stock_data[stock].loc[date, 'close']
                    if pd.notna(current_price):
                        position_value += holdings['quantity'] * current_price
            
            # SIP positions
            sip_value = 0
            for stock, sip_data in self.sip_positions.items():
                if date in self.stock_data[stock].index:
                    current_price = self.stock_data[stock].loc[date, 'close']
                    if pd.notna(current_price):
                        sip_value += sip_data['quantity'] * current_price
            
            total_portfolio_value = cash + position_value + sip_value
            
            daily_results.append({
                'date': date,
                'portfolio_value': total_portfolio_value,
                'cash': cash,
                'position_value': position_value,
                'sip_value': sip_value,
                'num_positions': len(portfolio),
                'num_sip_positions': len(self.sip_positions),
                'total_trades': total_trades,
                'current_position_size': current_position_size,
                'total_self_dividend': total_self_dividend
            })
            
            # Progress update
            if (i + 1) % 250 == 0:  # Every ~1 year
                progress = (i + 1) / len(trading_dates) * 100
                current_return = (total_portfolio_value / self.total_capital - 1) * 100
                print(f"⏳ Progress: {progress:.1f}% | Portfolio: ₹{total_portfolio_value:,.0f} (+{current_return:.1f}%)")
        
        # Calculate final metrics
        final_value = daily_results[-1]['portfolio_value'] if daily_results else self.total_capital
        total_return_pct = (final_value - self.total_capital) / self.total_capital
        
        # Calculate benchmark return (Nifty 50)
        benchmark_return = self.calculate_benchmark_return(trading_dates[0], trading_dates[-1])
        
        # Win rate
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Convert trade log to DataFrame
        trade_log_df = pd.DataFrame(self.trade_log)
        
        results = {
            'strategy_name': 'Nifty 50 RSI Strategy',
            'initial_capital': self.total_capital,
            'final_value': final_value,
            'total_return_pct': total_return_pct,
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate,
            'total_self_dividend': total_self_dividend,
            'nifty_benchmark_return': benchmark_return,
            'excess_return': total_return_pct - benchmark_return,
            'daily_results': pd.DataFrame(daily_results),
            'trade_log': trade_log_df,
            'final_portfolio': portfolio,
            'sip_positions': self.sip_positions,
            'corona_stocks': list(self.corona_stocks),
            'parameters': {
                'rsi_period': self.rsi_period,
                'entry_rsi_threshold': self.entry_rsi_threshold,
                'profit_target': self.profit_target,
                'position_size': self.position_size,
                'tax_rate': self.tax_rate
            }
        }
        
        return results
    
    def calculate_benchmark_return(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
        """Calculate Nifty 50 benchmark return for the period."""
        try:
            if self.nifty_index is None:
                return 0.0
            
            # Get benchmark data for the period
            benchmark_data = self.nifty_index[
                (self.nifty_index.index >= start_date) & 
                (self.nifty_index.index <= end_date)
            ]
            
            if len(benchmark_data) > 0:
                start_value = benchmark_data['nifty50_close'].iloc[0]
                end_value = benchmark_data['nifty50_close'].iloc[-1]
                return (end_value - start_value) / start_value
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Could not calculate benchmark return: {e}")
            return 0.0
    
    def create_visualizations(self, results: Dict) -> None:
        """Create comprehensive visualizations for the RSI strategy."""
        
        print("\n🎨 Creating RSI strategy visualizations...")
        
        # Create visualizations directory
        viz_dir = "results/visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        # Prepare data
        daily_results = results['daily_results'].copy()
        daily_results['date'] = pd.to_datetime(daily_results['date'])
        daily_results.set_index('date', inplace=True)
        
        # Get benchmark data
        benchmark_data = self.get_benchmark_data_for_period(
            daily_results.index[0], 
            daily_results.index[-1]
        )
        
        # Create comprehensive dashboard
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Portfolio Performance vs Benchmark
        ax1 = plt.subplot(3, 2, (1, 2))
        ax1.plot(daily_results.index, daily_results['portfolio_value'], 
                linewidth=3, label='RSI Strategy', color='#E74C3C')
        
        if benchmark_data is not None and len(benchmark_data) > 0:
            ax1.plot(benchmark_data.index, benchmark_data['normalized_nifty'], 
                    linewidth=2, label='Nifty 50 Benchmark', color='#3498DB', alpha=0.8)
        
        ax1.set_title('📈 RSI Strategy vs Nifty 50 Performance', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Portfolio Value (₹)', fontsize=12)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/100000:.1f}L'))
        
        # 2. Cash vs Invested vs SIP
        ax2 = plt.subplot(3, 2, 3)
        ax2.plot(daily_results.index, daily_results['cash'], 
                label='Cash', color='#2ECC71', linewidth=2)
        ax2.plot(daily_results.index, daily_results['position_value'], 
                label='Active Positions', color='#E67E22', linewidth=2)
        ax2.plot(daily_results.index, daily_results['sip_value'], 
                label='SIP Positions', color='#9B59B6', linewidth=2)
        ax2.set_title('💰 Capital Allocation', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Amount (₹)', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/1000:.0f}K'))
        
        # 3. Position Count
        ax3 = plt.subplot(3, 2, 4)
        ax3.plot(daily_results.index, daily_results['num_positions'], 
                label='Active Positions', color='#F39C12', linewidth=2, marker='o', markersize=1)
        ax3.plot(daily_results.index, daily_results['num_sip_positions'], 
                label='SIP Positions', color='#8E44AD', linewidth=2, marker='s', markersize=1)
        ax3.set_title('📊 Position Count Over Time', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Number of Positions', fontsize=12)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # 4. Self-Dividend Growth
        ax4 = plt.subplot(3, 2, 5)
        ax4.plot(daily_results.index, daily_results['total_self_dividend'], 
                color='#16A085', linewidth=3)
        ax4.set_title('💎 Cumulative Self-Dividend', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Self-Dividend (₹)', fontsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/1000:.0f}K'))
        
        # 5. Key Metrics Summary
        ax5 = plt.subplot(3, 2, 6)
        ax5.axis('off')
        
        # Calculate metrics
        final_return = results['total_return_pct'] * 100
        excess_return = results['excess_return'] * 100
        win_rate = results['win_rate'] * 100
        total_trades = results['total_trades']
        self_dividend = results['total_self_dividend']
        
        metrics_text = f"""🎯 RSI STRATEGY PERFORMANCE

📈 Total Return: {final_return:.1f}%
🚀 Excess Return: +{excess_return:.1f}%
🎲 Win Rate: {win_rate:.0f}%
🔄 Total Trades: {total_trades}

💰 Initial Capital: ₹{results['initial_capital']:,.0f}
💎 Final Value: ₹{results['final_value']:,.0f}
💸 Self-Dividend: ₹{self_dividend:,.0f}

🎪 Corona Stocks: {len(results['corona_stocks'])}
📊 RSI Threshold: < {self.entry_rsi_threshold}
🎯 Profit Target: {self.profit_target:.2%}"""
        
        ax5.text(0.1, 0.95, metrics_text, transform=ax5.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.8))
        
        plt.tight_layout(pad=3.0)
        
        # Save dashboard
        dashboard_file = os.path.join(viz_dir, "rsi_strategy_dashboard.png")
        plt.savefig(dashboard_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 RSI strategy dashboard saved to: {dashboard_file}")
        plt.close()
        
        # Create simple comparison chart
        self._create_simple_comparison_chart(results, viz_dir)
    
    def get_benchmark_data_for_period(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """Get normalized benchmark data for the given period."""
        try:
            if self.nifty_index is None:
                return None
            
            benchmark_subset = self.nifty_index[
                (self.nifty_index.index >= start_date) & 
                (self.nifty_index.index <= end_date)
            ].copy()
            
            if len(benchmark_subset) > 0:
                initial_value = benchmark_subset['nifty50_close'].iloc[0]
                benchmark_subset['normalized_nifty'] = (
                    benchmark_subset['nifty50_close'] / initial_value
                ) * self.total_capital
                return benchmark_subset
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Could not get benchmark data: {e}")
            return None
    
    def _create_simple_comparison_chart(self, results: Dict, viz_dir: str) -> None:
        """Create a simple strategy vs benchmark comparison chart."""
        
        daily_results = results['daily_results'].copy()
        daily_results['date'] = pd.to_datetime(daily_results['date'])
        
        # Normalize strategy to start at 100
        strategy_normalized = (daily_results['portfolio_value'] / self.total_capital) * 100
        
        # Get benchmark data
        benchmark_data = self.get_benchmark_data_for_period(
            daily_results['date'].iloc[0], 
            daily_results['date'].iloc[-1]
        )
        
        if benchmark_data is not None and len(benchmark_data) > 0:
            benchmark_normalized = (benchmark_data['nifty50_close'] / benchmark_data['nifty50_close'].iloc[0]) * 100
            
            plt.figure(figsize=(14, 8))
            plt.plot(daily_results['date'], strategy_normalized, 
                    linewidth=3, label='RSI Strategy', color='#E74C3C')
            plt.plot(benchmark_data.index, benchmark_normalized, 
                    linewidth=2, label='Nifty 50 Benchmark', color='#3498DB', alpha=0.8)
            
            plt.title('RSI Strategy vs Nifty 50 Performance\n(Normalized to 100)', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Normalized Value (Base = 100)', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Add final values
            final_strategy = strategy_normalized.iloc[-1]
            final_benchmark = benchmark_normalized.iloc[-1]
            
            plt.annotate(f'Final: {final_strategy:.1f}\n(+{final_strategy-100:.1f}%)', 
                       xy=(daily_results['date'].iloc[-1], final_strategy), 
                       xytext=(-80, 20), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.8),
                       fontsize=11, fontweight='bold', color='white', ha='center')
            
            plt.annotate(f'Final: {final_benchmark:.1f}\n(+{final_benchmark-100:.1f}%)', 
                       xy=(benchmark_data.index[-1], final_benchmark), 
                       xytext=(-80, -40), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='blue', alpha=0.8),
                       fontsize=11, fontweight='bold', color='white', ha='center')
            
            plt.tight_layout()
            
            # Save comparison chart
            comparison_file = os.path.join(viz_dir, "rsi_vs_benchmark_comparison.png")
            plt.savefig(comparison_file, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"📈 Comparison chart saved to: {comparison_file}")
            plt.close()
    
    def create_performance_report(self, results: Dict) -> str:
        """Create a detailed performance report."""
        
        # Calculate additional metrics
        total_investment = results['initial_capital']
        final_value = results['final_value']
        self_dividend = results['total_self_dividend']
        
        # Effective return including self-dividend
        effective_final_value = final_value + self_dividend
        effective_return = (effective_final_value - total_investment) / total_investment
        
        # Calculate annualized return
        daily_results = results['daily_results']
        years = len(daily_results) / 252  # Approximate trading days per year
        annualized_return = (final_value / total_investment) ** (1/years) - 1 if years > 0 else 0
        
        report = f"""
{'='*70}
🚀 NIFTY 50 RSI TRADING STRATEGY REPORT
{'='*70}
Strategy: {results['strategy_name']}

📊 PERFORMANCE SUMMARY:
- Initial Capital: ₹{results['initial_capital']:,.0f}
- Final Portfolio Value: ₹{results['final_value']:,.0f}
- Total Self-Dividend: ₹{results['total_self_dividend']:,.0f}
- Effective Final Value: ₹{effective_final_value:,.0f}

📈 RETURNS:
- Portfolio Return: {results['total_return_pct']:.2%}
- Effective Return (incl. dividend): {effective_return:.2%}
- Annualized Return: {annualized_return:.2%}
- Benchmark Return (Nifty 50): {results['nifty_benchmark_return']:.2%}
- Excess Return: {results['excess_return']:.2%}

🎯 TRADING STATISTICS:
- Total Trades: {results['total_trades']}
- Profitable Trades: {results['profitable_trades']}
- Win Rate: {results['win_rate']:.2%}
- RSI Entry Threshold: < {self.entry_rsi_threshold}
- Profit Target: {self.profit_target:.2%}

💼 RISK MANAGEMENT:
- Corona Stocks (Quarantined): {len(results['corona_stocks'])}
- Active SIP Positions: {len(results['sip_positions'])}
- Final Active Positions: {len(results['final_portfolio'])}

🏆 STRATEGY PARAMETERS:
- RSI Period: {self.rsi_period}
- Position Size: ₹{self.position_size:,.0f}
- Tax Rate: {self.tax_rate:.2%}
- Reinvestment Rate: {self.reinvestment_rate:.0%}
- Max Averaging Attempts: {self.max_averaging_attempts}

💡 KEY INSIGHTS:
- Mathematical approach with Pi-based thresholds (3.14%, 6.28%)
- Effective compounding through 50% reinvestment
- Risk control through corona quarantine and SIP recovery
- Tax-efficient profit booking with systematic dividend withdrawal

{'='*70}
✅ RSI STRATEGY ANALYSIS COMPLETE
{'='*70}
        """
        
        return report

def main():
    """Main function to run the RSI strategy backtest."""
    print("=" * 70)
    print("🚀 NIFTY 50 RSI TRADING STRATEGY")
    print("=" * 70)
    print("Mathematical Trading with Pi-Based Thresholds")
    print("Entry: RSI < 35 | Exit: 6.28% Profit | Averaging: 3.14% Drop")
    print("=" * 70)
    
    # Initialize RSI strategy
    strategy = NiftyRSIStrategy()
    
    try:
        # Load data
        print("\n📊 Loading Nifty 50 historical data...")
        strategy.load_data()
        print("✅ Data loaded successfully!")
        print(f"📈 Loaded {len(strategy.stock_data)} stocks")
        
        # Run backtest
        print("\n🔄 Running RSI strategy backtest...")
        print("This may take a few moments...")
        results = strategy.backtest_rsi_strategy()
        print("✅ Backtest completed successfully!")
        
        # Create and display report
        print("\n📋 Generating performance report...")
        report = strategy.create_performance_report(results)
        print(report)
        
        # Create visualizations
        strategy.create_visualizations(results)
        
        # Save results
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        print("\n💾 Saving results...")
        
        # Save trade log
        trade_log_file = os.path.join(results_dir, "rsi_strategy_trade_log.csv")
        results['trade_log'].to_csv(trade_log_file, index=False)
        print(f"📝 Trade log saved to: {trade_log_file}")
        
        # Save daily results
        daily_results_file = os.path.join(results_dir, "rsi_strategy_daily_results.csv")
        results['daily_results'].to_csv(daily_results_file, index=False)
        print(f"📊 Daily results saved to: {daily_results_file}")
        
        # Save performance report
        report_file = os.path.join(results_dir, "rsi_strategy_performance_report.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"📄 Performance report saved to: {report_file}")
        
        print("\n" + "=" * 70)
        print("🎯 RSI STRATEGY ANALYSIS COMPLETE!")
        print("=" * 70)
        print(f"📈 Total Return: {results['total_return_pct']:.2%}")
        print(f"🚀 Excess Return: {results['excess_return']:.2%}")
        print(f"🎲 Win Rate: {results['win_rate']:.2%}")
        print(f"🔄 Total Trades: {results['total_trades']}")
        print(f"💸 Self-Dividend: ₹{results['total_self_dividend']:,.0f}")
        print("=" * 70)
        print("💡 Key Features:")
        print("  • Mathematical precision with Pi-based thresholds")
        print("  • Compounding growth with systematic dividend withdrawal") 
        print("  • Risk management through corona quarantine and SIP")
        print("  • Tax-efficient profit booking")
        print("=" * 70)
        print("✅ RSI STRATEGY READY FOR IMPLEMENTATION!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error running RSI strategy: {e}")
        logger.error(f"Detailed error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
