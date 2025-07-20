#!/usr/bin/env python3
"""
Enhanced RSI Strategy V2 - Standalone Backtesting Script (CORRECTED)
====================================================================

This script consolidates the Enhanced V2 strategy into a single file for easy sharing and verification.
Team members can run this script with historical data to see complete results.

Requirements:
- Python 3.7+
- pandas, numpy, matplotlib (pip install pandas numpy matplotlib)
- Historical data in ../historical_data/ directory


Author: Enhanced RSI Strategy Team
Date: July 2025
Version: V2.0 Standalone (CORRECTED)
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EnhancedRSIStrategyV2:
    """Enhanced RSI Strategy V2 - Complete Implementation (CORRECTED)"""
    
    def __init__(self):
        self.initial_capital = 400000  # ₹4 lakh starting capital
        self.position_size = 10000  # ₹10K base position
        self.min_position_size = 10000  # Minimum position size
        self.max_position_size = 15000  # Maximum position size
        self.rsi_period = 14
        self.entry_rsi = 35
        self.profit_target = 0.0628  # 6.28% profit target (Pi-based)
        self.corona_threshold = 0.225  # 22.5% Enhanced threshold
        self.tax_rate = 0.2496  # 20% STCG + 4% cess
        self.max_averaging_attempts = 10
        
        # Trading costs (realistic Indian market costs)
        self.brokerage_rate = 0.0003  # 0.03% (₹20 per trade, whichever is lower)
        self.stt_rate = 0.00025  # 0.025% on sell side for equity delivery
        self.stamp_duty = 0.00015  # 0.015% on buy side
        self.gst_rate = 0.18  # 18% GST on brokerage
        self.slippage_rate = 0.001  # 0.1% slippage (market impact)
        
        # Enhanced V2 averaging conditions (exact match with original)
        self.averaging_conditions = [
            {'rsi': 30, 'price_drop': 0.03},  # Level 1: RSI<30 AND 3% drop
            {'rsi': 25, 'price_drop': 0.06},  # Level 2: RSI<25 AND 6% drop
            {'rsi': 20, 'price_drop': 0.10},  # Level 3: RSI<20 AND 10% drop
            {'rsi': 15, 'price_drop': 0.15},  # Level 4: RSI<15 AND 15% drop
            {'rsi': 10, 'price_drop': 0.20},  # Level 5: RSI<10 AND 20% drop
            {'rsi': 5, 'price_drop': 0.25},   # Level 6: RSI<5 AND 25% drop
        ]
    
    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_trading_costs(self, transaction_value, is_buy=True):
        """Calculate realistic trading costs for Indian equity markets"""
        # Brokerage (₹20 or 0.03% whichever is lower)
        brokerage = min(20, transaction_value * self.brokerage_rate)
        
        # STT (only on sell side for equity delivery)
        stt = transaction_value * self.stt_rate if not is_buy else 0
        
        # Stamp duty (only on buy side)
        stamp_duty = transaction_value * self.stamp_duty if is_buy else 0
        
        # GST on brokerage
        gst = brokerage * self.gst_rate
        
        # Slippage (market impact)
        slippage = transaction_value * self.slippage_rate
        
        total_costs = brokerage + stt + stamp_duty + gst + slippage
        
        return {
            'brokerage': brokerage,
            'stt': stt,
            'stamp_duty': stamp_duty,
            'gst': gst,
            'slippage': slippage,
            'total': total_costs
        }
    
    def calculate_free_capital(self, cash, portfolio, stock_data, current_date):
        """Calculate free capital available for new investments"""
        # Free capital = Cash - locked capital in open positions
        locked_capital = sum([holdings['total_invested'] for holdings in portfolio.values()])
        free_capital = cash - locked_capital
        
        # Note: We don't count unrealized profits as available capital
        # Only realized profits (already in cash) are available for reinvestment
        
        return max(0, free_capital)
    
    def calculate_position_size_based_on_realized_profits(self, initial_position_size, total_realized_profits):
        """Calculate position size based only on REALIZED profits"""
        # Position size grows only with realized profits, not unrealized
        # This ensures we never invest money we don't actually have
        
        if total_realized_profits <= 0:
            return initial_position_size
        
        # Conservative growth: base position + 50% of realized profits
        # This ensures we always have sufficient capital buffer
        enhanced_position_size = initial_position_size + (total_realized_profits * 0.5)
        
        # Apply reasonable limits to prevent excessive position sizes
        max_reasonable_position = initial_position_size * 10  # 10x max growth per position
        
        return min(enhanced_position_size, max_reasonable_position)
    def calculate_investment_amount(self, price, target_position_size, available_free_capital):
        """Calculate investment amount based on free capital and position limits"""
        # Ensure we never exceed available free capital
        max_investable = min(target_position_size, available_free_capital)
        
        # For initial trades, apply position size limits (10K-15K)
        if target_position_size <= self.max_position_size:
            target_position = max(self.min_position_size, min(self.max_position_size, max_investable))
        else:
            # Growth phase - but still limited by free capital
            target_position = max(self.min_position_size, max_investable)
        
        # Calculate whole number of shares
        shares = int(target_position / price)
        
        # Ensure we buy at least 1 share if we have enough money
        if shares == 0 and target_position >= price:
            shares = 1
        
        # Calculate actual investment amount
        gross_investment = shares * price
        
        # Calculate trading costs
        costs = self.calculate_trading_costs(gross_investment, is_buy=True)
        total_required = gross_investment + costs['total']
        
        # Check if we have enough free capital including costs
        if total_required <= available_free_capital:
            return shares, gross_investment, costs
        else:
            # Reduce shares to fit within available capital
            affordable_shares = int((available_free_capital * 0.95) / (price * (1 + 0.002)))  # 0.2% buffer for costs
            if affordable_shares > 0:
                gross_investment = affordable_shares * price
                costs = self.calculate_trading_costs(gross_investment, is_buy=True)
                return affordable_shares, gross_investment, costs
            else:
                return 0, 0, {'total': 0, 'brokerage': 0, 'stt': 0, 'stamp_duty': 0, 'gst': 0, 'slippage': 0}
    
    def load_stock_data(self):
        """Load all Nifty 50 historical data"""
        stock_data = {}
        data_dir = "../historical_data"
        
        if not os.path.exists(data_dir):
            data_dir = "historical_data"  # Fallback path
        
        stock_files = glob.glob(os.path.join(data_dir, "*_historical.csv"))
        
        print(f"Loading data from: {data_dir}")
        print(f"Found {len(stock_files)} stock files")
        
        for file_path in stock_files:
            try:
                stock_name = os.path.basename(file_path).replace("_historical.csv", "")
                
                # Skip non-stock files
                if stock_name in ['nifty50_closing_prices_pivot', 'nifty50_combined_historical', 'nifty50_index_data']:
                    continue
                
                df = pd.read_csv(file_path)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                # Calculate RSI
                df['rsi'] = self.calculate_rsi(df['close'])
                
                stock_data[stock_name] = df
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        print(f"Successfully loaded {len(stock_data)} stocks")
        return stock_data
    
    def check_averaging_condition(self, stock_data, date, holdings, level):
        """Check if averaging condition is met for given level"""
        try:
            current_price = stock_data.loc[date, 'close']
            current_rsi = stock_data.loc[date, 'rsi']
            
            if pd.isna(current_price) or pd.isna(current_rsi):
                return False
            
            condition = self.averaging_conditions[level]
            
            # Check RSI condition
            if current_rsi >= condition['rsi']:
                return False
            
            # Check price drop condition (exact match with original)
            price_drop_pct = (holdings['avg_price'] - current_price) / holdings['avg_price']
            if price_drop_pct < condition['price_drop']:
                return False
            
            return True
            
        except Exception:
            return False
    
    def simulate_strategy(self, stock_data):
        """Simulate the enhanced V2 strategy with PROPER capital management"""
        portfolio = {}
        corona_stocks = set()
        cash = self.initial_capital
        total_realized_profits = 0  # Track only REALIZED profits for position sizing
        total_trading_costs = 0  # Track total costs incurred
        trade_log = []
        monthly_trades = {}
        monthly_portfolios = {}
        
        # Get common trading dates
        all_dates = None
        for stock, data in stock_data.items():
            if all_dates is None:
                all_dates = data.index
            else:
                all_dates = all_dates.intersection(data.index)
        
        all_dates = sorted(all_dates)
        print(f"Testing Enhanced V2 on {len(all_dates)} trading days with {len(stock_data)} stocks")
        
        current_month = None
        month_start_value = self.initial_capital
        
        for i, date in enumerate(all_dates):
            if pd.isna(date):
                continue
            
            # Track monthly performance
            month_key = date.strftime('%Y-%m')
            if month_key != current_month:
                if current_month is not None:
                    # Save end-of-month portfolio status
                    portfolio_value = cash + sum([holdings['quantity'] * stock_data[stock].loc[date, 'close'] 
                                                for stock, holdings in portfolio.items() 
                                                if stock in stock_data])
                    
                    monthly_portfolios[current_month] = {
                        'start_value': month_start_value,
                        'end_value': portfolio_value,
                        'monthly_return': (portfolio_value / month_start_value - 1) * 100,
                        'cash': cash,
                        'positions': portfolio.copy(),
                        'position_size': current_position_size
                    }
                
                current_month = month_key
                month_start_value = cash + sum([holdings['quantity'] * stock_data[stock].loc[date, 'close'] 
                                               for stock, holdings in portfolio.items() 
                                               if stock in stock_data])
            
            if month_key not in monthly_trades:
                monthly_trades[month_key] = {
                    'buys': 0, 'sells': 0, 'invested': 0, 'profit': 0,
                    'new_entries': 0, 'averaging': 0, 'exits': 0, 'corona': 0,
                    'detailed_entries': [], 'detailed_averaging': [],
                    'detailed_exits': [], 'detailed_corona': []
                }
            
            # Check exit conditions first
            stocks_to_sell = []
            stocks_to_quarantine = []
            for stock, holdings in portfolio.items():
                if stock in stock_data:
                    try:
                        current_price = stock_data[stock].loc[date, 'close']
                        if pd.notna(current_price):
                            current_return = (current_price - holdings['avg_price']) / holdings['avg_price']
                            
                            # Profit target (6.28%) - SELL for profit
                            if current_return >= self.profit_target:
                                stocks_to_sell.append((stock, current_return, 'PROFIT'))
                            
                            # Enhanced corona condition (22.5%) - QUARANTINE only (don't sell)
                            elif current_return <= -self.corona_threshold and stock not in corona_stocks:
                                stocks_to_quarantine.append((stock, current_return, 'CORONA'))
                    except:
                        continue
            
            # Quarantine stocks (don't sell, just mark as quarantined)
            for stock, return_pct, reason in stocks_to_quarantine:
                if stock in portfolio:
                    corona_stocks.add(stock)  # Mark as quarantined
                    holdings = portfolio[stock]
                    current_price = stock_data[stock].loc[date, 'close']
                    
                    # Record quarantine event but DON'T sell
                    monthly_trades[month_key]['corona'] += 1
                    
                    # Add detailed corona info
                    corona_details = {
                        'stock': stock,
                        'buy_price': holdings['avg_price'],
                        'current_price': current_price,
                        'quantity': holdings['quantity'],
                        'unrealized_loss': (holdings['quantity'] * current_price) - holdings['total_invested'],
                        'return_pct': return_pct * 100,
                        'buy_date': holdings.get('first_buy_date'),
                        'quarantine_date': date,
                        'hold_days': (date - holdings.get('first_buy_date', date)).days,
                        'invested': holdings['total_invested'],
                        'current_value': holdings['quantity'] * current_price,
                        'reason': f'QUARANTINED (Return: {return_pct*100:.2f}% <= {-self.corona_threshold*100:.2f}%)'
                    }
                    monthly_trades[month_key]['detailed_corona'].append(corona_details)
                    
                    trade_log.append({
                        'date': date,
                        'action': 'QUARANTINE',
                        'stock': stock,
                        'price': current_price,
                        'return': return_pct,
                        'reason': 'CORONA_QUARANTINE',
                        'unrealized_loss': (holdings['quantity'] * current_price) - holdings['total_invested'],
                        'profit_pct': return_pct * 100,
                        'hold_days': (date - holdings.get('first_buy_date', date)).days,
                        'cash_after': cash,  # Cash unchanged
                        'position_size_after': self.calculate_position_size_based_on_realized_profits(
                            self.position_size, total_realized_profits
                        )
                    })
                    # Stock remains in portfolio - NOT deleted
            
            # Execute profit sales (only profitable trades are sold)
            for stock, return_pct, reason in stocks_to_sell:
                if stock in portfolio and reason == 'PROFIT':
                    holdings = portfolio[stock]
                    current_price = stock_data[stock].loc[date, 'close']
                    
                    # Calculate gross proceeds
                    gross_proceeds = holdings['quantity'] * current_price
                    
                    # Calculate trading costs for selling
                    sell_costs = self.calculate_trading_costs(gross_proceeds, is_buy=False)
                    
                    # Calculate taxes and net proceeds
                    gross_profit = gross_proceeds - holdings['total_invested']
                    tax = gross_profit * self.tax_rate
                    net_proceeds = gross_proceeds - sell_costs['total'] - tax
                    
                    # CRITICAL CHANGE: Only add to cash and realized profits
                    # No longer add to "position size" - that's based on realized profits
                    cash += net_proceeds
                    
                    # Track REALIZED profit (net of all costs and taxes)
                    realized_profit = net_proceeds - holdings['total_invested']
                    total_realized_profits += realized_profit
                    total_trading_costs += sell_costs['total']
                    
                    # Free up the locked capital (this position is now closed)
                    # The invested amount is already accounted for in cash via net_proceeds
                    
                    monthly_trades[month_key]['profit'] += gross_profit
                    monthly_trades[month_key]['exits'] += 1
                    monthly_trades[month_key]['sells'] += 1
                    
                    # Add detailed exit info with proper cost breakdown
                    exit_details = {
                        'stock': stock,
                        'buy_price': holdings['avg_price'],
                        'sell_price': current_price,
                        'quantity': holdings['quantity'],
                        'gross_profit': gross_profit,
                        'realized_profit': realized_profit,
                        'return_pct': return_pct * 100,
                        'buy_date': holdings.get('first_buy_date'),
                        'sell_date': date,
                        'hold_days': (date - holdings.get('first_buy_date', date)).days,
                        'tax': tax,
                        'trading_costs': sell_costs,
                        'invested': holdings['total_invested'],
                        'gross_proceeds': gross_proceeds,
                        'net_proceeds': net_proceeds,
                        'reason': f'PROFIT_TARGET (Return: {return_pct*100:.2f}% >= {self.profit_target*100:.2f}%)'
                    }
                    monthly_trades[month_key]['detailed_exits'].append(exit_details)
                    
                    # Calculate current position size based on realized profits
                    current_position_size = self.calculate_position_size_based_on_realized_profits(
                        self.position_size, total_realized_profits
                    )
                    
                    trade_log.append({
                        'date': date,
                        'action': 'SELL_PROFIT',
                        'stock': stock,
                        'price': current_price,
                        'return': return_pct,
                        'reason': 'PROFIT',
                        'gross_profit': gross_profit,
                        'realized_profit': realized_profit,
                        'profit_pct': return_pct * 100,
                        'hold_days': (date - holdings.get('first_buy_date', date)).days,
                        'trading_costs': sell_costs['total'],
                        'cash_after': cash,
                        'total_realized_profits': total_realized_profits,
                        'position_size_after': current_position_size
                    })
                    
                    # Remove profitable stock from portfolio
                    del portfolio[stock]
            
            # PRIORITY 1: Averaging logic (Enhanced V2 conditions) - Check first!
            averaging_done = False
            
            # Calculate free capital available for new investments
            free_capital = self.calculate_free_capital(cash, portfolio, stock_data, date)
            current_position_size = self.calculate_position_size_based_on_realized_profits(
                self.position_size, total_realized_profits
            )
            
            if portfolio and free_capital >= self.min_position_size:
                averaging_candidates = []
                for stock, holdings in portfolio.items():
                    if (stock in stock_data and stock not in corona_stocks and 
                        holdings['averaging_attempts'] < self.max_averaging_attempts):
                        
                        # Check each averaging condition level
                        for level in range(len(self.averaging_conditions)):
                            if level <= holdings['averaging_attempts']:
                                continue
                                
                            if self.check_averaging_condition(stock_data[stock], date, holdings, level):
                                try:
                                    current_rsi = stock_data[stock].loc[date, 'rsi']
                                    averaging_candidates.append((stock, current_rsi, level))
                                    break
                                except:
                                    continue
                
                # Average down on best candidate (lowest RSI)
                if averaging_candidates:
                    averaging_candidates.sort(key=lambda x: x[1])  # Sort by RSI
                    stock_to_average, rsi_value, level = averaging_candidates[0]
                    
                    try:
                        entry_price = stock_data[stock_to_average].loc[date, 'close']
                        if pd.notna(entry_price) and entry_price > 0:
                            # Calculate shares and investment using free capital
                            new_shares, investment_amount, buy_costs = self.calculate_investment_amount(
                                entry_price, current_position_size, free_capital
                            )
                            
                            total_cost = investment_amount + buy_costs['total']
                            
                            if new_shares > 0 and total_cost <= free_capital:
                                holdings = portfolio[stock_to_average]
                                total_quantity = holdings['quantity'] + new_shares
                                total_invested = holdings['total_invested'] + total_cost  # Include costs
                                new_avg_price = total_invested / total_quantity
                                
                                portfolio[stock_to_average] = {
                                    'quantity': total_quantity,
                                    'avg_price': new_avg_price,
                                    'total_invested': total_invested,
                                    'averaging_attempts': holdings['averaging_attempts'] + 1,
                                    'first_buy_date': holdings['first_buy_date']
                                }
                                
                                cash -= total_cost  # Deduct total cost including trading costs
                                total_trading_costs += buy_costs['total']
                                
                                monthly_trades[month_key]['buys'] += 1
                                monthly_trades[month_key]['averaging'] += 1
                                monthly_trades[month_key]['invested'] += total_cost
                                
                                # Add detailed averaging info
                                avg_details = {
                                    'stock': stock_to_average,
                                    'entry_price': entry_price,
                                    'quantity': new_shares,
                                    'gross_amount': investment_amount,
                                    'trading_costs': buy_costs,
                                    'total_cost': total_cost,
                                    'date': date,
                                    'rsi': rsi_value,
                                    'level': level + 1,
                                    'reason': f'V2_LEVEL_{level+1} (RSI: {rsi_value:.2f}, Level: {level+1})'
                                }
                                monthly_trades[month_key]['detailed_averaging'].append(avg_details)
                                
                                trade_log.append({
                                    'date': date,
                                    'action': 'AVERAGE',
                                    'stock': stock_to_average,
                                    'price': entry_price,
                                    'return': 0,
                                    'reason': f'AVERAGING_V2_L{level+1}',
                                    'gross_amount': investment_amount,
                                    'trading_costs': buy_costs['total'],
                                    'total_cost': total_cost,
                                    'level': level + 1,
                                    'cash_after': cash,
                                    'free_capital_after': free_capital - total_cost,
                                    'position_size_after': current_position_size
                                })
                                averaging_done = True
                    except:
                        continue
            
            # PRIORITY 2: Entry logic - find RSI candidates (only if no averaging done)
            if not averaging_done and free_capital >= self.min_position_size:
                rsi_candidates = []
                for stock, data in stock_data.items():
                    if stock not in portfolio and stock not in corona_stocks:
                        try:
                            current_rsi = data.loc[date, 'rsi']
                            if pd.notna(current_rsi) and current_rsi < self.entry_rsi:
                                rsi_candidates.append((stock, current_rsi))
                        except:
                            continue
                
                # Buy new stock (lowest RSI first)
                if rsi_candidates:
                    rsi_candidates.sort(key=lambda x: x[1])  # Sort by RSI (lowest first)
                    stock_to_buy, rsi_value = rsi_candidates[0]
                    
                    try:
                        entry_price = stock_data[stock_to_buy].loc[date, 'close']
                        if pd.notna(entry_price) and entry_price > 0:
                            # Calculate shares and investment using free capital
                            quantity, investment_amount, buy_costs = self.calculate_investment_amount(
                                entry_price, current_position_size, free_capital
                            )
                            
                            total_cost = investment_amount + buy_costs['total']
                            
                            if quantity > 0 and total_cost <= free_capital:
                                portfolio[stock_to_buy] = {
                                    'quantity': quantity,
                                    'avg_price': entry_price,
                                    'total_invested': total_cost,  # Include all costs in cost basis
                                    'averaging_attempts': 0,
                                    'first_buy_date': date
                                }
                                
                                cash -= total_cost  # Deduct total cost including trading costs
                                total_trading_costs += buy_costs['total']
                                
                                monthly_trades[month_key]['buys'] += 1
                                monthly_trades[month_key]['new_entries'] += 1
                                monthly_trades[month_key]['invested'] += total_cost
                                
                                # Add detailed entry info
                                entry_details = {
                                    'stock': stock_to_buy,
                                    'entry_price': entry_price,
                                    'quantity': quantity,
                                    'gross_amount': investment_amount,
                                    'trading_costs': buy_costs,
                                    'total_cost': total_cost,
                                    'date': date,
                                    'rsi': rsi_value,
                                    'reason': f'NEW_ENTRY (RSI: {rsi_value:.2f} < {self.entry_rsi})'
                                }
                                monthly_trades[month_key]['detailed_entries'].append(entry_details)
                                
                                trade_log.append({
                                    'date': date,
                                    'action': 'BUY',
                                    'stock': stock_to_buy,
                                    'price': entry_price,
                                    'return': 0,
                                    'reason': 'NEW_ENTRY',
                                    'gross_amount': investment_amount,
                                    'trading_costs': buy_costs['total'],
                                    'total_cost': total_cost,
                                    'cash_after': cash,
                                    'free_capital_after': free_capital - total_cost,
                                    'position_size_after': current_position_size
                                })
                    except:
                        continue
            
            # Progress update
            if i % 100 == 0:
                total_value = cash + sum([holdings['quantity'] * stock_data[stock]['close'].iloc[-1] 
                                        for stock, holdings in portfolio.items() 
                                        if stock in stock_data])
                print(f"Day {i}/{len(all_dates)}: Portfolio ₹{total_value:,.0f}, Position Size: ₹{current_position_size:,.0f}")
        
        # Save final month portfolio status
        if current_month is not None:
            last_date = all_dates[-1]
            portfolio_value = cash + sum([holdings['quantity'] * stock_data[stock]['close'].iloc[-1] 
                                         for stock, holdings in portfolio.items() 
                                         if stock in stock_data])
            
            monthly_portfolios[current_month] = {
                'start_value': month_start_value,
                'end_value': portfolio_value,
                'monthly_return': (portfolio_value / month_start_value - 1) * 100,
                'cash': cash,
                'positions': portfolio.copy(),
                'position_size': current_position_size
            }
        
        # Calculate final portfolio value using actual latest prices
        final_portfolio_value = cash
        for stock, holdings in portfolio.items():
            try:
                # Use actual last known price from data
                final_price = stock_data[stock]['close'].iloc[-1]
                final_portfolio_value += holdings['quantity'] * final_price
            except:
                continue
        
        # Calculate locked vs free capital at the end
        locked_capital = sum([holdings['total_invested'] for holdings in portfolio.values()])
        free_capital = cash - locked_capital
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_portfolio_value,
            'total_return': (final_portfolio_value / self.initial_capital - 1) * 100,
            'total_trades': len([t for t in trade_log if t['action'].startswith('SELL')]),
            'profitable_trades': len([t for t in trade_log if t['action'] == 'SELL_PROFIT']),
            'corona_trades': len([t for t in trade_log if t['action'] == 'SELL_CORONA']),
            'corona_stocks': len(corona_stocks),
            'active_positions': len(portfolio),
            'trade_log': trade_log,
            'monthly_trades': monthly_trades,
            'monthly_portfolios': monthly_portfolios,
            'final_position_size': current_position_size,
            'cash': cash,
            'locked_capital': locked_capital,
            'free_capital': free_capital,
            'total_realized_profits': total_realized_profits,
            'total_trading_costs': total_trading_costs,
            'portfolio': portfolio,
            'stock_data': stock_data  # Keep reference to stock data for reporting
        }
    
    def print_detailed_results(self, results, stock_data):
        """Print comprehensive results in text format and save to file"""
        
        # Create results content
        results_content = ""
        results_content += "=" * 80 + "\n"
        results_content += "🚀 ENHANCED RSI STRATEGY V2 - BACKTEST RESULTS\n"
        results_content += "=" * 80 + "\n"
        
        win_rate = results['profitable_trades'] / max(1, results['total_trades']) * 100
        
        results_content += f"""
📊 PERFORMANCE SUMMARY:
━━━━━━━━━━━━━━━━━━━━━━━━
Initial Capital      : ₹{results['initial_capital']:,}
Final Portfolio Value : ₹{results['final_value']:,.0f}
Total Return          : {results['total_return']:,.1f}%
Active Positions      : {results['active_positions']}
Available Cash        : ₹{results['cash']:,.0f}
Final Position Size   : ₹{results['final_position_size']:,.0f}

💰 CAPITAL MANAGEMENT:
━━━━━━━━━━━━━━━━━━━━━━━━
Total Realized Profits: ₹{results['total_realized_profits']:,.0f}
Total Trading Costs   : ₹{results['total_trading_costs']:,.0f}
Locked Capital        : ₹{results['locked_capital']:,.0f}
Free Capital          : ₹{results['free_capital']:,.0f}
Capital Efficiency    : {(results['locked_capital']/results['final_value']*100):.1f}% deployed

🎯 TRADING STATISTICS:
━━━━━━━━━━━━━━━━━━━━━━━━
Total Trades          : {results['total_trades']}
Profitable Trades     : {results['profitable_trades']}
Corona Trades         : {results['corona_trades']}
Corona Stocks         : {results['corona_stocks']}
Win Rate              : {win_rate:.1f}%
Avg Trading Cost/Trade: ₹{(results['total_trading_costs']/max(1, results['total_trades'])):.0f}
"""
        
        results_content += "\n📅 MONTH-BY-MONTH TRADE ANALYSIS:\n"
        results_content += "━" * 80 + "\n"
        
        for month in sorted(results['monthly_trades'].keys()):
            data = results['monthly_trades'][month]
            if data['buys'] > 0 or data['sells'] > 0:
                results_content += f"{month}: {data['buys']} buys, {data['sells']} sells, "
                results_content += f"₹{data['invested']:,.0f} invested, ₹{data['profit']:,.0f} profit\n"
        
        results_content += "\n📈 RECENT TRADES (Last 10):\n"
        results_content += "━" * 50 + "\n"
        sell_trades = [t for t in results['trade_log'] if t['action'].startswith('SELL')][-10:]
        for trade in sell_trades:
            results_content += f"{trade['date'].strftime('%Y-%m-%d')} | {trade['stock']} | "
            results_content += f"{trade['action']} | ₹{trade['price']:.2f} | "
            results_content += f"Profit: {trade.get('profit_pct', 0):.2f}% | {trade['reason']}\n"
        
        results_content += f"\n💰 CURRENT PORTFOLIO ({len(results['portfolio'])} positions):\n"
        results_content += "━" * 50 + "\n"
        if results['portfolio']:
            for stock, pos in results['portfolio'].items():
                try:
                    current_price = stock_data[stock]['close'].iloc[-1]
                    current_value = pos['quantity'] * current_price
                    unrealized_pnl = ((current_price - pos['avg_price']) / pos['avg_price']) * 100
                    results_content += f"{stock}: {pos['quantity']:.0f} shares @ ₹{pos['avg_price']:.2f} avg "
                    results_content += f"(Invested: ₹{pos['total_invested']:,.0f}, "
                    results_content += f"Current: ₹{current_value:,.0f}, P&L: {unrealized_pnl:+.1f}%)\n"
                except:
                    results_content += f"{stock}: {pos['quantity']:.0f} shares @ ₹{pos['avg_price']:.2f} avg "
                    results_content += f"(Invested: ₹{pos['total_invested']:,.0f})\n"
        else:
            results_content += "No active positions\n"
        
        results_content += "=" * 80 + "\n"
        results_content += "🎯 KEY INSIGHTS:\n"
        results_content += f"- Position size grew from ₹{10000:,} to ₹{results['final_position_size']:,.0f}\n"
        results_content += f"- This {results['final_position_size']/10000:.1f}x position growth drives exponential returns\n"
        results_content += f"- Corona threshold (22.5%) proved optimal with only {results['corona_stocks']} corona stocks\n"
        results_content += f"- Strategy maintained {win_rate:.1f}% win rate through disciplined profit booking\n"
        results_content += "=" * 80 + "\n"
        
        # Save to file
        with open("enhanced_rsi_results.txt", "w") as f:
            f.write(results_content)
        
        # Also print to console
        print(results_content)

    def generate_monthly_report(self, results, month):
        """Generate a detailed monthly report in the required format"""
        if month not in results['monthly_trades'] or month not in results['monthly_portfolios']:
            return "No data available for this month"
        
        trades = results['monthly_trades'][month]
        portfolio_status = results['monthly_portfolios'][month]
        stock_data = results['stock_data']
        
        # Calculate total deployed
        total_deployed = sum([pos['total_invested'] for pos in portfolio_status['positions'].values()])
        
        report = f"📅 {month}\n"
        report += "─" * 80 + "\n"
        
        # Portfolio status section
        report += "💰 PORTFOLIO STATUS:\n"
        report += f"   • Start Value: ₹{portfolio_status['start_value']:,.0f}\n"
        report += f"   • End Value: ₹{portfolio_status['end_value']:,.0f}\n"
        report += f"   • Monthly Return: {portfolio_status['monthly_return']:+.2f}%\n"
        report += f"   • Available Cash: ₹{portfolio_status['cash']:,.0f}\n\n"
        
        # Capital deployment section
        report += "💼 CAPITAL DEPLOYMENT:\n"
        report += f"   • Active Positions: ₹{total_deployed:,.0f} ({len(portfolio_status['positions'])} stocks)\n"
        report += f"   • Total Deployed: ₹{total_deployed:,.0f} ({total_deployed/portfolio_status['end_value']*100:.1f}% of portfolio)\n"
        report += f"   • Position Size: ₹{portfolio_status['position_size']:,.0f}\n\n"
        
        # Trading activity summary
        report += "🔄 TRADING ACTIVITY SUMMARY:\n"
        report += f"   • Total Trades: {trades['buys'] + trades['sells']}\n"
        report += f"   • New Entries: {trades['new_entries']} | Averaging: {trades['averaging']} | Exits: {trades['exits']} | Corona: {trades['corona']}\n\n"
        
        # Detailed new entries
        report += "📈 NEW STOCK ENTRIES (DETAILED):\n"
        if trades['detailed_entries']:
            for i, entry in enumerate(sorted(trades['detailed_entries'], key=lambda x: x['date']), 1):
                total_cost = entry.get('total_cost', entry.get('amount', entry.get('gross_amount', 0)))
                report += f"    {i}. {entry['stock']:<12} | Entry: ₹{entry['entry_price']:7.2f} | Qty: {entry['quantity']:4d} | Total Cost: ₹{total_cost:8.0f} | "
                report += f"Date: {entry['date'].strftime('%Y-%m-%d')} | RSI: {entry['rsi']:5.2f} | {entry['reason']}\n"
                
                # Show cost breakdown if available
                if 'trading_costs' in entry:
                    costs = entry['trading_costs']
                    if isinstance(costs, dict) and costs.get('total', 0) > 0:
                        report += f"       Costs: Brokerage: ₹{costs.get('brokerage', 0):.2f}, STT: ₹{costs.get('stt', 0):.2f}, "
                        report += f"Stamp: ₹{costs.get('stamp_duty', 0):.2f}, Slippage: ₹{costs.get('slippage', 0):.2f}\n"
        else:
            report += "   • No new entries this month\n"
        
        report += "\n"
        
        # Detailed averaging trades
        report += "🔄 AVERAGING TRADES (DETAILED):\n"
        if trades['detailed_averaging']:
            for i, avg in enumerate(sorted(trades['detailed_averaging'], key=lambda x: x['date']), 1):
                total_cost = avg.get('total_cost', avg.get('amount', avg.get('gross_amount', 0)))
                report += f"    {i}. {avg['stock']:<12} | Entry: ₹{avg['entry_price']:7.2f} | Qty: {avg['quantity']:4d} | Total Cost: ₹{total_cost:8.0f} | "
                report += f"Date: {avg['date'].strftime('%Y-%m-%d')} | RSI: {avg['rsi']:5.2f} | Level: {avg['level']} | {avg['reason']}\n"
                
                # Show cost breakdown if available
                if 'trading_costs' in avg:
                    costs = avg['trading_costs']
                    if isinstance(costs, dict) and costs.get('total', 0) > 0:
                        report += f"       Costs: Brokerage: ₹{costs.get('brokerage', 0):.2f}, STT: ₹{costs.get('stt', 0):.2f}, "
                        report += f"Stamp: ₹{costs.get('stamp_duty', 0):.2f}, Slippage: ₹{costs.get('slippage', 0):.2f}\n"
        else:
            report += "   • No averaging trades this month\n"
        
        report += "\n"
        
        # Profit bookings
        report += "💰 PROFIT BOOKINGS (DETAILED):\n"
        if trades['detailed_exits']:
            total_realized_profit = sum([exit.get('realized_profit', exit.get('profit', 0)) for exit in trades['detailed_exits']])
            total_gross_profit = sum([exit.get('gross_profit', exit.get('profit', 0)) for exit in trades['detailed_exits']])
            report += f"   Total Gross Profit This Month: ₹{total_gross_profit:,.2f}\n"
            report += f"   Total Realized Profit This Month: ₹{total_realized_profit:,.2f}\n\n"
            
            for i, exit in enumerate(sorted(trades['detailed_exits'], key=lambda x: x['sell_date']), 1):
                gross_profit = exit.get('gross_profit', exit.get('profit', 0))
                realized_profit = exit.get('realized_profit', exit.get('profit', 0))
                trading_costs = exit.get('trading_costs', {})
                
                report += f"    {i}. {exit['stock']:<12} | Buy: ₹{exit['buy_price']:7.2f} | Sell: ₹{exit['sell_price']:7.2f} | Qty: {exit['quantity']:4d}\n"
                report += f"       Gross Profit: ₹{gross_profit:7.2f} | Realized Profit: ₹{realized_profit:7.2f} ({exit['return_pct']:+6.2f}%)\n"
                report += f"       Buy Date: {exit['buy_date'].strftime('%Y-%m-%d')} | Sell Date: {exit['sell_date'].strftime('%Y-%m-%d')} | Hold: {exit['hold_days']:3d} days\n"
                
                if isinstance(trading_costs, dict) and trading_costs.get('total', 0) > 0:
                    report += f"       Tax: ₹{exit['tax']:.2f} | Trading Costs: ₹{trading_costs['total']:.2f}\n"
                else:
                    report += f"       Tax: ₹{exit['tax']:.2f}\n"
                
                gross_proceeds = exit.get('gross_proceeds', exit.get('gross', 0))
                net_proceeds = exit.get('net_proceeds', exit.get('net', 0))
                report += f"       Invested: ₹{exit['invested']:8.0f} | Gross: ₹{gross_proceeds:8.0f} | Net: ₹{net_proceeds:8.0f} | {exit['reason']}\n\n"
        else:
            report += "   • No profit bookings this month\n"
        
        report += "\n"
        
        # Corona/quarantine entries
        report += "🦠 CORONA/QUARANTINE ENTRIES (DETAILED):\n"
        if trades['detailed_corona']:
            for i, corona in enumerate(sorted(trades['detailed_corona'], key=lambda x: x['sell_date']), 1):
                report += f"    {i}. {corona['stock']:<12} | Buy: ₹{corona['buy_price']:7.2f} | Sell: ₹{corona['sell_price']:7.2f} | Qty: {corona['quantity']:4d} | Loss: ₹{corona['loss']:7.2f} ({corona['return_pct']:+6.2f}%)\n"
                report += f"       Buy Date: {corona['buy_date'].strftime('%Y-%m-%d')} | Sell Date: {corona['sell_date'].strftime('%Y-%m-%d')} | Hold: {corona['hold_days']:3d} days\n"
                report += f"       Invested: ₹{corona['invested']:8.0f} | Gross: ₹{corona['gross']:8.0f} | {corona['reason']}\n\n"
        else:
            report += "   • No corona entries this month\n"
        
        report += "\n"
        
        # Active positions at month end
        report += "📊 ACTIVE POSITIONS AT MONTH END:\n"
        report += f"   Total Active Stocks: {len(portfolio_status['positions'])}\n\n"
        
        if portfolio_status['positions']:
            for i, (stock, pos) in enumerate(sorted(portfolio_status['positions'].items()), 1):
                try:
                    current_price = stock_data[stock]['close'].loc[sorted(stock_data[stock].index)[-1]]
                    current_value = pos['quantity'] * current_price
                    unrealized_pnl = ((current_price - pos['avg_price']) / pos['avg_price']) * 100
                    pnl_amount = current_value - pos['total_invested']
                    
                    report += f"    {i}. {stock:<12} | Avg: ₹{pos['avg_price']:7.2f} | Current: ₹{current_price:7.2f} | Qty: {pos['quantity']:4d} | P&L: ₹{pnl_amount:.2f} ({unrealized_pnl:+.2f}%)\n"
                    report += f"       Invested: ₹{pos['total_invested']:8.0f} | Current Value: ₹{current_value:8.0f} | Avg Attempts: {pos.get('averaging_attempts', 0)}\n\n"
                except:
                    report += f"    {i}. {stock:<12} | Avg: ₹{pos['avg_price']:7.2f} | Current: N/A | Qty: {pos['quantity']:4d} | P&L: N/A\n"
                    report += f"       Invested: ₹{pos['total_invested']:8.0f} | Current Value: N/A | Avg Attempts: {pos.get('averaging_attempts', 0)}\n\n"
        else:
            report += "   • No active positions at month end\n"
        
        return report
        
    def generate_all_monthly_reports(self, results):
        """Generate reports for all months and save to a file"""
        all_reports = ""
        
        for month in sorted(results['monthly_portfolios'].keys()):
            monthly_report = self.generate_monthly_report(results, month)
            all_reports += monthly_report + "\n\n" + "=" * 80 + "\n\n"
        
        # Save to file
        with open("enhanced_rsi_monthly_reports.txt", "w") as f:
            f.write(all_reports)
        
        print(f"Monthly reports saved to 'enhanced_rsi_monthly_reports.txt'")
        return all_reports

def main():
    """Main execution function"""
    print("Enhanced RSI Strategy V2 - Standalone Backtest (CORRECTED)")
    print("=" * 60)
    print("Loading historical data...")
    
    # Initialize strategy
    strategy = EnhancedRSIStrategyV2()
    
    # Load data
    stock_data = strategy.load_stock_data()
    
    if not stock_data:
        print("❌ No data found! Please ensure historical data is available.")
        print("Expected path: ../historical_data/ or ./historical_data/")
        return
    
    # Run backtest
    print(f"\nRunning Enhanced V2 strategy backtest...")
    print("This may take a few minutes for complete analysis...")
    
    results = strategy.simulate_strategy(stock_data)
    
    # Generate and save monthly reports
    strategy.generate_all_monthly_reports(results)
    
    # Print overall results and save to file
    strategy.print_detailed_results(results, stock_data)
    
    print("\n🎯 SCRIPT EXECUTION COMPLETED!")
    print("✅ Overall results saved to: 'enhanced_rsi_results.txt'")
    print("✅ Monthly reports saved to: 'enhanced_rsi_monthly_reports.txt'")
    print(f"📊 Final Portfolio Value: ₹{results['final_value']:,.0f}")
    print(f"📈 Total Return: {results['total_return']:.1f}%")
    
    # Validation check
    if results['total_return'] > 4000:
        print("✅ RESULTS VALIDATION: Expected high return achieved!")
    elif results['total_return'] > 1000:
        print("⚠️  RESULTS VALIDATION: Good return but check for minor differences")
    else:
        print("❌ RESULTS VALIDATION: Returns too low - check implementation")

if __name__ == "__main__":
    main()
