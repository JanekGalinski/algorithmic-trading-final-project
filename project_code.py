#-----------------------------------------------------------------------------------------------------------------------------
#Algorithmic Trading
#MiBDS, 2nd Year, Part-Time
#Academic Year: 2024/2025
#Jan Galiński (40867)
#Individual Work Project
#"Multiple signals strategy"
#-----------------------------------------------------------------------------------------------------------------------------

# %%
#-----------------------------------------------------------------------------------------------------------------------------
# 1) Importing libraries
#-----------------------------------------------------------------------------------------------------------------------------

import backtrader as bt
import numpy as np
import pandas as pd
import yfinance as yf
import datetime
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from itertools import product

# %%
#-----------------------------------------------------------------------------------------------------------------------------
# 2) Strategy definition class with hyperparameters tuning
#-----------------------------------------------------------------------------------------------------------------------------

class AdvancedFeatureBasedStrategyWithStopLoss(bt.Strategy):
    def __init__(self, period_rsi, fast_period_macd, slow_period_macd, signal_window_size, stop_loss_threshold, fast_sma_period=10, slow_sma_period=50, pca_components=2, forecast_window=25, take_profit_threshold=0.05, stochastic_period=14, williams_period=14, macro_volume_period=20, risk_per_trade=0.01):
        self.period_rsi = period_rsi
        self.fast_period_macd = fast_period_macd
        self.slow_period_macd = slow_period_macd
        self.signal_window_size = signal_window_size
        self.stop_loss_threshold = stop_loss_threshold
        self.take_profit_threshold = take_profit_threshold
        self.fast_sma_period = fast_sma_period
        self.slow_sma_period = slow_sma_period
        self.forecast_window = forecast_window
        self.pca_components = pca_components
        self.stochastic_period = stochastic_period
        self.williams_period = williams_period
        self.macro_volume_period = macro_volume_period
        self.risk_per_trade = risk_per_trade

        # Technical indicators
        self.rsi_indicator = bt.indicators.RSI(self.data.close, period=self.period_rsi)
        self.macd_indicator = bt.indicators.MACD(self.data.close, period_me1=self.fast_period_macd, period_me2=self.slow_period_macd)
        self.sma_fast_indicator = bt.indicators.SimpleMovingAverage(self.data.close, period=self.fast_sma_period)
        self.sma_slow_indicator = bt.indicators.SimpleMovingAverage(self.data.close, period=self.slow_sma_period)
        self.stochastic_indicator = bt.indicators.Stochastic(self.data, period=self.stochastic_period)
        self.williams_indicator = bt.indicators.WilliamsR(self.data, period=self.williams_period)
        self.volume_sma_indicator = bt.indicators.SimpleMovingAverage(self.data.volume, period=self.macro_volume_period)
        self.atr_indicator = bt.indicators.ATR(self.data, period=14)
        self.adx_indicator = bt.indicators.ADX(self.data, period=14)

        # Logistic regression parameters
        self.lr_reg_strength = 1.0
        self.lr_penalty_type = 'l2'
        self.lr_optimization = 'liblinear'
        self.lr_iterations = 100

        # Logistic regression model
        self.lr_model = LogisticRegression(
            C=self.lr_reg_strength,
            penalty=self.lr_penalty_type,
            solver=self.lr_optimization,
            max_iter=self.lr_iterations
        )
        self.feature_data = [] 
        self.target_data = []

        self.entry_price = None
        self.trailing_stop_price = None

        self.scaler = StandardScaler()

    def next(self):
        if len(self.data) <= self.forecast_window:
            return

        current_index = len(self)

        remaining_data_points = self.data.buflen() - current_index

        if remaining_data_points <= self.forecast_window:
            return

        # Calculate indicators
        rsi_value = self.rsi_indicator[0]
        macd_value = self.macd_indicator.macd[0]
        macd_signal = self.macd_indicator.signal[0]
        fast_sma_value = self.sma_fast_indicator[0]
        slow_sma_value = self.sma_slow_indicator[0]
        stochastic_value = self.stochastic_indicator[0]
        williams_value = self.williams_indicator[0]
        macro_volume_value = self.volume_sma_indicator[0]
        atr_value = self.atr_indicator[0]
        adx_value = self.adx_indicator[0]

        # Check for valid or non-zero closing price
        if pd.isna(self.data.close[0]) or self.data.close[0] == 0:
            print(f"Invalid close value at index {len(self)}: {self.data.close[0]}. Skipping.")
            return

        # Ensure sufficient data for percentage change calculation
        if len(self.data.close) <= self.forecast_window:
            print(f"Insufficient data for percentage change calculation at index {len(self)}")
            return

        # Compute percentage change
        try:
            percent_change = (self.data.close[self.forecast_window] - self.data.close[0]) / self.data.close[0]
        except ZeroDivisionError:
            print(f"Division by zero at index {len(self)}. Skipping.")
            return

        # Determine target class based on percentage change
        if percent_change > 0.02:  
            target_class = 1  # Gain
        elif percent_change < -0.02: 
            target_class = -1  # Loss
        else:
            target_class = 0  # No significant change

        # Collect feature values
        feature_row = [rsi_value, macd_value, macd_signal, fast_sma_value, slow_sma_value, stochastic_value, williams_value, macro_volume_value, adx_value]

        self.feature_data.append(feature_row)  # Features
        self.target_data.append(target_class)  # Target

        if len(self.feature_data) >= self.signal_window_size:
            # Convert to numpy array
            X = np.array(self.feature_data)
            y = np.array(self.target_data)

            # Standardize features
            X_scaled = self.scaler.fit_transform(X)

            # Apply PCA
            pca = PCA(n_components=self.pca_components)
            X_pca = pca.fit_transform(X_scaled)

            # Hyperparameter tuning
            param_grid = {
                'C': [0.1, 1.0],
                'penalty': ['l2'],
                'solver': ['liblinear']
            }
            grid_search = GridSearchCV(
                LogisticRegression(max_iter=self.lr_iterations), param_grid, cv=2
            )
            grid_search.fit(X_pca, y)
            self.lr_model = grid_search.best_estimator_

            # Predict for the next step
            forecast = self.lr_model.predict([X_pca[-1]])[0]

            # Check trend alignment with SMA and ADX
            trend = fast_sma_value > slow_sma_value
            strong_trend = adx_value > 25

            # Execute trades based on forecast, trend, and ADX
            if forecast == 1 and trend and strong_trend and not self.position and len(self.broker.positions) < 10:  # Buy condition
                self.initiate_buy(atr_value)
            elif forecast == -1 and not trend and strong_trend and not self.position and len(self.broker.positions) < 10:  # Sell condition
                self.initiate_sell(atr_value)
            elif forecast == 1 and self.position.size < 0:  # Close short position
                self.close()
                self.initiate_buy(atr_value)
            elif forecast == -1 and self.position.size > 0:  # Close long position
                self.close()
                self.initiate_sell(atr_value)

        # Check stop-loss, trailing stop, and take-profit conditions
        if self.position:
            self.evaluate_stop_loss()
            self.evaluate_trailing_stop()
            self.evaluate_take_profit()

    def initiate_buy(self, atr_value):
        """Handle buy action."""
        cash_available = self.broker.get_cash()
        position_size = (cash_available * self.risk_per_trade) / atr_value
        position_size = max(1, int(position_size))  # Ensure at least 1 share
        self.buy(size=position_size)
        self.entry_price = self.data.close[0]
        self.trailing_stop_price = self.entry_price - atr_value

    def initiate_sell(self, atr_value):
        """Handle sell action."""
        cash_available = self.broker.get_cash()
        position_size = (cash_available * self.risk_per_trade) / atr_value
        position_size = max(1, int(position_size))  # Ensure at least 1 share
        self.sell(size=position_size)
        self.entry_price = self.data.close[0]
        self.trailing_stop_price = self.entry_price + atr_value

    def evaluate_stop_loss(self):
        """Close position if stop-loss is hit."""
        if self.stop_loss_threshold != 0:
            stop_loss_level = self.entry_price * (1 - self.stop_loss_threshold if self.position.size > 0 else 1 + self.stop_loss_threshold)
            if (self.position.size > 0 and self.data.close[0] < stop_loss_level) or \
               (self.position.size < 0 and self.data.close[0] > stop_loss_level):
                self.close()

    def evaluate_trailing_stop(self):
        """Adjust or trigger trailing stop."""
        if self.position.size > 0:  # Long position
            new_trailing_stop = self.data.close[0] - self.atr_indicator[0]
            if new_trailing_stop > self.trailing_stop_price:
                self.trailing_stop_price = new_trailing_stop
            elif self.data.close[0] < self.trailing_stop_price:
                self.close()
        elif self.position.size < 0:  # Short position
            new_trailing_stop = self.data.close[0] + self.atr_indicator[0]
            if new_trailing_stop < self.trailing_stop_price:
                self.trailing_stop_price = new_trailing_stop
            elif self.data.close[0] > self.trailing_stop_price:
                self.close()

    def evaluate_take_profit(self):
        """Close position if take-profit is hit."""
        if self.take_profit_threshold != 0:
            take_profit_level = self.entry_price * (1 + self.take_profit_threshold if self.position.size > 0 else 1 - self.take_profit_threshold)
            if (self.position.size > 0 and self.data.close[0] >= take_profit_level) or \
               (self.position.size < 0 and self.data.close[0] <= take_profit_level):
                self.close()


# %%
#-----------------------------------------------------------------------------------------------------------------------------
# 3) Backtest
#-----------------------------------------------------------------------------------------------------------------------------

def execute_backtest(strategy_class, strategy_args, stock_symbol, start_dt, end_dt,
                      initial_funds=1000, trade_slippage=0.002, trade_commission=0.004, allocation_percent=10, metrics_enabled=False, enable_plot=True):
    # Initialize Backtrader engine
    backtest_engine = bt.Cerebro()

    # Fetch market data from Yahoo Finance
    market_data = yf.download(stock_symbol, start=start_dt, end=end_dt)

    # Clean and format data
    if 'Adj Close' in market_data.columns:
        market_data = market_data.drop(columns=['Adj Close'])

    market_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Save data (optional)
    csv_name = f"{stock_symbol}_data.csv"
    market_data.to_csv(csv_name)

    # Verify data is in proper format
    if not isinstance(market_data, pd.DataFrame):
        raise ValueError(f"Expected a DataFrame, got: {type(market_data)}")

    # Load data into Backtrader
    data_feed = bt.feeds.PandasData(dataname=market_data)
    backtest_engine.adddata(data_feed)

    # Set initial capital
    backtest_engine.broker.set_cash(initial_funds)

    # Configure commission and slippage
    backtest_engine.broker.setcommission(commission=trade_commission)
    backtest_engine.broker.set_slippage_perc(trade_slippage)

    # Configure position size
    backtest_engine.addsizer(bt.sizers.PercentSizer, percents=allocation_percent)

    # Print initial portfolio value
    print(f"Starting Portfolio Value: {backtest_engine.broker.getvalue()}")

    # Add the strategy with parameters
    backtest_engine.addstrategy(strategy_class, **strategy_args)

    if metrics_enabled:
        # Add analyzers
        backtest_engine.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.03)
        backtest_engine.addanalyzer(bt.analyzers.DrawDown)
        backtest_engine.addanalyzer(bt.analyzers.Transactions, _name="transactions")
        backtest_engine.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trade_analysis")

    # Run backtest
    results = backtest_engine.run()

    # Print ending portfolio value
    ending_value = backtest_engine.broker.getvalue()
    print(f"Ending Portfolio Value: {ending_value}")

    if metrics_enabled:
        # Extract and print metrics
        sharpe_ratio = results[0].analyzers.sharperatio.get_analysis()
        drawdown = results[0].analyzers.drawdown.get_analysis()
        transactions = results[0].analyzers.transactions.get_analysis()
        trade_analysis = results[0].analyzers.trade_analysis.get_analysis()

        print(f"Sharpe Ratio: {sharpe_ratio}")
        print(f"Max Drawdown: {drawdown.max.drawdown}%")
        print(f"Max Drawdown Duration: {drawdown.max.len} days")
        print(f"Transactions: {transactions}")
        print(f"Trade Analysis: {trade_analysis}")

        # Calculate and print return rate
        return_rate = ((ending_value - initial_funds) / initial_funds) * 100
        print(f"Return Rate: {return_rate:.2f}%")

    # Plot results
    if enable_plot:
        backtest_engine.plot()

    return results

# %%
#-----------------------------------------------------------------------------------------------------------------------------
# 4) Strategy parameters with tuning
#-----------------------------------------------------------------------------------------------------------------------------

def tune_strategy(strategy_class, stock_symbol, start_dt, end_dt):
    parameter_grid = {
        "period_rsi": [10, 14],
        "fast_period_macd": [12, 15],
        "slow_period_macd": [26, 30],
        "signal_window_size": [200, 400],
        "stop_loss_threshold": [0.05],
        "fast_sma_period": [5, 10],
        "slow_sma_period": [10, 20],
        "pca_components": [2],
        "forecast_window": [25],
        "take_profit_threshold": [0.02],
        "stochastic_period": [14]
    }

    best_params = None
    best_performance = -float('inf')

    for params in product(*parameter_grid.values()):
        param_dict = dict(zip(parameter_grid.keys(), params))
        print(f"Testing parameters: {param_dict}")

        results = execute_backtest(
            strategy_class=strategy_class,
            strategy_args=param_dict,
            stock_symbol=stock_symbol,
            start_dt=start_dt,
            end_dt=end_dt,
            initial_funds=1000,
            trade_slippage=0.002,
            trade_commission=0.004,
            allocation_percent=10,
            metrics_enabled=False,
            enable_plot=False
        )

        final_value = results[0].broker.getvalue()
        performance = final_value / 1000 - 1

        if performance > best_performance:
            best_performance = performance
            best_params = param_dict

    print(f"Best parameters: {best_params} with performance: {best_performance:.2f}")
    return best_params

# %%
#-----------------------------------------------------------------------------------------------------------------------------
# 5) Data details
#-----------------------------------------------------------------------------------------------------------------------------

def run_strategy_with_tuning():
    best_params = tune_strategy(
        strategy_class=AdvancedFeatureBasedStrategyWithStopLoss,
        stock_symbol="^GSPC",
        start_dt="2000-01-01",
        end_dt="2002-01-01"
    )

    execute_backtest(
        strategy_class=AdvancedFeatureBasedStrategyWithStopLoss,
        strategy_args=best_params,
        stock_symbol="^GSPC",
        start_dt="2000-01-01",
        end_dt="2002-01-01",
        initial_funds=1000,
        trade_slippage=0.002,
        trade_commission=0.004,
        allocation_percent=10,
        metrics_enabled=True
    )

# %%
#-----------------------------------------------------------------------------------------------------------------------------
# 6) Running strategy
#-----------------------------------------------------------------------------------------------------------------------------

run_strategy_with_tuning()

# %%
