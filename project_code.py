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
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# %%
#-----------------------------------------------------------------------------------------------------------------------------
# 2) Strategy definition class with hyperparameters tuning
#-----------------------------------------------------------------------------------------------------------------------------
#Strategy description
#This trading strategy, uses combination of technical indicators (RSI, MACD, SMA, Stochastic, Williams, ADX)
#As well as logistic regression-based forecasting to make trading decisions
#It also leverages machine learning techniques including: PCA for dimensionality reduction and hyperparameter tuning to optimize model performance
#As a result it uses predictive analytics to classify market conditions (gains, losses, or neutral outcomes) and dynamically adjusts behavior 

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
    csv_name = f"{stock_symbol}.csv"
    market_data = pd.read_csv(csv_name, index_col=0, parse_dates=True)

    # If "Adj Close" exists, drop it
    if 'Adj Close' in market_data.columns:
        market_data.drop(columns=['Adj Close'], inplace=True)

    # Rename columns to match expected names
    market_data.rename(columns={'Max': 'High', 'Min': 'Low'}, inplace=True)

    # If Volume does not exist
    if 'Volume' not in market_data.columns:
        market_data['Volume'] = 0

    # Ensure final columns match the sequence expected
    market_data = market_data[['Open', 'High', 'Low', 'Close', 'Volume']]

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
#Thresholds justification
#Stop Loss Threshold
#A stop loss threshold of 5% (0.05) is chosen to limit potential losses on individual trades, this ensures risk management aligns with standard professional trading practices
#It is commonly and wide used threshold which balances providing trades with enough breathing room and limiting downside risk

#Take Profit Threshold
#A 2% take profit threshold ensures that small but consistent gains are taken when strategy accurately forecasts positive trends
#It also minimizes the risk of reversals eroding profits in volatile periods

#Trailing Stop
#In this strategy trailing stop threshold is dynamic - it adjusts based on the ATR (Average True Range) to secure gains while allowing to capitalize on favorable trends
#It enables adapting to market volatility, avoiding too early exits or over-tight stops, which is aligned with modern risk-adjusted trading practices

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

    initial_funds = 1000
    for params in product(*parameter_grid.values()):
        param_dict = dict(zip(parameter_grid.keys(), params))
        print(f"Testing parameters: {param_dict}")

        results = execute_backtest(
            strategy_class=strategy_class,
            strategy_args=param_dict,
            stock_symbol=stock_symbol,
            start_dt=start_dt,
            end_dt=end_dt,
            initial_funds=initial_funds,
            trade_slippage=0.002,
            trade_commission=0.004,
            allocation_percent=10,
            metrics_enabled=False,
            enable_plot=False
        )

        final_value = results[0].broker.getvalue()
        # performance calculation
        performance = (final_value - initial_funds) / initial_funds

        if performance > best_performance:
            best_performance = performance
            best_params = param_dict

    print(f"Best parameters: {best_params} with performance: {best_performance:.2f}")
    return best_params

# %%
#-----------------------------------------------------------------------------------------------------------------------------
# 5) Data details
#-----------------------------------------------------------------------------------------------------------------------------
#Choose particular asset
#usdjpy_w
#1990-01-07
#2009-03-01

#btc_v_w
#2010-07-18
#2025-02-02

#ge_us_m
#1962-01-31
#2025-01-31

#wig20_w
#2005-12-11
#2025-02-02

#zw=f_copper
#2000-07-17
#2019-09-02


def run_strategy_with_tuning():
    best_params = tune_strategy(
        strategy_class=AdvancedFeatureBasedStrategyWithStopLoss,
        stock_symbol="wig20_w",
        start_dt="2005-12-11",
        end_dt="2025-02-02"
    )

    execute_backtest(
        strategy_class=AdvancedFeatureBasedStrategyWithStopLoss,
        strategy_args=best_params,
        stock_symbol="wig20_w",
        start_dt="2005-12-11",
        end_dt="2025-02-02",
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
# %%
#-----------------------------------------------------------------------------------------------------------------------------
# 7) Testing comments
#-----------------------------------------------------------------------------------------------------------------------------

#Justification of Strategy Effectiveness
#Integration of multiple indicators (RSI for momentum, MACD for trend, ADX for trend strength) (momentum, trend, volatility, and volume indicators) creates a holistic market assessment, improving decision accuracy
#Using trend strength (ADX) and aligning it with SMA crossovers ensures trades are executed in strong, favorable trends
#Combination of trend-following and oscillating indicators reduces likelihood of false signals and is popular modern practice in trading strategies
#Logistic regression introduces a predictive layer to strategy, enhancing ability to forecast market movements based on historical patterns
#PCA simplifies complex data structures, improving model efficiency
#Hyperparameter tuning ensures optimal performance across different market environments
#Use of stop loss, take profit, and trailing stops mechanisms ensures that risk is effecitvely controlled, simulating real life scenarios
#Tuning of strategy parameters ensures optimal performance across different market conditions, periods and asset classes

#Limitations of strategy
#Please note that due to limitations of used PC compute power - the hyperparameters of regression as well as strategy parameters tuning process is highly limited and cannot fully leverage all possible optimization opportunities
#It is also worth to note that strategy has embedded preset factors algined with task requirements (slippage, commission threshold as well as basic trading rules)
#As a result it might limit the generalizibity of findings

#Strategy testing results and findings
#Portfolio performs reasonably well in different market conditions, time periods and with use of different assets, including:

#Index
#^GSPC - S&P 500 Index
#January 2000- January 2002 (2 years)
#Sharpe Ratio: 0.36
#Max Drawdown: 3.6%
#Return Rate: 10.07%

#Commodity Future
#GC=F Gold Future
#January 2009 - January 2012 (3 years)
#Sharpe Ratio: 0.21
#Max Drawdown: 6.7%
#Return Rate: 14.56%

#Individual stock
#AAPL - Apple Stock
#January 2005 - January 2008 (3 years)
#Sharpe Ratio: 0.36
#Max Drawdown: 6.41%
#Return Rate: 19.79%

#ETF
#VONG - Vanguard Russell 1000 Growth Index Fund ETF Shares
#January 2019 - January 2021 (2 years)
#Sharpe Ratio: 0.7
#Max Drawdown: 6.21%
#Return Rate: 21.82%

#Across different timeperiods and different assets, strategy keeps positive Sharpe Ratios in the range of 0.21 and 0.7
#It indicates that trades using this strategy provide returns above the given risk-free rate 
#What is more the Max Drawdowns vary between 3.6% and 6.7% suggesting that there are no very risky trades resulting in very big losses in this strategy
#Finally the Return Rates are between 10.07% to 21.82%
#Looking at Sharpe Ratios, the best performance looks to be noted at ETF case and the worst in case of Commodity Future
#It is worth to mention that similarly to parameters tuning, testing of strategy has certain limitations stemming from used PC compute power, limiting generalizibity of findings

# %%
#-----------------------------------------------------------------------------------------------------------------------------
# 8) Final results
#-----------------------------------------------------------------------------------------------------------------------------

#usdjpy_w
#1990-01-07
#2009-03-01
#Sharpe Ratio: -3.34
#Max Drawdown: 6.29%
#Return Rate: 0.32%
#Negative Sharpe indicates poor risk-adjusted performance, but drawdown is low. Overall returns are small but positive

#btc_v_w
#2010-07-18
#2025-02-02
#Sharpe Ratio: 0.35
#Max Drawdown: 101.05%
#Return Rate: 832.86%
#Positive Sharpe and extremely high returns, but the strategy experiences massive drawdowns probably due to volatility of this asset

#ge_us_m
#1962-01-31
#2025-01-31
#Sharpe Ratio: -1.61
#Max Drawdown: 6.701992144960003%
#Return Rate: 29.61%
#Negative Sharpe indicates poor risk-adjusted performance. However, drawdown is relatively small and the absolute return is positive, especially taking into account long time period

#wig20_w
#2005-12-11
#2025-02-02
#Sharpe Ratio: -0.04
#Max Drawdown: 15.64%
#Return Rate: 54.13%
#The near-zero Sharpe shows minimal risk-adjusted gains, but the absolute return is very high. Drawdown is moderate

#zw=f_copper
#2000-07-17
#2019-09-02
#Sharpe Ratio: -0.48
#Max Drawdown: 31.49%
#Return Rate: -4.33%
#Negative Sharpe and a substantial drawdown suggest the strategy struggled, ending with a net loss

#To summarize across all assets, the strategy demonstrates mixed performance
#Bitcoin shows the most extreme combination of high return and high drawdown
#Currencies and equities (USD/JPY, GE, WIG20) tend to have lower drawdowns but also lower risk-adjusted returns
#Copper stands out for both a negative Sharpe and negative overall return, indicating particularly poor performance in that market
#By looking at Sharpe ratio in 4/5 cases it is negative, indicating strategy did not performed very well when looking at risk-adjusted performance
#At the same time in 4/5 strategy provided relatively strong returns
