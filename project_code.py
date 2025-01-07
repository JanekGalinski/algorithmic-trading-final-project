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
# 2) Strategy definition class
#-----------------------------------------------------------------------------------------------------------------------------

class AdvancedFeatureBasedStrategyWithStopLoss(bt.Strategy):
    def __init__(self, period_rsi, fast_period_macd, slow_period_macd, signal_window_size, stop_loss_threshold, pca_components=2, forecast_window=25, take_profit_threshold=0.05):
        self.period_rsi = period_rsi
        self.fast_period_macd = fast_period_macd
        self.slow_period_macd = slow_period_macd
        self.signal_window_size = signal_window_size
        self.stop_loss_threshold = stop_loss_threshold
        self.take_profit_threshold = take_profit_threshold
        self.forecast_window = forecast_window
        self.pca_components = pca_components

        # Technical indicators
        self.rsi_indicator = bt.indicators.RSI(self.data.close, period=self.period_rsi)
        self.macd_indicator = bt.indicators.MACD(self.data.close, period_me1=self.fast_period_macd, period_me2=self.slow_period_macd)

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
        if percent_change > 0.01:  
            target_class = 1  # Gain
        elif percent_change < -0.01: 
            target_class = -1  # Loss
        else:
            target_class = 0  # No significant change

        # Collect feature values
        feature_row = [rsi_value, macd_value, macd_signal]

        self.feature_data.append(feature_row)  # Features
        self.target_data.append(target_class)  # Target

        if len(self.feature_data) >= self.signal_window_size:
            # Convert to numpy array
            X = np.array(self.feature_data)
            y = np.array(self.target_data)

            # Standardize features
            X_scaled = self.scaler.fit_transform(X)

            # Apply PCA
            pca = PCA(n_components=min(self.pca_components, X_scaled.shape[1]))
            pca_transformed = pca.fit_transform(X_scaled)

            # Hyperparameter tuning
            param_grid = {
                'C': [0.1, 1.0],
                'penalty': ['l2'],
                'solver': ['liblinear']
            }
            grid_search = GridSearchCV(
                LogisticRegression(max_iter=self.lr_iterations), param_grid, cv=2
            )
            grid_search.fit(X_scaled, y)
            self.lr_model = grid_search.best_estimator_

            # Predict for the next step
            forecast = self.lr_model.predict([X_scaled[-1]])[0]

            if forecast == 1 and not self.position and len(self.broker.positions) < 10:  # Buy condition
                self.initiate_buy()
            elif forecast == -1 and not self.position and len(self.broker.positions) < 10:  # Sell condition
                self.initiate_sell()
            elif forecast == 1 and self.position.size < 0:  # Close short position
                self.close()
                self.initiate_buy()
            elif forecast == -1 and self.position.size > 0:  # Close long position
                self.close()
                self.initiate_sell()

        # Check stop-loss, trailing stop, and take-profit conditions
        if self.position:
            self.evaluate_stop_loss()
            self.evaluate_trailing_stop()
            self.evaluate_take_profit()

    def initiate_buy(self):
        """Handle buy action."""
        self.buy(size=1)
        self.entry_price = self.data.close[0]
        self.trailing_stop_price = self.entry_price * (1 - self.stop_loss_threshold)

    def initiate_sell(self):
        """Handle sell action."""
        self.sell(size=1)
        self.entry_price = self.data.close[0]
        self.trailing_stop_price = self.entry_price * (1 + self.stop_loss_threshold)

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
            new_trailing_stop = self.data.close[0] * (1 - self.stop_loss_threshold)
            if new_trailing_stop > self.trailing_stop_price:
                self.trailing_stop_price = new_trailing_stop
            elif self.data.close[0] < self.trailing_stop_price:
                self.close()
        elif self.position.size < 0:  # Short position
            new_trailing_stop = self.data.close[0] * (1 + self.stop_loss_threshold)
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
