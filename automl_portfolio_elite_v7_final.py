"""
AutoML Portfolio Elite v7 - Integrated with yfinance
Sistema completo de análise de portfólio de ações com coleta de dados via yfinance
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

# Optimization imports
from scipy.optimize import minimize
import cvxpy as cp

# Statistical imports
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import ta

# Configuration
st.set_page_config(
    page_title="AutoML Portfolio Elite v7",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #2ca02c;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== YFINANCE DATA COLLECTION ====================

class YFinanceDataCollector:
    """Classe para coletar dados usando yfinance"""
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_stock_data(ticker, start_date, end_date):
        """Coleta dados históricos de preços"""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            if df.empty:
                st.warning(f"Nenhum dado encontrado para {ticker}")
                return None
            return df
        except Exception as e:
            st.error(f"Erro ao coletar dados para {ticker}: {str(e)}")
            return None
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_stock_info(ticker):
        """Coleta informações fundamentalistas"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return info
        except Exception as e:
            st.error(f"Erro ao coletar informações para {ticker}: {str(e)}")
            return {}
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_financials(ticker):
        """Coleta dados financeiros"""
        try:
            stock = yf.Ticker(ticker)
            financials = {
                'income_statement': stock.financials,
                'balance_sheet': stock.balance_sheet,
                'cash_flow': stock.cashflow,
                'quarterly_financials': stock.quarterly_financials
            }
            return financials
        except Exception as e:
            st.error(f"Erro ao coletar financials para {ticker}: {str(e)}")
            return {}
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_dividends(ticker):
        """Coleta histórico de dividendos"""
        try:
            stock = yf.Ticker(ticker)
            dividends = stock.dividends
            return dividends
        except Exception as e:
            return pd.Series()
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_multiple_stocks(tickers, start_date, end_date):
        """Coleta dados para múltiplas ações"""
        data = {}
        progress_bar = st.progress(0)
        for i, ticker in enumerate(tickers):
            df = YFinanceDataCollector.get_stock_data(ticker, start_date, end_date)
            if df is not None:
                data[ticker] = df
            progress_bar.progress((i + 1) / len(tickers))
        progress_bar.empty()
        return data

# ==================== TECHNICAL INDICATORS ====================

class TechnicalIndicators:
    """Classe para calcular indicadores técnicos"""
    
    @staticmethod
    def add_all_indicators(df):
        """Adiciona todos os indicadores técnicos ao dataframe"""
        df = df.copy()
        
        # Moving Averages
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
        df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_High'] = bollinger.bollinger_hband()
        df['BB_Low'] = bollinger.bollinger_lband()
        df['BB_Mid'] = bollinger.bollinger_mavg()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # ATR
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        
        # ADX
        df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
        
        # OBV
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        
        # CCI
        df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'])
        
        # Williams %R
        df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
        
        # ROC
        df['ROC'] = ta.momentum.roc(df['Close'])
        
        # Returns
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Volatility
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        return df

# ==================== PORTFOLIO METRICS ====================

class PortfolioMetrics:
    """Classe para calcular métricas de portfólio"""
    
    @staticmethod
    def calculate_returns(prices):
        """Calcula retornos"""
        return prices.pct_change().dropna()
    
    @staticmethod
    def calculate_cumulative_returns(returns):
        """Calcula retornos cumulativos"""
        return (1 + returns).cumprod() - 1
    
    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
        """Calcula Sharpe Ratio"""
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    @staticmethod
    def calculate_sortino_ratio(returns, risk_free_rate=0.02):
        """Calcula Sortino Ratio"""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        return np.sqrt(252) * excess_returns.mean() / downside_std
    
    @staticmethod
    def calculate_max_drawdown(returns):
        """Calcula Maximum Drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    @staticmethod
    def calculate_calmar_ratio(returns):
        """Calcula Calmar Ratio"""
        annual_return = (1 + returns.mean()) ** 252 - 1
        max_dd = abs(PortfolioMetrics.calculate_max_drawdown(returns))
        return annual_return / max_dd if max_dd != 0 else 0
    
    @staticmethod
    def calculate_var(returns, confidence=0.95):
        """Calcula Value at Risk"""
        return np.percentile(returns, (1 - confidence) * 100)
    
    @staticmethod
    def calculate_cvar(returns, confidence=0.95):
        """Calcula Conditional Value at Risk"""
        var = PortfolioMetrics.calculate_var(returns, confidence)
        return returns[returns <= var].mean()
    
    @staticmethod
    def calculate_beta(returns, market_returns):
        """Calcula Beta"""
        covariance = np.cov(returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        return covariance / market_variance
    
    @staticmethod
    def calculate_alpha(returns, market_returns, risk_free_rate=0.02):
        """Calcula Alpha"""
        beta = PortfolioMetrics.calculate_beta(returns, market_returns)
        return returns.mean() - (risk_free_rate / 252 + beta * (market_returns.mean() - risk_free_rate / 252))
    
    @staticmethod
    def calculate_information_ratio(returns, benchmark_returns):
        """Calcula Information Ratio"""
        active_returns = returns - benchmark_returns
        tracking_error = active_returns.std()
        return np.sqrt(252) * active_returns.mean() / tracking_error

# ==================== PORTFOLIO OPTIMIZATION ====================

class PortfolioOptimizer:
    """Classe para otimização de portfólio"""
    
    def __init__(self, returns):
        self.returns = returns
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        self.num_assets = len(returns.columns)
    
    def portfolio_performance(self, weights):
        """Calcula performance do portfólio"""
        returns = np.sum(self.mean_returns * weights) * 252
        std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
        return returns, std
    
    def negative_sharpe(self, weights, risk_free_rate=0.02):
        """Sharpe Ratio negativo para minimização"""
        returns, std = self.portfolio_performance(weights)
        return -(returns - risk_free_rate) / std
    
    def max_sharpe_ratio(self, risk_free_rate=0.02):
        """Otimização para máximo Sharpe Ratio"""
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_guess = np.array([1 / self.num_assets] * self.num_assets)
        
        result = minimize(
            self.negative_sharpe,
            initial_guess,
            args=(risk_free_rate,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
    
    def min_volatility(self):
        """Otimização para mínima volatilidade"""
        def portfolio_volatility(weights):
            return self.portfolio_performance(weights)[1]
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_guess = np.array([1 / self.num_assets] * self.num_assets)
        
        result = minimize(
            portfolio_volatility,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
    
    def efficient_frontier(self, num_portfolios=100):
        """Calcula fronteira eficiente"""
        results = np.zeros((3, num_portfolios))
        weights_record = []
        
        for i in range(num_portfolios):
            weights = np.random.random(self.num_assets)
            weights /= np.sum(weights)
            weights_record.append(weights)
            
            portfolio_return, portfolio_std = self.portfolio_performance(weights)
            results[0, i] = portfolio_std
            results[1, i] = portfolio_return
            results[2, i] = (portfolio_return - 0.02) / portfolio_std
        
        return results, weights_record
    
    def risk_parity(self):
        """Otimização Risk Parity"""
        def risk_contribution(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
            marginal_contrib = np.dot(self.cov_matrix * 252, weights)
            risk_contrib = weights * marginal_contrib / portfolio_vol
            return risk_contrib
        
        def risk_parity_objective(weights):
            risk_contrib = risk_contribution(weights)
            target = np.ones(self.num_assets) / self.num_assets
            return np.sum((risk_contrib - target) ** 2)
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_guess = np.array([1 / self.num_assets] * self.num_assets)
        
        result = minimize(
            risk_parity_objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
    
    def black_litterman(self, market_caps, views, view_confidences):
        """Modelo Black-Litterman"""
        # Implementação simplificada
        tau = 0.05
        market_weights = market_caps / market_caps.sum()
        
        # Prior returns
        pi = tau * np.dot(self.cov_matrix, market_weights)
        
        # Posterior returns
        omega = np.diag(view_confidences)
        P = np.eye(self.num_assets)
        
        bl_returns = pi + np.dot(
            np.dot(tau * self.cov_matrix, P.T),
            np.dot(
                np.linalg.inv(np.dot(P, np.dot(tau * self.cov_matrix, P.T)) + omega),
                views - np.dot(P, pi)
            )
        )
        
        return bl_returns

# ==================== MACHINE LEARNING MODELS ====================

class MLModels:
    """Classe para modelos de Machine Learning"""
    
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0),
            'SVR': SVR(kernel='rbf'),
            'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42)
        }
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_score = -np.inf
    
    def prepare_features(self, df):
        """Prepara features para ML"""
        feature_columns = [
            'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26',
            'MACD', 'MACD_Signal', 'RSI', 'BB_High', 'BB_Low',
            'Stoch_K', 'Stoch_D', 'ATR', 'ADX', 'OBV',
            'CCI', 'Williams_R', 'ROC', 'Volatility'
        ]
        
        df_ml = df[feature_columns + ['Close']].dropna()
        
        X = df_ml[feature_columns]
        y = df_ml['Close'].shift(-1).dropna()
        X = X.iloc[:-1]
        
        return X, y
    
    def train_models(self, X, y, test_size=0.2):
        """Treina todos os modelos"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        for name, model in self.models.items():
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'model': model,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'predictions': y_pred,
                    'actual': y_test
                }
                
                if r2 > self.best_score:
                    self.best_score = r2
                    self.best_model = model
                    
            except Exception as e:
                st.warning(f"Erro ao treinar {name}: {str(e)}")
        
        return results, X_test, y_test
    
    def predict_next(self, df, days=30):
        """Prediz próximos dias"""
        if self.best_model is None:
            return None
        
        X, _ = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        predictions = []
        last_features = X_scaled[-1:].copy()
        
        for _ in range(days):
            pred = self.best_model.predict(last_features)[0]
            predictions.append(pred)
            # Atualiza features (simplificado)
            last_features = np.roll(last_features, -1)
            last_features[0, -1] = pred
        
        return predictions
    
    def hyperparameter_tuning(self, X, y, model_name='Random Forest'):
        """Otimização de hiperparâmetros"""
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3]
            }
        }
        
        if model_name not in param_grids:
            return None
        
        X_scaled = self.scaler.fit_transform(X)
        
        grid_search = GridSearchCV(
            self.models[model_name],
            param_grids[model_name],
            cv=5,
            scoring='r2',
            n_jobs=-1
        )
        
        grid_search.fit(X_scaled, y)
        
        return grid_search.best_params_, grid_search.best_score_

# ==================== RISK MANAGEMENT ====================

class RiskManager:
    """Classe para gerenciamento de risco"""
    
    @staticmethod
    def calculate_position_size(capital, risk_per_trade, stop_loss_pct):
        """Calcula tamanho da posição"""
        risk_amount = capital * risk_per_trade
        position_size = risk_amount / stop_loss_pct
        return position_size
    
    @staticmethod
    def kelly_criterion(win_rate, avg_win, avg_loss):
        """Calcula Kelly Criterion"""
        if avg_loss == 0:
            return 0
        win_loss_ratio = avg_win / avg_loss
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        return max(0, kelly)
    
    @staticmethod
    def calculate_stop_loss(entry_price, atr, multiplier=2):
        """Calcula stop loss baseado em ATR"""
        return entry_price - (atr * multiplier)
    
    @staticmethod
    def calculate_take_profit(entry_price, atr, multiplier=3):
        """Calcula take profit baseado em ATR"""
        return entry_price + (atr * multiplier)
    
    @staticmethod
    def portfolio_heat(positions, capital):
        """Calcula exposição total do portfólio"""
        total_risk = sum([pos['risk'] for pos in positions])
        return total_risk / capital
    
    @staticmethod
    def correlation_risk(returns_df):
        """Analisa risco de correlação"""
        correlation_matrix = returns_df.corr()
        avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
        return correlation_matrix, avg_correlation
    
    @staticmethod
    def stress_test(returns, scenarios):
        """Testa portfólio em cenários de stress"""
        results = {}
        for scenario_name, scenario_returns in scenarios.items():
            portfolio_return = (returns * scenario_returns).sum()
            results[scenario_name] = portfolio_return
        return results

# ==================== BACKTESTING ====================

class Backtester:
    """Classe para backtesting de estratégias"""
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = []
        self.trades = []
        self.equity_curve = []
    
    def run_strategy(self, df, strategy_func):
        """Executa estratégia de backtesting"""
        self.capital = self.initial_capital
        self.positions = []
        self.trades = []
        self.equity_curve = [self.initial_capital]
        
        for i in range(len(df)):
            signal = strategy_func(df.iloc[:i+1])
            
            if signal == 'BUY' and len(self.positions) == 0:
                shares = self.capital // df.iloc[i]['Close']
                cost = shares * df.iloc[i]['Close']
                self.positions.append({
                    'entry_price': df.iloc[i]['Close'],
                    'shares': shares,
                    'entry_date': df.index[i]
                })
                self.capital -= cost
                
            elif signal == 'SELL' and len(self.positions) > 0:
                position = self.positions.pop()
                revenue = position['shares'] * df.iloc[i]['Close']
                self.capital += revenue
                
                profit = revenue - (position['shares'] * position['entry_price'])
                self.trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': df.index[i],
                    'entry_price': position['entry_price'],
                    'exit_price': df.iloc[i]['Close'],
                    'shares': position['shares'],
                    'profit': profit,
                    'return': profit / (position['shares'] * position['entry_price'])
                })
            
            # Calcula equity
            position_value = sum([p['shares'] * df.iloc[i]['Close'] for p in self.positions])
            total_equity = self.capital + position_value
            self.equity_curve.append(total_equity)
        
        return self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calcula métricas de performance"""
        if not self.trades:
            return {}
        
        trades_df = pd.DataFrame(self.trades)
        
        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital
        num_trades = len(self.trades)
        winning_trades = len(trades_df[trades_df['profit'] > 0])
        losing_trades = len(trades_df[trades_df['profit'] < 0])
        win_rate = winning_trades / num_trades if num_trades > 0 else 0
        
        avg_win = trades_df[trades_df['profit'] > 0]['profit'].mean() if winning_trades > 0 else 0
        avg_loss = abs(trades_df[trades_df['profit'] < 0]['profit'].mean()) if losing_trades > 0 else 0
        profit_factor = avg_win / avg_loss if avg_loss != 0 else 0
        
        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0
        
        cumulative = equity_series / self.initial_capital
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'final_capital': self.equity_curve[-1]
        }

# ==================== VISUALIZATION ====================

class Visualizer:
    """Classe para visualizações"""
    
    @staticmethod
    def plot_candlestick(df, title="Candlestick Chart"):
        """Plota gráfico de candlestick"""
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        )])
        
        fig.update_layout(
            title=title,
            yaxis_title='Price',
            xaxis_title='Date',
            template='plotly_white',
            height=600
        )
        
        return fig
    
    @staticmethod
    def plot_with_indicators(df, indicators=['SMA_20', 'SMA_50']):
        """Plota preço com indicadores"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=('Price & Indicators', 'Volume', 'RSI')
        )
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Indicators
        for indicator in indicators:
            if indicator in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[indicator], name=indicator, line=dict(width=1)),
                    row=1, col=1
                )
        
        # Volume
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='lightblue'),
            row=2, col=1
        )
        
        # RSI
        if 'RSI' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')),
                row=3, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        fig.update_layout(
            height=900,
            template='plotly_white',
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    @staticmethod
    def plot_correlation_matrix(corr_matrix):
        """Plota matriz de correlação"""
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='Correlation Matrix',
            template='plotly_white',
            height=600
        )
        
        return fig
    
    @staticmethod
    def plot_efficient_frontier(results, max_sharpe_weights, min_vol_weights, optimizer):
        """Plota fronteira eficiente"""
        fig = go.Figure()
        
        # Scatter de portfólios aleatórios
        fig.add_trace(go.Scatter(
            x=results[0],
            y=results[1],
            mode='markers',
            marker=dict(
                size=5,
                color=results[2],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Sharpe Ratio")
            ),
            name='Random Portfolios'
        ))
        
        # Máximo Sharpe
        max_sharpe_return, max_sharpe_std = optimizer.portfolio_performance(max_sharpe_weights)
        fig.add_trace(go.Scatter(
            x=[max_sharpe_std],
            y=[max_sharpe_return],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            name='Max Sharpe Ratio'
        ))
        
        # Mínima Volatilidade
        min_vol_return, min_vol_std = optimizer.portfolio_performance(min_vol_weights)
        fig.add_trace(go.Scatter(
            x=[min_vol_std],
            y=[min_vol_return],
            mode='markers',
            marker=dict(size=15, color='green', symbol='star'),
            name='Min Volatility'
        ))
        
        fig.update_layout(
            title='Efficient Frontier',
            xaxis_title='Volatility (Std Dev)',
            yaxis_title='Expected Return',
            template='plotly_white',
            height=600
        )
        
        return fig
    
    @staticmethod
    def plot_portfolio_composition(weights, tickers):
        """Plota composição do portfólio"""
        fig = go.Figure(data=[go.Pie(
            labels=tickers,
            values=weights,
            hole=0.3,
            textinfo='label+percent',
            textposition='auto'
        )])
        
        fig.update_layout(
            title='Portfolio Composition',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    @staticmethod
    def plot_equity_curve(equity_curve):
        """Plota curva de equity"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=equity_curve,
            mode='lines',
            name='Equity',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title='Equity Curve',
            xaxis_title='Time',
            yaxis_title='Portfolio Value',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    @staticmethod
    def plot_drawdown(equity_curve):
        """Plota drawdown"""
        equity_series = pd.Series(equity_curve)
        cumulative = equity_series / equity_series.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=drawdown * 100,
            mode='lines',
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title='Drawdown',
            xaxis_title='Time',
            yaxis_title='Drawdown (%)',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    @staticmethod
    def plot_returns_distribution(returns):
        """Plota distribuição de retornos"""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=returns * 100,
            nbinsx=50,
            name='Returns',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title='Returns Distribution',
            xaxis_title='Returns (%)',
            yaxis_title='Frequency',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    @staticmethod
    def plot_ml_predictions(actual, predicted, title="ML Predictions"):
        """Plota predições do modelo"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=actual,
            mode='lines',
            name='Actual',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            y=predicted,
            mode='lines',
            name='Predicted',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Price',
            template='plotly_white',
            height=500
        )
        
        return fig

# ==================== MAIN APPLICATION ====================

def main():
    """Função principal da aplicação"""
    
    # Header
    st.title("📊 AutoML Portfolio Elite v7 - Powered by yfinance")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data Collection
        st.subheader("📈 Data Collection")
        tickers_input = st.text_area(
            "Stock Tickers (one per line)",
            value="AAPL\nMSFT\nGOOG\nAMZN\nTSLA",
            height=150
        )
        tickers = [t.strip().upper() for t in tickers_input.split('\n') if t.strip()]
        
        # Date Range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365*2)
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now()
            )
        
        # Benchmark
        benchmark_ticker = st.text_input("Benchmark Ticker", value="^GSPC")
        
        # Risk Parameters
        st.subheader("⚠️ Risk Parameters")
        risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 2.0) / 100
        confidence_level = st.slider("VaR Confidence Level (%)", 90, 99, 95) / 100
        
        # ML Parameters
        st.subheader("🤖 ML Parameters")
        test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
        prediction_days = st.slider("Prediction Days", 7, 90, 30)
        
        # Load Data Button
        load_data = st.button("🔄 Load Data", type="primary", use_container_width=True)
    
    # Main Content
    if load_data:
        with st.spinner("Loading data from yfinance..."):
            # Collect data
            data_collector = YFinanceDataCollector()
            stock_data = data_collector.get_multiple_stocks(tickers, start_date, end_date)
            
            if not stock_data:
                st.error("No data collected. Please check your tickers and try again.")
                return
            
            # Store in session state
            st.session_state['stock_data'] = stock_data
            st.session_state['tickers'] = tickers
            st.session_state['start_date'] = start_date
            st.session_state['end_date'] = end_date
            st.session_state['benchmark_ticker'] = benchmark_ticker
            st.session_state['risk_free_rate'] = risk_free_rate
            st.session_state['confidence_level'] = confidence_level
            st.session_state['test_size'] = test_size
            st.session_state['prediction_days'] = prediction_days
            
            st.success(f"✅ Data loaded successfully for {len(stock_data)} stocks!")
    
    # Check if data is loaded
    if 'stock_data' not in st.session_state:
        st.info("👈 Please configure parameters and load data from the sidebar to begin analysis.")
        return
    
    # Get data from session state
    stock_data = st.session_state['stock_data']
    tickers = st.session_state['tickers']
    risk_free_rate = st.session_state['risk_free_rate']
    confidence_level = st.session_state['confidence_level']
    test_size = st.session_state['test_size']
    prediction_days = st.session_state['prediction_days']
    
    # Tabs
    tabs = st.tabs([
        "📊 Overview",
        "📈 Technical Analysis",
        "🤖 Machine Learning",
        "💼 Portfolio Optimization",
        "⚠️ Risk Management",
        "🔄 Backtesting",
        "📉 Advanced Analytics"
    ])
    
    # ==================== TAB 1: OVERVIEW ====================
    with tabs[0]:
        st.header("Portfolio Overview")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Stocks", len(stock_data))
        
        with col2:
            total_days = sum([len(df) for df in stock_data.values()])
            st.metric("Total Data Points", f"{total_days:,}")
        
        with col3:
            date_range = (st.session_state['end_date'] - st.session_state['start_date']).days
            st.metric("Date Range (days)", date_range)
        
        with col4:
            st.metric("Benchmark", st.session_state['benchmark_ticker'])
        
        st.markdown("---")
        
        # Stock Information
        st.subheader("📋 Stock Information")
        
        info_data = []
        for ticker in tickers:
            if ticker in stock_data:
                info = YFinanceDataCollector.get_stock_info(ticker)
                df = stock_data[ticker]
                
                info_data.append({
                    'Ticker': ticker,
                    'Company': info.get('longName', 'N/A'),
                    'Sector': info.get('sector', 'N/A'),
                    'Industry': info.get('industry', 'N/A'),
                    'Market Cap': f"${info.get('marketCap', 0) / 1e9:.2f}B" if info.get('marketCap') else 'N/A',
                    'Current Price': f"${df['Close'].iloc[-1]:.2f}",
                    'YTD Return': f"{((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100:.2f}%"
                })
        
        info_df = pd.DataFrame(info_data)
        st.dataframe(info_df, use_container_width=True)
        
        st.markdown("---")
        
        # Price Performance
        st.subheader("📈 Price Performance")
        
        # Normalize prices
        normalized_prices = pd.DataFrame()
        for ticker, df in stock_data.items():
            normalized_prices[ticker] = df['Close'] / df['Close'].iloc[0] * 100
        
        fig = go.Figure()
        for ticker in normalized_prices.columns:
            fig.add_trace(go.Scatter(
                x=normalized_prices.index,
                y=normalized_prices[ticker],
                mode='lines',
                name=ticker
            ))
        
        fig.update_layout(
            title='Normalized Price Performance (Base 100)',
            xaxis_title='Date',
            yaxis_title='Normalized Price',
            template='plotly_white',
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Returns Distribution
        st.subheader("📊 Returns Distribution")
        
        returns_data = pd.DataFrame()
        for ticker, df in stock_data.items():
            returns_data[ticker] = df['Close'].pct_change()
        
        fig = go.Figure()
        for ticker in returns_data.columns:
            fig.add_trace(go.Box(
                y=returns_data[ticker] * 100,
                name=ticker,
                boxmean='sd'
            ))
        
        fig.update_layout(
            title='Daily Returns Distribution',
            yaxis_title='Returns (%)',
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation Matrix
        st.subheader("🔗 Correlation Matrix")
        
        corr_matrix = returns_data.corr()
        fig = Visualizer.plot_correlation_matrix(corr_matrix)
        st.plotly_chart(fig, use_container_width=True)
    
    # ==================== TAB 2: TECHNICAL ANALYSIS ====================
    with tabs[1]:
        st.header("Technical Analysis")
        
        # Select stock
        selected_ticker = st.selectbox("Select Stock", tickers, key='tech_ticker')
        
        if selected_ticker in stock_data:
            df = stock_data[selected_ticker].copy()
            
            # Add technical indicators
            with st.spinner("Calculating technical indicators..."):
                df = TechnicalIndicators.add_all_indicators(df)
            
            # Candlestick with indicators
            st.subheader("📊 Price Chart with Indicators")
            
            indicators_to_plot = st.multiselect(
                "Select Indicators",
                ['SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26', 'BB_High', 'BB_Low'],
                default=['SMA_20', 'SMA_50']
            )
            
            fig = Visualizer.plot_with_indicators(df, indicators_to_plot)
            st.plotly_chart(fig, use_container_width=True)
            
            # Indicator Values
            st.subheader("📈 Current Indicator Values")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")
                st.metric("MACD", f"{df['MACD'].iloc[-1]:.4f}")
            
            with col2:
                st.metric("Stochastic K", f"{df['Stoch_K'].iloc[-1]:.2f}")
                st.metric("Stochastic D", f"{df['Stoch_D'].iloc[-1]:.2f}")
            
            with col3:
                st.metric("ATR", f"{df['ATR'].iloc[-1]:.2f}")
                st.metric("ADX", f"{df['ADX'].iloc[-1]:.2f}")
            
            with col4:
                st.metric("CCI", f"{df['CCI'].iloc[-1]:.2f}")
                st.metric("Williams %R", f"{df['Williams_R'].iloc[-1]:.2f}")
            
            # Trading Signals
            st.subheader("🎯 Trading Signals")
            
            signals = []
            
            # RSI Signal
            rsi_value = df['RSI'].iloc[-1]
            if rsi_value < 30:
                signals.append(("RSI", "🟢 OVERSOLD - Buy Signal", rsi_value))
            elif rsi_value > 70:
                signals.append(("RSI", "🔴 OVERBOUGHT - Sell Signal", rsi_value))
            else:
                signals.append(("RSI", "🟡 NEUTRAL", rsi_value))
            
            # MACD Signal
            if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
                signals.append(("MACD", "🟢 BULLISH - Buy Signal", df['MACD'].iloc[-1]))
            else:
                signals.append(("MACD", "🔴 BEARISH - Sell Signal", df['MACD'].iloc[-1]))
            
            # Moving Average Signal
            if df['Close'].iloc[-1] > df['SMA_50'].iloc[-1]:
                signals.append(("MA Cross", "🟢 ABOVE SMA50 - Bullish", df['Close'].iloc[-1]))
            else:
                signals.append(("MA Cross", "🔴 BELOW SMA50 - Bearish", df['Close'].iloc[-1]))
            
            signals_df = pd.DataFrame(signals, columns=['Indicator', 'Signal', 'Value'])
            st.dataframe(signals_df, use_container_width=True)
    
    # ==================== TAB 3: MACHINE LEARNING ====================
    with tabs[2]:
        st.header("Machine Learning Predictions")
        
        # Select stock
        selected_ticker = st.selectbox("Select Stock", tickers, key='ml_ticker')
        
        if selected_ticker in stock_data:
            df = stock_data[selected_ticker].copy()
            
            # Add technical indicators
            with st.spinner("Preparing data..."):
                df = TechnicalIndicators.add_all_indicators(df)
            
            # Train models
            st.subheader("🤖 Model Training")
            
            if st.button("Train Models", type="primary"):
                with st.spinner("Training models..."):
                    ml_models = MLModels()
                    X, y = ml_models.prepare_features(df)
                    
                    results, X_test, y_test = ml_models.train_models(X, y, test_size)
                    
                    # Store in session state
                    st.session_state['ml_results'] = results
                    st.session_state['ml_models'] = ml_models
                    st.session_state['ml_ticker'] = selected_ticker
                    
                    st.success("✅ Models trained successfully!")
            
            # Display results
            if 'ml_results' in st.session_state and st.session_state['ml_ticker'] == selected_ticker:
                results = st.session_state['ml_results']
                ml_models = st.session_state['ml_models']
                
                # Model Comparison
                st.subheader("📊 Model Comparison")
                
                comparison_data = []
                for name, result in results.items():
                    comparison_data.append({
                        'Model': name,
                        'R² Score': f"{result['r2']:.4f}",
                        'RMSE': f"{result['rmse']:.4f}",
                        'MAE': f"{result['mae']:.4f}"
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df = comparison_df.sort_values('R² Score', ascending=False)
                st.dataframe(comparison_df, use_container_width=True)
                
                # Best Model
                best_model_name = max(results.items(), key=lambda x: x[1]['r2'])[0]
                st.success(f"🏆 Best Model: {best_model_name} (R² = {results[best_model_name]['r2']:.4f})")
                
                # Predictions Plot
                st.subheader("📈 Predictions vs Actual")
                
                best_result = results[best_model_name]
                fig = Visualizer.plot_ml_predictions(
                    best_result['actual'].values,
                    best_result['predictions'],
                    title=f"{best_model_name} Predictions"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Future Predictions
                st.subheader("🔮 Future Predictions")
                
                if st.button("Generate Future Predictions"):
                    with st.spinner("Generating predictions..."):
                        future_predictions = ml_models.predict_next(df, prediction_days)
                        
                        if future_predictions:
                            # Create future dates
                            last_date = df.index[-1]
                            future_dates = pd.date_range(
                                start=last_date + timedelta(days=1),
                                periods=prediction_days,
                                freq='D'
                            )
                            
                            # Plot
                            fig = go.Figure()
                            
                            # Historical
                            fig.add_trace(go.Scatter(
                                x=df.index[-60:],
                                y=df['Close'].iloc[-60:],
                                mode='lines',
                                name='Historical',
                                line=dict(color='blue')
                            ))
                            
                            # Predictions
                            fig.add_trace(go.Scatter(
                                x=future_dates,
                                y=future_predictions,
                                mode='lines',
                                name='Predictions',
                                line=dict(color='red', dash='dash')
                            ))
                            
                            fig.update_layout(
                                title=f'{selected_ticker} - {prediction_days} Days Forecast',
                                xaxis_title='Date',
                                yaxis_title='Price',
                                template='plotly_white',
                                height=500
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Prediction table
                            pred_df = pd.DataFrame({
                                'Date': future_dates,
                                'Predicted Price': future_predictions
                            })
                            st.dataframe(pred_df, use_container_width=True)
    
    # ==================== TAB 4: PORTFOLIO OPTIMIZATION ====================
    with tabs[3]:
        st.header("Portfolio Optimization")
        
        # Calculate returns
        returns_data = pd.DataFrame()
        for ticker, df in stock_data.items():
            returns_data[ticker] = df['Close'].pct_change()
        returns_data = returns_data.dropna()
        
        # Optimization
        st.subheader("🎯 Optimization Methods")
        
        optimization_method = st.selectbox(
            "Select Optimization Method",
            ["Maximum Sharpe Ratio", "Minimum Volatility", "Risk Parity", "Efficient Frontier"]
        )
        
        if st.button("Optimize Portfolio", type="primary"):
            with st.spinner("Optimizing portfolio..."):
                optimizer = PortfolioOptimizer(returns_data)
                
                if optimization_method == "Maximum Sharpe Ratio":
                    weights = optimizer.max_sharpe_ratio(risk_free_rate)
                    method_name = "Maximum Sharpe Ratio"
                    
                elif optimization_method == "Minimum Volatility":
                    weights = optimizer.min_volatility()
                    method_name = "Minimum Volatility"
                    
                elif optimization_method == "Risk Parity":
                    weights = optimizer.risk_parity()
                    method_name = "Risk Parity"
                    
                else:  # Efficient Frontier
                    results, weights_record = optimizer.efficient_frontier(num_portfolios=1000)
                    max_sharpe_weights = optimizer.max_sharpe_ratio(risk_free_rate)
                    min_vol_weights = optimizer.min_volatility()
                    
                    fig = Visualizer.plot_efficient_frontier(
                        results, max_sharpe_weights, min_vol_weights, optimizer
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    weights = max_sharpe_weights
                    method_name = "Efficient Frontier (Max Sharpe)"
                
                # Store results
                st.session_state['optimal_weights'] = weights
                st.session_state['optimization_method'] = method_name
                st.session_state['optimizer'] = optimizer
                
                st.success(f"✅ Portfolio optimized using {method_name}")
        
        # Display results
        if 'optimal_weights' in st.session_state:
            weights = st.session_state['optimal_weights']
            optimizer = st.session_state['optimizer']
            method_name = st.session_state['optimization_method']
            
            # Portfolio Composition
            st.subheader("📊 Portfolio Composition")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Pie chart
                fig = Visualizer.plot_portfolio_composition(weights, tickers)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Weights table
                weights_df = pd.DataFrame({
                    'Ticker': tickers,
                    'Weight': weights,
                    'Percentage': [f"{w*100:.2f}%" for w in weights]
                })
                weights_df = weights_df.sort_values('Weight', ascending=False)
                st.dataframe(weights_df, use_container_width=True)
            
            # Portfolio Metrics
            st.subheader("📈 Portfolio Metrics")
            
            portfolio_return, portfolio_std = optimizer.portfolio_performance(weights)
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Expected Annual Return", f"{portfolio_return*100:.2f}%")
            
            with col2:
                st.metric("Annual Volatility", f"{portfolio_std*100:.2f}%")
            
            with col3:
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            
            with col4:
                st.metric("Optimization Method", method_name)
            
            # Portfolio Performance
            st.subheader("📊 Portfolio Performance")
            
            # Calculate portfolio returns
            portfolio_returns = (returns_data * weights).sum(axis=1)
            cumulative_returns = PortfolioMetrics.calculate_cumulative_returns(portfolio_returns)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns * 100,
                mode='lines',
                name='Portfolio',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title='Cumulative Returns',
                xaxis_title='Date',
                yaxis_title='Cumulative Return (%)',
                template='plotly_white',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional Metrics
            st.subheader("📊 Additional Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sortino = PortfolioMetrics.calculate_sortino_ratio(portfolio_returns, risk_free_rate)
                st.metric("Sortino Ratio", f"{sortino:.2f}")
                
                max_dd = PortfolioMetrics.calculate_max_drawdown(portfolio_returns)
                st.metric("Max Drawdown", f"{max_dd*100:.2f}%")
            
            with col2:
                calmar = PortfolioMetrics.calculate_calmar_ratio(portfolio_returns)
                st.metric("Calmar Ratio", f"{calmar:.2f}")
                
                var = PortfolioMetrics.calculate_var(portfolio_returns, confidence_level)
                st.metric(f"VaR ({confidence_level*100:.0f}%)", f"{var*100:.2f}%")
            
            with col3:
                cvar = PortfolioMetrics.calculate_cvar(portfolio_returns, confidence_level)
                st.metric(f"CVaR ({confidence_level*100:.0f}%)", f"{cvar*100:.2f}%")
                
                annual_return = (1 + portfolio_returns.mean()) ** 252 - 1
                st.metric("Annualized Return", f"{annual_return*100:.2f}%")
    
    # ==================== TAB 5: RISK MANAGEMENT ====================
    with tabs[4]:
        st.header("Risk Management")
        
        # Calculate returns
        returns_data = pd.DataFrame()
        for ticker, df in stock_data.items():
            returns_data[ticker] = df['Close'].pct_change()
        returns_data = returns_data.dropna()
        
        # Risk Metrics
        st.subheader("⚠️ Risk Metrics")
        
        risk_metrics = []
        for ticker in tickers:
            if ticker in returns_data.columns:
                returns = returns_data[ticker]
                
                volatility = returns.std() * np.sqrt(252)
                var = PortfolioMetrics.calculate_var(returns, confidence_level)
                cvar = PortfolioMetrics.calculate_cvar(returns, confidence_level)
                max_dd = PortfolioMetrics.calculate_max_drawdown(returns)
                
                risk_metrics.append({
                    'Ticker': ticker,
                    'Volatility': f"{volatility*100:.2f}%",
                    f'VaR ({confidence_level*100:.0f}%)': f"{var*100:.2f}%",
                    f'CVaR ({confidence_level*100:.0f}%)': f"{cvar*100:.2f}%",
                    'Max Drawdown': f"{max_dd*100:.2f}%"
                })
        
        risk_df = pd.DataFrame(risk_metrics)
        st.dataframe(risk_df, use_container_width=True)
        
        # Correlation Risk
        st.subheader("🔗 Correlation Risk")
        
        corr_matrix, avg_corr = RiskManager.correlation_risk(returns_data)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = Visualizer.plot_correlation_matrix(corr_matrix)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("Average Correlation", f"{avg_corr:.2f}")
            
            if avg_corr > 0.7:
                st.warning("⚠️ High correlation - Portfolio may lack diversification")
            elif avg_corr < 0.3:
                st.success("✅ Low correlation - Good diversification")
            else:
                st.info("ℹ️ Moderate correlation")
        
        # Position Sizing
        st.subheader("💰 Position Sizing Calculator")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            capital = st.number_input("Portfolio Capital ($)", value=100000, step=1000)
        
        with col2:
            risk_per_trade = st.slider("Risk per Trade (%)", 0.5, 5.0, 2.0) / 100
        
        with col3:
            stop_loss_pct = st.slider("Stop Loss (%)", 1.0, 10.0, 5.0) / 100
        
        position_size = RiskManager.calculate_position_size(capital, risk_per_trade, stop_loss_pct)
        
        st.success(f"💵 Recommended Position Size: ${position_size:,.2f}")
        
        # Kelly Criterion
        st.subheader("🎲 Kelly Criterion")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            win_rate = st.slider("Win Rate (%)", 30, 80, 55) / 100
        
        with col2:
            avg_win = st.number_input("Average Win ($)", value=1000, step=100)
        
        with col3:
            avg_loss = st.number_input("Average Loss ($)", value=500, step=100)
        
        kelly = RiskManager.kelly_criterion(win_rate, avg_win, avg_loss)
        kelly_half = kelly / 2  # Conservative Kelly
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Full Kelly", f"{kelly*100:.2f}%")
        
        with col2:
            st.metric("Half Kelly (Conservative)", f"{kelly_half*100:.2f}%")
        
        if kelly > 0.25:
            st.warning("⚠️ Kelly suggests high position size - Consider using Half Kelly")
        
        # Drawdown Analysis
        st.subheader("📉 Drawdown Analysis")
        
        selected_ticker = st.selectbox("Select Stock for Drawdown", tickers, key='dd_ticker')
        
        if selected_ticker in stock_data:
            df = stock_data[selected_ticker]
            returns = df['Close'].pct_change().dropna()
            
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown * 100,
                mode='lines',
                fill='tozeroy',
                name='Drawdown',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title=f'{selected_ticker} Drawdown',
                xaxis_title='Date',
                yaxis_title='Drawdown (%)',
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            max_dd = drawdown.min()
            st.metric("Maximum Drawdown", f"{max_dd*100:.2f}%")
    
    # ==================== TAB 6: BACKTESTING ====================
    with tabs[5]:
        st.header("Strategy Backtesting")
        
        # Select stock
        selected_ticker = st.selectbox("Select Stock", tickers, key='backtest_ticker')
        
        if selected_ticker in stock_data:
            df = stock_data[selected_ticker].copy()
            df = TechnicalIndicators.add_all_indicators(df)
            
            # Strategy Selection
            st.subheader("📋 Strategy Selection")
            
            strategy_type = st.selectbox(
                "Select Strategy",
                ["SMA Crossover", "RSI Strategy", "MACD Strategy", "Bollinger Bands"]
            )
            
            # Strategy Parameters
            st.subheader("⚙️ Strategy Parameters")
            
            if strategy_type == "SMA Crossover":
                col1, col2 = st.columns(2)
                with col1:
                    fast_period = st.slider("Fast SMA Period", 5, 50, 20)
                with col2:
                    slow_period = st.slider("Slow SMA Period", 20, 200, 50)
                
                def strategy(data):
                    if len(data) < slow_period:
                        return 'HOLD'
                    fast_sma = data['Close'].rolling(fast_period).mean().iloc[-1]
                    slow_sma = data['Close'].rolling(slow_period).mean().iloc[-1]
                    prev_fast = data['Close'].rolling(fast_period).mean().iloc[-2]
                    prev_slow = data['Close'].rolling(slow_period).mean().iloc[-2]
                    
                    if fast_sma > slow_sma and prev_fast <= prev_slow:
                        return 'BUY'
                    elif fast_sma < slow_sma and prev_fast >= prev_slow:
                        return 'SELL'
                    return 'HOLD'
            
            elif strategy_type == "RSI Strategy":
                col1, col2 = st.columns(2)
                with col1:
                    rsi_oversold = st.slider("RSI Oversold", 20, 40, 30)
                with col2:
                    rsi_overbought = st.slider("RSI Overbought", 60, 80, 70)
                
                def strategy(data):
                    if 'RSI' not in data.columns or len(data) < 2:
                        return 'HOLD'
                    rsi = data['RSI'].iloc[-1]
                    prev_rsi = data['RSI'].iloc[-2]
                    
                    if rsi < rsi_oversold and prev_rsi >= rsi_oversold:
                        return 'BUY'
                    elif rsi > rsi_overbought and prev_rsi <= rsi_overbought:
                        return 'SELL'
                    return 'HOLD'
            
            elif strategy_type == "MACD Strategy":
                def strategy(data):
                    if 'MACD' not in data.columns or len(data) < 2:
                        return 'HOLD'
                    macd = data['MACD'].iloc[-1]
                    signal = data['MACD_Signal'].iloc[-1]
                    prev_macd = data['MACD'].iloc[-2]
                    prev_signal = data['MACD_Signal'].iloc[-2]
                    
                    if macd > signal and prev_macd <= prev_signal:
                        return 'BUY'
                    elif macd < signal and prev_macd >= prev_signal:
                        return 'SELL'
                    return 'HOLD'
            
            else:  # Bollinger Bands
                def strategy(data):
                    if 'BB_High' not in data.columns or len(data) < 2:
                        return 'HOLD'
                    close = data['Close'].iloc[-1]
                    bb_low = data['BB_Low'].iloc[-1]
                    bb_high = data['BB_High'].iloc[-1]
                    prev_close = data['Close'].iloc[-2]
                    
                    if close < bb_low and prev_close >= bb_low:
                        return 'BUY'
                    elif close > bb_high and prev_close <= bb_high:
                        return 'SELL'
                    return 'HOLD'
            
            # Initial Capital
            initial_capital = st.number_input(
                "Initial Capital ($)",
                value=100000,
                step=10000
            )
            
            # Run Backtest
            if st.button("Run Backtest", type="primary"):
                with st.spinner("Running backtest..."):
                    backtester = Backtester(initial_capital)
                    metrics = backtester.run_strategy(df, strategy)
                    
                    # Store results
                    st.session_state['backtest_metrics'] = metrics
                    st.session_state['backtest_equity'] = backtester.equity_curve
                    st.session_state['backtest_trades'] = backtester.trades
                    
                    st.success("✅ Backtest completed!")
            
            # Display Results
            if 'backtest_metrics' in st.session_state:
                metrics = st.session_state['backtest_metrics']
                equity_curve = st.session_state['backtest_equity']
                trades = st.session_state['backtest_trades']
                
                # Performance Metrics
                st.subheader("📊 Performance Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Return", f"{metrics['total_return']*100:.2f}%")
                    st.metric("Number of Trades", metrics['num_trades'])
                
                with col2:
                    st.metric("Win Rate", f"{metrics['win_rate']*100:.2f}%")
                    st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
                
                with col3:
                    st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                    st.metric("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%")
                
                with col4:
                    st.metric("Avg Win", f"${metrics['avg_win']:.2f}")
                    st.metric("Avg Loss", f"${metrics['avg_loss']:.2f}")
                
                # Equity Curve
                st.subheader("📈 Equity Curve")
                fig = Visualizer.plot_equity_curve(equity_curve)
                st.plotly_chart(fig, use_container_width=True)
                
                # Drawdown
                st.subheader("📉 Drawdown")
                fig = Visualizer.plot_drawdown(equity_curve)
                st.plotly_chart(fig, use_container_width=True)
                
                # Trade History
                st.subheader("📋 Trade History")
                if trades:
                    trades_df = pd.DataFrame(trades)
                    trades_df['profit'] = trades_df['profit'].apply(lambda x: f"${x:.2f}")
                    trades_df['return'] = trades_df['return'].apply(lambda x: f"{x*100:.2f}%")
                    st.dataframe(trades_df, use_container_width=True)
                else:
                    st.info("No trades executed")
    
    # ==================== TAB 7: ADVANCED ANALYTICS ====================
    with tabs[6]:
        st.header("Advanced Analytics")
        
        # Monte Carlo Simulation
        st.subheader("🎲 Monte Carlo Simulation")
        
        selected_ticker = st.selectbox("Select Stock", tickers, key='mc_ticker')
        
        col1, col2 = st.columns(2)
        with col1:
            num_simulations = st.slider("Number of Simulations", 100, 10000, 1000)
        with col2:
            forecast_days = st.slider("Forecast Days", 30, 365, 90)
        
        if st.button("Run Monte Carlo Simulation"):
            if selected_ticker in stock_data:
                with st.spinner("Running simulations..."):
                    df = stock_data[selected_ticker]
                    returns = df['Close'].pct_change().dropna()
                    
                    last_price = df['Close'].iloc[-1]
                    mean_return = returns.mean()
                    std_return = returns.std()
                    
                    # Run simulations
                    simulations = np.zeros((forecast_days, num_simulations))
                    
                    for i in range(num_simulations):
                        prices = [last_price]
                        for j in range(forecast_days):
                            price = prices[-1] * (1 + np.random.normal(mean_return, std_return))
                            prices.append(price)
                        simulations[:, i] = prices[1:]
                    
                    # Plot
                    fig = go.Figure()
                    
                    # Plot sample paths
                    for i in range(min(100, num_simulations)):
                        fig.add_trace(go.Scatter(
                            y=simulations[:, i],
                            mode='lines',
                            line=dict(width=0.5),
                            opacity=0.3,
                            showlegend=False
                        ))
                    
                    # Plot mean
                    fig.add_trace(go.Scatter(
                        y=simulations.mean(axis=1),
                        mode='lines',
                        name='Mean',
                        line=dict(color='red', width=3)
                    ))
                    
                    # Confidence intervals
                    fig.add_trace(go.Scatter(
                        y=np.percentile(simulations, 95, axis=1),
                        mode='lines',
                        name='95th Percentile',
                        line=dict(color='green', dash='dash')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        y=np.percentile(simulations, 5, axis=1),
                        mode='lines',
                        name='5th Percentile',
                        line=dict(color='orange', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title=f'{selected_ticker} Monte Carlo Simulation ({num_simulations} paths)',
                        xaxis_title='Days',
                        yaxis_title='Price',
                        template='plotly_white',
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistics
                    final_prices = simulations[-1, :]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Mean Final Price", f"${final_prices.mean():.2f}")
                    
                    with col2:
                        st.metric("Median Final Price", f"${np.median(final_prices):.2f}")
                    
                    with col3:
                        st.metric("5th Percentile", f"${np.percentile(final_prices, 5):.2f}")
                    
                    with col4:
                        st.metric("95th Percentile", f"${np.percentile(final_prices, 95):.2f}")
        
        st.markdown("---")
        
        # Seasonality Analysis
        st.subheader("📅 Seasonality Analysis")
        
        selected_ticker = st.selectbox("Select Stock", tickers, key='season_ticker')
        
        if selected_ticker in stock_data:
            df = stock_data[selected_ticker].copy()
            
            # Monthly returns
            df['Month'] = df.index.month
            df['Year'] = df.index.year
            df['Returns'] = df['Close'].pct_change()
            
            monthly_returns = df.groupby('Month')['Returns'].mean() * 100
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                y=monthly_returns.values,
                marker_color=['green' if x > 0 else 'red' for x in monthly_returns.values]
            ))
            
            fig.update_layout(
                title=f'{selected_ticker} Average Monthly Returns',
                xaxis_title='Month',
                yaxis_title='Average Return (%)',
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Day of week analysis
            df['DayOfWeek'] = df.index.dayofweek
            dow_returns = df.groupby('DayOfWeek')['Returns'].mean() * 100
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                y=dow_returns.values,
                marker_color=['green' if x > 0 else 'red' for x in dow_returns.values]
            ))
            
            fig.update_layout(
                title=f'{selected_ticker} Average Returns by Day of Week',
                xaxis_title='Day',
                yaxis_title='Average Return (%)',
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Volatility Analysis
        st.subheader("📊 Volatility Analysis")
        
        # Calculate rolling volatility for all stocks
        volatility_data = pd.DataFrame()
        for ticker, df in stock_data.items():
            returns = df['Close'].pct_change()
            volatility_data[ticker] = returns.rolling(window=30).std() * np.sqrt(252) * 100
        
        fig = go.Figure()
        for ticker in volatility_data.columns:
            fig.add_trace(go.Scatter(
                x=volatility_data.index,
                y=volatility_data[ticker],
                mode='lines',
                name=ticker
            ))
        
        fig.update_layout(
            title='30-Day Rolling Volatility',
            xaxis_title='Date',
            yaxis_title='Annualized Volatility (%)',
            template='plotly_white',
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Current volatility comparison
        current_vol = volatility_data.iloc[-1].sort_values(ascending=False)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=current_vol.index,
            y=current_vol.values,
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title='Current Volatility Comparison',
            xaxis_title='Ticker',
            yaxis_title='Volatility (%)',
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ==================== RUN APPLICATION ====================

if __name__ == "__main__":
    main()
