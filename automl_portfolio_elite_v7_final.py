# =============================================================================
# SISTEMA DE ANÁLISE E OTIMIZAÇÃO DE PORTFÓLIO DE INVESTIMENTOS
# =============================================================================
# Descrição: Unificação: Lógica de Otimização (portfolio_analyzer.py) + Dashboard (v4.py)
# =============================================================================

# =============================================================================
# IMPORTAÇÕES E CONFIGURAÇÕES INICIAIS (Baseado em portfolio_analyzer.py)
# =============================================================================

import warnings
import numpy as np
import pandas as pd
import subprocess
import sys
import streamlit as st
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.stats import zscore 
from sklearn.model_selection import TimeSeriesSplit, cross_val_score # Mantido do v4 para ML
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Desativa avisos desnecessários
warnings.filterwarnings("ignore")

# =============================================================================
# CONSTANTES GLOBAIS DO SISTEMA (Baseado em portfolio_analyzer.py)
# =============================================================================

RISK_FREE_RATE = 0.1475 # Taxa livre de risco
MIN_WEIGHT = 0.10       # Peso mínimo por ativo
MAX_WEIGHT = 0.30       # Peso máximo por ativo
NUM_ASSETS_IN_PORTFOLIO = 5
FIXED_ANALYSIS_PERIOD = '2y'
MIN_HISTORY_DAYS = 60

# MAPEAMENTOS DE PONTUAÇÃO (Mantido do portfolio_analyzer.py)
SCORE_MAP = {
    'CT: Concordo Totalmente': 5, 'C: Concordo': 4, 'N: Neutro': 3,
    'D: Discordo': 2, 'DT: Discordo Totalmente': 1, 'A': 5, 'B': 3, 'C': 1
}
SCORE_MAP_INV = {
    'CT: Concordo Totalmente': 1, 'C: Concordo': 2, 'N: Neutro': 3,
    'D: Discordo': 4, 'DT: Discordo Totalmente': 5
}
SCORE_MAP_CONHECIMENTO = {
    'A: Avançado': 5, 'B: Intermediário': 3, 'C: Iniciante': 1
}
SCORE_MAP_REACTION = {
    'A: Venderia': 1, 'B: Manteria': 3, 'C: Compraria mais': 5
}

# LISTA DE ATIVOS DISPONÍVEIS (Baseado em v4.py para maior cobertura)
ATIVOS_POR_SETOR = {
    'Bens Industriais': ['ALOS3.SA', 'AZZA3.SA', 'CMIN3.SA', 'CURY3.SA', 'VAMO3.SA'],
    'Consumo Cíclico': ['CVCB3.SA', 'CYRE3.SA', 'DIRR3.SA', 'LREN3.SA', 'MGLU3.SA', 'RENT3.SA', 'YDUQ3.SA'],
    'Consumo não Cíclico': ['ABEV3.SA', 'ASAI3.SA', 'BEEF3.SA', 'NATU3.SA', 'PCAR3.SA', 'VIVA3.SA'],
    'Financeiro': ['BBSE3.SA', 'BBDC3.SA', 'BBDC4.SA', 'BPAC11.SA', 'BBAS3.SA', 'ITSA4.SA', 'ITUB4.SA', 'IRBR3.SA', 'PSSA3.SA', 'SANB11.SA'],
    'Materiais Básicos': ['BRAP4.SA', 'BRKM5.SA', 'CSNA3.SA', 'GGBR4.SA', 'GOAU4.SA', 'SUZB3.SA', 'USIM5.SA', 'VALE3.SA'],
    'Petróleo, Gás e Biocombustíveis': ['AURE3.SA', 'PRIO3.SA', 'PETR3.SA', 'PETR4.SA', 'RECV3.SA', 'UGPA3.SA', 'VBBR3.SA'],
    'Saúde': ['FLRY3.SA', 'HAPV3.SA', 'HYPE3.SA', 'RADL3.SA', 'RDOR3.SA'],
    'Tecnologia da Informação': ['TOTS3.SA', 'WEGE3.SA'],
    'Telecomunicações': ['TIMS3.SA', 'VIVT3.SA'],
    'Utilidade Pública': ['BRAV3.SA', 'CMIG4.SA', 'CPLE6.SA', 'CPFE3.SA', 'ELET3.SA', 'ELET6.SA', 'ENGI11.SA', 'ENEV3.SA', 'EQTL3.SA', 'EGIE3.SA', 'SBSP3.SA', 'TAEE11.SA']
}
TODOS_ATIVOS = sorted(list(set([ativo for ativos in ATIVOS_POR_SETOR.values() for ativo in ativos])))
ATIVOS_IBOVESPA = sorted(list(set([a for a in TODOS_ATIVOS if a.endswith('.SA')])))

# =============================================================================
# CLASSE: ANALISADOR DE PERFIL DO INVESTIDOR (De portfolio_analyzer.py)
# =============================================================================

class InvestorProfileAnalyzer:
    """
    Classe responsável por analisar o perfil de risco do investidor
    baseado em respostas de questionário e determinar horizonte temporal.
    """
    
    def __init__(self):
        """Inicializa o analisador com valores padrão."""
        self.risk_level = ""
        self.time_horizon = ""
        self.ml_lookback_days = 5

    def determine_risk_level(self, score):
        if score <= 18:
            return "CONSERVADOR"
        elif score <= 30:
            return "INTERMEDIÁRIO"
        elif score <= 45:
            return "MODERADO"
        elif score <= 60:
            return "MODERADO-ARROJADO"
        else:
            return "AVANÇADO"

    def determine_ml_lookback(self, liquidity_key, purpose_key):
        time_map = {
            'A': 5,    # Curto prazo: 5 dias
            'B': 20,   # Médio prazo: 20 dias
            'C': 30    # Longo prazo: 30 dias
        }
        
        final_lookback = max(
            time_map.get(liquidity_key, 5),
            time_map.get(purpose_key, 5)
        )
        
        if final_lookback >= 30:
            self.time_horizon = "LONGO PRAZO"
            self.ml_lookback_days = 30
        elif final_lookback >= 20:
            self.time_horizon = "MÉDIO PRAZO"
            self.ml_lookback_days = 20
        else:
            self.time_horizon = "CURTO PRAZO"
            self.ml_lookback_days = 5

        return self.time_horizon, self.ml_lookback_days
        
    def calculate_profile(self, risk_answers):
        score = (
            SCORE_MAP[risk_answers['risk_accept']] * 5 +
            SCORE_MAP[risk_answers['max_gain']] * 5 +
            SCORE_MAP_INV[risk_answers['stable_growth']] * 5 +
            SCORE_MAP_INV[risk_answers['avoid_loss']] * 5 +
            SCORE_MAP_CONHECIMENTO[risk_answers['level']] * 3 +
            SCORE_MAP_REACTION[risk_answers['reaction']] * 3
        )
        
        risk_level = self.determine_risk_level(score)
        
        time_horizon, ml_lookback = self.determine_ml_lookback(
            risk_answers['liquidity'],
            risk_answers['time_purpose']
        )
        
        return risk_level, time_horizon, ml_lookback, score

# =============================================================================
# FUNÇÕES DE ANÁLISE (De portfolio_analyzer.py)
# =============================================================================

def calculate_technical_indicators(df):
    """Calcula indicadores técnicos (RSI, MACD, Volatilidade, SMA)"""
    df['Returns'] = df['Close'].pct_change()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
    df['Momentum'] = df['Close'] / df['Close'].shift(10) - 1
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    return df.dropna(subset=['SMA_200', 'MACD', 'RSI'])

def calculate_risk_metrics(returns):
    """Calcula métricas de risco e retorno (Retorno Anual, Volatilidade, Sharpe, Max Drawdown)"""
    annual_return = returns.mean() * 252
    annual_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = (annual_return - RISK_FREE_RATE) / annual_volatility if annual_volatility > 0 else 0
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    return {
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }

# =============================================================================
# CLASSE: OTIMIZADOR DE PORTFÓLIO (MARKOWITZ - De portfolio_analyzer.py)
# =============================================================================

class PortfolioOptimizer:
    """Implementa otimização de portfólio usando teoria moderna de Markowitz."""

    def __init__(self, returns_df):
        self.returns = returns_df
        self.mean_returns = returns_df.mean() * 252
        self.cov_matrix = returns_df.cov() * 252
        self.num_assets = len(returns_df.columns)

    def portfolio_stats(self, weights):
        p_return = np.dot(weights, self.mean_returns)
        p_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return p_return, p_std

    def negative_sharpe(self, weights):
        p_return, p_std = self.portfolio_stats(weights)
        return -(p_return - RISK_FREE_RATE) / p_std if p_std != 0 else -100

    def minimize_volatility(self, weights):
        return self.portfolio_stats(weights)[1]

    def optimize(self, strategy):
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((MIN_WEIGHT, MAX_WEIGHT) for _ in range(self.num_assets))
        initial_guess = np.array([1/self.num_assets] * self.num_assets)

        if strategy == 'MinVolatility':
            objective = self.minimize_volatility
        elif strategy == 'MaxSharpe':
            objective = self.negative_sharpe
        else:
            return None

        try:
            result = minimize(
                objective,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                return {asset: weight for asset, weight in zip(self.returns.columns, result.x)}
            else:
                return None
        except Exception:
            return None

# =============================================================================
# CLASSE PRINCIPAL: CONSTRUTOR DE PORTFÓLIO (De portfolio_analyzer.py)
# =============================================================================

class PortfolioBuilder:
    """Orquestra coleta, cálculo de features, ML, pontuação, seleção e otimização."""
    
    def __init__(self, investment_amount, period=FIXED_ANALYSIS_PERIOD):
        self.investment_amount = investment_amount
        self.period = period
        
        self.data_by_asset = {}
        self.performance_data = pd.DataFrame()
        self.fundamental_data = pd.DataFrame()
        self.ml_predictions = {}
        self.successful_assets = []
        self.combined_scores = pd.DataFrame()
        
        self.portfolio_metrics = {}
        self.portfolio_allocation = {}
        self.selected_assets = []
        self.current_allocation_method = "Não Aplicado"
        self.selection_justifications = {}
        self.dashboard_profile = {}
        self.current_weights = {}

    def get_asset_sector(self, symbol):
        """Retorna o setor do ativo (adaptado do v4.py para usar a lista global)."""
        for sector, assets in ATIVOS_POR_SETOR.items():
            if symbol in assets:
                return sector
        return 'Unknown'
        
    def collect_market_data(self, symbols):
        """
        Coleta dados de mercado para uma lista de símbolos. (Função adaptada para Streamlit/simplificação).
        """
        self.successful_assets = []
        self.data_by_asset = {}
        fundamentals_list = []
        
        print(f"INICIANDO COLETA DE DADOS - {len(symbols)} ativos")
        
        for symbol in tqdm(symbols, desc=f"[Coleta {FIXED_ANALYSIS_PERIOD}] Baixando dados"):
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=self.period)
                info = ticker.info

                if not hist.empty and len(hist) >= MIN_HISTORY_DAYS:
                    df = calculate_technical_indicators(hist.copy())
                    
                    if not df.empty and len(df) >= MIN_HISTORY_DAYS:
                        self.data_by_asset[symbol] = df
                        self.successful_assets.append(symbol)
                        
                        sector = self.get_asset_sector(symbol)
                        
                        fundamentals_list.append({
                            'Ticker': symbol,
                            'Sector': sector,
                            'PE_Ratio': info.get('trailingPE', np.nan),
                            'PB_Ratio': info.get('priceToBook', np.nan),
                            'Div_Yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else np.nan,
                            'ROE': info.get('returnOnEquity', np.nan) * 100 if info.get('returnOnEquity') else np.nan
                        })
                        
            except Exception:
                continue

        if len(self.successful_assets) < NUM_ASSETS_IN_PORTFOLIO:
            print(f"ERRO: Apenas {len(self.successful_assets)} ativos coletados. Necessário: {NUM_ASSETS_IN_PORTFOLIO}")
            return False

        # Prepara DataFrame de dados fundamentalistas
        self.fundamental_data = pd.DataFrame(fundamentals_list).set_index('Ticker').loc[self.successful_assets]
        self.fundamental_data = self.fundamental_data.replace([np.inf, -np.inf], np.nan)
        self.fundamental_data.fillna(self.fundamental_data.median(numeric_only=True), inplace=True)
        self.fundamental_data.fillna(0, inplace=True)

        # Calcula métricas de performance para cada ativo
        metrics = {
            s: calculate_risk_metrics(self.data_by_asset[s]['Returns'])
            for s in self.successful_assets
            if 'Returns' in self.data_by_asset[s]
        }
        self.performance_data = pd.DataFrame(metrics).T
        
        return True

    def calculate_cross_sectional_features(self):
        # Implementação simplificada do portfolio_analyzer.py
        df_fund = self.fundamental_data.copy()
        
        sector_means = df_fund.groupby('Sector')[['PE_Ratio', 'PB_Ratio']].transform('mean')
        
        df_fund['pe_rel_sector'] = df_fund['PE_Ratio'] / sector_means['PE_Ratio']
        df_fund['pb_rel_sector'] = df_fund['PB_Ratio'] / sector_means['PB_Ratio']
        
        df_fund = df_fund.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        
        self.fundamental_data = df_fund
        return self.fundamental_data

    def apply_clustering_and_ml(self, ml_lookback_days):
        # 1. CLUSTERING
        clustering_df = self.fundamental_data[['PE_Ratio', 'PB_Ratio', 'Div_Yield', 'ROE']].join(
            self.performance_data[['sharpe_ratio', 'annual_volatility']],
            how='inner'
        ).fillna(0)

        if len(clustering_df) >= 5:
            # Lógica de clustering simplificada
            data_scaled = StandardScaler().fit_transform(clustering_df)
            n_comp = min(data_scaled.shape[1], 3)
            data_pca = PCA(n_components=n_comp).fit_transform(data_scaled)
            n_clusters_fit = min(len(data_pca), 5)
            kmeans = KMeans(n_clusters=max(2, n_clusters_fit), random_state=42, n_init=10)
            clusters = kmeans.fit_predict(data_pca)
            
            cluster_series = pd.Series(clusters, index=clustering_df.index)
            self.fundamental_data['Cluster'] = self.fundamental_data.index.map(cluster_series).fillna(-1).astype(int)
        else:
            self.fundamental_data['Cluster'] = 0

        # 2. RANDOM FOREST CLASSIFIER (MODELO DE portfolio_analyzer.py)
        features_ml = [
            'RSI', 'MACD', 'Volatility', 'Momentum', 'SMA_50', 'SMA_200',
            'PE_Ratio', 'PB_Ratio', 'Div_Yield', 'ROE',
            'pe_rel_sector', 'pb_rel_sector', 'Cluster'
        ]
        
        for symbol in tqdm(self.successful_assets, desc="[ML] Treinando Random Forest"):
            df = self.data_by_asset[symbol].copy()
            
            df['Future_Direction'] = np.where(
                df['Close'].pct_change(ml_lookback_days).shift(-ml_lookback_days) > 0,
                1, 0
            )
            
            if symbol in self.fundamental_data.index:
                fund_data = self.fundamental_data.loc[symbol].to_dict()
                for col in [f for f in features_ml if f not in df.columns]:
                    if col in fund_data:
                        df[col] = fund_data[col]
            
            current_features = [f for f in features_ml if f in df.columns]
            df.dropna(subset=current_features + ['Future_Direction'], inplace=True)
            
            if len(df) < MIN_HISTORY_DAYS:
                continue
            
            X = df[current_features].iloc[:-ml_lookback_days].copy()
            y = df['Future_Direction'].iloc[:-ml_lookback_days]

            if 'Cluster' in X.columns:
                X['Cluster'] = X['Cluster'].astype(str)
            
            numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = ['Cluster'] if 'Cluster' in X.columns else []

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), [f for f in numeric_cols if 'rel_sector' not in f]),
                    ('rel', 'passthrough', [f for f in numeric_cols if 'rel_sector' in f]),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
                ],
                remainder='passthrough'
            )
            
            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced'))
            ])
            
            try:
                if len(np.unique(y)) < 2:
                    self.ml_predictions[symbol] = {
                        'predicted_proba_up': 0.5, 'auc_roc_score': 0.5, 'model_name': 'Inconclusivo (Classe Única)'
                    }
                    continue
                
                model.fit(X, y)
                
                scores = cross_val_score(
                    model, X, y,
                    cv=TimeSeriesSplit(n_splits=5),
                    scoring='roc_auc'
                )
                auc_roc_score = scores.mean()
                
                last_features = df[current_features].iloc[[-ml_lookback_days]].copy()
                if 'Cluster' in last_features.columns:
                    last_features['Cluster'] = last_features['Cluster'].astype(str)
                
                proba = model.predict_proba(last_features)[0][1]
                
                self.ml_predictions[symbol] = {
                    'predicted_proba_up': proba,
                    'auc_roc_score': auc_roc_score,
                    'model_name': 'RandomForestClassifier'
                }
                
            except Exception:
                self.ml_predictions[symbol] = {
                    'predicted_proba_up': 0.5, 'auc_roc_score': np.nan, 'model_name': 'Erro no Treino'
                }
                pass

    def _normalize_score(self, series, higher_better=True):
        series_clean = series.replace([np.inf, -np.inf], np.nan).dropna()
        if series_clean.empty or series_clean.std() == 0:
            return pd.Series(50, index=series.index)
        z = zscore(series_clean, nan_policy='omit')
        normalized_series = pd.Series(
            50 + (z.clip(-3, 3) / 3) * 50,
            index=series_clean.index
        )
        if not higher_better:
            normalized_series = 100 - normalized_series
        return normalized_series.reindex(series.index, fill_value=50)

    def score_and_rank_assets(self, time_horizon):
        
        # Ponderações adaptadas da lógica de v4.py para o score base do portfolio_analyzer.py
        if time_horizon == "CURTO PRAZO":
            WEIGHT_PERF, WEIGHT_FUND, WEIGHT_TECH = 0.40, 0.10, 0.50
        elif time_horizon == "LONGO PRAZO":
            WEIGHT_PERF, WEIGHT_FUND, WEIGHT_TECH = 0.40, 0.50, 0.10
        else:
            WEIGHT_PERF, WEIGHT_FUND, WEIGHT_TECH = 0.40, 0.30, 0.30

        self.current_weights = {
            'Performance': WEIGHT_PERF, 'Fundamentos': WEIGHT_FUND,
            'Técnicos': WEIGHT_TECH
        }
        
        combined = self.performance_data.join(self.fundamental_data, how='inner').copy()
        
        for symbol in combined.index:
            if symbol in self.data_by_asset and 'RSI' in self.data_by_asset[symbol].columns:
                df = self.data_by_asset[symbol]
                combined.loc[symbol, 'RSI_current'] = df['RSI'].iloc[-1]
                combined.loc[symbol, 'MACD_current'] = df['MACD'].iloc[-1]

        scores = pd.DataFrame(index=combined.index)
        
        # SCORE DE PERFORMANCE
        scores['performance_score'] = self._normalize_score(
            combined['sharpe_ratio'], higher_better=True
        ) * WEIGHT_PERF
        
        # SCORE FUNDAMENTALISTA
        fund_score = self._normalize_score(
            combined.get('PE_Ratio', 50), higher_better=False
        ) * 0.5
        fund_score += self._normalize_score(
            combined.get('ROE', 50), higher_better=True
        ) * 0.5
        scores['fundamental_score'] = fund_score * WEIGHT_FUND
        
        # SCORE TÉCNICO
        tech_score = 0
        if WEIGHT_TECH > 0:
            rsi_proximity_score = 100 - abs(combined['RSI_current'] - 50)
            tech_score += self._normalize_score(
                rsi_proximity_score.clip(0, 100), higher_better=True
            ) * 0.5
            tech_score += self._normalize_score(
                combined.get('MACD_current', 50), higher_better=True
            ) * 0.5

        scores['technical_score'] = tech_score * WEIGHT_TECH
        
        # SCORE DE MACHINE LEARNING (PONDERADO PELO AUC)
        ml_scores = pd.Series({
            s: self.ml_predictions.get(s, {'predicted_proba_up': 0.5})['predicted_proba_up']
            for s in combined.index
        })
        confidence_scores = pd.Series({
            s: self.ml_predictions.get(s, {'auc_roc_score': np.nan})['auc_roc_score']
            for s in combined.index
        })
        
        combined['ML_Proba'] = ml_scores
        combined['ML_Confidence'] = confidence_scores.fillna(0.5)
        
        # O peso do ML é 30% no portfolio_analyzer.py, mas a soma dos demais é 100%.
        # Vamos usar a lógica final de pontuação: 70% da Base + Score ML Ponderado (30% do total)
        ml_score_base = (
             self._normalize_score(combined['ML_Proba'], higher_better=True) * 0.5 +
             self._normalize_score(combined['ML_Confidence'], higher_better=True) * 0.5
        ) * 0.3 # 30% do total (peso máximo)
        
        scores['ml_score_weighted'] = ml_score_base * combined['ML_Confidence'] 
        
        score_sum = (WEIGHT_PERF + WEIGHT_FUND + WEIGHT_TECH)
        scores['base_score'] = scores[[
            'performance_score', 'fundamental_score', 'technical_score'
        ]].sum(axis=1) / score_sum if score_sum > 0 else 50
        
        scores['total_score'] = scores['base_score'] * 0.7 + scores['ml_score_weighted']
        
        self.combined_scores = scores.join(combined).sort_values('total_score', ascending=False)
        return self.combined_scores

    def select_final_portfolio(self):
        ranked_assets = self.combined_scores.index.tolist()
        final_portfolio = []
        selected_sectors = set()

        for asset in ranked_assets:
            sector = self.get_asset_sector(asset)
            
            is_broad_asset = any(key in sector for key in [
                'ETF', 'Fundo', 'Fiagro', 'Fixed Income', 'Commodities', 'Unknown'
            ])
            
            if is_broad_asset:
                current_funds = len([
                    a for a in final_portfolio
                    if any(key in self.get_asset_sector(a) for key in ['ETF', 'Fundo', 'Fiagro'])
                ])
                
                if asset not in final_portfolio and len(final_portfolio) < NUM_ASSETS_IN_PORTFOLIO and current_funds < 2:
                    final_portfolio.append(asset)
            
            elif sector not in selected_sectors:
                final_portfolio.append(asset)
                selected_sectors.add(sector)
            
            if len(final_portfolio) >= NUM_ASSETS_IN_PORTFOLIO:
                break
        
        self.selected_assets = final_portfolio
        return self.selected_assets

    def optimize_allocation(self, risk_level):
        if len(self.selected_assets) < NUM_ASSETS_IN_PORTFOLIO:
            self.current_allocation_method = "ERRO: Ativos Insuficientes para Otimização Mínima"
            weights = {asset: 1/len(self.selected_assets) for asset in self.selected_assets} if self.selected_assets else {}
        else:
            final_returns_df = pd.concat(
                [self.data_by_asset[s]['Returns'] for s in self.selected_assets if s in self.data_by_asset],
                axis=1,
                keys=self.selected_assets
            ).dropna()
            
            if final_returns_df.shape[0] < 50:
                weights = {asset: 1/len(self.selected_assets) for asset in self.selected_assets}
                self.current_allocation_method = 'PESOS IGUAIS (Dados insuficientes)'
            else:
                optimizer = PortfolioOptimizer(final_returns_df)
                
                if 'CONSERVADOR' in risk_level or 'INTERMEDIÁRIO' in risk_level:
                    weights = optimizer.optimize('MinVolatility')
                    self.current_allocation_method = 'MINIMIZAÇÃO DE VOLATILIDADE'
                else:
                    weights = optimizer.optimize('MaxSharpe')
                    self.current_allocation_method = 'MAXIMIZAÇÃO DE SHARPE'

        if weights is None:
            weights = {asset: 1/len(self.selected_assets) for asset in self.selected_assets}
            self.current_allocation_method += " | FALLBACK (Otimização Falhou)"
        
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

        self.portfolio_allocation = {
            s: {
                'weight': w,
                'amount': self.investment_amount * w
            }
            for s, w in weights.items()
        }
        
        return self.portfolio_allocation
        
    def calculate_portfolio_metrics(self):
        if not self.selected_assets or not self.portfolio_allocation:
            return {}

        returns_data = {
            s: self.data_by_asset[s]['Returns']
            for s in self.selected_assets
            if s in self.data_by_asset
        }
        returns_df = pd.DataFrame(returns_data).dropna()

        if returns_df.empty:
            return {}

        weights_dict = {s: self.portfolio_allocation[s]['weight'] for s in self.selected_assets}
        weights = np.array([weights_dict[s] for s in returns_df.columns])
        weights = weights / weights.sum()

        portfolio_returns = (returns_df * weights).sum(axis=1)

        metrics = calculate_risk_metrics(portfolio_returns)
        self.portfolio_metrics = {
            **metrics,
            'total_investment': self.investment_amount
        }
        
        return self.portfolio_metrics
        
    def generate_justifications(self):
        for symbol in self.selected_assets:
            justification = []
            
            if symbol in self.performance_data.index:
                perf = self.performance_data.loc[symbol]
                justification.append(
                    f"Perf: Sharpe {perf['sharpe_ratio']:.3f}, Retorno Anual {perf['annual_return']*100:.2f}%."
                )
            
            if symbol in self.fundamental_data.index:
                fund = self.fundamental_data.loc[symbol]
                fund_list = [f"P/L: {fund['PE_Ratio']:.2f}", f"ROE: {fund['ROE']:.2f}%"]
                fund_list = [item for item in fund_list if not pd.isna(float(item.split(': ')[1].replace('%', '').replace('P/L: ', '').replace('ROE: ', '')))]
                if fund_list:
                    justification.append(f"Fund: {', '.join(fund_list)}.")
                
            if symbol in self.ml_predictions:
                ml = self.ml_predictions[symbol]
                proba_up = ml.get('predicted_proba_up', 0.5)
                auc_score = ml.get('auc_roc_score', np.nan)
                auc_str = f"{auc_score:.3f}" if not pd.isna(auc_score) else "N/A"
                
                justification.append(
                    f"ML: Prob. Alta {proba_up*100:.1f}% (AUC {auc_str})."
                )
            
            self.selection_justifications[symbol] = " | ".join(justification)
        
        return self.selection_justifications
        
    def run_complete_pipeline(self, custom_symbols=None, profile_inputs=None):
        """Executa o pipeline completo de construção de portfólio."""
        self.dashboard_profile = profile_inputs
        ml_lookback_days = profile_inputs['ml_lookback_days']
        risk_level = profile_inputs['risk_level']
        time_horizon = profile_inputs['time_horizon']
        
        if not self.collect_market_data(custom_symbols):
            return False

        self.calculate_cross_sectional_features()
        self.performance_data = self.performance_data.dropna(subset=['sharpe_ratio'])
        
        self.apply_clustering_and_ml(ml_lookback_days=ml_lookback_days)
        
        self.score_and_rank_assets(time_horizon=time_horizon)
        
        self.select_final_portfolio()
        
        self.optimize_allocation(risk_level=risk_level)
        
        self.calculate_portfolio_metrics()
        
        self.generate_justifications()
        
        return True

# =============================================================================
# FUNÇÕES DE ESTILO E VISUALIZAÇÃO (De v4.py)
# =============================================================================

def obter_template_grafico():
    """Retorna template de layout para gráficos Plotly com estilo Times New Roman."""
    return {
        'plot_bgcolor': 'white', 'paper_bgcolor': 'white',
        'font': {'family': 'Times New Roman, serif', 'size': 12, 'color': 'black'},
        'title': {'font': {'family': 'Times New Roman, serif', 'size': 16, 'color': '#2c3e50', 'weight': 'bold'}, 'x': 0.5, 'xanchor': 'center'},
        'xaxis': {'showgrid': True, 'gridcolor': 'lightgray', 'showline': True, 'linecolor': 'black', 'linewidth': 1, 'tickfont': {'family': 'Times New Roman, serif', 'color': 'black'}, 'title': {'font': {'family': 'Times New Roman, serif', 'color': 'black'}}, 'zeroline': False},
        'yaxis': {'showgrid': True, 'gridcolor': 'lightgray', 'showline': True, 'linecolor': 'black', 'linewidth': 1, 'tickfont': {'family': 'Times New Roman, serif', 'color': 'black'}, 'title': {'font': {'family': 'Times New Roman, serif', 'color': 'black'}}, 'zeroline': False},
        'legend': {'font': {'family': 'Times New Roman, serif', 'color': 'black'}, 'bgcolor': 'rgba(255, 255, 255, 0.8)', 'bordercolor': 'lightgray', 'borderwidth': 1},
        'colorway': ['#2c3e50', '#7f8c8d', '#3498db', '#e74c3c', '#27ae60']
    }

# =============================================================================
# ESTRUTURA STREAMLIT (De v4.py)
# =============================================================================

# As funções aba_introducao, aba_selecao_ativos, aba_analise_individual e aba_governanca
# são adaptadas a seguir para usar as classes simplificadas.

def aba_introducao():
    # Mantém o conteúdo de v4.py
    st.markdown("## 📚 Bem-vindo ao Sistema AutoML de Otimização de Portfólio")
    # ... (Restante do conteúdo da aba_introducao de v4.py) ...
    st.markdown("""
    <div class="info-box">
    <h3>🎯 O que este sistema faz?</h3>
    <p>Este é um sistema avançado de construção e otimização de portfólios de investimento que utiliza 
    <strong>Machine Learning</strong>, <strong>modelagem estatística</strong> e <strong>teoria moderna de portfólio</strong> 
    para criar carteiras personalizadas baseadas no seu perfil de risco e objetivos.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔬 Metodologia Científica")
        st.markdown("""
        **1. Análise de Perfil do Investidor**
        - Questionário baseado em normas CVM
        - Avaliação de tolerância ao risco
        - Definição de horizonte temporal
        
        **2. Coleta e Processamento de Dados**
        - Dados históricos de preços (máximo disponível)
        - Indicadores técnicos (RSI, MACD, etc.)
        - Fundamentos financeiros (P/L, ROE, etc.)
        
        **3. Engenharia de Features**
        - Indicadores técnicos avançados
        - Fatores fundamentalistas de Valor e Qualidade
        - Análise Cross-Sectional e Clusterização
        """)
    
    with col2:
        st.markdown("### 🤖 Tecnologias Utilizadas")
        st.markdown("""
        **Machine Learning Ensemble (Simplificado)**
        - RandomForestClassifier
        - Ponderação por AUC-ROC
        
        **Modelagem de Risco e Volatilidade**
        - Volatilidade Histórica (Base)
        
        **Otimização de Portfólio**
        - Teoria de Markowitz (Max Sharpe / Min Volatility)
        - Restrições de peso (10-30% por ativo)
        
        **Governança de Modelo**
        - Monitoramento de AUC-ROC (Implementação simplificada)
        """)
    
    st.markdown("---")
    
    st.markdown("### 📊 Como Funciona a Seleção dos 5 Ativos?")
    
    st.markdown("""
    <div class="info-box">
    <h4>Sistema de Pontuação Multi-Fator Adaptativo</h4>
    <p>O sistema utiliza um <strong>score composto</strong> que combina múltiplas dimensões de análise, com <strong>ponderações adaptativas</strong> ao seu perfil.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        **📈 Score de Performance (até 40%)**
        - Sharpe Ratio, Retorno anualizado
        """)
    
    with col2:
        st.markdown("""
        **💼 Score Fundamentalista (até 50%)**
        - Qualidade: ROE, Valor: P/L
        """)
    
    with col3:
        st.markdown("""
        **🔧 Score Técnico (até 50%)**
        - Indicadores de Momentum (MACD, RSI)
        """)
    
    with col4:
        st.markdown("""
        **🤖 Score de Machine Learning (até 30%)**
        - Probabilidade de alta (RandomForest)
        - Confiança do modelo (AUC-ROC)
        """)
    
    st.markdown("---")
    
    st.markdown("### ⚖️ Ponderação Adaptativa por Perfil")
    
    perfil_table = pd.DataFrame({
        'Perfil': ['Conservador', 'Intermediário', 'Moderado', 'Moderado-Arrojado', 'Avançado'],
        'Horizonte': ['Longo Prazo', 'Médio Prazo', 'Médio Prazo', 'Curto Prazo', 'Curto Prazo'],
        'Performance': ['40%', '40%', '40%', '40%', '40%'],
        'Fundamentos': ['50%', '30%', '30%', '10%', '10%'],
        'Técnicos': ['10%', '30%', '30%', '50%', '50%'],
        'ML (Máx)': ['30%', '30%', '30%', '30%', '30%'],
        'Foco': ['Qualidade e Estabilidade', 'Equilíbrio', 'Equilíbrio', 'Momentum', 'Curto Prazo e Momentum']
    })
    
    st.table(perfil_table)

def aba_selecao_ativos():
    # Adapta a aba_selecao_ativos de v4.py para usar as constantes simplificadas
    st.markdown("## 🎯 Seleção de Ativos para Análise")
    
    # ... (Lógica de seleção de ativos mantida de v4.py) ...
    modo_selecao = st.radio(
        "**Modo de Seleção:**",
        [
            f"📊 Ibovespa Completo ({len(ATIVOS_IBOVESPA)} ativos)",
            f"🌐 Lista Completa de Ativos ({len(TODOS_ATIVOS)} ativos)",
            "🏢 Setores Específicos",
            "✍️ Digitar Ativos Manualmente"
        ],
        index=0
    )
    
    ativos_selecionados = []
    
    # (Lógica para cada modo de seleção usando ATIVOS_IBOVESPA, TODOS_ATIVOS, ATIVOS_POR_SETOR)
    if "Ibovespa Completo" in modo_selecao:
        ativos_selecionados = ATIVOS_IBOVESPA.copy()
        # ... (Display logic)
        st.success(f"✓ **{len(ativos_selecionados)} ativos do Ibovespa** selecionados")
        
    elif "Lista Completa" in modo_selecao:
        ativos_selecionados = TODOS_ATIVOS.copy()
        st.success(f"✓ **{len(ativos_selecionados)} ativos** selecionados de todos os setores")
        
    elif "Setores Específicos" in modo_selecao:
        setores_disponiveis = list(ATIVOS_POR_SETOR.keys())
        setores_selecionados = st.multiselect(
            "Escolha um ou mais setores:", options=setores_disponiveis, default=setores_disponiveis[:3]
        )
        for setor in setores_selecionados:
            ativos_selecionados.extend(ATIVOS_POR_SETOR[setor])
        
    elif "Digitar Ativos" in modo_selecao:
        ativos_da_lista = st.multiselect(
            "Pesquise e selecione ativos:",
            options=TODOS_ATIVOS,
            format_func=lambda x: x.replace('.SA', '')
        )
        novos_ativos_input = st.text_area("Digite os códigos dos ativos (um por linha):", placeholder="PETR4\nVALE3...")
        novos_ativos = []
        if novos_ativos_input.strip():
            linhas = novos_ativos_input.strip().split('\n')
            for linha in linhas:
                ticker = linha.strip().upper()
                if ticker and not ticker.endswith('.SA'):
                    novos_ativos.append(f"{ticker}.SA")
                elif ticker:
                    novos_ativos.append(ticker)
        
        ativos_selecionados = list(set(ativos_da_lista + novos_ativos))


    if ativos_selecionados:
        st.session_state.ativos_para_analise = ativos_selecionados
        st.markdown("---")
        st.success(f"✓ Seleção confirmada: **{len(ativos_selecionados)} ativos** para análise.")
    else:
        st.warning("⚠️ Nenhum ativo selecionado.")

def aba_construtor_portfolio():
    # Adaptação da aba_construtor_portfolio de v4.py para usar a lógica de portfolio_analyzer.py
    
    if 'ativos_para_analise' not in st.session_state or not st.session_state.ativos_para_analise:
        st.warning("⚠️ Por favor, selecione os ativos na aba **'Seleção de Ativos'** primeiro.")
        return
    
    if 'builder' not in st.session_state: st.session_state.builder = None
    if 'profile' not in st.session_state: st.session_state.profile = {}
    if 'builder_complete' not in st.session_state: st.session_state.builder_complete = False
    
    # FASE 1: QUESTIONÁRIO
    if not st.session_state.builder_complete:
        st.markdown('## 📋 Questionário de Perfil do Investidor')
        st.info(f"✓ {len(st.session_state.ativos_para_analise)} ativos selecionados para análise")
        
        with st.form("investor_profile_form"):
            col_question1, col_question2 = st.columns(2)
            
            # ... (Lógica do questionário mantida) ...
            options_score = ['CT: Concordo Totalmente', 'C: Concordo', 'N: Neutro', 'D: Discordo', 'DT: Discordo Totalmente']
            options_reaction = ['A: Venderia', 'B: Manteria', 'C: Compraria mais']
            options_level_abc = ['A: Avançado', 'B: Intermediário', 'C: Iniciante']
            options_time_horizon = ['A: Curto (até 1 ano)', 'B: Médio (1-5 anos)', 'C: Longo (5+ anos)']
            options_liquidity = ['A: Menos de 6 meses', 'B: Entre 6 meses e 2 anos', 'C: Mais de 2 anos']

            with col_question1:
                st.markdown("#### Tolerância ao Risco")
                p2_risk = st.radio("**1. Aceito risco de curto prazo por retorno de longo prazo**", options=options_score, index=2)
                p3_gain = st.radio("**2. Ganhar o máximo é minha prioridade, mesmo com risco**", options=options_score, index=2)
                p4_stable = st.radio("**3. Prefiro crescimento constante, sem volatilidade**", options=options_score, index=2)
                p5_loss = st.radio("**4. Evitar perdas é mais importante que crescimento**", options=options_score, index=2)
                p511_reaction = st.radio("**5. Se meus investimentos caíssem 10%, eu:**", options=options_reaction, index=1)
                p_level = st.radio("**6. Meu nível de conhecimento em investimentos:**", options=options_level_abc, index=1)
            
            with col_question2:
                st.markdown("#### Horizonte Temporal e Capital")
                p211_time = st.radio("**7. Prazo máximo para reavaliação de estratégia:**", options=options_time_horizon, index=2)[0]
                p311_liquid = st.radio("**8. Necessidade de liquidez (prazo mínimo para resgate):**", options=options_liquidity, index=2)[0]
                st.markdown("---")
                investment = st.number_input("Valor de Investimento (R$)", min_value=1000, max_value=10000000, value=100000, step=10000)
            
            # Opções avançadas (simplificadas - o otimizar_ml não faz nada no core do portfolio_analyzer.py, mas mantido para UI)
            with st.expander("Opções Avançadas"):
                st.info("A otimização avançada de ML (Optuna) não está implementada nesta versão simplificada do pipeline.")
                otimizar_ml = False

            submitted = st.form_submit_button("🚀 Gerar Portfólio Otimizado", type="primary")
            
            if submitted:
                risk_answers = {'risk_accept': p2_risk, 'max_gain': p3_gain, 'stable_growth': p4_stable, 'avoid_loss': p5_loss, 'reaction': p511_reaction, 'level': p_level, 'time_purpose': p211_time, 'liquidity': p311_liquid}
                
                analyzer = InvestorProfileAnalyzer()
                risk_level, horizon, lookback, score = analyzer.calculate_profile(risk_answers)
                
                st.session_state.profile = {'risk_level': risk_level, 'time_horizon': horizon, 'ml_lookback_days': lookback, 'risk_score': score}
                
                builder = PortfolioBuilder(investment)
                st.session_state.builder = builder
                
                with st.spinner(f'Criando portfólio para **PERFIL {risk_level}** ({horizon})...'):
                    success = builder.run_complete_pipeline(
                        custom_symbols=st.session_state.ativos_para_analise,
                        profile_inputs=st.session_state.profile
                    )
                    
                    if not success:
                        st.error("Falha fatal: Não foi possível coletar dados suficientes. Tente novamente.")
                        return
                    
                    st.session_state.builder_complete = True
                    st.rerun()
    
    # FASE 2: RESULTADOS
    else:
        builder = st.session_state.builder
        profile = st.session_state.profile
        assets = builder.selected_assets
        allocation = builder.portfolio_allocation
        
        st.markdown('## ✅ Portfólio Otimizado Gerado')
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Perfil de Risco", profile.get('risk_level', 'N/A'), f"Score: {profile.get('risk_score', 'N/A')}")
        col2.metric("Horizonte", profile.get('time_horizon', 'N/A'))
        col3.metric("Sharpe Ratio", f"{builder.portfolio_metrics.get('sharpe_ratio', 0):.3f}")
        col4.metric("Estratégia", builder.current_allocation_method.split('(')[0].strip())
        
        if st.button("🔄 Recomeçar Análise", key='recomecar_analysis'):
            st.session_state.builder_complete = False
            st.session_state.builder = None
            st.session_state.profile = {}
            st.session_state.ativos_para_analise = []
            st.rerun()
        
        st.markdown("---")
        
        # Dashboard de resultados (adaptado para o formato de dados do PortfolioBuilder)
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Alocação", "📈 Performance", "🔬 Análise ML", "📉 Volatilidade", "❓ Justificativas" # Volatilidade substitui GARCH
        ])
        
        with tab1:
            col_alloc, col_table = st.columns([1, 2])
            with col_alloc:
                st.markdown('#### Alocação de Capital')
                alloc_data = pd.DataFrame([
                    {'Ativo': a, 'Peso (%)': allocation[a]['weight'] * 100}
                    for a in assets if a in allocation and allocation[a]['weight'] > 0.001
                ])
                if not alloc_data.empty:
                    fig_alloc = px.pie(alloc_data, values='Peso (%)', names='Ativo', hole=0.3)
                    fig_alloc.update_layout(**obter_template_grafico(), title={'text': "Distribuição do Portfólio"})
                    st.plotly_chart(fig_alloc, use_container_width=True)
            
            with col_table:
                st.markdown('#### Detalhamento dos Ativos')
                alloc_table = []
                for asset in assets:
                    if asset in allocation and allocation[asset]['weight'] > 0:
                        weight = allocation[asset]['weight']
                        amount = allocation[asset]['amount']
                        sector = builder.get_asset_sector(asset)
                        ml_info = builder.ml_predictions.get(asset, {})
                        
                        auc_score = ml_info.get('auc_roc_score', np.nan)
                        auc_display = f"{auc_score:.3f}" if not pd.isna(auc_score) else "N/A"

                        alloc_table.append({
                            'Ativo': asset.replace('.SA', ''),
                            'Setor': sector,
                            'Peso (%)': f"{weight * 100:.2f}",
                            'Valor (R$)': f"R$ {amount:,.2f}",
                            'ML Prob. Alta (%)': f"{ml_info.get('predicted_proba_up', 0.5)*100:.1f}",
                            'ML AUC': auc_display
                        })
                df_alloc = pd.DataFrame(alloc_table)
                st.dataframe(df_alloc, use_container_width=True, hide_index=True)
        
        with tab2:
            st.markdown('#### Métricas de Performance do Portfólio')
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Retorno Anual", f"{builder.portfolio_metrics.get('annual_return', 0)*100:.2f}%")
            col2.metric("Volatilidade Anual", f"{builder.portfolio_metrics.get('annual_volatility', 0)*100:.2f}%")
            col3.metric("Sharpe Ratio", f"{builder.portfolio_metrics.get('sharpe_ratio', 0):.3f}")
            col4.metric("Max Drawdown", f"{builder.portfolio_metrics.get('max_drawdown', 0)*100:.2f}%")
            
            st.markdown("---")
            st.markdown('#### Evolução dos Retornos Cumulativos dos Ativos')
            fig_cum = go.Figure()
            for asset in assets:
                if asset in builder.data_by_asset and 'Returns' in builder.data_by_asset[asset]:
                    returns = builder.data_by_asset[asset]['Returns']
                    cum_returns = (1 + returns).cumprod()
                    fig_cum.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns.values, name=asset.replace('.SA', ''), mode='lines'))
            fig_cum.update_layout(**obter_template_grafico(), title={'text': "Evolução dos Retornos Cumulativos"})
            st.plotly_chart(fig_cum, use_container_width=True)
            
        with tab3:
            st.markdown('#### Análise de Machine Learning')
            ml_data = []
            for asset in assets:
                if asset in builder.ml_predictions:
                    ml_info = builder.ml_predictions[asset]
                    ml_data.append({
                        'Ativo': asset.replace('.SA', ''),
                        'Prob. Alta (%)': ml_info.get('predicted_proba_up', 0.5) * 100,
                        'AUC-ROC (CV)': ml_info.get('auc_roc_score', np.nan),
                        'Modelo': ml_info.get('model_name', 'N/A')
                    })
            df_ml = pd.DataFrame(ml_data)
            if not df_ml.empty:
                df_ml_display = df_ml.copy()
                df_ml_display['Prob. Alta (%)'] = df_ml_display['Prob. Alta (%)'].round(2)
                df_ml_display['AUC-ROC (CV)'] = df_ml_display['AUC-ROC (CV)'].apply(
                    lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A"
                )
                st.dataframe(df_ml_display, use_container_width=True, hide_index=True)
            else:
                st.warning("Não há dados de Machine Learning para exibir.")

        with tab4:
            st.markdown('#### Análise de Volatilidade e Risco (Histórica)')
            dados_vol = []
            for ativo in assets:
                if ativo in builder.performance_data.index:
                    perf = builder.performance_data.loc[ativo]
                    dados_vol.append({
                        'Ativo': ativo.replace('.SA', ''),
                        'Vol. Anual Histórica (%)': perf['annual_volatility'] * 100,
                        'Max Drawdown (%)': perf['max_drawdown'] * 100
                    })
            df_vol = pd.DataFrame(dados_vol)
            st.dataframe(df_vol, use_container_width=True, hide_index=True)

        with tab5:
            st.markdown('#### Justificativas de Seleção e Alocação')
            if not builder.selection_justifications:
                st.warning("Nenhuma justificativa gerada.")
            else:
                for asset, justification in builder.selection_justifications.items():
                    weight = builder.portfolio_allocation.get(asset, {}).get('weight', 0)
                    st.markdown(f"""
                    <div class="info-box">
                    <h4>{asset.replace('.SA', '')} ({weight*100:.2f}%)</h4>
                    <p>{justification}</p>
                    </div>
                    """, unsafe_allow_html=True)

def aba_analise_individual():
    # Adaptação da aba_analise_individual de v4.py para usar a lógica de portfolio_analyzer.py
    
    st.markdown("## 🔍 Análise Individual Completa de Ativos")
    
    ativos_disponiveis = ATIVOS_IBOVESPA 
    if 'ativos_para_analise' in st.session_state and st.session_state.ativos_para_analise:
        ativos_disponiveis = st.session_state.ativos_para_analise
            
    if not ativos_disponiveis:
        st.error("Nenhum ativo disponível para análise.")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        ativo_selecionado = st.selectbox(
            "Selecione um ativo para análise detalhada:", options=ativos_disponiveis,
            format_func=lambda x: x.replace('.SA', '') if isinstance(x, str) else x, key='individual_asset_select'
        )
    with col2:
        if st.button("🔄 Analisar Ativo", key='analyze_asset_button', type="primary"):
            st.session_state.analisar_ativo_triggered = True
    
    if 'analisar_ativo_triggered' not in st.session_state or not st.session_state.analisar_ativo_triggered:
        st.info("👆 Selecione um ativo e clique em 'Analisar Ativo' para começar a análise completa.")
        return
    
    with st.spinner(f"Analisando {ativo_selecionado}..."):
        try:
            ticker = yf.Ticker(ativo_selecionado)
            hist = ticker.history(period=FIXED_ANALYSIS_PERIOD) # Usa 2 anos para consistência
            
            if hist.empty:
                st.error(f"Não foi possível obter dados históricos para {ativo_selecionado}.")
                return
            
            df_completo = calculate_technical_indicators(hist.copy())
            features_fund = ticker.info # Usa .info diretamente para dados fundamentalistas
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Visão Geral", "📈 Análise Técnica", "💼 Análise Fundamentalista", "🤖 Machine Learning", "🔬 Clusterização (Proxy)"
            ])
            
            # ... (Lógica para as 5 abas de análise individual adaptada para as novas classes) ...
            with tab1:
                st.markdown(f"### {ativo_selecionado.replace('.SA', '')} - Visão Geral")
                preco_atual = df_completo['Close'].iloc[-1] if not df_completo.empty else np.nan
                col1, col2, col3 = st.columns(3)
                col1.metric("Preço Atual", f"R$ {preco_atual:.2f}")
                col2.metric("Setor", features_fund.get('sector', 'N/A'))
                col3.metric("P/L (TTM)", f"{features_fund.get('trailingPE', np.nan):.2f}")
                
                # Candlestick Plot
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                fig.add_trace(go.Candlestick(x=df_completo.index, open=df_completo['Open'], high=df_completo['High'], low=df_completo['Low'], close=df_completo['Close'], name='Preço'), row=1, col=1)
                fig.add_trace(go.Bar(x=df_completo.index, y=df_completo['Volume'], name='Volume'), row=2, col=1)
                fig.update_layout(**obter_template_grafico(), height=600)
                st.plotly_chart(fig, use_container_width=True)
                
            with tab2:
                st.markdown("### Indicadores Técnicos")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("RSI (14)", f"{df_completo['RSI'].iloc[-1]:.2f}" if 'RSI' in df_completo.columns else "N/A")
                col2.metric("MACD", f"{df_completo['MACD'].iloc[-1]:.4f}" if 'MACD' in df_completo.columns else "N/A")
                col3.metric("SMA 50", f"{df_completo['SMA_50'].iloc[-1]:.2f}" if 'SMA_50' in df_completo.columns else "N/A")
                col4.metric("Volatilidade Anual", f"{df_completo['Volatility'].iloc[-1]*100:.2f}%" if 'Volatility' in df_completo.columns else "N/A")
                
            with tab3:
                st.markdown("### Análise Fundamentalista")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("ROE", f"{features_fund.get('returnOnEquity', np.nan)*100:.2f}%")
                col2.metric("Div. Yield", f"{features_fund.get('dividendYield', np.nan)*100:.2f}%")
                col3.metric("Market Cap", f"R$ {features_fund.get('marketCap', np.nan):,.0f}")
                col4.metric("Beta", f"{features_fund.get('beta', np.nan):.2f}")
                
            with tab4:
                st.markdown("### Análise de Machine Learning (Proxy)")
                # Implementação de um proxy simples para a previsão ML
                ml_lookback = 5 # Curto prazo fixo
                # Re-executar a lógica ML simples de PortfolioBuilder para este ativo
                
                builder_proxy = PortfolioBuilder(1000)
                builder_proxy.data_by_asset[ativo_selecionado] = df_completo
                builder_proxy.successful_assets = [ativo_selecionado]
                builder_proxy.fundamental_data = pd.DataFrame([
                    {'Ticker': ativo_selecionado, 'PE_Ratio': features_fund.get('trailingPE', 0), 'PB_Ratio': features_fund.get('priceToBook', 0), 'Div_Yield': features_fund.get('dividendYield', 0)*100, 'ROE': features_fund.get('returnOnEquity', 0)*100, 'Sector': features_fund.get('sector', 'Unknown')}
                ]).set_index('Ticker')
                builder_proxy.calculate_cross_sectional_features()
                builder_proxy.apply_clustering_and_ml(ml_lookback_days=ml_lookback)
                
                ml_info = builder_proxy.ml_predictions.get(ativo_selecionado, {'predicted_proba_up': 0.5, 'auc_roc_score': np.nan})
                
                col1, col2 = st.columns(2)
                col1.metric("Probabilidade de Alta (5 dias)", f"{ml_info['predicted_proba_up']*100:.2f}%")
                col2.metric("AUC-ROC (Validação Cruzada)", f"{ml_info['auc_roc_score']:.3f}" if not pd.isna(ml_info['auc_roc_score']) else "N/A")
                st.info(f"Modelo: {ml_info['model_name']}")
                
            with tab5:
                st.markdown("### Clusterização e Similaridade (Proxy)")
                st.warning("Esta funcionalidade exige o `PortfolioBuilder` para ter acesso a dados de múltiplos ativos. Use a aba **'Construtor de Portfólio'** para ver o resultado da clusterização.")

        except Exception as e:
            st.error(f"Erro ao analisar o ativo {ativo_selecionado}: {str(e)}")
            st.session_state.analisar_ativo_triggered = False


def aba_governanca():
    # Adaptação da aba_governanca de v4.py. A governança será um proxy simples
    # baseada no AUC do último treinamento do PortfolioBuilder.
    st.markdown("## 🛡️ Governança de Modelo - Monitoramento de Performance (Proxy)")
    
    if 'builder' not in st.session_state or st.session_state.builder is None:
        st.warning("⚠️ Execute o **Construtor de Portfólio** primeiro para visualizar métricas de governança.")
        return
    
    builder = st.session_state.builder
    
    ativos_com_ml = [a for a in builder.successful_assets if a in builder.ml_predictions and not pd.isna(builder.ml_predictions[a].get('auc_roc_score'))]
    
    if not ativos_com_ml:
        st.info("📊 Nenhum ativo com dados de ML válidos após o treinamento.")
        return
    
    ativo_selecionado = st.selectbox(
        "Selecione um ativo para análise de governança:",
        options=ativos_com_ml,
        format_func=lambda x: x.replace('.SA', '')
    )
    
    ml_info = builder.ml_predictions[ativo_selecionado]
    auc_atual = ml_info.get('auc_roc_score', 0.5)
    
    # Lógica de Alerta Simplificada
    AUC_THRESHOLD_MIN = 0.60
    if auc_atual < AUC_THRESHOLD_MIN:
        severidade = 'error'
        status_msg = f'🚨 AUC ({auc_atual:.3f}) abaixo do mínimo aceitável ({AUC_THRESHOLD_MIN}). Requer atenção imediata.'
    elif auc_atual < 0.70:
        severidade = 'warning'
        status_msg = f'⚠️ AUC ({auc_atual:.3f}) em zona de atenção. Modelo em monitoramento.'
    else:
        severidade = 'success'
        status_msg = f'✅ AUC ({auc_atual:.3f}). Modelo operando normalmente.'
        
    st.markdown(f'<div class="alert-{severidade}"><strong>{status_msg}</strong></div>', unsafe_allow_html=True)
    
    st.markdown("### 📊 Métricas de Performance (Último Treinamento)")
    col1, col2 = st.columns(2)
    col1.metric("AUC Atual", f"{auc_atual:.3f}")
    col2.metric("Probabilidade Média de Alta", f"{ml_info.get('predicted_proba_up', 0.5)*100:.2f}%")
    
    st.markdown("---")
    st.info("O histórico e métricas detalhadas de Precision/Recall não estão disponíveis na versão simplificada do pipeline.")

def configurar_pagina():
    """Configura a página Streamlit e injeta CSS customizado."""
    st.set_page_config(
        page_title="Portfolio Adaptativo",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # CSS customizado (mantido de v4.py)
    st.markdown("""
        <style>
        .main-header { font-family: 'Times New Roman', serif; color: #2c3e50; text-align: center; border-bottom: 2px solid #e0e0e0; padding-bottom: 10px; font-size: 2.2rem !important; margin-bottom: 20px;}
        html, body, [class*="st-"] { font-family: 'Times New Roman', serif;}
        .stButton button { border: 1px solid #2c3e50; color: #2c3e50; border-radius: 4px; padding: 8px 16px;}
        .stButton button:hover { background-color: #7f8c8d; color: white;}
        .stButton button[kind="primary"] { background-color: #2c3e50; color: white; border: none;}
        .info-box { background-color: #f8f9fa; border-left: 4px solid #2c3e50; padding: 15px; margin: 10px 0; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
        .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f8f9fa; border-radius: 4px 4px 0 0; gap: 1px; padding-top: 10px; padding-bottom: 10px;}
        .stTabs [aria-selected="true"] { background-color: #ffffff; border-top: 2px solid #2c3e50; border-left: 1px solid #e0e0e0; border-right: 1px solid #e0e0e0;}
        .alert-success { background-color: #d4edda; border-color: #c3e6cb; color: #155724; padding: .75rem 1.25rem; margin-bottom: 1rem; border: 1px solid transparent; border-radius: .25rem; }
        .alert-warning { background-color: #fff3cd; border-color: #ffeeba; color: #856404; padding: .75rem 1.25rem; margin-bottom: 1rem; border: 1px solid transparent; border-radius: .25rem; }
        .alert-error { background-color: #f8d7da; border-color: #f5c6cb; color: #721c24; padding: .75rem 1.25rem; margin-bottom: 1rem; border: 1px solid transparent; border-radius: .25rem; }
        </style>
        """, unsafe_allow_html=True)

def main():
    """Função principal que inicializa e gerencia a aplicação Streamlit com 5 abas."""
    
    if 'builder' not in st.session_state:
        st.session_state.builder = None
        st.session_state.builder_complete = False
        st.session_state.profile = {}
        st.session_state.ativos_para_analise = ATIVOS_IBOVESPA.copy() # Default to Ibovespa
        st.session_state.analisar_ativo_triggered = False
        
    configurar_pagina()
    
    st.sidebar.markdown('<p style="font-size: 26px; font-weight: bold; color: #2c3e50;">📈 Portfolio Adaptativo</p>', unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Recursos Core")
    st.sidebar.markdown("""
    - **Lógica de Otimização**: Markowitz (Max Sharpe / Min Vol)
    - **Modelo ML**: RandomForestClassifier
    - **Análise Técnica**: RSI, MACD, SMAs
    - **Análise Fundamentalista**: P/L, ROE, etc.
    - **Risco**: Volatilidade Histórica
    """)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Base de Código**: portfolio_analyzer.py (Lógica) + v4.py (Dashboard)")

    st.markdown('<h1 class="main-header">Sistema de Análise e Otimização de Portfólio</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📚 Introdução", "🎯 Seleção de Ativos", "🏗️ Construtor de Portfólio", "🔍 Análise Individual", "🛡️ Governança de Modelo"
    ])
    
    with tab1: aba_introducao()
    with tab2: aba_selecao_ativos()
    with tab3: aba_construtor_portfolio()
    with tab4: aba_analise_individual()
    with tab5: aba_governanca()

if __name__ == "__main__":
    main()
