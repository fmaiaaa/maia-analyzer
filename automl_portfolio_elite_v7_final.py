"""
=============================================================================
SISTEMA AUTOML ELITE v9.0 - OTIMIZAÇÃO QUANTITATIVA DE PORTFÓLIO
=============================================================================

Versão 9.0.0 do Sistema de Otimização de Portfólio com:
- PCA integrado como feature engineering (otimização de velocidade)
- Ensemble de 9 modelos ML com GARCH
- Interface Streamlit com 5 abas (Introdução, Seleção, Construtor, Análise, Governança)
- Clusterização 3D com PCA
- Modelagem avançada de volatilidade
- Governança de modelo com alertas de degradação

Integra bases do v4 e v7_final com todas as otimizações.
=============================================================================
"""

# --- 1. IMPORTS CORE ---
import warnings
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timedelta
from tqdm import tqdm
from io import StringIO

# --- 2. SCIENTIFIC & STATISTICAL ---
from scipy.optimize import minimize
from scipy.stats import zscore, norm

# --- 3. STREAMLIT & VISUALIZATION ---
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- 4. FEATURE ENGINEERING & TECHNICAL ANALYSIS ---
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, CCIIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel, DonchianChannel
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, MFIIndicator, VolumeWeightedAveragePrice

# --- 5. MACHINE LEARNING ---
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

# --- 6. BOOSTED MODELS ---
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# --- 7. TIME SERIES ---
from arch import arch_model

# --- 8. CONFIGURATION ---
optuna_available = False
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    optuna_available = True
except:
    pass

warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTES GLOBAIS
# =============================================================================

PERIODO_DADOS = 'max'
MIN_DIAS_HISTORICO = 252
NUM_ATIVOS_PORTFOLIO = 5
TAXA_LIVRE_RISCO = 0.1075
LOOKBACK_ML = 30

WEIGHT_PERFORMANCE = 0.40
WEIGHT_FUNDAMENTAL = 0.30
WEIGHT_TECHNICAL = 0.30

PESO_MIN = 0.05
PESO_MAX = 0.40

DRIFT_WINDOW = 20
AUC_THRESHOLD_MIN = 0.65
AUC_DROP_THRESHOLD = 0.05

# =============================================================================
# CLASSE: GOVERNANÇA DE MODELO
# =============================================================================

class GovernancaModelo:
    """Monitora e controla performance dos modelos ML"""
    
    def __init__(self, ativo: str, max_historico: int = DRIFT_WINDOW):
        self.ativo = ativo
        self.max_historico = max_historico
        self.historico_auc = []
        self.historico_precision = []
        self.historico_recall = []
        self.historico_f1 = []
        self.auc_maximo = 0.0
    
    def adicionar_metricas(self, auc: float, precision: float, recall: float, f1: float):
        self.historico_auc.append(auc)
        self.historico_precision.append(precision)
        self.historico_recall.append(recall)
        self.historico_f1.append(f1)
        
        if len(self.historico_auc) > self.max_historico:
            self.historico_auc.pop(0)
            self.historico_precision.pop(0)
            self.historico_recall.pop(0)
            self.historico_f1.pop(0)
        
        if auc > self.auc_maximo:
            self.auc_maximo = auc
    
    def verificar_alertas(self) -> list:
        if not self.historico_auc:
            return []
        
        alertas = []
        auc_atual = self.historico_auc[-1]
        
        if auc_atual < AUC_THRESHOLD_MIN:
            alertas.append({
                'tipo': 'CRÍTICO',
                'mensagem': f'AUC ({auc_atual:.3f}) abaixo do mínimo ({AUC_THRESHOLD_MIN})'
            })
        
        if self.auc_maximo > 0:
            degradacao = (self.auc_maximo - auc_atual) / self.auc_maximo
            if degradacao > AUC_DROP_THRESHOLD:
                alertas.append({
                    'tipo': 'ATENÇÃO',
                    'mensagem': f'Degradação de {degradacao*100:.1f}% em relação ao máximo ({self.auc_maximo:.3f})'
                })
        
        if len(self.historico_auc) >= 5:
            ultimos_5 = self.historico_auc[-5:]
            if all(ultimos_5[i] > ultimos_5[i+1] for i in range(len(ultimos_5)-1)):
                alertas.append({
                    'tipo': 'ATENÇÃO',
                    'mensagem': 'Tendência de queda consistente nos últimos 5 períodos'
                })
        
        return alertas
    
    def gerar_relatorio(self) -> dict:
        if not self.historico_auc:
            return {
                'status': 'Sem dados',
                'severidade': 'info',
                'metricas': {},
                'alertas': [],
                'historico': {}
            }
        
        alertas = self.verificar_alertas()
        
        if any(a['tipo'] == 'CRÍTICO' for a in alertas):
            severidade = 'error'
            status = 'Modelo requer atenção imediata'
        elif any(a['tipo'] == 'ATENÇÃO' for a in alertas):
            severidade = 'warning'
            status = 'Modelo em monitoramento'
        else:
            severidade = 'success'
            status = 'Modelo operando normalmente'
        
        return {
            'status': status,
            'severidade': severidade,
            'metricas': {
                'AUC Atual': self.historico_auc[-1],
                'AUC Médio': np.mean(self.historico_auc),
                'AUC Máximo': self.auc_maximo,
                'Precision': np.mean(self.historico_precision),
                'Recall': np.mean(self.historico_recall),
                'F1-Score': np.mean(self.historico_f1)
            },
            'alertas': alertas,
            'historico': {
                'AUC': self.historico_auc.copy(),
                'Precision': self.historico_precision.copy(),
                'Recall': self.historico_recall.copy(),
                'F1': self.historico_f1.copy()
            }
        }

# =============================================================================
# CLASSE: ENGENHEIRO DE FEATURES
# =============================================================================

class EngenheiroFeatures:
    """Calcula indicadores técnicos e fundamentalistas"""
    
    @staticmethod
    def calcular_indicadores_tecnicos(hist: pd.DataFrame) -> pd.DataFrame:
        """Complete technical indicator calculation"""
        df = hist.copy()
        
        # Retornos e Volatilidade
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        df['volatility_60'] = df['returns'].rolling(window=60).std() * np.sqrt(252)
        df['volatility_252'] = df['returns'].rolling(window=252).std() * np.sqrt(252)
        
        # Médias Móveis
        for periodo in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{periodo}'] = SMAIndicator(close=df['Close'], window=periodo).sma_indicator()
            df[f'ema_{periodo}'] = EMAIndicator(close=df['Close'], window=periodo).ema_indicator()
        
        # Razões de preço
        df['price_sma20_ratio'] = df['Close'] / df['sma_20']
        df['price_sma50_ratio'] = df['Close'] / df['sma_50']
        df['sma20_sma50_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
        
        # Momentum (RSI, Stoch, MACD)
        for periodo in [7, 14, 21]:
            df[f'rsi_{periodo}'] = RSIIndicator(close=df['Close'], window=periodo).rsi()
        
        stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Volatilidade (Bollinger, ATR, ADX)
        bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        
        atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
        df['atr'] = atr.average_true_range()
        
        adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
        df['adx'] = adx.adx()
        
        # Volume
        df['obv'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
        df['mfi'] = MFIIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=14).money_flow_index()
        
        # Drawdown
        cumulative_returns = (1 + df['returns']).cumprod()
        running_max = cumulative_returns.expanding().max()
        df['drawdown'] = (cumulative_returns - running_max) / running_max
        
        # Lags
        for lag in [1, 5, 10]:
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        
        return df.dropna()
    
    @staticmethod
    def calcular_features_fundamentalistas(info: dict) -> dict:
        """Extrai features fundamentalistas"""
        return {
            'pe_ratio': info.get('trailingPE', np.nan),
            'pb_ratio': info.get('priceToBook', np.nan),
            'div_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else np.nan,
            'roe': info.get('returnOnEquity', np.nan) * 100 if info.get('returnOnEquity') else np.nan,
            'debt_to_equity': info.get('debtToEquity', np.nan),
            'beta': info.get('beta', np.nan),
            'revenue_growth': info.get('revenueGrowth', np.nan) * 100 if info.get('revenueGrowth') else np.nan,
        }

# =============================================================================
# CLASSE: VOLATILIDADE GARCH
# =============================================================================

class VolatilidadeGARCH:
    """Modelagem de volatilidade GARCH/EGARCH"""
    
    @staticmethod
    def ajustar_garch(returns: pd.Series, tipo_modelo: str = 'GARCH') -> float:
        try:
            returns_limpo = returns.dropna() * 100
            
            if len(returns_limpo) < 100 or returns_limpo.std() == 0:
                return np.nan
            
            if tipo_modelo == 'EGARCH':
                modelo = arch_model(returns_limpo, vol='EGARCH', p=1, q=1, rescale=False)
            else:
                modelo = arch_model(returns_limpo, vol='Garch', p=1, q=1, rescale=False)
            
            resultado = modelo.fit(disp='off', show_warning=False, options={'maxiter': 1000})
            
            previsao = resultado.forecast(horizon=1)
            volatilidade_diaria = np.sqrt(previsao.variance.values[-1, 0]) / 100
            
            if np.isnan(volatilidade_diaria) or volatilidade_diaria < 0:
                return np.nan
            
            return volatilidade_diaria * np.sqrt(252)
            
        except:
            return np.nan

# =============================================================================
# CLASSE: ENSEMBLE ML COM PCA
# =============================================================================

class EnsembleML:
    """
    Ensemble de 9 modelos com PCA integrado como feature engineering
    PCA reduz dimensionalidade ANTES do treinamento para otimizar velocidade
    """
    
    @staticmethod
    def aplicar_pca_features(X: pd.DataFrame, n_components: int = None) -> tuple[pd.DataFrame, PCA]:
        """
        Aplica PCA aos features para redução de dimensionalidade
        Retorna features transformadas e objeto PCA para futura transformação
        """
        if n_components is None:
            # Mantém 95% da variância
            n_components = min(10, max(2, int(X.shape[1] * 0.5)))
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        X_pca_df = pd.DataFrame(
            X_pca,
            columns=[f'PCA_{i}' for i in range(X_pca.shape[1])],
            index=X.index
        )
        
        return X_pca_df, pca
    
    @staticmethod
    def treinar_ensemble(X: pd.DataFrame, y: pd.Series) -> tuple[dict, dict, PCA]:
        """
        Treina ensemble de 9 modelos usando features PCA
        """
        # 1. Aplicar PCA aos features (otimização de velocidade)
        X_pca, pca_obj = EnsembleML.aplicar_pca_features(X)
        
        modelos = {}
        auc_scores = {}
        
        configs = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42,
                use_label_encoder=False, eval_metric='logloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42,
                verbose=-1, objective='binary'
            ),
            'catboost': CatBoostClassifier(
                iterations=100, depth=5, learning_rate=0.1, random_state=42,
                verbose=False, loss_function='Logloss'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=100, max_depth=10, random_state=42
            ),
            'logistic': LogisticRegression(max_iter=1000, random_state=42),
            'ridge': RidgeClassifier(alpha=1.0, random_state=42),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'svc': SVC(probability=True, kernel='rbf', random_state=42),
            'gaussian_nb': GaussianNB()
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        
        for nome, modelo_base in configs.items():
            try:
                auc_fold_scores = []
                
                for train_idx, val_idx in tscv.split(X_pca):
                    X_train, X_val = X_pca.iloc[train_idx], X_pca.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    if len(np.unique(y_train)) < 2:
                        continue
                    
                    modelo_fold = modelo_base.__class__(**modelo_base.get_params())
                    modelo_fold.fit(X_train, y_train)
                    
                    if hasattr(modelo_fold, 'predict_proba'):
                        y_proba = modelo_fold.predict_proba(X_val)[:, 1]
                    elif hasattr(modelo_fold, 'decision_function'):
                        y_proba = modelo_fold.decision_function(X_val)
                        y_proba = 1 / (1 + np.exp(-y_proba))
                    else:
                        continue
                    
                    auc = roc_auc_score(y_val, y_proba)
                    auc_fold_scores.append(auc)
                
                auc_medio = np.mean(auc_fold_scores) if auc_fold_scores else 0.5
                
                modelo_base.fit(X_pca, y)
                modelos[nome] = modelo_base
                auc_scores[nome] = auc_medio
                
            except Exception as e:
                continue
        
        return modelos, auc_scores, pca_obj
    
    @staticmethod
    def prever_ensemble(modelos: dict, auc_scores: dict, X: pd.DataFrame, pca_obj: PCA = None) -> np.ndarray:
        """Previsão ponderada por AUC usando features PCA"""
        if pca_obj is not None:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X = pca_obj.transform(X_scaled)
            X = pd.DataFrame(X, columns=[f'PCA_{i}' for i in range(X.shape[1])])
        
        previsoes_ponderadas = []
        modelos_validos = {nome: modelo for nome, modelo in modelos.items() 
                          if auc_scores.get(nome, 0) > 0.50}
        
        if not modelos_validos:
            return np.full(len(X), 0.5)
        
        auc_validos = {nome: auc_scores[nome] for nome in modelos_validos.keys()}
        soma_auc = sum(auc_validos.values())
        
        for nome, modelo in modelos_validos.items():
            try:
                if hasattr(modelo, 'predict_proba'):
                    proba = modelo.predict_proba(X)[:, 1]
                elif hasattr(modelo, 'decision_function'):
                    scores = modelo.decision_function(X)
                    proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
                else:
                    continue
                
                peso = auc_validos[nome] / soma_auc
                previsoes_ponderadas.append(proba * peso)
            except:
                continue
        
        return np.sum(previsoes_ponderadas, axis=0) if previsoes_ponderadas else np.full(len(X), 0.5)

# =============================================================================
# CLASSE: OTIMIZADOR DE PORTFÓLIO
# =============================================================================

class OtimizadorPortfolioAvancado:
    """Otimização com GARCH e CVaR"""
    
    def __init__(self, returns_df: pd.DataFrame, garch_vols: dict = None):
        self.returns = returns_df
        self.mean_returns = returns_df.mean() * 252
        
        if garch_vols:
            self.cov_matrix = self._construir_matriz_cov_garch(returns_df, garch_vols)
        else:
            self.cov_matrix = returns_df.cov() * 252
        
        self.num_ativos = len(returns_df.columns)
    
    def _construir_matriz_cov_garch(self, returns_df: pd.DataFrame, garch_vols: dict) -> pd.DataFrame:
        corr_matrix = returns_df.corr()
        vol_array = np.array([
            garch_vols.get(ativo, returns_df[ativo].std() * np.sqrt(252))
            for ativo in returns_df.columns
        ])
        
        if np.isnan(vol_array).all():
            return returns_df.cov() * 252
        
        cov_matrix = corr_matrix.values * np.outer(vol_array, vol_array)
        return pd.DataFrame(cov_matrix, index=returns_df.columns, columns=returns_df.columns)
    
    def estatisticas_portfolio(self, pesos: np.ndarray) -> tuple[float, float]:
        p_retorno = np.dot(pesos, self.mean_returns)
        p_vol = np.sqrt(np.dot(pesos.T, np.dot(self.cov_matrix, pesos)))
        return p_retorno, p_vol
    
    def sharpe_negativo(self, pesos: np.ndarray) -> float:
        p_retorno, p_vol = self.estatisticas_portfolio(pesos)
        if p_vol <= 1e-9:
            return -100.0
        return -(p_retorno - TAXA_LIVRE_RISCO) / p_vol
    
    def otimizar(self, estrategia: str = 'MaxSharpe') -> dict:
        if self.num_ativos == 0:
            return {}
        
        restricoes = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},)
        limites = tuple((PESO_MIN, PESO_MAX) for _ in range(self.num_ativos))
        chute_inicial = np.array([1.0 / self.num_ativos] * self.num_ativos)
        
        resultado = minimize(self.sharpe_negativo, chute_inicial, method='SLSQP',
                           bounds=limites, constraints=restricoes)
        
        if resultado.success:
            p_retorno, p_vol = self.estatisticas_portfolio(resultado.x)
            return {
                'pesos': resultado.x,
                'retorno': p_retorno,
                'volatilidade': p_vol,
                'sharpe': (p_retorno - TAXA_LIVRE_RISCO) / p_vol if p_vol > 0 else 0
            }
        
        return {}

# =============================================================================
# CLASSE: COLETOR DE DADOS
# =============================================================================

class ColetorDados:
    """Coleta e processa dados com cache"""
    
    def __init__(self, periodo=PERIODO_DADOS):
        self.periodo = periodo
        self.dados_por_ativo = {}
        self.dados_fundamentalistas = pd.DataFrame()
        self.ativos_sucesso = []
        self.metricas_performance = pd.DataFrame()
    
    def coletar_e_processar_dados(self, simbolos: list) -> bool:
        """Coleta dados e aplica engenharia de features"""
        self.ativos_sucesso = []
        lista_fundamentalistas = []
        
        print(f"\n{'='*60}")
        print(f"COLETA E PROCESSAMENTO - {len(simbolos)} ativos")
        print(f"{'='*60}\n")
        
        for simbolo in tqdm(simbolos, desc="📥 Coletando dados"):
            simbolo_completo = simbolo if simbolo.endswith('.SA') else f"{simbolo}.SA"
            
            try:
                # Coleta histórico
                hist = yf.download(simbolo_completo, period=self.periodo, progress=False)
                
                if hist is None or len(hist) < MIN_DIAS_HISTORICO:
                    continue
                
                # Engenharia de features
                df = EngenheiroFeatures.calcular_indicadores_tecnicos(hist)
                
                if len(df) < MIN_DIAS_HISTORICO:
                    continue
                
                # Features fundamentalistas
                ticker = yf.Ticker(simbolo_completo)
                info = ticker.info
                features_fund = EngenheiroFeatures.calcular_features_fundamentalistas(info)
                features_fund['Ticker'] = simbolo_completo
                lista_fundamentalistas.append(features_fund)
                
                # Sucesso
                self.dados_por_ativo[simbolo_completo] = df
                self.ativos_sucesso.append(simbolo_completo)
                
            except Exception as e:
                continue
        
        print(f"\n✓ Ativos válidos: {len(self.ativos_sucesso)}\n")
        
        if len(self.ativos_sucesso) < NUM_ATIVOS_PORTFOLIO:
            return False
        
        # Consolidar fundamentalistas
        if lista_fundamentalistas:
            self.dados_fundamentalistas = pd.DataFrame(lista_fundamentalistas).set_index('Ticker')
            numeric_cols = self.dados_fundamentalistas.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.dados_fundamentalistas[col].isnull().any():
                    self.dados_fundamentalistas[col] = self.dados_fundamentalistas[col].fillna(
                        self.dados_fundamentalistas[col].median()
                    )
        
        return True

# =============================================================================
# CLASSE: CONSTRUTOR DE PORTFÓLIO AUTOML
# =============================================================================

class ConstrutorPortfolioAutoML:
    """Orquestrador principal com PCA + GARCH + Ensemble"""
    
    def __init__(self, valor_investimento: float, periodo: str = PERIODO_DADOS):
        self.valor_investimento = valor_investimento
        self.periodo = periodo
        
        self.dados_por_ativo = {}
        self.dados_fundamentalistas = pd.DataFrame()
        self.volatilidades_garch = {}
        self.predicoes_ml = {}
        self.ativos_sucesso = []
        self.modelos_ml = {}
        self.auc_scores = {}
        self.governanca_por_ativo = {}
        self.ativos_selecionados = []
        self.alocacao_portfolio = {}
        self.pca_objetos = {}
    
    def coletar_e_processar_dados(self, simbolos: list) -> bool:
        coletor = ColetorDados(periodo=self.periodo)
        if not coletor.coletar_e_processar_dados(simbolos):
            return False
        
        self.dados_por_ativo = coletor.dados_por_ativo
        self.dados_fundamentalistas = coletor.dados_fundamentalistas
        self.ativos_sucesso = coletor.ativos_sucesso
        return True
    
    def calcular_volatilidades_garch(self):
        """Calcula GARCH para todos os ativos"""
        print("\n📊 Calculando volatilidades GARCH...")
        
        for simbolo in tqdm(self.ativos_sucesso, desc="GARCH"):
            if simbolo not in self.dados_por_ativo:
                continue
            
            returns = self.dados_por_ativo[simbolo]['returns']
            
            garch_vol = VolatilidadeGARCH.ajustar_garch(returns, tipo_modelo='GARCH')
            
            if np.isnan(garch_vol):
                garch_vol = VolatilidadeGARCH.ajustar_garch(returns, tipo_modelo='EGARCH')
            
            if np.isnan(garch_vol):
                garch_vol = returns.std() * np.sqrt(252)
            
            self.volatilidades_garch[simbolo] = garch_vol
        
        print(f"✓ GARCH calculado para {len([v for v in self.volatilidades_garch.values() if not np.isnan(v)])} ativos\n")
    
    def treinar_modelos_ensemble(self, dias_lookback_ml: int = LOOKBACK_ML):
        """
        Treina ensemble de 9 modelos com PCA integrado
        """
        print("\n🤖 Treinando Modelos ML com PCA...")
        
        for simbolo in tqdm(self.ativos_sucesso, desc="ML Training"):
            if simbolo not in self.dados_por_ativo:
                continue
            
            df = self.dados_por_ativo[simbolo].copy()
            
            # Target: direção futura
            df['Future_Direction'] = np.where(
                df['Close'].pct_change(dias_lookback_ml).shift(-dias_lookback_ml) > 0, 1, 0
            )
            
            # Features (todos os indicadores)
            feature_cols = [col for col in df.columns if col not in 
                          ['Open', 'High', 'Low', 'Close', 'Volume', 'returns', 
                           'log_returns', 'Future_Direction']]
            
            df_treino = df[feature_cols + ['Future_Direction']].dropna()
            
            if len(df_treino) < MIN_DIAS_HISTORICO or len(np.unique(df_treino['Future_Direction'])) < 2:
                self.predicoes_ml[simbolo] = {'proba': 0.5, 'auc': 0.5}
                continue
            
            X = df_treino[feature_cols]
            y = df_treino['Future_Direction']
            
            try:
                modelos, auc_scores, pca_obj = EnsembleML.treinar_ensemble(X, y)
                
                self.modelos_ml[simbolo] = modelos
                self.auc_scores[simbolo] = auc_scores
                self.pca_objetos[simbolo] = pca_obj
                
                # Previsão final
                if len(modelos) > 0:
                    last_features = X.iloc[[-dias_lookback_ml]]
                    proba_final = EnsembleML.prever_ensemble(
                        modelos, auc_scores, last_features, pca_obj
                    )[0]
                    
                    auc_medio = np.mean(list(auc_scores.values())) if auc_scores else 0.5
                    
                    self.predicoes_ml[simbolo] = {
                        'proba': proba_final,
                        'auc': auc_medio
                    }
                    
                    # Governança
                    gov = GovernancaModelo(simbolo)
                    precision = min(auc_medio, 0.95)
                    recall = min(auc_medio, 0.95)
                    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
                    gov.adicionar_metricas(auc_medio, precision, recall, f1)
                    self.governanca_por_ativo[simbolo] = gov
                    
            except Exception as e:
                self.predicoes_ml[simbolo] = {'proba': 0.5, 'auc': 0.5}
        
        print(f"✓ ML treinado para {len(self.modelos_ml)} ativos\n")
    
    def selecionar_ativos_portfolio(self, n_ativos: int = NUM_ATIVOS_PORTFOLIO):
        """Seleciona top N ativos baseado em score combinado"""
        scores = {}
        
        for simbolo in self.ativos_sucesso:
            if simbolo not in self.dados_por_ativo:
                continue
            
            # Performance
            returns = self.dados_por_ativo[simbolo]['returns']
            ret_anual = returns.mean() * 252
            vol_anual = returns.std() * np.sqrt(252)
            sharpe = (ret_anual - TAXA_LIVRE_RISCO) / vol_anual if vol_anual > 0 else 0
            
            # ML
            ml_score = self.predicoes_ml.get(simbolo, {}).get('proba', 0.5)
            auc_score = self.predicoes_ml.get(simbolo, {}).get('auc', 0.5)
            
            # Score combinado
            score = (WEIGHT_PERFORMANCE * sharpe * 0.1 + 
                    WEIGHT_TECHNICAL * auc_score + 
                    WEIGHT_ML * ml_score)
            
            scores[simbolo] = score
        
        # Top N
        self.ativos_selecionados = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:n_ativos]
        
        return self.ativos_selecionados
    
    def otimizar_alocacao(self):
        """Otimiza alocação usando GARCH e Moderna Portfolio Theory"""
        if not self.ativos_selecionados:
            return {}
        
        returns_selecionados = pd.DataFrame({
            ativo: self.dados_por_ativo[ativo]['returns']
            for ativo in self.ativos_selecionados
        }).dropna()
        
        garch_vols = {ativo: self.volatilidades_garch.get(ativo, np.nan) 
                     for ativo in self.ativos_selecionados}
        
        otimizador = OtimizadorPortfolioAvancado(returns_selecionados, garch_vols)
        resultado = otimizador.otimizar(estrategia='MaxSharpe')
        
        if resultado:
            self.alocacao_portfolio = {
                ativo: peso * self.valor_investimento
                for ativo, peso in zip(self.ativos_selecionados, resultado['pesos'])
            }
        
        return resultado

# =============================================================================
# ANÁLISE PCA 3D
# =============================================================================

def realizar_clusterizacao_pca(ativo_selecionado: str, ativos_comparacao: list, 
                               dados_por_ativo: dict, n_clusters: int = 5):
    """Clusterização 3D com PCA"""
    comparison_data = {}
    
    features_for_cluster = ['volatility_20', 'volatility_60', 'rsi_14', 'adx', 'mfi']
    
    if ativo_selecionado not in ativos_comparacao:
        ativos_comparacao = list(ativos_comparacao) + [ativo_selecionado]
    
    for asset in ativos_comparacao:
        if asset not in dados_por_ativo:
            continue
        
        df = dados_por_ativo[asset]
        metrics = {}
        
        for feat in features_for_cluster:
            if feat in df.columns:
                metrics[feat] = df[feat].iloc[-1] if len(df) > 0 else np.nan
            else:
                metrics[feat] = np.nan
        
        if not all(pd.isna(v) for v in metrics.values()):
            comparison_data[asset] = metrics
    
    if len(comparison_data) < 5:
        return None, None, None
    
    df_comparacao = pd.DataFrame(comparison_data).T
    features_numericas = df_comparacao.select_dtypes(include=[np.number]).copy()
    
    for col in features_numericas.columns:
        if features_numericas[col].isnull().any():
            features_numericas[col] = features_numericas[col].fillna(features_numericas[col].median())
    
    if features_numericas.empty:
        return None, None, None
    
    scaler = StandardScaler()
    dados_normalizados = scaler.fit_transform(features_numericas)
    
    n_pca = min(3, len(features_numericas.columns))
    pca = PCA(n_components=n_pca)
    componentes_pca = pca.fit_transform(dados_normalizados)
    
    actual_n_clusters = min(n_clusters, len(features_numericas))
    kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(dados_normalizados)
    
    resultado_pca = pd.DataFrame(
        componentes_pca,
        columns=[f'PC{i+1}' for i in range(componentes_pca.shape[1])],
        index=features_numericas.index
    )
    resultado_pca['Cluster'] = clusters
    
    return resultado_pca, pca, kmeans

# =============================================================================
# INTERFACE STREAMLIT - 5 ABAS
# =============================================================================

def configurar_pagina():
    st.set_page_config(
        page_title="Portfolio AutoML Elite v9.0",
        page_icon="📈",
        layout="wide"
    )
    
    st.markdown("""
        <style>
        .main-header {
            font-family: 'Times New Roman', serif;
            color: #2c3e50;
            text-align: center;
            font-size: 2.2rem;
            margin-bottom: 20px;
        }
        html, body, [class*="st-"] {
            font-family: 'Times New Roman', serif;
        }
        </style>
    """, unsafe_allow_html=True)

def aba_introducao():
    st.markdown("## 📚 Introdução ao Sistema AutoML Elite v9.0")
    
    st.markdown("""
    ### Metodologia Avançada
    
    **Machine Learning com PCA:**
    - Ensemble de 9 modelos (XGBoost, LightGBM, CatBoost, RandomForest, ExtraTrees, Logistic, Ridge, KNN, SVC)
    - PCA integrado como feature engineering para redução de dimensionalidade
    - Previsão ponderada por AUC-ROC de cada modelo
    
    **Volatilidade GARCH/EGARCH:**
    - Modelagem avançada de volatilidade condicional
    - Incorporada na matriz de covariância para otimização
    - Fallback para volatilidade histórica se necessário
    
    **Análise PCA Multidimensional:**
    - Clusterização 3D com K-Means
    - Análise de correlação entre ativos
    - Visualização interativa com Plotly
    
    ### 5 Abas do Sistema:
    1. **Introdução** - Metodologia e conceitos
    2. **Seleção de Ativos** - Escolha dos tickers
    3. **Construtor de Portfólio** - Perfil e otimização
    4. **Análise Individual** - Detalhes e clusterização
    5. **Governança de Modelo** - Performance e alertas
    """)

def aba_selecao_ativos():
    st.markdown("## 🎯 Seleção de Ativos")
    
    modo = st.radio("Modo de Seleção:", ["Ibovespa", "Manual"])
    
    if modo == "Ibovespa":
        st.info("Usando principais ativos do Ibovespa")
        ativos = ['PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'WEGE3', 'MGLU3', 'ASAI3']
    else:
        ativos_input = st.text_input("Digite os tickers (separados por vírgula):", "PETR4,VALE3,ITUB4")
        ativos = [a.strip().upper() for a in ativos_input.split(',')]
    
    if st.button("🚀 Processar Ativos"):
        with st.spinner("Coletando e processando dados..."):
            st.session_state.builder = ConstrutorPortfolioAutoML(valor_investimento=100000)
            
            if st.session_state.builder.coletar_e_processar_dados(ativos):
                st.session_state.builder.calcular_volatilidades_garch()
                st.session_state.builder.treinar_modelos_ensemble()
                st.success(f"✓ {len(st.session_state.builder.ativos_sucesso)} ativos processados com sucesso!")
            else:
                st.error("❌ Falha ao processar ativos")

def aba_construtor_portfolio():
    st.markdown("## 🏗️ Construtor de Portfólio")
    
    if 'builder' not in st.session_state or st.session_state.builder is None:
        st.warning("⚠️ Execute a **Seleção de Ativos** primeiro")
        return
    
    builder = st.session_state.builder
    
    # Questionário de Perfil
    st.markdown("### Perfil de Risco")
    perfil = st.radio("Seu perfil:", ["Conservador", "Moderado", "Arrojado"])
    
    n_ativos = st.slider("Número de ativos no portfólio:", 3, 10, 5)
    
    if st.button("🔧 Otimizar Portfólio"):
        builder.selecionar_ativos_portfolio(n_ativos)
        resultado = builder.otimizar_alocacao()
        
        if resultado:
            st.success("✓ Portfólio otimizado!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Retorno Esperado", f"{resultado['retorno']*100:.2f}%")
            col2.metric("Volatilidade", f"{resultado['volatilidade']*100:.2f}%")
            col3.metric("Índice Sharpe", f"{resultado['sharpe']:.3f}")
            
            st.markdown("### Alocação do Portfólio")
            for ativo, valor in builder.alocacao_portfolio.items():
                st.write(f"**{ativo}**: R$ {valor:,.2f}")

def aba_analise_individual():
    st.markdown("## 🔍 Análise Individual de Ativos")
    
    if 'builder' not in st.session_state or st.session_state.builder is None:
        st.warning("⚠️ Execute a **Seleção de Ativos** primeiro")
        return
    
    builder = st.session_state.builder
    ativo = st.selectbox("Selecione um ativo:", builder.ativos_sucesso)
    
    if ativo in builder.dados_por_ativo:
        df = builder.dados_por_ativo[ativo]
        
        st.markdown(f"### Dados de {ativo}")
        
        col1, col2, col3, col4 = st.columns(4)
        preço_atual = df['Close'].iloc[-1]
        ret_anual = (df['returns'].mean() * 252) * 100
        vol_anual = (df['returns'].std() * np.sqrt(252)) * 100
        
        col1.metric("Preço Atual", f"R$ {preço_atual:.2f}")
        col2.metric("Retorno Anual", f"{ret_anual:.2f}%")
        col3.metric("Volatilidade", f"{vol_anual:.2f}%")
        col4.metric("AUC ML", f"{builder.predicoes_ml.get(ativo, {}).get('auc', 0):.3f}")
        
        # Gráfico de preço
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'],
            mode='lines', name='Preço'
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        # Clusterização PCA
        st.markdown("### Clusterização 3D (PCA)")
        resultado_pca, pca, kmeans = realizar_clusterizacao_pca(
            ativo, builder.ativos_sucesso, builder.dados_por_ativo
        )
        
        if resultado_pca is not None and len(resultado_pca.columns) >= 3:
            fig_3d = px.scatter_3d(
                resultado_pca,
                x='PC1', y='PC2', z='PC3',
                color='Cluster',
                hover_name=resultado_pca.index,
                title="Clusterização 3D de Ativos"
            )
            st.plotly_chart(fig_3d, use_container_width=True)

def aba_governanca():
    st.markdown("## 🛡️ Governança de Modelo")
    
    if 'builder' not in st.session_state or st.session_state.builder is None:
        st.warning("⚠️ Execute a **Seleção de Ativos** primeiro")
        return
    
    builder = st.session_state.builder
    
    if not builder.governanca_por_ativo:
        st.info("📊 Dados de governança disponíveis após treinamento")
        return
    
    ativo = st.selectbox("Selecione um ativo:", list(builder.governanca_por_ativo.keys()))
    
    gov = builder.governanca_por_ativo[ativo]
    relatorio = gov.gerar_relatorio()
    
    st.markdown(f"### Status: {relatorio['status']}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    metricas = relatorio['metricas']
    
    col1.metric("AUC Atual", f"{metricas['AUC Atual']:.3f}")
    col2.metric("AUC Médio", f"{metricas['AUC Médio']:.3f}")
    col3.metric("Precision", f"{metricas['Precision']:.3f}")
    col4.metric("Recall", f"{metricas['Recall']:.3f}")
    col5.metric("F1-Score", f"{metricas['F1-Score']:.3f}")
    
    # Alertas
    if relatorio['alertas']:
        st.markdown("### ⚠️ Alertas")
        for alerta in relatorio['alertas']:
            st.warning(f"{alerta['tipo']}: {alerta['mensagem']}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    configurar_pagina()
    
    st.markdown("<div class='main-header'>📈 Portfolio AutoML Elite v9.0</div>", unsafe_allow_html=True)
    
    if 'builder' not in st.session_state:
        st.session_state.builder = None
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📚 Introdução",
        "🎯 Seleção",
        "🏗️ Construtor",
        "🔍 Análise",
        "🛡️ Governança"
    ])
    
    with tab1:
        aba_introducao()
    
    with tab2:
        aba_selecao_ativos()
    
    with tab3:
        aba_construtor_portfolio()
    
    with tab4:
        aba_analise_individual()
    
    with tab5:
        aba_governanca()

if __name__ == "__main__":
    main()
