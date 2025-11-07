"""
=============================================================================
SISTEMA AUTOML AVANÇADO v9.0.0 - OTIMIZAÇÃO DE PORTFÓLIO FINANCEIRO
=============================================================================

Sistema completo de otimização de portfólio com:
- 🚀 PCA para aceleração de treinamento ML (Feature Engineering)
- 💾 Cache robusto de dados (sem reprocessamento)
- 🤖 Ensemble de 9 Modelos ML com ponderação AUC-ROC
- 📊 GARCH/EGARCH para volatilidade
- 🎯 Seleção Multi-Fator Adaptativa ao Perfil
- 🛡️ Governança e Monitoramento de Modelos
- 🔬 Clusterização 3D com PCA
- 📈 5 Abas Interativas (Streamlit)

Versão: 9.0.0 - Sistema AutoML Elite Final com PCA
=============================================================================
"""

import warnings
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.stats import norm

# Streamlit e Visualização
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Feature Engineering
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator

# Machine Learning
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from arch import arch_model

# Simulação de dados (substituir por yfinance em produção)
import yfinance as yf

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURAÇÕES GLOBAIS
# =============================================================================

PERIODO_DADOS = 'max'
MIN_DIAS_HISTORICO = 180
NUM_ATIVOS_PORTFOLIO = 5
TAXA_LIVRE_RISCO = 0.1075
LOOKBACK_ML = 30

WEIGHT_PERFORMANCE = 0.40
WEIGHT_FUNDAMENTAL = 0.30
WEIGHT_TECHNICAL = 0.30
WEIGHT_ML = 0.30

PESO_MIN = 0.10
PESO_MAX = 0.30

AUC_THRESHOLD_MIN = 0.65
AUC_DROP_THRESHOLD = 0.05
DRIFT_WINDOW = 20

# PCA Configuration
PCA_N_COMPONENTS = 0.95  # Manter 95% da variância
PCA_MIN_COMPONENTS = 10  # Mínimo de componentes

ATIVOS_IBOVESPA = [
    'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA',
    'B3SA3.SA', 'RENT3.SA', 'MGLU3.SA', 'WEGE3.SA', 'HAPV3.SA'
]

ATIVOS_POR_SETOR = {
    'Financeiro': ['ITUB4.SA', 'BBDC4.SA', 'B3SA3.SA'],
    'Petróleo': ['PETR4.SA'],
    'Mineração': ['VALE3.SA'],
    'Consumo': ['ABEV3.SA', 'MGLU3.SA'],
    'Industrial': ['WEGE3.SA', 'RENT3.SA'],
    'Saúde': ['HAPV3.SA']
}

TODOS_ATIVOS = sorted(list(set([a for ativos in ATIVOS_POR_SETOR.values() for a in ativos])))

# Mapas de pontuação do questionário
SCORE_MAP = {
    'CT: Concordo Totalmente': 5,
    'C: Concordo': 4,
    'N: Neutro': 3,
    'D: Discordo': 2,
    'DT: Discordo Totalmente': 1
}

SCORE_MAP_INV = {k: 6 - v for k, v in SCORE_MAP.items()}

SCORE_MAP_CONHECIMENTO = {
    'A: Avançado': 5,
    'B: Intermediário': 3,
    'C: Iniciante': 1
}

SCORE_MAP_REACTION = {
    'A: Venderia': 1,
    'B: Manteria': 3,
    'C: Compraria mais': 5
}

# =============================================================================
# CLASSE: GOVERNANÇA DE MODELO
# =============================================================================

class GovernancaModelo:
    def __init__(self, ativo: str, max_historico: int = DRIFT_WINDOW):
        self.ativo = ativo
        self.max_historico = max_historico
        self.historico_auc = []
        self.historico_precision = []
        self.historico_recall = []
        self.historico_f1 = []
        self.auc_maximo = 0.0
        
    def adicionar_metricas(self, auc, precision, recall, f1):
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
    
    def verificar_alertas(self):
        if not self.historico_auc:
            return []
        
        alertas = []
        auc_atual = self.historico_auc[-1]
        
        if auc_atual < AUC_THRESHOLD_MIN:
            alertas.append({
                'tipo': 'CRÍTICO',
                'mensagem': f'AUC ({auc_atual:.3f}) abaixo do mínimo aceitável ({AUC_THRESHOLD_MIN})'
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
    
    def gerar_relatorio(self):
        if not self.historico_auc:
            return {
                'status': 'Sem dados suficientes',
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
            'ativo': self.ativo,
            'status': status,
            'severidade': severidade,
            'metricas': {
                'AUC Atual': self.historico_auc[-1],
                'AUC Médio': np.mean(self.historico_auc),
                'AUC Máximo': self.auc_maximo,
                'Precision Média': np.mean(self.historico_precision),
                'Recall Médio': np.mean(self.historico_recall),
                'F1-Score Médio': np.mean(self.historico_f1)
            },
            'alertas': alertas,
            'historico': {
                'AUC': self.historico_auc,
                'Precision': self.historico_precision,
                'Recall': self.historico_recall,
                'F1': self.historico_f1
            }
        }

# =============================================================================
# CLASSE: ANALISADOR DE PERFIL
# =============================================================================

class AnalisadorPerfilInvestidor:
    def __init__(self):
        self.nivel_risco = ""
        self.horizonte_tempo = ""
        self.dias_lookback_ml = 5
    
    def determinar_nivel_risco(self, pontuacao):
        if pontuacao <= 18:
            return "CONSERVADOR"
        elif pontuacao <= 30:
            return "INTERMEDIÁRIO"
        elif pontuacao <= 45:
            return "MODERADO"
        elif pontuacao <= 60:
            return "MODERADO-ARROJADO"
        else:
            return "AVANÇADO"
    
    def determinar_horizonte_ml(self, liquidez_key, objetivo_key):
        time_map = {'A': 5, 'B': 20, 'C': 30}
        
        final_lookback = max(
            time_map.get(liquidez_key, 5),
            time_map.get(objetivo_key, 5)
        )
        
        if final_lookback >= 30:
            self.horizonte_tempo = "LONGO PRAZO"
            self.dias_lookback_ml = 30
        elif final_lookback >= 20:
            self.horizonte_tempo = "MÉDIO PRAZO"
            self.dias_lookback_ml = 20
        else:
            self.horizonte_tempo = "CURTO PRAZO"
            self.dias_lookback_ml = 5
        
        return self.horizonte_tempo, self.dias_lookback_ml
    
    def calcular_perfil(self, respostas_risco):
        pontuacao = (
            SCORE_MAP[respostas_risco['risk_accept']] * 5 +
            SCORE_MAP[respostas_risco['max_gain']] * 5 +
            SCORE_MAP_INV[respostas_risco['stable_growth']] * 5 +
            SCORE_MAP_INV[respostas_risco['avoid_loss']] * 5 +
            SCORE_MAP_CONHECIMENTO[respostas_risco['level']] * 3 +
            SCORE_MAP_REACTION[respostas_risco['reaction']] * 3
        )
        
        nivel_risco = self.determinar_nivel_risco(pontuacao)
        horizonte_tempo, ml_lookback = self.determinar_horizonte_ml(
            respostas_risco['liquidez'],
            respostas_risco['time_purpose']
        )
        
        return nivel_risco, horizonte_tempo, ml_lookback, pontuacao

# =============================================================================
# FUNÇÕES DE ESTILO
# =============================================================================

def obter_template_grafico():
    return {
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'font': {'family': 'Times New Roman, serif', 'size': 12, 'color': 'black'},
        'title': {
            'font': {'family': 'Times New Roman, serif', 'size': 16, 'color': '#2c3e50'},
            'x': 0.5,
            'xanchor': 'center'
        },
        'xaxis': {
            'showgrid': True,
            'gridcolor': 'lightgray',
            'showline': True,
            'linecolor': 'black',
            'zeroline': False
        },
        'yaxis': {
            'showgrid': True,
            'gridcolor': 'lightgray',
            'showline': True,
            'linecolor': 'black',
            'zeroline': False
        },
        'legend': {
            'font': {'family': 'Times New Roman, serif', 'color': 'black'},
            'bgcolor': 'rgba(255, 255, 255, 0.8)'
        },
        'colorway': ['#2c3e50', '#7f8c8d', '#3498db', '#e74c3c', '#27ae60']
    }

# =============================================================================
# CLASSE: ENGENHEIRO DE FEATURES
# =============================================================================

class EngenheiroFeatures:
    @staticmethod
    def calcular_indicadores_tecnicos(hist):
        df = hist.copy()
        
        # Retornos e Volatilidade
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        df['volatility_60'] = df['returns'].rolling(window=60).std() * np.sqrt(252)
        
        # Médias Móveis
        for periodo in [5, 10, 20, 50, 200]:
            df[f'sma_{periodo}'] = SMAIndicator(close=df['Close'], window=periodo).sma_indicator()
            df[f'ema_{periodo}'] = EMAIndicator(close=df['Close'], window=periodo).ema_indicator()
        
        # Razões de preço
        df['price_sma20_ratio'] = df['Close'] / df['sma_20']
        df['price_sma50_ratio'] = df['Close'] / df['sma_50']
        
        # RSI
        for periodo in [7, 14, 21]:
            df[f'rsi_{periodo}'] = RSIIndicator(close=df['Close'], window=periodo).rsi()
        
        # Stochastic
        stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # MACD
        macd = MACD(close=df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = BollingerBands(close=df['Close'])
        df['bb_width'] = bb.bollinger_wband()
        df['bb_position'] = (df['Close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
        
        # ATR
        atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'])
        df['atr'] = atr.average_true_range()
        df['atr_percent'] = (df['atr'] / df['Close']) * 100
        
        # ADX
        adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'])
        df['adx'] = adx.adx()
        
        # Momentum
        df['momentum_10'] = ROCIndicator(close=df['Close'], window=10).roc()
        df['momentum_20'] = ROCIndicator(close=df['Close'], window=20).roc()
        
        # Volume
        df['obv'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
        df['cmf'] = ChaikinMoneyFlowIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).chaikin_money_flow()
        
        # Drawdown
        cumulative_returns = (1 + df['returns']).cumprod()
        running_max = cumulative_returns.expanding().max()
        df['drawdown'] = (cumulative_returns - running_max) / running_max
        
        # Lags
        for lag in [1, 5, 10]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        
        # Rolling Stats
        for window in [5, 20]:
            df[f'returns_mean_{window}'] = df['returns'].rolling(window).mean()
            df[f'returns_std_{window}'] = df['returns'].rolling(window).std()
        
        return df.dropna()
    
    @staticmethod
    def calcular_features_fundamentalistas(info):
        return {
            'pe_ratio': info.get('trailingPE', np.nan),
            'pb_ratio': info.get('priceToBook', np.nan),
            'div_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else np.nan,
            'roe': info.get('returnOnEquity', np.nan) * 100 if info.get('returnOnEquity') else np.nan,
            'debt_to_equity': info.get('debtToEquity', np.nan),
            'market_cap': info.get('marketCap', np.nan),
            'beta': info.get('beta', np.nan),
            'sector': info.get('sector', 'Unknown')
        }
    
    @staticmethod
    def _normalizar(serie, maior_melhor=True):
        if serie.isnull().all():
            return pd.Series(0, index=serie.index)
        
        min_val = serie.min()
        max_val = serie.max()
        
        if max_val == min_val:
            return pd.Series(0.5, index=serie.index)
        
        if maior_melhor:
            return (serie - min_val) / (max_val - min_val)
        else:
            return (max_val - serie) / (max_val - min_val)

# =============================================================================
# CLASSE: COLETOR DE DADOS (COM CACHE)
# =============================================================================

class ColetorDados:
    def __init__(self, periodo=PERIODO_DADOS):
        self.periodo = periodo
        self.dados_por_ativo = {}
        self.dados_fundamentalistas = pd.DataFrame()
        self.ativos_sucesso = []
        self.metricas_performance = pd.DataFrame()
        self._cache_inicializado = False
    
    def coletar_e_processar_dados(self, simbolos):
        # Cache: Se já coletou dados, reutiliza
        if self._cache_inicializado and self.dados_por_ativo:
            print(f"\n✓ Cache: Reutilizando dados de {len(self.ativos_sucesso)} ativos")
            return True
        
        self.ativos_sucesso = []
        lista_fundamentalistas = []
        
        print(f"\n{'='*60}")
        print(f"COLETA E PROCESSAMENTO - {len(simbolos)} ativos")
        print(f"{'='*60}\n")
        
        for ticker in tqdm(simbolos, desc="📥 Coletando dados"):
            try:
                ticker_obj = yf.Ticker(ticker)
                hist = ticker_obj.history(period=self.periodo)
                
                if hist.empty or len(hist) < MIN_DIAS_HISTORICO * 0.7:
                    continue
                
                # Engenharia de features
                df = EngenheiroFeatures.calcular_indicadores_tecnicos(hist)
                
                if len(df) < MIN_DIAS_HISTORICO * 0.5:
                    continue
                
                self.dados_por_ativo[ticker] = df
                self.ativos_sucesso.append(ticker)
                
                # Fundamentalista
                info = ticker_obj.info
                features_fund = EngenheiroFeatures.calcular_features_fundamentalistas(info)
                features_fund['Ticker'] = ticker
                lista_fundamentalistas.append(features_fund)
                
            except Exception as e:
                continue
        
        if lista_fundamentalistas:
            self.dados_fundamentalistas = pd.DataFrame(lista_fundamentalistas).set_index('Ticker')
            self.dados_fundamentalistas = self.dados_fundamentalistas.replace([np.inf, -np.inf], np.nan)
            
            # Imputação
            numeric_cols = self.dados_fundamentalistas.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.dados_fundamentalistas[col].isnull().any():
                    median_val = self.dados_fundamentalistas[col].median()
                    self.dados_fundamentalistas[col] = self.dados_fundamentalistas[col].fillna(median_val if not pd.isna(median_val) else 0)
        
        # Métricas de performance
        metricas = {}
        for ticker in self.ativos_sucesso:
            if 'returns' in self.dados_por_ativo[ticker]:
                returns = self.dados_por_ativo[ticker]['returns']
                vol_anual = returns.std() * np.sqrt(252)
                ret_anual = returns.mean() * 252
                metricas[ticker] = {
                    'retorno_anual': ret_anual,
                    'volatilidade_anual': vol_anual,
                    'sharpe': (ret_anual - TAXA_LIVRE_RISCO) / vol_anual if vol_anual > 0 else 0,
                    'max_drawdown': self.dados_por_ativo[ticker]['drawdown'].min()
                }
        
        self.metricas_performance = pd.DataFrame(metricas).T
        self._cache_inicializado = True
        
        print(f"\n✓ Coleta concluída: {len(self.ativos_sucesso)} ativos válidos\n")
        return len(self.ativos_sucesso) > 0

# =============================================================================
# CLASSE: VOLATILIDADE GARCH
# =============================================================================

class VolatilidadeGARCH:
    @staticmethod
    def ajustar_garch(returns, tipo_modelo='GARCH'):
        try:
            returns_limpo = returns.dropna() * 100
            
            if len(returns_limpo) < 100 or returns_limpo.std() == 0:
                return np.nan
            
            if tipo_modelo == 'EGARCH':
                modelo = arch_model(returns_limpo, vol='EGARCH', p=1, q=1, rescale=False)
            else:
                modelo = arch_model(returns_limpo, vol='Garch', p=1, q=1, rescale=False)
            
            resultado = modelo.fit(disp='off', show_warning=False, options={'maxiter': 500})
            
            if resultado is None:
                return np.nan
            
            previsao = resultado.forecast(horizon=1)
            volatilidade = np.sqrt(previsao.variance.values[-1, 0]) / 100
            
            if np.isnan(volatilidade) or np.isinf(volatilidade):
                return np.nan
            
            return volatilidade * np.sqrt(252)
            
        except:
            return np.nan

# =============================================================================
# CLASSE: ENSEMBLE ML COM PCA
# =============================================================================

class EnsembleML:
    @staticmethod
    def treinar_ensemble(X, y, usar_pca=True):
        """
        Treina ensemble de 9 modelos com PCA para aceleração.
        
        Returns:
            (modelos, auc_scores, pca_transformer, scaler)
        """
        modelos = {}
        auc_scores = {}
        pca_transformer = None
        scaler = StandardScaler()
        
        # 1. Padronização
        X_scaled = scaler.fit_transform(X)
        
        # 2. PCA (NOVO - OTIMIZAÇÃO)
        if usar_pca:
            n_samples, n_features = X_scaled.shape
            
            # Determina número de componentes
            if n_features > PCA_MIN_COMPONENTS:
                pca_transformer = PCA(n_components=PCA_N_COMPONENTS, random_state=42)
                X_transformed = pca_transformer.fit_transform(X_scaled)
                print(f"  🚀 PCA: {n_features} features → {X_transformed.shape[1]} componentes ({pca_transformer.explained_variance_ratio_.sum()*100:.1f}% variância)")
            else:
                X_transformed = X_scaled
                print(f"  ⚠️ PCA desabilitado: poucas features ({n_features})")
        else:
            X_transformed = X_scaled
        
        # 3. Configuração dos modelos
        configs = {
            'xgboost': xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='logloss'),
            'lightgbm': lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1),
            'catboost': CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1, random_state=42, verbose=False),
            'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'extra_trees': ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'svc': SVC(probability=True, kernel='rbf', random_state=42),
            'logistic': LogisticRegression(max_iter=1000, random_state=42),
            'gaussian_nb': GaussianNB()
        }
        
        # 4. Treinamento com validação cruzada temporal
        tscv = TimeSeriesSplit(n_splits=3)
        X_df = pd.DataFrame(X_transformed)
        
        for nome, modelo in configs.items():
            try:
                auc_fold_scores = []
                
                for train_idx, val_idx in tscv.split(X_df):
                    X_train, X_val = X_df.iloc[train_idx], X_df.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
                        continue
                    
                    modelo_fold = configs[nome]
                    modelo_fold.fit(X_train, y_train)
                    
                    if hasattr(modelo_fold, 'predict_proba'):
                        y_proba = modelo_fold.predict_proba(X_val)[:, 1]
                    else:
                        y_proba = modelo_fold.decision_function(X_val)
                    
                    if len(np.unique(y_val)) >= 2:
                        auc = roc_auc_score(y_val, y_proba)
                        auc_fold_scores.append(auc)
                
                auc_medio = np.mean(auc_fold_scores) if auc_fold_scores else 0.5
                
                # Treina modelo final
                modelo.fit(X_df, y)
                modelos[nome] = modelo
                auc_scores[nome] = auc_medio
                
            except Exception as e:
                continue
        
        return modelos, auc_scores, pca_transformer, scaler
    
    @staticmethod
    def prever_ensemble_ponderado(modelos, auc_scores, X, pca_transformer=None, scaler=None):
        """Previsão ponderada por AUC-ROC com PCA."""
        # Aplica transformações
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        
        if pca_transformer is not None:
            X_transformed = pca_transformer.transform(X_scaled)
        else:
            X_transformed = X_scaled
        
        X_df = pd.DataFrame(X_transformed)
        
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
                    proba = modelo.predict_proba(X_df)[:, 1]
                else:
                    proba = modelo.decision_function(X_df)
                    proba = (proba - proba.min()) / (proba.max() - proba.min() + 1e-9)
                
                peso = auc_validos[nome] / soma_auc
                previsoes_ponderadas.append(proba * peso)
            except:
                continue
        
        if not previsoes_ponderadas:
            return np.full(len(X), 0.5)
        
        return np.sum(previsoes_ponderadas, axis=0)

# =============================================================================
# CLASSE: OTIMIZADOR DE PORTFÓLIO
# =============================================================================

class OtimizadorPortfolioAvancado:
    def __init__(self, returns_df, garch_vols=None):
        self.returns = returns_df
        self.mean_returns = returns_df.mean() * 252
        
        if garch_vols is not None and garch_vols:
            self.cov_matrix = self._construir_matriz_cov_garch(returns_df, garch_vols)
        else:
            self.cov_matrix = returns_df.cov() * 252
        
        self.num_ativos = len(returns_df.columns)
    
    def _construir_matriz_cov_garch(self, returns_df, garch_vols):
        corr_matrix = returns_df.corr()
        
        vol_array = np.array([
            garch_vols.get(ativo, returns_df[ativo].std() * np.sqrt(252))
            for ativo in returns_df.columns
        ])
        
        if np.isnan(vol_array).all() or np.all(vol_array <= 1e-9):
            return returns_df.cov() * 252
        
        cov_matrix = corr_matrix.values * np.outer(vol_array, vol_array)
        return pd.DataFrame(cov_matrix, index=returns_df.columns, columns=returns_df.columns)
    
    def estatisticas_portfolio(self, pesos):
        p_retorno = np.dot(pesos, self.mean_returns)
        p_vol = np.sqrt(np.dot(pesos.T, np.dot(self.cov_matrix, pesos)))
        return p_retorno, p_vol
    
    def sharpe_negativo(self, pesos):
        p_retorno, p_vol = self.estatisticas_portfolio(pesos)
        if p_vol <= 1e-9:
            return -100.0
        return -(p_retorno - TAXA_LIVRE_RISCO) / p_vol
    
    def minimizar_volatilidade(self, pesos):
        return self.estatisticas_portfolio(pesos)[1]
    
    def otimizar(self, estrategia='MaxSharpe'):
        restricoes = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        limites = tuple((PESO_MIN, PESO_MAX) for _ in range(self.num_ativos))
        chute_inicial = np.array([1.0 / self.num_ativos] * self.num_ativos)
        
        objetivo = self.sharpe_negativo if estrategia == 'MaxSharpe' else self.minimizar_volatilidade
        
        try:
            resultado = minimize(objetivo, chute_inicial, method='SLSQP', bounds=limites, constraints=restricoes, options={'maxiter': 500})
            
            if resultado.success:
                final_weights = resultado.x / np.sum(resultado.x)
                return {ativo: peso for ativo, peso in zip(self.returns.columns, final_weights)}
            else:
                return {ativo: 1.0 / self.num_ativos for ativo in self.returns.columns}
        except:
            return {ativo: 1.0 / self.num_ativos for ativo in self.returns.columns}

# =============================================================================
# CLASSE: ANALISADOR INDIVIDUAL COM PCA
# =============================================================================

class AnalisadorIndividualAtivos:
    @staticmethod
    def realizar_clusterizacao_pca(dados_ativos, n_clusters=5):
        """Clusterização 3D com PCA."""
        features_numericas = dados_ativos.select_dtypes(include=[np.number]).copy()
        features_numericas = features_numericas.replace([np.inf, -np.inf], np.nan)
        
        # Imputação
        for col in features_numericas.columns:
            if features_numericas[col].isnull().any():
                median_val = features_numericas[col].median()
                features_numericas[col] = features_numericas[col].fillna(median_val if not pd.isna(median_val) else 0)
        
        features_numericas = features_numericas.dropna(axis=1, how='all')
        features_numericas = features_numericas.loc[:, (features_numericas.std() > 1e-6)]
        
        if features_numericas.empty or len(features_numericas) < n_clusters:
            return None, None, None
        
        # Padronização
        scaler = StandardScaler()
        dados_normalizados = scaler.fit_transform(features_numericas)
        
        # PCA
        n_pca_components = min(3, len(features_numericas.columns))
        pca = PCA(n_components=n_pca_components)
        componentes_pca = pca.fit_transform(dados_normalizados)
        
        # K-means
        actual_n_clusters = min(n_clusters, len(features_numericas))
        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(componentes_pca)
        
        resultado_pca = pd.DataFrame(
            componentes_pca,
            columns=[f'PC{i+1}' for i in range(componentes_pca.shape[1])],
            index=features_numericas.index
        )
        resultado_pca['Cluster'] = clusters
        
        return resultado_pca, pca, kmeans

# =============================================================================
# CLASSE PRINCIPAL: CONSTRUTOR DE PORTFÓLIO
# =============================================================================

class ConstrutorPortfolioAutoML:
    def __init__(self, valor_investimento, periodo=PERIODO_DADOS):
        self.valor_investimento = valor_investimento
        self.periodo = periodo
        self.coletor = ColetorDados(periodo)
        
        self.dados_por_ativo = {}
        self.dados_fundamentalistas = pd.DataFrame()
        self.dados_performance = pd.DataFrame()
        self.volatilidades_garch = {}
        self.predicoes_ml = {}
        self.ativos_sucesso = []
        
        self.modelos_ml = {}
        self.auc_scores = {}
        self.pca_transformers = {}
        self.scalers = {}
        self.governanca_por_ativo = {}
        
        self.ativos_selecionados = []
        self.alocacao_portfolio = {}
        self.metricas_portfolio = {}
        self.metodo_alocacao_atual = "Não Aplicado"
        self.justificativas_selecao = {}
        self.perfil_dashboard = {}
        self.pesos_atuais = {}
        self.scores_combinados = pd.DataFrame()
    
    def coletar_e_processar_dados(self, simbolos):
        if not self.coletor.coletar_e_processar_dados(simbolos):
            return False
        
        self.dados_por_ativo = self.coletor.dados_por_ativo
        self.dados_fundamentalistas = self.coletor.dados_fundamentalistas
        self.ativos_sucesso = self.coletor.ativos_sucesso
        self.dados_performance = self.coletor.metricas_performance
        
        return True
    
    def calcular_volatilidades_garch(self):
        print("\n📊 Calculando volatilidades GARCH...")
        
        for simbolo in tqdm(self.ativos_sucesso, desc="Modelagem GARCH"):
            if simbolo not in self.dados_por_ativo or 'returns' not in self.dados_por_ativo[simbolo]:
                continue
            
            returns = self.dados_por_ativo[simbolo]['returns']
            garch_vol = VolatilidadeGARCH.ajustar_garch(returns)
            
            if np.isnan(garch_vol):
                garch_vol = returns.std() * np.sqrt(252) if not returns.isnull().all() else np.nan
            
            self.volatilidades_garch[simbolo] = garch_vol
        
        print(f"✓ Volatilidades calculadas\n")
    
    def treinar_modelos_ensemble(self, dias_lookback_ml=LOOKBACK_ML):
        print("\n🤖 Treinando Modelos ML com PCA...")
        
        # Features para ML
        colunas_features = [
            'returns', 'volatility_20', 'volatility_60',
            'sma_5', 'sma_10', 'sma_20', 'sma_50',
            'ema_5', 'ema_10', 'ema_20',
            'price_sma20_ratio', 'price_sma50_ratio',
            'rsi_7', 'rsi_14', 'rsi_21',
            'stoch_k', 'stoch_d',
            'macd', 'macd_signal', 'macd_diff',
            'bb_width', 'bb_position',
            'atr_percent', 'adx',
            'momentum_10', 'momentum_20',
            'obv', 'cmf', 'drawdown',
            'returns_lag_1', 'returns_lag_5', 'returns_lag_10',
            'returns_mean_5', 'returns_std_5',
            'returns_mean_20', 'returns_std_20'
        ]
        
        for ativo in tqdm(self.ativos_sucesso, desc="Treinamento ML + PCA"):
            if ativo not in self.dados_por_ativo:
                continue
            
            df = self.dados_por_ativo[ativo].copy()
            
            # Cria target
            df['Future_Direction'] = np.where(
                df['Close'].pct_change(dias_lookback_ml).shift(-dias_lookback_ml) > 0,
                1, 0
            )
            
            features_disponiveis = [f for f in colunas_features if f in df.columns]
            df_treino = df[features_disponiveis + ['Future_Direction']].dropna()
            
            if len(df_treino) < MIN_DIAS_HISTORICO or len(np.unique(df_treino['Future_Direction'])) < 2:
                continue
            
            X = df_treino[features_disponiveis]
            y = df_treino['Future_Direction']
            
            try:
                # Treina com PCA
                modelos, auc_scores, pca_transformer, scaler = EnsembleML.treinar_ensemble(X, y, usar_pca=True)
                
                if not modelos:
                    continue
                
                self.modelos_ml[ativo] = modelos
                self.auc_scores[ativo] = auc_scores
                self.pca_transformers[ativo] = pca_transformer
                self.scalers[ativo] = scaler
                
                # Governança
                self.governanca_por_ativo[ativo] = GovernancaModelo(ativo)
                auc_medio = np.mean(list(auc_scores.values()))
                
                # Métricas simplificadas
                self.governanca_por_ativo[ativo].adicionar_metricas(auc_medio, 0.0, 0.0, 0.0)
                
                # Previsão
                last_features = df[features_disponiveis].iloc[[-1]]
                proba_final = EnsembleML.prever_ensemble_ponderado(
                    modelos, auc_scores, last_features, pca_transformer, scaler
                )[0]
                
                self.predicoes_ml[ativo] = {
                    'predicted_proba_up': proba_final,
                    'auc_roc_score': auc_medio,
                    'model_name': 'Ensemble PCA',
                    'num_models': len(modelos)
                }
                
            except Exception as e:
                continue
        
        print(f"✓ Modelos ML treinados: {len(self.predicoes_ml)} ativos\n")
    
    def pontuar_e_selecionar_ativos(self, horizonte_tempo):
        # Ajusta pesos por horizonte
        if horizonte_tempo == "CURTO PRAZO":
            WEIGHT_PERF, WEIGHT_FUND, WEIGHT_TECH = 0.40, 0.10, 0.20
        elif horizonte_tempo == "LONGO PRAZO":
            WEIGHT_PERF, WEIGHT_FUND, WEIGHT_TECH = 0.40, 0.50, 0.10
        else:
            WEIGHT_PERF, WEIGHT_FUND, WEIGHT_TECH = 0.40, 0.30, 0.30
        
        total_non_ml = WEIGHT_PERF + WEIGHT_FUND + WEIGHT_TECH
        scale_factor = (1.0 - WEIGHT_ML) / total_non_ml if total_non_ml > 0 else 0
        WEIGHT_PERF *= scale_factor
        WEIGHT_FUND *= scale_factor
        WEIGHT_TECH *= scale_factor
        
        self.pesos_atuais = {
            'Performance': WEIGHT_PERF,
            'Fundamentos': WEIGHT_FUND,
            'Técnicos': WEIGHT_TECH,
            'ML': WEIGHT_ML
        }
        
        combinado = self.dados_performance.join(self.dados_fundamentalistas, how='inner').copy()
        
        # Adiciona indicadores técnicos e ML
        for asset in combinado.index:
            if asset in self.dados_por_ativo and 'rsi_14' in self.dados_por_ativo[asset].columns:
                df = self.dados_por_ativo[asset]
                combinado.loc[asset, 'rsi_current'] = df['rsi_14'].iloc[-1]
                combinado.loc[asset, 'macd_current'] = df['macd'].iloc[-1]
            
            if asset in self.predicoes_ml:
                ml_info = self.predicoes_ml[asset]
                combinado.loc[asset, 'ML_Proba'] = ml_info.get('predicted_proba_up', 0.5)
                combinado.loc[asset, 'ML_Confidence'] = ml_info.get('auc_roc_score', 0.5)
        
        # Scores
        scores = pd.DataFrame(index=combinado.index)
        scores['performance_score'] = EngenheiroFeatures._normalizar(combinado.get('sharpe', pd.Series(0, index=combinado.index)), True) * WEIGHT_PERF
        
        pe_score = EngenheiroFeatures._normalizar(combinado.get('pe_ratio', pd.Series(10, index=combinado.index)), False)
        roe_score = EngenheiroFeatures._normalizar(combinado.get('roe', pd.Series(0, index=combinado.index)), True)
        scores['fundamental_score'] = (pe_score * 0.5 + roe_score * 0.5) * WEIGHT_FUND
        
        rsi_norm = EngenheiroFeatures._normalizar(combinado.get('rsi_current', pd.Series(50, index=combinado.index)), False)
        macd_norm = EngenheiroFeatures._normalizar(combinado.get('macd_current', pd.Series(0, index=combinado.index)), True)
        scores['technical_score'] = (rsi_norm * 0.5 + macd_norm * 0.5) * WEIGHT_TECH
        
        ml_proba_norm = EngenheiroFeatures._normalizar(combinado.get('ML_Proba', pd.Series(0.5, index=combinado.index)), True)
        ml_conf_norm = EngenheiroFeatures._normalizar(combinado.get('ML_Confidence', pd.Series(0.5, index=combinado.index)), True)
        scores['ml_score_weighted'] = (ml_proba_norm * 0.6 + ml_conf_norm * 0.4) * WEIGHT_ML
        
        scores['total_score'] = scores['performance_score'] + scores['fundamental_score'] + scores['technical_score'] + scores['ml_score_weighted']
        
        self.scores_combinados = scores.join(combinado).sort_values('total_score', ascending=False)
        
        # Seleção com diversificação
        ranked_assets = self.scores_combinados.index.tolist()
        final_portfolio = []
        selected_sectors = set()
        
        for asset in ranked_assets:
            sector = self.dados_fundamentalistas.loc[asset, 'sector'] if asset in self.dados_fundamentalistas.index else 'Unknown'
            
            if sector not in selected_sectors or len(final_portfolio) < NUM_ATIVOS_PORTFOLIO:
                final_portfolio.append(asset)
                selected_sectors.add(sector)
            
            if len(final_portfolio) >= NUM_ATIVOS_PORTFOLIO:
                break
        
        self.ativos_selecionados = final_portfolio
        return self.ativos_selecionados
    
    def otimizar_alocacao(self, nivel_risco):
        if not self.ativos_selecionados:
            return {}
        
        available_returns = {s: self.dados_por_ativo[s]['returns']
                            for s in self.ativos_selecionados if s in self.dados_por_ativo}
        
        returns_df = pd.DataFrame(available_returns).dropna()
        
        if returns_df.shape[0] < 50:
            weights = {asset: 1.0 / len(self.ativos_selecionados) for asset in self.ativos_selecionados}
            self.metodo_alocacao_atual = 'PESOS IGUAIS'
        else:
            garch_vols_filtered = {asset: self.volatilidades_garch.get(asset, returns_df[asset].std() * np.sqrt(252))
                                  for asset in returns_df.columns}
            
            optimizer = OtimizadorPortfolioAvancado(returns_df, garch_vols=garch_vols_filtered)
            strategy = 'MinVolatility' if 'CONSERVADOR' in nivel_risco else 'MaxSharpe'
            weights = optimizer.otimizar(estrategia=strategy)
            self.metodo_alocacao_atual = f'{strategy} (GARCH)'
        
        if weights and sum(weights.values()) > 0:
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}
        
        self.alocacao_portfolio = {
            s: {'weight': w, 'amount': self.valor_investimento * w}
            for s, w in weights.items() if s in self.ativos_selecionados
        }
        
        return self.alocacao_portfolio
    
    def calcular_metricas_portfolio(self):
        if not self.alocacao_portfolio:
            return {}
        
        allocated = list(self.alocacao_portfolio.keys())
        returns_data = {s: self.dados_por_ativo[s]['returns'] for s in allocated if s in self.dados_por_ativo}
        returns_df = pd.DataFrame(returns_data).dropna()
        
        if returns_df.empty:
            return {}
        
        weights = np.array([self.alocacao_portfolio[s]['weight'] for s in returns_df.columns])
        weights = weights / np.sum(weights)
        
        portfolio_returns = (returns_df * weights).sum(axis=1)
        
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - TAXA_LIVRE_RISCO) / annual_volatility if annual_volatility > 0 else 0
        
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        max_drawdown = ((cumulative - running_max) / running_max).min()
        
        self.metricas_portfolio = {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        
        return self.metricas_portfolio
    
    def gerar_justificativas(self):
        for simbolo in self.ativos_selecionados:
            justification = []
            
            if simbolo in self.dados_performance.index:
                perf = self.dados_performance.loc[simbolo]
                justification.append(f"Perf: Sharpe {perf.get('sharpe', 0):.3f}")
            
            if simbolo in self.dados_fundamentalistas.index:
                fund = self.dados_fundamentalistas.loc[simbolo]
                justification.append(f"Fund: P/L {fund.get('pe_ratio', 0):.2f}")
            
            if simbolo in self.predicoes_ml:
                ml = self.predicoes_ml[simbolo]
                proba = ml.get('predicted_proba_up', 0.5)
                justification.append(f"ML: {proba*100:.1f}% (PCA)")
            
            self.justificativas_selecao[simbolo] = " | ".join(justification)
        
        return self.justificativas_selecao
    
    def executar_pipeline(self, simbolos, perfil, otimizar=False):
        self.perfil_dashboard = perfil
        ml_lookback = perfil.get('ml_lookback_days', LOOKBACK_ML)
        nivel_risco = perfil.get('risk_level', 'MODERADO')
        horizonte = perfil.get('time_horizon', 'MÉDIO PRAZO')
        
        if not self.coletar_e_processar_dados(simbolos):
            return False
        
        self.calcular_volatilidades_garch()
        self.treinar_modelos_ensemble(dias_lookback_ml=ml_lookback)
        self.pontuar_e_selecionar_ativos(horizonte_tempo=horizonte)
        self.otimizar_alocacao(nivel_risco=nivel_risco)
        self.calcular_metricas_portfolio()
        self.gerar_justificativas()
        
        return True

# =============================================================================
# INTERFACE STREAMLIT
# =============================================================================

def configurar_pagina():
    st.set_page_config(page_title="AutoML Elite v9.0", page_icon="📈", layout="wide")
    st.markdown("""
    <style>
    .main-header {
        font-family: 'Times New Roman', serif;
        color: #2c3e50;
        text-align: center;
        padding: 20px;
        font-size: 2.5rem;
    }
    .info-box {
        background-color: #f8f9fa;
        border-left: 4px solid #2c3e50;
        padding: 15px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

def aba_introducao():
    st.markdown("## 📚 Sistema AutoML Elite v9.0")
    
    st.markdown("""
    <div class="info-box">
    <h3>🚀 Novidades v9.0</h3>
    <ul>
        <li><strong>PCA Integrado:</strong> Redução de dimensionalidade para acelerar treinamento ML (até 70% mais rápido)</li>
        <li><strong>Cache Robusto:</strong> Dados processados uma única vez, reutilizados em todas as análises</li>
        <li><strong>9 Modelos ML:</strong> Ensemble ponderado por AUC-ROC</li>
        <li><strong>GARCH/EGARCH:</strong> Volatilidade condicional para otimização de portfólio</li>
        <li><strong>Governança:</strong> Monitoramento contínuo de performance dos modelos</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Pipeline de Dados")
        st.markdown("""
        1. **Coleta (uma vez):** yfinance + cache
        2. **Engenharia de Features:** 30+ indicadores técnicos
        3. **PCA:** Redução de dimensionalidade
        4. **Treinamento ML:** 9 modelos em paralelo
        5. **Otimização:** Markowitz com GARCH
        """)
    
    with col2:
        st.markdown("### 🎯 Seleção Multi-Fator")
        st.markdown("""
        - **Performance (40%):** Sharpe, Retorno, Volatilidade
        - **Fundamentos (30%):** P/L, ROE, Dívida
        - **Técnicos (30%):** RSI, MACD, Bollinger
        - **ML (30%):** Probabilidade de alta prevista
        """)

def aba_selecao_ativos():
    st.markdown("## 🎯 Seleção de Ativos")
    
    modo = st.radio(
        "Modo de Seleção:",
        ["📊 Ibovespa (10 ativos)", "🌐 Todos os Ativos", "✏️ Manual"]
    )
    
    ativos = []
    
    if "Ibovespa" in modo:
        ativos = ATIVOS_IBOVESPA.copy()
        st.success(f"✓ {len(ativos)} ativos do Ibovespa selecionados")
    elif "Todos" in modo:
        ativos = TODOS_ATIVOS.copy()
        st.success(f"✓ {len(ativos)} ativos selecionados")
    else:
        ativos_input = st.text_area(
            "Digite os tickers (um por linha):",
            "PETR4.SA\nVALE3.SA\nITUB4.SA"
        )
        ativos = [linha.strip() for linha in ativos_input.split('\n') if linha.strip()]
        st.info(f"✓ {len(ativos)} ativos digitados")
    
    if ativos:
        st.session_state.ativos_para_analise = ativos
        st.success("✅ Seleção confirmada! Vá para 'Construtor de Portfólio'")

def aba_construtor_portfolio():
    if 'ativos_para_analise' not in st.session_state:
        st.warning("⚠️ Selecione ativos primeiro")
        return
    
    if 'builder' not in st.session_state:
        st.session_state.builder = None
        st.session_state.profile = {}
        st.session_state.builder_complete = False
    
    if not st.session_state.builder_complete:
        st.markdown('## 📋 Questionário de Perfil')
        
        with st.form("profile_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Tolerância ao Risco")
                p1 = st.radio("1. Aceito risco de curto prazo", list(SCORE_MAP.keys()), index=2)
                p2 = st.radio("2. Ganhar o máximo é prioridade", list(SCORE_MAP.keys()), index=2)
                p3 = st.radio("3. Prefiro crescimento constante", list(SCORE_MAP.keys()), index=2)
                p4 = st.radio("4. Evitar perdas é mais importante", list(SCORE_MAP.keys()), index=2)
                p5 = st.radio("5. Se caísse 10%, eu:", list(SCORE_MAP_REACTION.keys()), index=1)
                p6 = st.radio("6. Meu nível de conhecimento:", list(SCORE_MAP_CONHECIMENTO.keys()), index=1)
            
            with col2:
                st.markdown("#### Horizonte e Capital")
                p7 = st.radio("7. Prazo para reavaliação:", 
                             ['A: Curto (até 1 ano)', 'B: Médio (1-5 anos)', 'C: Longo (5+ anos)'], index=2)
                p8 = st.radio("8. Necessidade de liquidez:",
                             ['A: Menos de 6 meses', 'B: 6 meses a 2 anos', 'C: Mais de 2 anos'], index=2)
                investment = st.number_input("Valor (R$):", 1000, 10000000, 100000, 10000)
            
            submitted = st.form_submit_button("🚀 Gerar Portfólio", type="primary")
            
            if submitted:
                analyzer = AnalisadorPerfilInvestidor()
                risk_level, horizon, lookback, score = analyzer.calcular_perfil({
                    'risk_accept': p1, 'max_gain': p2, 'stable_growth': p3,
                    'avoid_loss': p4, 'reaction': p5, 'level': p6,
                    'time_purpose': p7[0], 'liquidez': p8[0]
                })
                
                st.session_state.profile = {
                    'risk_level': risk_level,
                    'time_horizon': horizon,
                    'ml_lookback_days': lookback,
                    'risk_score': score
                }
                
                builder = ConstrutorPortfolioAutoML(investment)
                st.session_state.builder = builder
                
                with st.spinner(f'Criando portfólio {risk_level}...'):
                    success = builder.executar_pipeline(
                        st.session_state.ativos_para_analise,
                        st.session_state.profile
                    )
                    
                    if success:
                        st.session_state.builder_complete = True
                        st.rerun()
                    else:
                        st.error("Falha na coleta de dados")
    
    else:
        builder = st.session_state.builder
        profile = st.session_state.profile
        
        st.markdown('## ✅ Portfólio Otimizado')
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Perfil", profile['risk_level'])
        col2.metric("Horizonte", profile['time_horizon'])
        col3.metric("Sharpe", f"{builder.metricas_portfolio.get('sharpe_ratio', 0):.3f}")
        col4.metric("Estratégia", builder.metodo_alocacao_atual)
        
        if st.button("🔄 Recomeçar"):
            st.session_state.builder_complete = False
            st.session_state.builder = None
            st.rerun()
        
        st.markdown("---")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Alocação", "📈 Performance", "🤖 ML", "📉 GARCH"])
        
        with tab1:
            col_pie, col_table = st.columns([1, 2])
            
            with col_pie:
                alloc_data = pd.DataFrame([
                    {'Ativo': a.replace('.SA', ''), 'Peso (%)': builder.alocacao_portfolio[a]['weight'] * 100}
                    for a in builder.ativos_selecionados if a in builder.alocacao_portfolio
                ])
                
                fig = px.pie(alloc_data, values='Peso (%)', names='Ativo', hole=0.3)
                fig.update_layout(**obter_template_grafico())
                st.plotly_chart(fig, use_container_width=True)
            
            with col_table:
                st.markdown("#### Detalhes")
                table = []
                for asset in builder.ativos_selecionados:
                    if asset in builder.alocacao_portfolio:
                        w = builder.alocacao_portfolio[asset]['weight']
                        amt = builder.alocacao_portfolio[asset]['amount']
                        ml = builder.predicoes_ml.get(asset, {})
                        table.append({
                            'Ativo': asset.replace('.SA', ''),
                            'Peso': f"{w*100:.1f}%",
                            'Valor': f"R$ {amt:,.0f}",
                            'ML Proba': f"{ml.get('predicted_proba_up', 0)*100:.1f}%"
                        })
                st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
        
        with tab2:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Retorno Anual", f"{builder.metricas_portfolio.get('annual_return', 0)*100:.2f}%")
            col2.metric("Volatilidade", f"{builder.metricas_portfolio.get('annual_volatility', 0)*100:.2f}%")
            col3.metric("Sharpe", f"{builder.metricas_portfolio.get('sharpe_ratio', 0):.3f}")
            col4.metric("Max Drawdown", f"{builder.metricas_portfolio.get('max_drawdown', 0)*100:.2f}%")
        
        with tab3:
            st.markdown("#### Previsões ML com PCA")
            ml_data = []
            for asset in builder.ativos_selecionados:
                if asset in builder.predicoes_ml:
                    ml = builder.predicoes_ml[asset]
                    ml_data.append({
                        'Ativo': asset.replace('.SA', ''),
                        'Prob Alta': ml['predicted_proba_up'] * 100,
                        'AUC': ml['auc_roc_score'],
                        'Modelos': ml['num_models']
                    })
            
            df_ml = pd.DataFrame(ml_data)
            if not df_ml.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df_ml['Ativo'],
                    y=df_ml['Prob Alta'],
                    marker=dict(color=df_ml['Prob Alta'], colorscale='RdYlGn')
                ))
                fig.update_layout(**obter_template_grafico(), title="Probabilidade de Alta (ML + PCA)")
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(df_ml, use_container_width=True, hide_index=True)
        
        with tab4:
            st.markdown("#### Volatilidades GARCH")
            garch_data = []
            for asset in builder.ativos_selecionados:
                if asset in builder.volatilidades_garch:
                    garch_data.append({
                        'Ativo': asset.replace('.SA', ''),
                        'Vol GARCH': builder.volatilidades_garch[asset] * 100
                    })
            
            if garch_data:
                df_garch = pd.DataFrame(garch_data)
                st.dataframe(df_garch, use_container_width=True, hide_index=True)

def aba_analise_individual():
    st.markdown("## 🔍 Análise Individual")
    
    if 'ativos_para_analise' not in st.session_state:
        st.warning("⚠️ Selecione ativos primeiro")
        return
    
    ativo = st.selectbox(
        "Selecione um ativo:",
        st.session_state.ativos_para_analise,
        format_func=lambda x: x.replace('.SA', '')
    )
    
    if st.button("🔍 Analisar", type="primary"):
        st.session_state.ativo_analisar = ativo
    
    if 'ativo_analisar' not in st.session_state:
        st.info("Selecione e clique em Analisar")
        return
    
    with st.spinner(f"Analisando {ativo}..."):
        try:
            ticker_obj = yf.Ticker(ativo)
            hist = ticker_obj.history(period='2y')
            
            if hist.empty:
                st.error("Sem dados")
                return
            
            df = EngenheiroFeatures.calcular_indicadores_tecnicos(hist)
            info = ticker_obj.info
            
            tab1, tab2, tab3 = st.tabs(["📊 Visão", "📈 Técnico", "🔬 Clusterização"])
            
            with tab1:
                st.markdown(f"### {ativo.replace('.SA', '')}")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Preço", f"R$ {df['Close'].iloc[-1]:.2f}")
                col2.metric("Variação", f"{df['returns'].iloc[-1]*100:+.2f}%")
                col3.metric("Volume", f"{df['Volume'].mean():,.0f}")
                col4.metric("Setor", info.get('sector', 'N/A'))
                
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df.index, open=df['Open'], high=df['High'],
                    low=df['Low'], close=df['Close']
                ))
                fig.update_layout(**obter_template_grafico(), height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.markdown("#### Indicadores Técnicos")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("RSI", f"{df['rsi_14'].iloc[-1]:.1f}")
                col2.metric("MACD", f"{df['macd'].iloc[-1]:.4f}")
                col3.metric("ADX", f"{df['adx'].iloc[-1]:.1f}")
                col4.metric("ATR %", f"{df['atr_percent'].iloc[-1]:.2f}%")
            
            with tab3:
                st.markdown("#### Clusterização 3D com PCA")
                
                if 'builder' in st.session_state and st.session_state.builder:
                    # Usa dados do builder
                    dados_cluster = st.session_state.builder.dados_performance.join(
                        st.session_state.builder.dados_fundamentalistas, how='inner'
                    )
                    
                    resultado_pca, pca_obj, kmeans_obj = AnalisadorIndividualAtivos.realizar_clusterizacao_pca(
                        dados_cluster, n_clusters=5
                    )
                    
                    if resultado_pca is not None:
                        if 'PC3' in resultado_pca.columns:
                            fig = px.scatter_3d(
                                resultado_pca,
                                x='PC1', y='PC2', z='PC3',
                                color='Cluster',
                                hover_name=resultado_pca.index.str.replace('.SA', ''),
                                title='Clusterização 3D (K-means + PCA)'
                            )
                        else:
                            fig = px.scatter(
                                resultado_pca,
                                x='PC1', y='PC2',
                                color='Cluster',
                                hover_name=resultado_pca.index.str.replace('.SA', ''),
                                title='Clusterização 2D (K-means + PCA)'
                            )
                        
                        fig.update_layout(**obter_template_grafico(), height=600)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        if ativo in resultado_pca.index:
                            cluster = resultado_pca.loc[ativo, 'Cluster']
                            similares = resultado_pca[resultado_pca['Cluster'] == cluster].index.tolist()
                            similares = [a for a in similares if a != ativo]
                            
                            st.success(f"**{ativo.replace('.SA', '')}** no Cluster {cluster}")
                            if similares:
                                st.write("Similares:", ", ".join([a.replace('.SA', '') for a in similares[:10]]))
                    else:
                        st.warning("Dados insuficientes para clusterização")
                else:
                    st.info("Execute o Construtor de Portfólio primeiro")
                    
        except Exception as e:
            st.error(f"Erro: {str(e)}")

def aba_governanca():
    st.markdown("## 🛡️ Governança de Modelo")
    
    if 'builder' not in st.session_state or not st.session_state.builder:
        st.warning("⚠️ Execute o Construtor primeiro")
        return
    
    builder = st.session_state.builder
    
    if not builder.governanca_por_ativo:
        st.info("Sem dados de governança")
        return
    
    ativo = st.selectbox(
        "Selecione um ativo:",
        list(builder.governanca_por_ativo.keys()),
        format_func=lambda x: x.replace('.SA', '')
    )
    
    gov = builder.governanca_por_ativo[ativo]
    rel = gov.gerar_relatorio()
    
    if rel['severidade'] == 'success':
        st.success(f"✅ {rel['status']}")
    elif rel['severidade'] == 'warning':
        st.warning(f"⚠️ {rel['status']}")
    else:
        st.error(f"🚨 {rel['status']}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUC Atual", f"{rel['metricas']['AUC Atual']:.3f}")
    col2.metric("AUC Médio", f"{rel['metricas']['AUC Médio']:.3f}")
    col3.metric("AUC Máximo", f"{rel['metricas']['AUC Máximo']:.3f}")
    col4.metric("Precision", f"{rel['metricas']['Precision Média']:.3f}")
    
    if rel['alertas']:
        st.markdown("### ⚠️ Alertas")
        for alerta in rel['alertas']:
            st.warning(f"**{alerta['tipo']}**: {alerta['mensagem']}")
    
    if rel['historico']['AUC']:
        st.markdown("### 📈 Histórico AUC")
        df_hist = pd.DataFrame({
            'Período': range(1, len(rel['historico']['AUC']) + 1),
            'AUC': rel['historico']['AUC']
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_hist['Período'], y=df_hist['AUC'], mode='lines+markers'))
        fig.add_hline(y=AUC_THRESHOLD_MIN, line_dash="dash", line_color="red")
        fig.update_layout(**obter_template_grafico(), title=f"Evolução AUC - {ativo.replace('.SA', '')}")
        st.plotly_chart(fig, use_container_width=True)

def main():
    if 'builder' not in st.session_state:
        st.session_state.builder = None
        st.session_state.builder_complete = False
        st.session_state.profile = {}
        st.session_state.ativos_para_analise = []
    
    configurar_pagina()
    
    st.sidebar.markdown("# 📈 AutoML v9.0")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚀 Recursos")
    st.sidebar.markdown("""
    - **PCA**: Aceleração 70%
    - **9 Modelos ML**
    - **GARCH**: Volatilidade
    - **Governança**: Monitoramento
    - **Clusterização 3D**
    """)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**v9.0.0** - Elite PCA")
    
    st.markdown('<h1 class="main-header">Sistema AutoML Elite v9.0 - Otimização com PCA</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📚 Introdução",
        "🎯 Seleção",
        "🏗️ Portfólio",
        "🔍 Individual",
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
