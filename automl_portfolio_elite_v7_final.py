# -*- coding: utf-8 -*-
"""
=============================================================================
SISTEMA DE OTIMIZAÇÃO QUANTITATIVA
=============================================================================

Modelo de Alocação de Ativos com Métodos Adaptativos.
- Preços: Estratégia Linear com Fail-Fast (YFinance -> TvDatafeed -> Estático Global). 
- Fundamentos: Coleta Exaustiva Pynvest (50+ indicadores).
- GARCH: Modelagem de Volatilidade Condicional com Critério Hannan-Quinn.
- ML: Score Ponderado (AUC x Probabilidade) com ElasticNet + LogReg/Ensemble RF+XGB.

Versão: 10.0.0 (Professional Build: GARCH + ML Avançado + Score Ponderado)
=============================================================================
"""


import warnings
warnings.filterwarnings('ignore') 

import numpy as np
import pandas as pd
import subprocess
import sys
import os
import time
from datetime import datetime, timedelta
import json
import traceback

from scipy.optimize import minimize
from scipy.stats import zscore, norm
import math

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests


try:
    from tvDatafeed import TvDatafeed, Interval
except ImportError:
    st.error("""
    Biblioteca 'tvdatafeed' não encontrada. 
    Instale usando: pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git
    """)

import yfinance as yf

try:
    from pynvest.scrappers.fundamentus import Fundamentus
except ImportError:
    pass

from sklearn.ensemble import IsolationForest, RandomForestClassifier 
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer 
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, ElasticNet

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    from arch import arch_model
    from arch.univariate import GARCH, ConstantMean
except ImportError:
    arch_model = None
    st.warning("Biblioteca 'arch' não encontrada. GARCH desabilitado. Instale: pip install arch")

try:
    import hdbscan
except ImportError:
    pass


PERIODO_DADOS = 'max'
MIN_DIAS_HISTORICO = 252
NUM_ATIVOS_PORTFOLIO = 5
TAXA_LIVRE_RISCO = 0.1075
LOOKBACK_ML = 30
SCORE_PERCENTILE_THRESHOLD = 0.85

PESO_MIN = 0.10
PESO_MAX = 0.30

MIN_TRAIN_DAYS_ML = 120 

def get_ml_horizons(ml_lookback_days: int):
    """Adapta os horizontes de predição ML com base no lookback do perfil (CP/MP/LP)."""
    
    if ml_lookback_days >= 252:
        return [80, 160, 240]
    elif ml_lookback_days >= 168:
        return [50, 100, 150]
    else:
        return [20, 40, 60]

LOOKBACK_ML_DAYS_MAP = {
    'curto_prazo': 84,
    'medio_prazo': 168,
    'longo_prazo': 252
}

LGBM_FEATURES = ["ret", "vol20", "ma20", "z20", "trend", "volrel"]


ATIVOS_IBOVESPA = [
    'ALOS3.SA', 'ABEV3.SA', 'ASAI3.SA', 'AESB3.SA', 'AZZA3.SA', 'B3SA3.SA',
    'BBSE3.SA', 'BBDC3.SA', 'BBDC4.SA', 'BRAP4.SA', 'BBAS3.SA', 'BRKM5.SA',
    'BRAV3.SA', 'BPAC11.SA', 'CXSE3.SA', 'CEAB3.SA', 'CMIG4.SA', 'COGN3.SA',
    'CPLE6.SA', 'CSAN3.SA', 'CPFE3.SA', 'CMIN3.SA', 'CURY3.SA', 'CVCB3.SA',
    'CYRE3.SA', 'DIRR3.SA', 'ELET3.SA', 'ELET6.SA', 'EMBR3.SA', 'ENGI11.SA',
    'ENEV3.SA', 'EGIE3.SA', 'EQTL3.SA', 'FLRY3.SA', 'GGBR4.SA', 'GOAU4.SA',
    'HAPV3.SA', 'HYPE3.SA', 'IGTI11.SA', 'IRBR3.SA', 'ISAE4.SA', 'ITSA4.SA',
    'ITUB4.SA', 'KLBN11.SA', 'RENT3.SA', 'LREN3.SA', 'MGLU3.SA', 'POMO4.SA',
    'BEEF3.SA', 'MRVE3.SA', 'MULT3.SA', 'NATU3.SA',
    'PCAR3.SA', 'PETR3.SA', 'PETR4.SA', 'RECV3.SA', 'PRIO3.SA', 'PSSA3.SA',
    'RADL3.SA', 'RAIZ4.SA', 'RDOR3.SA', 'RAIL3.SA', 'SBSP3.SA', 'SANB11.SA',
    'CSNA3.SA', 'SLCE3.SA', 'SMFT3.SA', 'SUZB3.SA', 'TAEE11.SA', 'VIVT3.SA',
    'TIMS3.SA', 'TOTS3.SA', 'UGPA3.SA', 'USIM5.SA', 'VALE3.SA', 'VAMO3.SA',
    'VBBR3.SA', 'VIVA3.SA', 'WEGE3.SA', 'YDUQ3.SA'
]


ATIVOS_POR_SETOR_IBOV = {
    'Bens Industriais': ['EMBR3.SA', 'VAMO3.SA', 'WEGE3.SA', 'VIVA3.SA', 'ASAI3.SA', 'SMFT3.SA', 'CMIN3.SA', 'SLCE3.SA'],
    'Consumo Cíclico': ['AZZA3.SA', 'ALOS3.SA', 'CEAB3.SA', 'COGN3.SA', 'CURY3.SA', 'CVCB3.SA', 'CYRE3.SA', 'DIRR3.SA', 'LREN3.SA', 'MGLU3.SA', 'MRVE3.SA', 'MULT3.SA', 'NATU3.SA', 'PCAR3.SA', 'RENT3.SA', 'YDUQ3.SA'],
    'Consumo não Cíclico': ['BEEF3.SA', 'NATU3.SA', 'PCAR3.SA', 'VIVA3.SA'], 
    'Financeiro': ['B3SA3.SA', 'BBSE3.SA', 'BBDC3.SA', 'BBDC4.SA', 'BBAS3.SA', 'BPAC11.SA', 'CXSE3.SA', 'HYPE3.SA', 'IGTI11.SA', 'IRBR3.SA', 'ITSA4.SA', 'ITUB4.SA', 'MULT3.SA', 'PSSA3.SA', 'RDOR3.SA', 'SANB11.SA'],
    'Materiais Básicos': ['BRAP4.SA', 'BRKM5.SA', 'CSNA3.SA', 'GGBR4.SA', 'GOAU4.SA', 'KLBN11.SA', 'POMO4.SA', 'SUZB3.SA', 'USIM5.SA', 'VALE3.SA'],
    'Petróleo, Gás e Biocombustíveis': ['ENEV3.SA', 'PETR3.SA', 'PETR4.SA', 'PRIO3.SA', 'RAIZ4.SA', 'RECV3.SA', 'UGPA3.SA', 'VBBR3.SA'],
    'Saúde': ['FLRY3.SA', 'HAPV3.SA', 'RADL3.SA'],
    'Tecnologia da Informação': ['TOTS3.SA'],
    'Telecomunicações': ['TIMS3.SA', 'VIVT3.SA'],
    'Utilidade Pública': ['AESB3.SA', 'AURE3.SA', 'BRAV3.SA', 'CMIG4.SA', 'CPLE6.SA', 'CPFE3.SA', 'EGIE3.SA', 'ELET3.SA', 'ELET6.SA', 'ENGI11.SA', 'EQTL3.SA', 'ISAE4.SA', 'RAIL3.SA', 'SBSP3.SA', 'TAEE11.SA']
}

FALLBACK_SETORES = {}
for setor, tickers in ATIVOS_POR_SETOR_IBOV.items():
    for t in tickers:
        FALLBACK_SETORES[t] = setor

TODOS_ATIVOS = sorted(list(set(ATIVOS_IBOVESPA)))

ATIVOS_POR_SETOR = {
    setor: [ativo for ativo in ativos if ativo in ATIVOS_IBOVESPA] 
    for setor, ativos in ATIVOS_POR_SETOR_IBOV.items()
    if any(ativo in ATIVOS_IBOVESPA for ativo in ativos)
}


SCORE_MAP_ORIGINAL = {
    'CT: Concordo Totalmente': 5, 'C: Concordo': 4, 'N: Neutro': 3, 'D: Discordo': 2, 'DT: Discordo Totalmente': 1
}
SCORE_MAP_INV_ORIGINAL = {
    'CT: Concordo Totalmente': 1, 'C: Concordo': 2, 'N: Neutro': 3, 'D: Discordo': 4, 'DT: Discordo Totalmente': 5
}
SCORE_MAP_CONHECIMENTO_ORIGINAL = {
    'A: Avançado (Análise fundamentalista, macro e técnica)': 5, 
    'B: Intermediário (Conhecimento básico sobre mercados e ativos)': 3, 
    'C: Iniciante (Pouca ou nenhuma experiência em investimentos)': 1
}
SCORE_MAP_REACTION_ORIGINAL = {
    'A: Venderia imediatamente': 1, 
    'B: Manteria e reavaliaria a tese': 3, 
    'C: Compraria mais para aproveitar preços baixos': 5
}

OPTIONS_CONCORDA = [
    "CT: (Concordo Totalmente) - Estou confortável com altas flutuações, pois entendo que são o preço para retornos potencialmente maiores.",
    "C: (Concordo) - Aceito alguma volatilidade, mas espero que os ganhos compensem o risco assumido de forma clara.",
    "N: (Neutro) - Tenho dificuldade em opinar; minha decisão dependeria do momento e do ativo específico.",
    "D: (Discordo) - Prefiro estratégias mais cautelosas, mesmo que isso signifique um potencial de retorno menor.",
    "DT: (Discordo Totalmente) - Não estou disposto a ver meu patrimônio flutuar significativamente; prefiro segurança absoluta."
]
MAP_CONCORDA = {OPTIONS_CONCORDA[0]: 'CT: Concordo Totalmente', OPTIONS_CONCORDA[1]: 'C: Concordo', OPTIONS_CONCORDA[2]: 'N: Neutro', OPTIONS_CONCORDA[3]: 'D: Discordo', OPTIONS_CONCORDA[4]: 'DT: Discordo Totalmente'}

OPTIONS_DISCORDA = [
    "CT: (Concordo Totalmente) - A preservação do capital é minha prioridade máxima, acima de qualquer ganho potencial.",
    "C: (Concordo) - É muito importante para mim evitar perdas, mesmo que isso limite o crescimento do meu portfólio.",
    "N: (Neutro) - Busco um equilíbrio; não quero perdas excessivas, mas sei que algum risco é necessário para crescer.",
    "D: (Discordo) - Estou focado no crescimento de longo prazo e entendo que perdas de curto prazo fazem parte do processo.",
    "DT: (Discordo Totalmente) - Meu foco é maximizar o retorno; perdas de curto prazo são irrelevantes se a tese de longo prazo for válida."
]
MAP_DISCORDA = {OPTIONS_DISCORDA[0]: 'CT: Concordo Totalmente', OPTIONS_DISCORDA[1]: 'C: Concordo', OPTIONS_DISCORDA[2]: 'N: Neutro', OPTIONS_DISCORDA[3]: 'D: Discordo', OPTIONS_DISCORDA[4]: 'DT: Discordo Totalmente'}

OPTIONS_REACTION_DETALHADA = [
    "A: (Vender Imediatamente) - Venderia a posição para evitar perdas maiores; prefiro realizar o prejuízo e reavaliar.",
    "B: (Manter e Reavaliar) - Manteria a calma, reavaliaria os fundamentos do ativo e o cenário macro para tomar uma decisão.",
    "C: (Comprar Mais) - Encararia como uma oportunidade de compra, aumentando a posição a um preço menor, se os fundamentos estiverem intactos."
]
MAP_REACTION = {OPTIONS_REACTION_DETALHADA[0]: 'A: Venderia imediatamente', OPTIONS_REACTION_DETALHADA[1]: 'B: Manteria e reavaliaria a tese', OPTIONS_REACTION_DETALHADA[2]: 'C: Compraria mais para aproveitar preços baixos'}

OPTIONS_CONHECIMENTO_DETALHADA = [
    "A: (Avançado) - Sinto-me confortável analisando balanços (fundamentalista), gráficos (técnica) e cenários macroeconômicos.",
    "B: (Intermediário) - Entendo os conceitos básicos (Renda Fixa vs. Variável, risco vs. retorno) e acompanho o mercado.",
    "C: (Iniciante) - Tenho pouca ou nenhuma experiência prática em investimentos além da poupança ou produtos bancários simples."
]
MAP_CONHECIMENTO = {OPTIONS_CONHECIMENTO_DETALHADA[0]: 'A: Avançado (Análise fundamentalista, macro e técnica)', OPTIONS_CONHECIMENTO_DETALHADA[1]: 'B: Intermediário (Conhecimento básico sobre mercados e ativos)', OPTIONS_CONHECIMENTO_DETALHADA[2]: 'C: Iniciante (Pouca ou nenhuma experiência em investimentos)'}

OPTIONS_TIME_HORIZON_DETALHADA = [
    'A: Curto (até 1 ano) - Meu objetivo é preservar capital ou realizar um ganho rápido, com alta liquidez.', 
    'B: Médio (1-5 anos) - Busco um crescimento balanceado e posso tolerar alguma flutuação neste período.', 
    'C: Longo (5+ anos) - Meu foco é a acumulação de patrimônio; flutuações de curto/médio prazo não me afetam.'
]
OPTIONS_LIQUIDEZ_DETALHADA = [
    'A: Menos de 6 meses - Posso precisar resgatar o valor a qualquer momento (ex: reserva de emergência).', 
    'B: Entre 6 meses e 2 anos - Não preciso do dinheiro imediatamente, mas tenho um objetivo de curto/médio prazo.', 
    'C: Mais de 2 anos - Este é um investimento de longo prazo; não tenho planos de resgatar nos próximos anos.'
]

def safe_format(value):
    """Formata valor float para string com 2 casas, tratando strings e NaNs."""
    if pd.isna(value):
        return "N/A"
    try:
        float_val = float(value)
        return f"{float_val:.2f}"
    except (ValueError, TypeError):
        return str(value)

def log_debug(message: str):
    """Adiciona uma mensagem formatada ao log de debug da sessão."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"DEBUG {timestamp} | {message}") 
    if 'debug_logs' in st.session_state:
        st.session_state.debug_logs.append(f"[{timestamp}] {message}")

def obter_template_grafico() -> dict:
    """Retorna o template padrão de cores e estilo para gráficos Plotly."""
    corporate_colors = ['#2E86C1', '#D35400', '#27AE60', '#8E44AD', '#C0392B', '#16A085', '#F39C12', '#34495E']
    return {
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'font': {'family': 'Inter, sans-serif', 'size': 12, 'color': '#343a40'},
        'title': {'font': {'family': 'Inter, sans-serif', 'size': 16, 'color': '#212529', 'weight': 'bold'}, 'x': 0.5, 'xanchor': 'center'},
        'xaxis': {'showgrid': True, 'gridcolor': '#ecf0f1', 'showline': True, 'linecolor': '#bdc3c7', 'linewidth': 1},
        'yaxis': {'showgrid': True, 'gridcolor': '#ecf0f1', 'showline': True, 'linecolor': '#bdc3c7', 'linewidth': 1},
        'legend': {'bgcolor': 'rgba(255,255,255,0.5)', 'bordercolor': '#ecf0f1'},
        'colorway': corporate_colors
    }

# =============================================================================
# 8. CLASSE: ENGENHEIRO DE FEATURES
# =============================================================================

class EngenheiroFeatures:
    @staticmethod
    def _normalize_score(serie: pd.Series, maior_melhor: bool = True) -> pd.Series:
        """Normaliza uma série de scores para uma escala de 0 a 100."""
        
        series_clean = serie.replace([np.inf, -np.inf], np.nan).dropna()
        if series_clean.empty or series_clean.std() == 0:
            return pd.Series(50, index=serie.index)
        z = zscore(series_clean, nan_policy='omit')
        normalized_values = 50 + (z.clip(-3, 3) / 3) * 50
        if not maior_melhor:
            normalized_values = 100 - normalized_values
        normalized_series = pd.Series(normalized_values, index=series_clean.index)
        return normalized_series.reindex(serie.index, fill_value=50)

# =============================================================================
# 8.1. CLASSE: CALCULADORA TÉCNICA
# =============================================================================

class CalculadoraTecnica:
    """
    Classe para enriquecer dados técnicos com features LGBM e indicadores tradicionais.
    """
    @staticmethod
    def enriquecer_dados_tecnicos(df_ativo: pd.DataFrame) -> pd.DataFrame:
        if df_ativo.empty: return df_ativo
        df = df_ativo.sort_index().copy()
        
        df.rename(columns={'Close': 'close', 'High': 'high', 'Low': 'low', 'Open': 'open', 'Volume': 'volume'}, inplace=True)

        df['returns'] = df['close'].pct_change()
        
        # === fast_features (LGBM) ===
        df["ret"] = np.log(df["close"] / df["close"].shift(1))
        df["vol20"] = df["ret"].rolling(20).std()
        df["ma20"] = df["close"].rolling(20).mean()
        df["z20"] = (df["close"] - df["ma20"]) / (df["close"].rolling(20).std() + 1e-9)
        df["trend"] = df["close"].diff(5)
        df["volrel"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-9)
        
        # --- MÉTRICAS DE SCORING / DISPLAY TRADICIONAL (RSI, MACD, BBands) ---
        
        df['vol_20d'] = df["ret"].rolling(20).std() * np.sqrt(252) 
        
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']

        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi_14'] = 100 - (100 / (1 + rs)) 
        
        rolling_mean_20 = df['close'].rolling(window=20).mean()
        rolling_std_20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = rolling_mean_20 + (rolling_std_20 * 2)
        df['bb_lower'] = rolling_mean_20 - (rolling_std_20 * 2)

        df.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low', 'open': 'Open', 'volume': 'Volume'}, inplace=True)
        
        cols_to_keep = ['Close', 'High', 'Low', 'Open', 'Volume', 'returns', 
                        'ret', 'vol20', 'ma20', 'z20', 'trend', 'volrel',
                        'vol_20d', 'macd_diff', 'rsi_14', 'macd', 'macd_signal', 'bb_upper', 'bb_lower']
        
        cols_to_drop = [c for c in df.columns if c not in cols_to_keep and c not in ['Close', 'High', 'Low', 'Open', 'Volume', 'returns']]
        df.drop(columns=cols_to_drop, errors='ignore', inplace=True)

        return df.dropna(subset=['Close']).fillna(0)

# =============================================================================
# =============================================================================

class ModeladorGARCH:
    """
    Classe para modelagem GARCH usando critério Hannan-Quinn para seleção de lags.
    """
    
    @staticmethod
    def selecionar_ordem_garch_hq(returns: pd.Series, max_p: int = 3, max_q: int = 3) -> tuple:
        """
        Seleciona a melhor ordem (p, q) para GARCH usando critério Hannan-Quinn.
        
        Args:
            returns: Série de retornos
            max_p: Máximo valor de p a testar
            max_q: Máximo valor de q a testar
            
        Returns:
            Tupla (p, q) com a melhor ordem segundo HQ
        """
        if arch_model is None:
            return (1, 1)
        
        returns_clean = returns.dropna()
        if len(returns_clean) < 100:
            return (1, 1)
        
        returns_scaled = returns_clean * 100
        
        best_hq = np.inf
        best_order = (1, 1)
        
        for p in range(1, max_p + 1):
            for q in range(1, max_q + 1):
                try:
                    model = arch_model(
                        returns_scaled,
                        vol='Garch',
                        p=p,
                        q=q,
                        dist='normal',
                        rescale=False
                    )
                    
                    result = model.fit(disp='off', show_warning=False)
                    
                    # Critério Hannan-Quinn: HQ = -2*log_likelihood + 2*k*log(log(n))
                    n = len(returns_scaled)
                    k = len(result.params)
                    log_likelihood = result.loglikelihood
                    hq = -2 * log_likelihood + 2 * k * np.log(np.log(n))
                    
                    if hq < best_hq:
                        best_hq = hq
                        best_order = (p, q)
                        
                except:
                    continue
        
        log_debug(f"GARCH: Ordem selecionada por HQ: GARCH({best_order[0]},{best_order[1]}) com HQ={best_hq:.2f}")
        return best_order
    
    @staticmethod
    def calcular_volatilidade_garch(returns: pd.Series, use_hq: bool = True) -> tuple:
        """
        Calcula volatilidade condicional usando GARCH.
        
        Args:
            returns: Série de retornos
            use_hq: Se True, usa Hannan-Quinn para seleção de ordem
            
        Returns:
            Tupla (volatilidade_anualizada, nome_modelo)
        """
        if arch_model is None:
            vol_hist = returns.std() * np.sqrt(252)
            return (vol_hist, "Vol. Histórica (GARCH não disponível)")
        
        returns_clean = returns.dropna()
        if len(returns_clean) < 100:
            vol_hist = returns.std() * np.sqrt(252)
            return (vol_hist, "Vol. Histórica (Dados Insuficientes)")
        
        try:
            if use_hq:
                p, q = ModeladorGARCH.selecionar_ordem_garch_hq(returns_clean)
            else:
                p, q = 1, 1
            
            returns_scaled = returns_clean * 100
            
            model = arch_model(
                returns_scaled,
                vol='Garch',
                p=p,
                q=q,
                dist='normal',
                rescale=False
            )
            
            result = model.fit(disp='off', show_warning=False)
            
            forecast = result.forecast(horizon=1)
            vol_forecast = np.sqrt(forecast.variance.values[-1, 0])
            vol_anualizada = (vol_forecast / 100) * np.sqrt(252)
            
            model_name = f"GARCH({p},{q})"
            
            return (vol_anualizada, model_name)
            
        except Exception as e:
            log_debug(f"GARCH: Falha no ajuste - {str(e)[:50]}")
            vol_hist = returns.std() * np.sqrt(252)
            return (vol_hist, "Vol. Histórica (Fallback)")

# =============================================================================
# 6. CLASSE: ANALISADOR DE PERFIL DO INVESTIDOR
# =============================================================================

class AnalisadorPerfilInvestidor:
    def __init__(self):
        self.nivel_risco = ""
        self.horizonte_tempo = ""
        self.dias_lookback_ml = 5
    
    def determinar_nivel_risco(self, pontuacao: int) -> str:
        if pontuacao <= 46: return "CONSERVADOR"
        elif pontuacao <= 67: return "INTERMEDIÁRIO"
        elif pontuacao <= 88: return "MODERADO"
        elif pontuacao <= 109: return "MODERADO-ARROJADO"
        else: return "AVANÇADO"
    
    def determinar_horizonte_ml(self, liquidez_key: str, objetivo_key: str) -> tuple[str, int]:
        time_map = { 'A': 84, 'B': 168, 'C': 252 } 
        final_lookback = max(time_map.get(liquidez_key[0], 168), time_map.get(objetivo_key[0], 168))
        
        if final_lookback >= 252:
            return ("LONGO PRAZO", 252)
        elif final_lookback >= 168:
            return ("MÉDIO PRAZO", 168)
        else:
            return ("CURTO PRAZO", 84)
    
    def calcular_perfil(self, respostas: dict) -> tuple[str, str, int, int]:
        p1 = SCORE_MAP_ORIGINAL.get(respostas.get('risk_accept', 'N: Neutro'), 3)
        p2 = SCORE_MAP_ORIGINAL.get(respostas.get('max_gain', 'N: Neutro'), 3)
        p3 = SCORE_MAP_INV_ORIGINAL.get(respostas.get('stable_growth', 'N: Neutro'), 3)
        p4 = SCORE_MAP_INV_ORIGINAL.get(respostas.get('avoid_loss', 'N: Neutro'), 3)
        p5 = SCORE_MAP_REACTION_ORIGINAL.get(respostas.get('reaction', 'B: Manteria e reavaliaria a tese'), 3)
        p6 = SCORE_MAP_CONHECIMENTO_ORIGINAL.get(respostas.get('level', 'B: Intermediário (Conhecimento básico sobre mercados e ativos)'), 3)
        
        pontuacao_total = (p1 * 4) + (p2 * 3) + (p3 * 3) + (p4 * 3) + (p5 * 2) + (p6 * 5)
        
        nivel_risco = self.determinar_nivel_risco(pontuacao_total)
        horizonte, lookback = self.determinar_horizonte_ml(
            respostas.get('liquidity', 'B'),
            respostas.get('time_purpose', 'B')
        )
        
        return (nivel_risco, horizonte, lookback, pontuacao_total)

# =============================================================================
# 7. CLASSE: OTIMIZADOR DE PORTFÓLIO
# =============================================================================

class OtimizadorPortfolioAvancado:
    def __init__(self, returns_df: pd.DataFrame, garch_vols: dict = None):
        self.returns = returns_df
        self.mean_returns = returns_df.mean() * 252
        
        garch_vols = garch_vols or {}
        self.cov_matrix = returns_df.cov() * 252
            
        self.num_ativos = len(returns_df.columns)
    
    def estatisticas_portfolio(self, pesos: np.ndarray) -> tuple[float, float]:
        p_retorno = np.dot(pesos, self.mean_returns)
        p_vol = np.sqrt(np.dot(pesos.T, np.dot(self.cov_matrix, pesos)))
        return p_retorno, p_vol
    
    def sharpe_negativo(self, pesos: np.ndarray) -> float:
        p_retorno, p_vol = self.estatisticas_portfolio(pesos)
        if p_vol <= 1e-9: return -100.0
        return -(p_retorno - TAXA_LIVRE_RISCO) / p_vol
    
    def volatilidade(self, pesos: np.ndarray) -> float:
        _, p_vol = self.estatisticas_portfolio(pesos)
        return p_vol
    
    def otimizar(self, estrategia: str = 'MaxSharpe') -> dict:
        x0 = np.array([1 / self.num_ativos] * self.num_ativos)
        bounds = tuple((PESO_MIN, PESO_MAX) for _ in range(self.num_ativos))
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        if estrategia == 'MaxSharpe':
            result = minimize(self.sharpe_negativo, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000})
        elif estrategia == 'MinVolatility':
            result = minimize(self.volatilidade, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000})
        else:
            return {}
        
        if result.success:
            weights = result.x
            return {asset: float(w) for asset, w in zip(self.returns.columns, weights) if w > 0.001}
        return {}

# =============================================================================
# 9. CLASSE: COLETOR DE DADOS LIVE
# =============================================================================

class ColetorDadosLive(object):
    def __init__(self, periodo=PERIODO_DADOS):
        self.periodo = periodo
        self.dados_por_ativo = {} 
        self.dados_fundamentalistas = pd.DataFrame() 
        self.metricas_performance = pd.DataFrame() 
        self.volatilidades_garch_raw = {}
        self.metricas_simples = {}
        
        log_debug("Inicializando ColetorDadosLive...")
        try:
            self.tv = TvDatafeed() 
            self.tv_ativo = True
            log_debug("TvDatafeed inicializado com sucesso.")
        except Exception as e:
            self.tv_ativo = False
            log_debug(f"ERRO: Falha ao inicializar TvDatafeed: {str(e)[:50]}...")
            st.error(f"Erro ao inicializar tvDatafeed: {e}")
        
        try:
            self.pynvest_scrapper = Fundamentus()
            self.pynvest_ativo = True
            log_debug("Pynvest (Fundamentus) inicializado com sucesso.")
        except Exception:
            self.pynvest_ativo = False
            log_debug("AVISO: Pynvest falhou ao inicializar. Coleta de fundamentos desativada.")
            st.warning("Biblioteca pynvest não inicializada corretamente.")
    
    def _mapear_colunas_pynvest(self, df_pynvest: pd.DataFrame) -> dict:
        if df_pynvest.empty: return {}
        row = df_pynvest.iloc[0]
        
        mapping = {
            'vlr_ind_p_sobre_l': 'pe_ratio', 
            'vlr_ind_p_sobre_vp': 'pb_ratio', 
            'vlr_ind_roe': 'roe',
            'vlr_ind_roic': 'roic', 
            'vlr_ind_margem_liq': 'net_margin', 
            'vlr_ind_div_yield': 'div_yield',
            'vlr_ind_divida_bruta_sobre_patrim': 'debt_to_equity', 
            'vlr_liquidez_corr': 'current_ratio',
            'pct_cresc_rec_liq_ult_5a': 'revenue_growth', 
            'vlr_ind_ev_sobre_ebitda': 'ev_ebitda',
            'nome_setor': 'sector', 
            'nome_subsetor': 'industry', 
            'vlr_mercado': 'market_cap',
            'vlr_ind_margem_ebit': 'operating_margin',
            'vlr_ind_beta': 'beta',
            'nome_papel': 'nome_papel', 'tipo_papel': 'tipo_papel', 'nome_empresa': 'nome_empresa',
            'vlr_cot': 'vlr_cot', 'dt_ult_cot': 'dt_ult_cot', 'vlr_min_52_sem': 'vlr_min_52_sem',
            'vlr_max_52_sem': 'vlr_max_52_sem', 'vol_med_neg_2m': 'vol_med_neg_2m', 'vlr_firma': 'vlr_firma',
            'num_acoes': 'num_acoes', 'pct_var_dia': 'pct_var_dia', 'pct_var_mes': 'pct_var_mes',
            'pct_var_30d': 'pct_var_30d', 'pct_var_12m': 'pct_var_12m', 'pct_var_ano_a0': 'pct_var_ano_a0',
            'pct_var_ano_a1': 'pct_var_ano_a1', 'pct_var_ano_a2': 'pct_var_ano_a2', 'pct_var_ano_a3': 'pct_var_ano_a3',
            'pct_var_ano_a4': 'pct_var_ano_a4', 'pct_var_ano_a5': 'pct_var_ano_a5', 
            'vlr_ind_p_sobre_ebit': 'p_ebit', 'vlr_ind_psr': 'psr', 'vlr_ind_p_sobre_ativ': 'p_ativo',
            'vlr_ind_p_sobre_cap_giro': 'p_cap_giro', 'vlr_ind_p_sobre_ativ_circ_liq': 'p_ativ_circ_liq',
            'vlr_ind_ev_sobre_ebit': 'ev_ebit', 'vlr_ind_lpa': 'lpa', 'vlr_ind_vpa': 'vpa',
            'vlr_ind_margem_bruta': 'margem_bruta', 'vlr_ind_ebit_sobre_ativo': 'ebit_ativo',
            'vlr_ind_giro_ativos': 'giro_ativos', 'vlr_ativo': 'ativo_total', 'vlr_disponibilidades': 'disponibilidades',
            'vlr_ativ_circulante': 'ativo_circulante', 'vlr_divida_bruta': 'divida_bruta', 'vlr_divida_liq': 'divida_liquida',
            'vlr_patrim_liq': 'patrimonio_liquido', 'vlr_receita_liq_ult_12m': 'receita_liq_12m',
            'vlr_ebit_ult_12m': 'ebit_12m', 'vlr_lucro_liq_ult_12m': 'lucro_liq_12m'
        }
        dados_formatados = {}
        for col_orig, col_dest in mapping.items():
            if col_orig in row:
                val = row[col_orig]
                if isinstance(val, str):
                    if val.strip() == '':
                        val = np.nan
                    
                    if isinstance(val, str):
                        val = val.replace('.', '').replace(',', '.')
                        if val.endswith('%'):
                            val = val.replace('%', '')
                            try: val = float(val) / 100.0
                            except (ValueError, TypeError): pass
                try: 
                    dados_formatados[col_dest] = float(val)
                except (ValueError, TypeError): 
                    dados_formatados[col_dest] = val
        return dados_formatados

    def coletar_fundamentos_em_lote(self, simbolos: list) -> pd.DataFrame:
        if not self.pynvest_ativo: 
            log_debug("Coleta de fundamentos em lote ignorada (Pynvest inativo).")
            return pd.DataFrame()
        lista_fund = []
        log_debug(f"Iniciando coleta de fundamentos para {len(simbolos)} ativos via Pynvest...")
        for i, simbolo in enumerate(simbolos):
            try:
                ticker_pynvest = simbolo.replace('.SA', '').lower()
                df_fund_raw = self.pynvest_scrapper.coleta_indicadores_de_ativo(ticker_pynvest)
                if df_fund_raw is not None and not df_fund_raw.empty:
                    fund_data = self._mapear_colunas_pynvest(df_fund_raw)
                    fund_data['Ticker'] = simbolo
                    if 'sector' not in fund_data or fund_data['sector'] == 'Unknown':
                        fund_data['sector'] = FALLBACK_SETORES.get(simbolo, 'Outros')
                    lista_fund.append(fund_data)
                    log_debug(f"Fundamentos para {simbolo} coletados com sucesso.")
                else:
                     log_debug(f"AVISO: Pynvest não retornou dados para {simbolo}.")
            except Exception as e:
                log_debug(f"ERRO: Falha ao coletar fundamentos para {simbolo}: {str(e)[:50]}...")
                pass
            time.sleep(0.05) 
        
        if lista_fund:
            log_debug(f"Coleta de fundamentos finalizada. {len(lista_fund)} ativos com dados.")
            return pd.DataFrame(lista_fund).set_index('Ticker')
        return pd.DataFrame()

    def coletar_e_processar_dados(self, simbolos: list, check_min_ativos: bool = True) -> bool:
        self.ativos_sucesso = []
        lista_fundamentalistas = []
        metricas_simples_list = []
        
        use_garch_hq = True
        
        consecutive_failures = 0
        FAILURE_THRESHOLD = 3 
        global_static_mode = False 
        log_debug(f"Iniciando ciclo de coleta de preços para {len(simbolos)} ativos. GARCH: {'HQ-Auto' if use_garch_hq else 'GARCH(1,1)'}")
        
        for simbolo in simbolos:
            df_tecnicos = pd.DataFrame()
            usando_fallback_estatico = False 
            tem_dados = False
            
            vol_anual, ret_anual, sharpe, max_dd = 0.20, 0.0, 0.0, 0.0
            garch_vol = 0.20
            garch_model_name = "N/A"

            if not global_static_mode:
                
                try:
                    log_debug(f"Iniciando Tentativa 1 (YFinance) para {simbolo}...")
                    session = requests.Session()
                    session.headers.update({
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                    })
                    ticker_obj = yf.Ticker(simbolo, session=session)
                    df_tecnicos = ticker_obj.history(period=self.periodo)
                except Exception:
                    pass
                
                if df_tecnicos is not None and not df_tecnicos.empty and 'Close' in df_tecnicos.columns:
                    tem_dados = True
                    log_debug(f"Tentativa 1 (YFinance): Sucesso. {len(df_tecnicos)} pontos.")

                if not tem_dados and self.tv_ativo:
                    log_debug(f"Tentativa 1 falhou. Iniciando Tentativa 2 (TvDatafeed) para {simbolo}...")
                    simbolo_tv = simbolo.replace('.SA', '')
                    try:
                        df_tecnicos = self.tv.get_hist(
                            symbol=simbolo_tv, 
                            exchange='BMFBOVESPA', 
                            interval=Interval.in_daily, 
                            n_bars=1260
                        )
                    except Exception as e:
                        if "Connection timed out" in str(e):
                             pass
                
                    if df_tecnicos is not None and not df_tecnicos.empty:
                        tem_dados = True
                        log_debug(f"Tentativa 2 (TvDatafeed): Sucesso. {len(df_tecnicos)} pontos.")

                if not tem_dados:
                    consecutive_failures += 1
                    log_debug(f"Coleta de preços para {simbolo} falhou. Falhas consecutivas: {consecutive_failures}")
                    if consecutive_failures >= FAILURE_THRESHOLD:
                        global_static_mode = True
                        log_debug(f"ATIVANDO MODO ESTÁTICO GLOBAL (Limite de {FAILURE_THRESHOLD} falhas atingido).")
                        st.warning(f"⚠️ Falha na coleta de preços para {consecutive_failures} ativos consecutivos. Ativando MODO FUNDAMENTALISTA GLOBAL (sem histórico de preços) para o restante da lista.")
                else:
                    consecutive_failures = 0 
            
            if global_static_mode or not tem_dados:
                usando_fallback_estatico = True
                log_debug(f"Ativo {simbolo}: Processando em MODO ESTÁTICO (Sem preços históricos).")
                df_tecnicos = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'rsi_14', 'macd', 'vol_20d'])
                df_tecnicos.loc[pd.Timestamp.today()] = [np.nan] * len(df_tecnicos.columns)
            else:
                log_debug(f"Ativo {simbolo}: Enriquecendo dados técnicos...")
                rename_map = {
                    'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
                }
                df_tecnicos.rename(columns=rename_map, inplace=True)
                
                for col in df_tecnicos.columns:
                    if ':' in str(col):
                        base_col = str(col).split(':')[-1]
                        if base_col in rename_map:
                            df_tecnicos.rename(columns={col: rename_map[base_col]}, inplace=True)

                if 'Close' in df_tecnicos.columns:
                    if not df_tecnicos.empty:
                        df_tecnicos = CalculadoraTecnica.enriquecer_dados_tecnicos(df_tecnicos)
                    else:
                       usando_fallback_estatico = True 
                else:
                    usando_fallback_estatico = True

            fund_data = {}
            if self.pynvest_ativo:
                try:
                    ticker_pynvest = simbolo.replace('.SA', '').lower()
                    df_fund_raw = self.pynvest_scrapper.coleta_indicadores_de_ativo(ticker_pynvest)
                    if df_fund_raw is not None and not df_fund_raw.empty:
                        fund_data = self._mapear_colunas_pynvest(df_fund_raw)
                    else:
                        fund_data = {'sector': 'Unknown', 'industry': 'Unknown'}
                except Exception:
                    fund_data = {'sector': 'Unknown', 'industry': 'Unknown'}
            
            if 'sector' not in fund_data or fund_data['sector'] == 'Unknown':
                 fund_data['sector'] = FALLBACK_SETORES.get(simbolo, 'Outros')
                 log_debug(f"Ativo {simbolo}: Setor definido via FALLBACK_SETORES.")

            if usando_fallback_estatico and (not fund_data or fund_data.get('pe_ratio') is None):
                log_debug(f"Ativo {simbolo} pulado: Modo estático, mas dados fundamentais críticos ausentes.")
                continue 

            if not usando_fallback_estatico and 'returns' in df_tecnicos.columns:
                retornos = df_tecnicos['returns'].dropna()
                log_debug(f"Ativo {simbolo}: Calculando métricas de performance (Sharpe/DD)...")
                
                if len(retornos) > 30:
                    vol_anual = retornos.std() * np.sqrt(252)
                    ret_anual = retornos.mean() * 252
                    sharpe = (ret_anual - TAXA_LIVRE_RISCO) / vol_anual if vol_anual > 0 else 0
                    cum_prod = (1 + retornos).cumprod()
                    peak = cum_prod.expanding(min_periods=1).max()
                    dd = (cum_prod - peak) / peak
                    max_dd = dd.min()
                else:
                    vol_anual, ret_anual, sharpe, max_dd = 0.20, 0.0, 0.0, 0.0 
                
                garch_vol, garch_model_name = ModeladorGARCH.calcular_volatilidade_garch(retornos, use_hq=use_garch_hq)
            
            fund_data.update({
                'Ticker': simbolo, 'sharpe_ratio': sharpe, 'annual_return': ret_anual,
                'annual_volatility': vol_anual, 'max_drawdown': max_dd, 'garch_volatility': garch_vol,
                'static_mode': usando_fallback_estatico, 'garch_model': garch_model_name
            })
            
            self.dados_por_ativo[simbolo] = df_tecnicos
            self.ativos_sucesso.append(simbolo)
            lista_fundamentalistas.append(fund_data)
            self.volatilidades_garch_raw[simbolo] = garch_vol
            
            metricas_simples_list.append({
                'Ticker': simbolo, 'sharpe': sharpe, 'retorno_anual': ret_anual,
                'volatilidade_anual': vol_anual, 'max_drawdown': max_dd,
            })
            
            if not global_static_mode:
                time.sleep(0.1) 

        self.dados_fundamentalistas = pd.DataFrame(lista_fundamentalistas)
        if not self.dados_fundamentalistas.empty:
             self.dados_fundamentalistas = self.dados_fundamentalistas.set_index('Ticker')
             
        self.metricas_performance = pd.DataFrame(metricas_simples_list)
        if not self.metricas_performance.empty:
             self.metricas_performance = self.metricas_performance.set_index('Ticker')
        
        for simbolo in self.ativos_sucesso:
            if simbolo in self.dados_por_ativo:
                df_local = self.dados_por_ativo[simbolo]
                if not df_local.empty:
                    last_idx = df_local.index[-1]
                    if simbolo in self.dados_fundamentalistas.index:
                        for k, v in self.dados_fundamentalistas.loc[simbolo].items():
                            if k not in df_local.columns:
                                df_local.loc[last_idx, k] = v

        if check_min_ativos and len(self.ativos_sucesso) < NUM_ATIVOS_PORTFOLIO: 
             log_debug(f"AVISO: Coleta finalizada com {len(self.ativos_sucesso)} ativos, abaixo do mínimo requerido.")
             return False

        log_debug(f"Coleta de dados finalizada com sucesso. {len(self.ativos_sucesso)} ativos processados.")
        return True

    def coletar_ativo_unico_gcs(self, ativo_selecionado: str):
        """
        Função para coletar e processar um único ativo para a aba de análise individual,
        incluindo o pipeline ML.
        """
        log_debug(f"Iniciando coleta e análise de ativo único: {ativo_selecionado}")
        
        self.coletar_e_processar_dados([ativo_selecionado], check_min_ativos=False)
        
        if ativo_selecionado not in self.dados_por_ativo:
            log_debug(f"ERRO: Dados não encontrados após coleta para {ativo_selecionado}.")
            return None, None, None

        df_tec = self.dados_por_ativo[ativo_selecionado]
        fund_row = {}
        if ativo_selecionado in self.dados_fundamentalistas.index:
            fund_row = self.dados_fundamentalistas.loc[ativo_selecionado].to_dict()
        
        df_ml_meta = pd.DataFrame()
        
        # Placeholder para ML individual (pode ser expandido posteriormente)
        
        return df_tec, fund_row, df_ml_meta

    def calculate_cross_sectional_features(self):
        df_fund = self.dados_fundamentalistas.copy()
        if 'sector' not in df_fund.columns or 'pe_ratio' not in df_fund.columns: return
        
        log_debug("Calculando features cross-sectional (P/L e P/VP relativos ao setor).")
        
        cols_numeric = ['pe_ratio', 'pb_ratio']
        for col in cols_numeric:
             if col in df_fund.columns:
                 df_fund[col] = pd.to_numeric(df_fund[col], errors='coerce')

        sector_means = df_fund.groupby('sector')[['pe_ratio', 'pb_ratio']].transform('mean')
        
        valid_pe_mean = sector_means['pe_ratio'].replace(0, np.nan).fillna(1.0)
        valid_pb_mean = sector_means['pb_ratio'].replace(0, np.nan).fillna(1.0)

        df_fund['pe_rel_sector'] = df_fund['pe_ratio'] / valid_pe_mean
        df_fund['pb_rel_sector'] = df_fund['pb_ratio'] / valid_pb_mean
        
        df_fund = df_fund.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        self.dados_fundamentalistas = df_fund
        log_debug("Features cross-sectional concluídas.")

    def calcular_volatilidades_garch(self):
        log_debug("Volatilidades GARCH já calculadas durante a coleta de dados.")

# =============================================================================
# =============================================================================

class ConstrutorPortfolioAutoML(ColetorDadosLive):
    """
    Classe principal para construção de portfólio com ML avançado.
    Herda de ColetorDadosLive e adiciona funcionalidades de ML e otimização.
    """
    
    def __init__(self, valor_investimento: float):
        super().__init__()
        self.valor_investimento = valor_investimento
        self.ativos_selecionados = []
        self.alocacao_portfolio = {}
        self.metricas_portfolio = {}
        self.scores_combinados = pd.DataFrame()
        self.predicoes_ml = {}
        self.justificativas_selecao = {}
        self.metodo_alocacao_atual = ""
        self.pesos_atuais = {}
        self.perfil_dashboard = {}
        self.volatilidades_garch = {}
        
        log_debug(f"ConstrutorPortfolioAutoML inicializado. Capital: R$ {valor_investimento:,.2f}")
    
    def treinar_modelos_ensemble(self, ml_mode: str = 'simple', progress_callback=None):
        """
        Treina modelos de ML usando scores como features.
        
        ml_mode:
            - 'simple': ElasticNet + LogisticRegression
            - 'complex': ElasticNet + Ensemble (RandomForest + XGBoost)
        """
        ativos_com_dados = [s for s in self.ativos_sucesso if s in self.dados_por_ativo]
        log_debug(f"Iniciando Pipeline de Treinamento ML (Modo: {ml_mode}).")
        
        SCORE_BASED_FEATURES = ['Close', 'annual_return', 'annual_volatility', 'sharpe', 
                                'pe_ratio', 'pb_ratio', 'div_yield', 'roe']
        ALL_FUND_FEATURES = ['pe_ratio', 'pb_ratio', 'div_yield', 'roe', 'pe_rel_sector', 
                            'pb_rel_sector', 'Cluster', 'roic', 'net_margin', 'debt_to_equity', 
                            'current_ratio', 'revenue_growth', 'ev_ebitda', 'operating_margin']
        
        if ml_mode == 'simple':
            # Modelo simples: ElasticNet para seleção + LogisticRegression
            FEATURE_SELECTOR = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=2000)
            CLASSIFIER = LogisticRegression
            MODEL_PARAMS = dict(penalty='l2', solver='liblinear', class_weight='balanced', random_state=42)
            MODEL_NAME = 'ElasticNet + LogReg'
        else:  # complex
            # Modelo complexo: ElasticNet + Ensemble RF + XGBoost
            FEATURE_SELECTOR = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=2000)
            CLASSIFIER = 'ensemble'
            MODEL_PARAMS = {}
            MODEL_NAME = 'ElasticNet + Ensemble(RF+XGB)'
        
        # Clusterização inicial
        required_cols_cluster = ['pe_ratio', 'pb_ratio', 'div_yield', 'roe']
        available_fund_cols = [col for col in required_cols_cluster if col in self.dados_fundamentalistas.columns]
        
        if len(available_fund_cols) >= 4 and len(self.dados_fundamentalistas) >= 5:
            log_debug("Executando Clusterização inicial (KMeans + PCA) nos fundamentos.")
            clustering_df = self.dados_fundamentalistas[available_fund_cols].fillna(0)
            
            if len(clustering_df) >= 5:
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(clustering_df)
                pca = PCA(n_components=min(data_scaled.shape[1], 3))
                data_pca = pca.fit_transform(data_scaled)
                kmeans = KMeans(n_clusters=min(len(data_pca), 5), random_state=42, n_init=10)
                clusters = kmeans.fit_predict(data_pca)
                self.dados_fundamentalistas['Cluster'] = pd.Series(clusters, index=clustering_df.index).fillna(-1).astype(int)
                log_debug(f"Clusterização inicial concluída. {self.dados_fundamentalistas['Cluster'].nunique()} clusters formados.")
            else:
                self.dados_fundamentalistas['Cluster'] = 0
                log_debug("AVISO: Dados insuficientes após o JOIN. Clusterização inicial ignorada.")
        else:
            self.dados_fundamentalistas['Cluster'] = 0
            log_debug("AVISO: Falha na coleta de dados fundamentais. Usando Cluster = 0.")
        
        # Treinamento ML
        ML_HORIZONS = get_ml_horizons(self.perfil_dashboard.get('ml_lookback_days', 168))
        all_ml_results = {}
        total_ml_success = 0
        
        for idx, ativo in enumerate(ativos_com_dados):
            if progress_callback:
                progress = 50 + int((idx / len(ativos_com_dados)) * 20)
                progress_callback.progress(progress, text=f"Treinando ML: {ativo} ({idx+1}/{len(ativos_com_dados)})...")
            
            try:
                df = self.dados_por_ativo[ativo].copy()
                
                # Adiciona métricas históricas
                if ativo in self.metricas_performance.index:
                    df['annual_return'] = self.metricas_performance.loc[ativo, 'retorno_anual']
                    df['annual_volatility'] = self.metricas_performance.loc[ativo, 'volatilidade_anual']
                    df['sharpe'] = self.metricas_performance.loc[ativo, 'sharpe']
                
                # Adiciona features fundamentalistas
                last_idx = df.index[-1] if not df.empty else None
                if last_idx:
                    for f_col in ALL_FUND_FEATURES:
                        if ativo in self.dados_fundamentalistas.index:
                            fund_val = self.dados_fundamentalistas.loc[ativo, f_col] if f_col in self.dados_fundamentalistas.columns else np.nan
                            df.loc[last_idx, f_col] = fund_val
                        elif f_col not in df.columns:
                            df[f_col] = np.nan
                
                # Cria targets futuros
                for d in ML_HORIZONS:
                    df[f"t_{d}"] = (df["Close"].shift(-d) > df["Close"]).astype(int)
                
                # Seleciona features disponíveis
                ML_FEATURES_FINAL = [f for f in SCORE_BASED_FEATURES if f in df.columns]
                df_model = df.dropna(subset=ML_FEATURES_FINAL + [f"t_{ML_HORIZONS[-1]}"])
                
                if len(df_model) < MIN_TRAIN_DAYS_ML:
                    log_debug(f"ML: Dados insuficientes para {ativo} ({len(df_model)} < {MIN_TRAIN_DAYS_ML})")
                    raise ValueError("Dados insuficientes")
                
                X_full = df_model[ML_FEATURES_FINAL]
                
                # Ensemble de horizontes
                probabilities = []
                auc_scores = []
                
                for horizon in ML_HORIZONS:
                    y = df_model[f"t_{horizon}"]
                    
                    # Split temporal
                    split_point = int(len(X_full) * 0.7)
                    X_train = X_full.iloc[:split_point]
                    y_train = y.iloc[:split_point]
                    X_test = X_full.iloc[split_point:]
                    y_test = y.iloc[split_point:]
                    
                    if len(np.unique(y_train)) < 2:
                        continue
                    
                    feature_selector = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=2000)
                    X_train_filled = X_train.fillna(X_train.mean())
                    y_train_reg = y_train.astype(float)
                    
                    feature_selector.fit(X_train_filled, y_train_reg)
                    selected_features = [feat for feat, coef in zip(ML_FEATURES_FINAL, feature_selector.coef_) if abs(coef) > 0.01]
                    
                    if not selected_features:
                        selected_features = ML_FEATURES_FINAL[:min(3, len(ML_FEATURES_FINAL))]
                    
                    X_train_selected = X_train[selected_features]
                    X_test_selected = X_test[selected_features]
                    X_predict_selected = X_full.iloc[[-1]][selected_features]
                    
                    if CLASSIFIER == 'ensemble':
                        # Ensemble RF + XGBoost
                        scaler = StandardScaler().fit(X_train_selected.fillna(0))
                        X_train_scaled = scaler.transform(X_train_selected.fillna(0))
                        X_test_scaled = scaler.transform(X_test_selected.fillna(0))
                        X_predict_scaled = scaler.transform(X_predict_selected.fillna(0))
                        
                        # Random Forest
                        rf_model = RandomForestClassifier(
                            n_estimators=100, 
                            max_depth=5, 
                            random_state=42, 
                            class_weight='balanced', 
                            n_jobs=-1
                        )
                        rf_model.fit(X_train_scaled, y_train)
                        rf_prob = rf_model.predict_proba(X_predict_scaled)[0, 1]
                        
                        # XGBoost
                        if xgb is not None:
                            xgb_model = xgb.XGBClassifier(
                                n_estimators=100,
                                max_depth=3,
                                learning_rate=0.1,
                                random_state=42,
                                n_jobs=-1
                            )
                            xgb_model.fit(X_train_scaled, y_train)
                            xgb_prob = xgb_model.predict_proba(X_predict_scaled)[0, 1]
                            
                            # Média dos dois modelos
                            ensemble_prob = (rf_prob + xgb_prob) / 2
                        else:
                            ensemble_prob = rf_prob
                        
                        probabilities.append(ensemble_prob)
                        
                        # AUC
                        if len(y_test) > 0 and len(np.unique(y_test)) >= 2:
                            rf_prob_test = rf_model.predict_proba(X_test_scaled)[:, 1]
                            if xgb is not None:
                                xgb_prob_test = xgb_model.predict_proba(X_test_scaled)[:, 1]
                                ensemble_prob_test = (rf_prob_test + xgb_prob_test) / 2
                            else:
                                ensemble_prob_test = rf_prob_test
                            auc_scores.append(roc_auc_score(y_test, ensemble_prob_test))
                    
                    else:
                        # Modelo simples (LogisticRegression)
                        scaler = StandardScaler().fit(X_train_selected.fillna(0))
                        X_train_scaled = scaler.transform(X_train_selected.fillna(0))
                        X_test_scaled = scaler.transform(X_test_selected.fillna(0))
                        X_predict_scaled = scaler.transform(X_predict_selected.fillna(0))
                        
                        model = LogisticRegression(**MODEL_PARAMS)
                        model.fit(X_train_scaled, y_train)
                        
                        prob_now = model.predict_proba(X_predict_scaled)[0, 1]
                        probabilities.append(prob_now)
                        
                        if len(y_test) > 0 and len(np.unique(y_test)) >= 2:
                            prob_test = model.predict_proba(X_test_scaled)[:, 1]
                            auc_scores.append(roc_auc_score(y_test, prob_test))
                
                ensemble_proba = np.mean(probabilities) if probabilities else 0.5
                conf_final = np.mean(auc_scores) if auc_scores else 0.5
                
                result_for_ativo = {
                    'predicted_proba_up': ensemble_proba, 
                    'auc_roc_score': conf_final, 
                    'model_name': MODEL_NAME
                }
                
                self.dados_por_ativo[ativo].loc[df.index[-1], 'ML_Proba'] = ensemble_proba
                self.dados_por_ativo[ativo].loc[df.index[-1], 'ML_Confidence'] = conf_final
                log_debug(f"ML: {ativo} sucesso. Prob: {ensemble_proba:.2f}, AUC: {conf_final:.2f}, Modelo: {MODEL_NAME}")
                total_ml_success += 1
                
            except Exception as e:
                log_debug(f"ML: {ativo} falhou - {str(e)[:50]}")
                result_for_ativo = {'predicted_proba_up': 0.5, 'auc_roc_score': 0.0, 'model_name': 'Training Failed'}
            
            all_ml_results[ativo] = result_for_ativo
        
        if total_ml_success == 0 and len(ativos_com_dados) > 0:
            log_debug("AVISO: Falha total no ML. Score ML será neutro.")
            for ativo in ativos_com_dados:
                all_ml_results[ativo]['auc_roc_score'] = 0.0
                all_ml_results[ativo]['model_name'] = 'Total Fallback'
        
        self.predicoes_ml = all_ml_results
        log_debug("Pipeline de Treinamento ML concluído.")

    def realizar_clusterizacao_final(self):
        if self.scores_combinados.empty: return
        log_debug("Iniciando Clusterização Final nos Scores (KMeans).")
        
        features_cluster = ['fundamental_score', 'technical_score', 'ml_score_weighted']
        data_cluster = self.scores_combinados[features_cluster].fillna(50)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_cluster)
        pca = PCA(n_components=min(data_scaled.shape[1], 2))
        data_pca = pca.fit_transform(data_scaled)
        kmeans = KMeans(n_clusters=min(len(data_pca), 4), random_state=42, n_init=10)
        clusters = kmeans.fit_predict(data_pca)
        self.scores_combinados['Final_Cluster'] = clusters
        log_debug(f"Clusterização Final concluída. {self.scores_combinados['Final_Cluster'].nunique()} perfis identificados.")

    def pontuar_e_selecionar_ativos(self, horizonte_tempo: str):
        if horizonte_tempo == "CURTO PRAZO": share_tech, share_fund = 0.7, 0.3
        elif horizonte_tempo == "LONGO PRAZO": share_tech, share_fund = 0.3, 0.7
        else: share_tech, share_fund = 0.5, 0.5

        W_ML_GLOBAL_BASE = 0.20
        W_REMAINING = 0.80
        w_tech_final = W_REMAINING * share_tech
        w_fund_final = W_REMAINING * share_fund
        self.pesos_atuais = {'Fundamentos': w_fund_final, 'Técnicos': w_tech_final, 'ML': W_ML_GLOBAL_BASE}
        
        # JOIN seguro
        cols_to_drop = [col for col in self.dados_fundamentalistas.columns if col in self.metricas_performance.columns]
        df_fund_clean = self.dados_fundamentalistas.drop(columns=cols_to_drop, errors='ignore')
        combined = self.metricas_performance.join(df_fund_clean, how='inner').copy()
        
        for symbol in combined.index:
            if symbol in self.dados_por_ativo:
                df = self.dados_por_ativo[symbol]
                
                if not df.empty and 'rsi_14' in df.columns:
                    combined.loc[symbol, 'rsi_current'] = df['rsi_14'].iloc[-1]
                    combined.loc[symbol, 'macd_current'] = df['macd_diff'].iloc[-1]
                    combined.loc[symbol, 'vol_current'] = df['vol_20d'].iloc[-1]
                    combined.loc[symbol, 'macd_diff'] = df['macd_diff'].iloc[-1]
                    combined.loc[symbol, 'rsi_14'] = df['rsi_14'].iloc[-1]
                    combined.loc[symbol, 'vol_20d'] = df['vol_20d'].iloc[-1]
                    
                    for lgbm_f in LGBM_FEATURES:
                         if lgbm_f in df.columns:
                             combined.loc[symbol, lgbm_f] = df[lgbm_f].iloc[-1]
                else:
                    combined.loc[symbol, 'rsi_current'] = 50
                    combined.loc[symbol, 'macd_current'] = 0
                    combined.loc[symbol, 'vol_current'] = 0
                    combined.loc[symbol, 'macd_diff'] = np.nan
                    combined.loc[symbol, 'rsi_14'] = np.nan
                    combined.loc[symbol, 'vol_20d'] = np.nan

        scores = pd.DataFrame(index=combined.index)
        
        def _get_score_series(df: pd.DataFrame, col: str, default_val: float) -> pd.Series:
            if col in df.columns:
                return df[col]
            return pd.Series(default_val, index=df.index)

        s_pl = EngenheiroFeatures._normalize_score(_get_score_series(combined, 'pe_ratio', 50), False)
        s_pvp = EngenheiroFeatures._normalize_score(_get_score_series(combined, 'pb_ratio', 50), False)
        s_roe = EngenheiroFeatures._normalize_score(_get_score_series(combined, 'roe', 50), True)
        s_dy = EngenheiroFeatures._normalize_score(_get_score_series(combined, 'div_yield', 0), True)
        
        scores['fundamental_score'] = ((s_pl + s_pvp + s_roe + s_dy) / 4) * w_fund_final
        
        s_rsi = EngenheiroFeatures._normalize_score(100 - abs(combined.get('rsi_current', 50) - 50), True)
        s_macd = EngenheiroFeatures._normalize_score(combined.get('macd_current', 0), True)
        s_vol = EngenheiroFeatures._normalize_score(combined.get('vol_current', 0), False)
        scores['technical_score'] = (s_rsi * 0.3 + s_macd * 0.4 + s_vol * 0.3) * w_tech_final
        
        ml_probs = pd.Series({s: self.predicoes_ml.get(s, {}).get('predicted_proba_up', 0.5) for s in combined.index})
        ml_conf = pd.Series({s: self.predicoes_ml.get(s, {}).get('auc_roc_score', 0.5) for s in combined.index})
        
        scores['ml_score_weighted'] = ml_conf.fillna(0) * ml_probs.fillna(0) * W_ML_GLOBAL_BASE * 100

        scores['total_score'] = scores['fundamental_score'] + scores['technical_score'] + scores['ml_score_weighted']
        
        self.scores_combinados = scores.join(combined).sort_values('total_score', ascending=False)
        
        log_debug(f"Scores calculados. Horizonte: {horizonte_tempo}. Pesos: Fund={w_fund_final:.2f}, Tec={w_tech_final:.2f}, ML={W_ML_GLOBAL_BASE:.2f}.")

        # Filtro de score mínimo
        if len(self.scores_combinados) > NUM_ATIVOS_PORTFOLIO:
            cutoff_index = min(15, len(self.scores_combinados) - 1)
            base_score = self.scores_combinados['total_score'].iloc[cutoff_index]
            min_score = base_score * SCORE_PERCENTILE_THRESHOLD
            ativos_filtrados = self.scores_combinados[self.scores_combinados['total_score'] >= min_score]
            num_eliminados = len(self.scores_combinados) - len(ativos_filtrados)
            log_debug(f"Score mínimo: {min_score:.3f}. Eliminados: {num_eliminados}")
            self.scores_combinados = ativos_filtrados
        
        # Seleção final por cluster
        self.realizar_clusterizacao_final()
        final_selection = []
        
        if not self.scores_combinados.empty and 'Final_Cluster' in self.scores_combinados.columns:
            clusters_present = self.scores_combinados['Final_Cluster'].unique()
            for c in clusters_present:
                best = self.scores_combinados[self.scores_combinados['Final_Cluster'] == c].head(1).index[0]
                final_selection.append(best)
        
        if len(final_selection) < NUM_ATIVOS_PORTFOLIO:
            others = [x for x in self.scores_combinados.index if x not in final_selection]
            remaining_to_add = NUM_ATIVOS_PORTFOLIO - len(final_selection)
            
            if remaining_to_add > 0:
                 others_df = self.scores_combinados.loc[others].sort_values('total_score', ascending=False)
                 final_selection.extend(others_df.index[:remaining_to_add].tolist())

        self.ativos_selecionados = final_selection[:NUM_ATIVOS_PORTFOLIO]
        log_debug(f"Seleção final: {len(self.ativos_selecionados)} ativos: {self.ativos_selecionados}")

        return self.ativos_selecionados
    
    def otimizar_alocacao(self, nivel_risco: str):
        if not self.ativos_selecionados or len(self.ativos_selecionados) < 1:
            self.metodo_alocacao_atual = "ERRO: Ativos Insuficientes"
            return {}
        
        available_assets_returns = {}
        ativos_sem_dados = []
        ativos_static_mode = []

        for s in self.ativos_selecionados:
            is_static = self.dados_fundamentalistas.loc[s, 'static_mode'] if s in self.dados_fundamentalistas.index else True
            
            if is_static:
                 ativos_static_mode.append(s)
            
            if s in self.dados_por_ativo and 'returns' in self.dados_por_ativo[s] and not self.dados_por_ativo[s]['returns'].dropna().empty:
                available_assets_returns[s] = self.dados_por_ativo[s]['returns']
            else:
                ativos_sem_dados.append(s)

        final_returns_df = pd.DataFrame(available_assets_returns).dropna()
        
        is_markowitz_possible = (
            final_returns_df.shape[0] >= 50 and 
            len(final_returns_df.columns) == len(self.ativos_selecionados) and
            not any(self.dados_fundamentalistas.loc[s, 'static_mode'] for s in final_returns_df.columns if s in self.dados_fundamentalistas.index)
        )
        
        if not is_markowitz_possible or 'PESOS_IGUAIS' in nivel_risco:
            log_debug(f"Markowitz Indisponível. Usando ponderação por score.")

            valid_selection = [a for a in self.ativos_selecionados if a in self.scores_combinados.index]
            
            if valid_selection:
                 scores = self.scores_combinados.loc[valid_selection, 'total_score']
                 total_score = scores.sum()
                 
                 if total_score > 0:
                     weights = (scores / total_score).to_dict()
                     self.metodo_alocacao_atual = 'PONDERAÇÃO POR SCORE'
                 else:
                     weights = {asset: 1.0 / len(valid_selection) for asset in valid_selection}
                     self.metodo_alocacao_atual = 'PESOS IGUAIS (Fallback)'
            else:
                 weights = {asset: 1.0 / len(self.ativos_selecionados) for asset in self.ativos_selecionados}
                 self.metodo_alocacao_atual = 'PESOS IGUAIS (Fallback)'
                 
            return self._formatar_alocacao(weights)

        garch_vols_filtered = {asset: self.volatilidades_garch_raw.get(asset, final_returns_df[asset].std() * np.sqrt(252)) for asset in final_returns_df.columns}
        optimizer = OtimizadorPortfolioAvancado(final_returns_df, garch_vols=garch_vols_filtered)
        
        if 'CONSERVADOR' in nivel_risco or 'INTERMEDIÁRIO' in nivel_risco:
            strategy = 'MinVolatility'
            self.metodo_alocacao_atual = 'MINIMIZAÇÃO DE VOLATILIDADE'
        else:
            strategy = 'MaxSharpe'
            self.metodo_alocacao_atual = 'MAXIMIZAÇÃO DE SHARPE'
            
        log_debug(f"Otimizando Markowitz. Estratégia: {self.metodo_alocacao_atual}")
            
        weights = optimizer.otimizar(estrategia=strategy)
        
        if not weights:
             log_debug("Markowitz falhou. Fallback para ponderação por score.")
             
             valid_selection = [a for a in self.ativos_selecionados if a in self.scores_combinados.index]
             if valid_selection:
                 scores = self.scores_combinados.loc[valid_selection, 'total_score']
                 total_score = scores.sum()
                 if total_score > 0:
                     weights = (scores / total_score).to_dict()
                     self.metodo_alocacao_atual = 'PONDERAÇÃO POR SCORE (Fallback)'
                 else:
                     weights = {asset: 1.0 / len(valid_selection) for asset in valid_selection}
                     self.metodo_alocacao_atual = 'PESOS IGUAIS (Fallback)'
             else:
                 weights = {asset: 1.0 / len(self.ativos_selecionados) for asset in self.ativos_selecionados}
                 self.metodo_alocacao_atual = 'PESOS IGUAIS (Fallback)'
        
        total_weight = sum(weights.values())
        log_debug(f"Otimização finalizada. Peso total: {total_weight:.2f}")
        return self._formatar_alocacao(weights)
        
    def _formatar_alocacao(self, weights: dict) -> dict:
        if not weights or sum(weights.values()) == 0: 
            return {s: {'weight': 0.0, 'amount': 0.0} for s in self.ativos_selecionados}

        total_weight = sum(weights.values())
        return {s: {'weight': w / total_weight, 'amount': self.valor_investimento * (w / total_weight)} for s, w in weights.items() if s in self.ativos_selecionados}
    
    def calcular_metricas_portfolio(self):
        if not self.alocacao_portfolio: 
             self.metricas_portfolio = {
                'annual_return': 0, 'annual_volatility': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'total_investment': self.valor_investimento
            }
             return self.metricas_portfolio

        weights_dict = {s: data['weight'] for s, data in self.alocacao_portfolio.items()}
        available_returns = {s: self.dados_por_ativo[s]['returns'] for s in weights_dict.keys() if s in self.dados_por_ativo and 'returns' in self.dados_por_ativo[s] and not self.dados_por_ativo[s]['returns'].dropna().empty}
        
        if not available_returns:
             self.metricas_portfolio = {
                'annual_return': 0, 'annual_volatility': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'total_investment': self.valor_investimento
            }
             return self.metricas_portfolio

        returns_df = pd.DataFrame(available_returns).dropna()
        if returns_df.empty: 
             self.metricas_portfolio = {
                'annual_return': 0, 'annual_volatility': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'total_investment': self.valor_investimento
            }
             return self.metricas_portfolio
        
        valid_assets = returns_df.columns
        valid_weights = np.array([weights_dict[s] for s in valid_assets])
        
        if valid_weights.sum() > 0:
            valid_weights = valid_weights / valid_weights.sum()
            portfolio_returns = (returns_df * valid_weights).sum(axis=1)
            metrics = {
                'annual_return': portfolio_returns.mean() * 252,
                'annual_volatility': portfolio_returns.std() * np.sqrt(252),
                'sharpe_ratio': (portfolio_returns.mean() * 252 - TAXA_LIVRE_RISCO) / (portfolio_returns.std() * np.sqrt(252)) if portfolio_returns.std() > 0 else 0,
                'max_drawdown': ((1 + portfolio_returns).cumprod() / (1 + portfolio_returns).cumprod().expanding().max() - 1).min(),
                'total_investment': self.valor_investimento
            }
        else:
             metrics = {'annual_return': 0, 'annual_volatility': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'total_investment': self.valor_investimento}
             
        self.metricas_portfolio = metrics
        return metrics

    def gerar_justificativas(self):
        self.justificativas_selecao = {}
        
        if self.scores_combinados.empty and self.ativos_selecionados:
             for simbolo in self.ativos_selecionados:
                 self.justificativas_selecao[simbolo] = "Dados de ranqueamento indisponíveis. Usando Fallback."
             return self.justificativas_selecao
        elif self.scores_combinados.empty:
             return self.justificativas_selecao

        for simbolo in self.ativos_selecionados:
            justification = []
            
            score_row = self.scores_combinados.loc[simbolo] if simbolo in self.scores_combinados.index else {}
            
            ml_data = self.predicoes_ml.get(simbolo, {})
            is_ml_failed = ml_data.get('auc_roc_score', 0.0) == 0.0
            
            is_static = False
            if simbolo in self.dados_fundamentalistas.index and 'static_mode' in self.dados_fundamentalistas.columns:
                is_static = self.dados_fundamentalistas.loc[simbolo, 'static_mode']
            
            if is_static:
                justification.append("⚠️ Modo Estático (Sem Preços)")
            
            justification.append(f"Score Fund: {score_row.get('fundamental_score', 0.0):.3f}")
            justification.append(f"Score Téc: {score_row.get('technical_score', 0.0):.3f}")
            
            if is_ml_failed:
                justification.append("Score ML: N/A (Falha)")
                justification.append("✅ Selecionado por Fundamentos")
            else:
                ml_prob = ml_data.get('predicted_proba_up', np.nan)
                ml_auc = ml_data.get('auc_roc_score', np.nan)
                justification.append(f"Score ML: {score_row.get('ml_score_weighted', 0.0):.3f} (Prob {ml_prob*100:.1f}%, AUC {ml_auc:.2f})")
            
            cluster = score_row.get('Final_Cluster', 'N/A')
            sector = self.dados_fundamentalistas.loc[simbolo, 'sector'] if simbolo in self.dados_fundamentalistas.index else 'N/A'
            justification.append(f"Cluster: {cluster}")
            justification.append(f"Setor: {sector}")
            
            self.justificativas_selecao[simbolo] = " | ".join(justification)
        return self.justificativas_selecao
        
    def executar_pipeline(self, simbolos_customizados: list, perfil_inputs: dict, ml_mode: str, pipeline_mode: str, progress_bar=None) -> bool:
        """
        Executa o pipeline completo de construção de portfólio.
        
        ml_mode: 'simple' ou 'complex'
        pipeline_mode: 'general' ou 'fundamentalista'
        """
        self.perfil_dashboard = perfil_inputs
        
        if pipeline_mode == 'fundamentalista':
             log_debug("Modo de Pipeline: FUNDAMENTALISTA (ML e Markowitz ignorados).")
             ml_mode = 'fallback'

        try:
            if progress_bar: progress_bar.progress(10, text="Coletando dados LIVE (YFinance + Pynvest)...")
            
            if not self.coletar_e_processar_dados(simbolos_customizados):
                log_debug("AVISO: Coleta inicial falhou parcialmente.")
            
            if not self.ativos_sucesso: 
                 st.error("Falha na aquisição: Nenhum ativo processado.")
                 return False

            if progress_bar: progress_bar.progress(30, text="Calculando métricas setoriais e GARCH...")
            self.calculate_cross_sectional_features()
            self.calcular_volatilidades_garch()
            
            if pipeline_mode == 'general':
                if progress_bar: progress_bar.progress(50, text=f"Executando Pipeline ML ({ml_mode.upper()})...")
                self.treinar_modelos_ensemble(ml_mode=ml_mode, progress_callback=progress_bar)
            else:
                 for ativo in self.ativos_sucesso:
                    self.predicoes_ml[ativo] = {'predicted_proba_up': 0.5, 'auc_roc_score': 0.0, 'model_name': 'Fundamental Mode'}
                 if progress_bar: progress_bar.progress(60, text="Pulando ML (Modo Fundamentalista)...")
            
            if progress_bar: progress_bar.progress(70, text="Ranqueando e selecionando...")
            self.pontuar_e_selecionar_ativos(horizonte_tempo=perfil_inputs.get('time_horizon', 'MÉDIO PRAZO')) 
            
            if pipeline_mode == 'general':
                if progress_bar: progress_bar.progress(85, text="Otimizando alocação (Markowitz)...")
                self.alocacao_portfolio = self.otimizar_alocacao(nivel_risco=perfil_inputs.get('risk_level', 'MODERADO'))
            else:
                if progress_bar: progress_bar.progress(85, text="Alocação por Score...")
                self.alocacao_portfolio = self.otimizar_alocacao(nivel_risco='PESOS_IGUAIS')

            if progress_bar: progress_bar.progress(95, text="Calculando métricas finais...")
            self.calcular_metricas_portfolio()
            self.gerar_justificativas()
            if progress_bar: progress_bar.progress(100, text="Pipeline concluído!")
            time.sleep(1)
        except Exception as e:
            st.error(f"Erro durante execução do pipeline: {e}")
            st.code(traceback.format_exc())
            return False
        return True


# Resto do código (funções de visualização, abas do Streamlit, etc.) permanece inalterado
# mas usando a classe ConstrutorPortfolioAutoML corrigida

def configurar_pagina():
    st.set_page_config(
        page_title="Sistema de Otimização Quantitativa",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

def main():
    if 'builder' not in st.session_state:
        st.session_state.builder = None
        st.session_state.builder_complete = False
        st.session_state.profile = {}
        st.session_state.ativos_para_analise = []
        st.session_state.analisar_ativo_triggered = False
        
    configurar_pagina()
    if 'debug_logs' not in st.session_state:
        st.session_state.debug_logs = []

    st.markdown('<h1 class="main-header">Sistema de Otimização Quantitativa</h1>', unsafe_allow_html=True)
    
    tabs_list = ["📚 Metodologia", "🎯 Seleção de Ativos", "🏗️ Construtor de Portfólio", "🔍 Análise Individual", "📖 Referências"]
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(tabs_list)
    
    with tab1: 
        st.markdown("## Metodologia")
        st.write("""
        ### Sistema Profissional de Otimização de Portfólio
        
        **Novidades da Versão 10.0:**
        - ✅ **GARCH com Hannan-Quinn**: Modelagem avançada de volatilidade condicional
        - ✅ **ML com ElasticNet**: Seleção automática de features relevantes
        - ✅ **Score Ponderado**: ML baseado em AUC × Probabilidade para maior robustez
        - ✅ **Ensemble Complexo**: Random Forest + XGBoost para predições refinadas
        
        **Pipeline:**
        1. Coleta de dados (YFinance + TvDatafeed + Pynvest)
        2. Modelagem GARCH com seleção automática de ordem (Hannan-Quinn)
        3. Features técnicas e fundamentalistas
        4. ML com ElasticNet para seleção + LogReg/Ensemble
        5. Score ponderado: 80% (Fundamentalista+Técnico) + 20% (ML)
        6. Otimização Markowitz com volatilidades GARCH
        """)
    
    with tab2: 
        st.markdown("## Seleção de Ativos")
        st.info(f"✔️ {len(ATIVOS_IBOVESPA)} ativos do IBOVESPA disponíveis")
        st.session_state.ativos_para_analise = ATIVOS_IBOVESPA
    
    with tab3: 
        aba_construtor_portfolio()
    
    with tab4: 
        st.markdown("## Análise Individual")
        st.info("Selecione um ativo específico para análise detalhada")
    
    with tab5: 
        st.markdown("## Referências")
        st.write("""
        - GARCH: Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity
        - Hannan-Quinn: Hannan & Quinn (1979). The Determination of the Order of an Autoregression
        - ElasticNet: Zou & Hastie (2005). Regularization and Variable Selection via the Elastic Net
        - Markowitz: Markowitz, H. (1952). Portfolio Selection
        """)

def aba_construtor_portfolio():
    """Aba 3: Construtor de Portfólio"""
    
    if 'ativos_para_analise' not in st.session_state or not st.session_state.ativos_para_analise:
        st.warning("⚠️ Por favor, defina o universo de análise na aba **'Seleção de Ativos'** primeiro.")
        return
    
    if 'builder' not in st.session_state: st.session_state.builder = None
    if 'profile' not in st.session_state: st.session_state.profile = {}
    if 'builder_complete' not in st.session_state: st.session_state.builder_complete = False
    
    if not st.session_state.builder_complete:
        st.markdown('## 📋 Calibração do Perfil de Risco')
        
        st.info(f"✔️ **{len(st.session_state.ativos_para_analise)} ativos** prontos. Responda o questionário para calibrar a otimização.")
        
        col_question1, col_question2 = st.columns(2)
        
        with st.form("investor_profile_form_v10", clear_on_submit=False): 
            
            with col_question1:
                st.markdown("#### Tolerância ao Risco")
                
                p2_risk_desc = st.radio(
                    "**1. Tolerância à Volatilidade:**", 
                    options=OPTIONS_CONCORDA, index=2, key='risk_accept_radio_v10_q1'
                )
                
                p3_gain_desc = st.radio(
                    "**2. Foco em Retorno Máximo:**", 
                    options=OPTIONS_CONCORDA, index=2, key='max_gain_radio_v10_q2'
                )
                
                p4_stable_desc = st.radio(
                    "**3. Prioridade de Estabilidade:**", 
                    options=OPTIONS_DISCORDA, index=2, key='stable_growth_radio_v10_q3'
                )
                
                p5_loss_desc = st.radio(
                    "**4. Aversão à Perda:**", 
                    options=OPTIONS_DISCORDA, index=2, key='avoid_loss_radio_v10_q4'
                )
                
                p511_reaction_desc = st.radio(
                    "**5. Reação a Queda de 10%:**", 
                    options=OPTIONS_REACTION_DETALHADA, index=1, key='reaction_radio_v10_q5'
                )
                
                p_level_desc = st.radio(
                    "**6. Nível de Conhecimento:**", 
                    options=OPTIONS_CONHECIMENTO_DETALHADA, index=1, key='level_radio_v10_q6'
                )
            
            with col_question2:
                st.markdown("#### Horizonte e Capital")
                
                p211_time_desc = st.radio(
                    "**7. Horizonte de Investimento:**", 
                    options=OPTIONS_TIME_HORIZON_DETALHADA, index=2, key='time_purpose_radio_v10_q7'
                )
                
                p311_liquid_desc = st.radio(
                    "**8. Necessidade de Liquidez:**", 
                    options=OPTIONS_LIQUIDEZ_DETALHADA, index=2, key='liquidity_radio_v10_q8'
                )
                
                st.markdown("---")
                investment = st.number_input(
                    "Capital Total a ser Alocado (R$)",
                    min_value=1000, max_value=10000000, value=10000, step=1000, key='investment_amount_input_v10'
                )

                st.markdown("---")
                st.markdown("#### Modo de Execução do Pipeline")

                pipeline_mode = st.radio(
                    "**1. Modo de Construção:**",
                    ["Modo Geral (ML + Otimização Markowitz)", "Modo Fundamentalista (Cluster/Anomalias)"],
                    index=0,
                    key='pipeline_mode_radio_v10'
                )

                ml_mode = 'simple'
                if pipeline_mode == 'Modo Geral (ML + Otimização Markowitz)':
                    ml_mode = st.selectbox(
                        "**2. Seleção de Modelo ML:**",
                        ['simple', 'complex'],
                        format_func=lambda x: "Simples (ElasticNet + LogReg)" if x == 'simple' else "Complexo (ElasticNet + Ensemble RF+XGB)",
                        index=0,
                        key='ml_model_mode_select_v10'
                    )
            
            st.markdown("---")
            col_btn_start, col_btn_center, col_btn_end = st.columns([1, 2, 1])
            with col_btn_center:
                submitted = st.form_submit_button("🚀 Gerar Alocação Otimizada", type="primary", use_container_width=True)
            
            progress_bar_placeholder = st.empty()
            
            if submitted:
                log_debug("Questionário de perfil submetido.")
                risk_answers_originais = {
                    'risk_accept': MAP_CONCORDA.get(p2_risk_desc, 'N: Neutro'),
                    'max_gain': MAP_CONCORDA.get(p3_gain_desc, 'N: Neutro'),
                    'stable_growth': MAP_DISCORDA.get(p4_stable_desc, 'N: Neutro'),
                    'avoid_loss': MAP_DISCORDA.get(p5_loss_desc, 'N: Neutro'),
                    'reaction': MAP_REACTION.get(p511_reaction_desc, 'B: Manteria e reavaliaria a tese'),
                    'level': MAP_CONHECIMENTO.get(p_level_desc, 'B: Intermediário (Conhecimento básico sobre mercados e ativos)'),
                    'time_purpose': p211_time_desc, 
                    'liquidity': p311_liquid_desc,
                }
                
                analyzer = AnalisadorPerfilInvestidor()
                risk_level, horizon, lookback, score = analyzer.calcular_perfil(risk_answers_originais)
                
                st.session_state.profile = {
                    'risk_level': risk_level, 'time_horizon': horizon, 'ml_lookback_days': lookback, 'risk_score': score
                }
                log_debug(f"Perfil calculado: {risk_level} (Score {score}). Horizonte ML: {lookback} dias.")
                
                try:
                    builder_local = ConstrutorPortfolioAutoML(investment)
                    st.session_state.builder = builder_local
                except Exception as e:
                    st.error(f"Erro fatal ao inicializar o construtor do portfólio: {e}")
                    st.code(traceback.format_exc())
                    return

                progress_widget = progress_bar_placeholder.progress(0, text=f"Iniciando pipeline para PERFIL {risk_level}...")
                
                # Extrai o modo correto do pipeline
                pipeline_mode_key = 'general' if 'Geral' in pipeline_mode else 'fundamentalista'
                
                success = builder_local.executar_pipeline(
                    simbolos_customizados=st.session_state.ativos_para_analise,
                    perfil_inputs=st.session_state.profile,
                    ml_mode=ml_mode,
                    pipeline_mode=pipeline_mode_key,
                    progress_bar=progress_widget
                )
                
                progress_widget.empty()
                    
                if not success:
                    st.error("Falha na aquisição ou processamento dos dados.")
                    st.session_state.builder_complete = True
                    st.rerun() 
                
                st.session_state.builder_complete = True
                st.rerun()
    
    else:
        builder = st.session_state.builder
        if builder is None: 
            st.error("Objeto construtor não encontrado. Recomece a análise.")
            st.session_state.builder_complete = False
            return
            
        profile = st.session_state.profile
        assets = builder.ativos_selecionados
        allocation = builder.alocacao_portfolio
        
        st.markdown('## ✅ Relatório de Alocação Otimizada')
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        col1.metric("Perfil Identificado", profile.get('risk_level', 'N/A'))
        col2.metric("Score Risco", profile.get('risk_score', 'N/A'))
        col3.metric("Horizonte Estratégico", profile.get('time_horizon', 'N/A'))
        
        sharpe_val = builder.metricas_portfolio.get('sharpe_ratio', 0)
        if pd.isna(sharpe_val) or sharpe_val == np.inf or sharpe_val == -np.inf:
             sharpe_display = "0.000"
        else:
             sharpe_display = f"{sharpe_val:.3f}"
             
        col4.metric("Sharpe (Portfólio)", sharpe_display)
        
        strategy_name = builder.metodo_alocacao_atual.split('(')[0].strip()
        if len(strategy_name) > 15:
            strategy_name = strategy_name.replace("MINIMIZAÇÃO DE ", "MIN ").replace("MAXIMIZAÇÃO DE ", "MAX ")
            
        col5.metric("Estratégia", strategy_name)
        
        st.markdown("---")
        
        # Exibição de resultados
        st.markdown("### 📊 Alocação Recomendada")
        
        if allocation:
            df_allocation = pd.DataFrame([
                {
                    'Ticker': ticker,
                    'Peso (%)': data['weight'] * 100,
                    'Valor (R$)': data['amount'],
                    'Justificativa': builder.justificativas_selecao.get(ticker, 'N/A')
                }
                for ticker, data in allocation.items()
            ])
            
            st.dataframe(df_allocation, use_container_width=True)
            
            # Gráfico de pizza
            fig = go.Figure(data=[go.Pie(
                labels=df_allocation['Ticker'],
                values=df_allocation['Peso (%)'],
                hole=0.3
            )])
            fig.update_layout(title="Distribuição de Alocação", template=obter_template_grafico())
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Nenhuma alocação disponível.")

if __name__ == "__main__":
    main()
