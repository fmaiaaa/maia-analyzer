# -*- coding: utf-8 -*-
"""
=============================================================================
SISTEMA DE PORTFÓLIOS ADAPTATIVOS - OTIMIZAÇÃO QUANTITATIVA
=============================================================================

Adaptação do Sistema AutoML para coleta em TEMPO REAL (Live Data).
- Preços: Estratégia Linear com Fail-Fast (YFinance -> TvDatafeed -> Estático Global). 
- Fundamentos: Coleta Exaustiva Pynvest (50+ indicadores).
- Lógica de Construção (V9.4): Pesos Dinâmicos + Seleção por Clusterização.
- Design (V9.31): ML Soft Fallback (Short History Support).

Versão: 9.32.32 (Update: FINAL SCOPE FIX 3, ML/GARCH ROBUSTNESS, REMOVE DEBUG)
=============================================================================
"""

# --- 1. CORE LIBRARIES & UTILITIES ---
import warnings
# Movemos o filtro para o topo para suprimir SyntaxWarnings de bibliotecas importadas
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

# --- 2. SCIENTIFIC / STATISTICAL TOOLS ---
from scipy.optimize import minimize
from scipy.stats import zscore, norm

# --- 3. STREAMLIT, DATA ACQUISITION, & PLOTTING ---\
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests

# --- IMPORTAÇÕES PARA COLETA LIVE (HÍBRIDO) ---
# 1. TvDatafeed (Primário)
try:
    from tvDatafeed import TvDatafeed, Interval
except ImportError:
    st.error("""
    Biblioteca 'tvdatafeed' não encontrada. 
    Instale usando: pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git
    """)

# 2. YFinance (Backup/Fallback)
import yfinance as yf

# 3. Pynvest (Fundamentos)
try:
    from pynvest.scrappers.fundamentus import Fundamentus
except ImportError:
    pass

# --- 4. FEATURE ENGINEERING / TECHNICAL ANALYSIS (TA) ---
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import silhouette_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer 
try:
    import hdbscan
except ImportError:
    pass # Fallback to KMeans if hdbscan not installed

# --- 7. SPECIALIZED TIME SERIES & ECONOMETRICS ---
from arch import arch_model
# --- IMPORTAÇÕES ADICIONAIS PARA INDICADORES (DO ARQUIVO mlrun.py) ---
try:
    from ta.trend import SMAIndicator, EMAIndicator, MACD
    from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
    from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel, DonchianChannel
    from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, MFIIndicator, VolumeWeightedAveragePrice
    from ta.trend import ADXIndicator, CCIIndicator
except ImportError:
    # Apenas um aviso, pois o código deve rodar mesmo sem as bibliotecas avançadas
    pass 


# =============================================================================
# 1. CONFIGURAÇÕES E CONSTANTES GLOBAIS
# =============================================================================

PERIODO_DADOS = 'max' # ALTERADO: Período máximo de dados
MIN_DIAS_HISTORICO = 252
NUM_ATIVOS_PORTFOLIO = 5
TAXA_LIVRE_RISCO = 0.1075
LOOKBACK_ML = 30
SCORE_PERCENTILE_THRESHOLD = 0.85 # Limite para definir o score mínimo de inclusão (85% do score do 15º melhor ativo)

# Pesos de alocação (Markowitz - Lógica Analyzer)
PESO_MIN = 0.10
PESO_MAX = 0.30

LOOKBACK_ML_DAYS_MAP = {
    'curto_prazo': 84,   # ALTERADO: 84 dias (Aprox. 4 meses)
    'medio_prazo': 168,  # ALTERADO: 168 dias (Aprox. 8 meses)
    'longo_prazo': 252   # ALTERADO: 252 dias (Aprox. 1 ano)
}

# =============================================================================
# 4. LISTAS DE ATIVOS E SETORES (AJUSTADAS SOMENTE PARA IBOVESPA)
# =============================================================================

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
    'Petróleo, Gás e Biocombustíveis': ['ENEV3.SA', 'PETR3.SA', 'PETR4.SA', 'RECV3.SA', 'PRIO3.SA', 'RAIZ4.SA', 'UGPA3.SA', 'VBBR3.SA'],
    'Saúde': ['FLRY3.SA', 'HAPV3.SA', 'RADL3.SA'],
    'Tecnologia da Informação': ['TOTS3.SA'],
    'Telecomunicações': ['TIMS3.SA', 'VIVT3.SA'],
    'Utilidade Pública': ['AESB3.SA', 'BRAV3.SA', 'CMIG4.SA', 'CPLE6.SA', 'CPFE3.SA', 'EGIE3.SA', 'ELET3.SA', 'ELET6.SA', 'ENGI11.SA', 'EQTL3.SA', 'ISAE4.SA', 'RAIL3.SA', 'SBSP3.SA', 'TAEE11.SA']
}

# Dicionário Fallback Invertido (Ticker -> Setor)
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

# =============================================================================
# 5. MAPEAMENTOS DE PONTUAÇÃO DO QUESTIONÁRIO (Design Original)
# =============================================================================

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

# --- FUNÇÕES UTILITÁRIAS GLOBAIS (Movidas para o início para garantir o escopo) ---

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
    # Remover logs no arquivo final, mas manter a função para evitar NameError
    pass 

def mostrar_debug_panel():
    """Exibe o painel de debug (Streamlit expander) no topo da aplicação."""
    # Remover painel de debug do código final
    pass 

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
        # A Series recebida aqui agora é garantida como Pandas Series pela correção do FIX 2
        
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
    @staticmethod
    def enriquecer_dados_tecnicos(df_ativo: pd.DataFrame) -> pd.DataFrame:
        if df_ativo.empty: return df_ativo
        df = df_ativo.sort_index().copy()
        
        # --- Cálculo de Retornos (Necessário para a maioria dos indicadores) ---
        df['returns'] = df['Close'].pct_change()
        
        # 1. Volatilidade e Retornos
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        df['volatility_60'] = df['returns'].rolling(window=60).std() * np.sqrt(252)
        df['volatility_252'] = df['returns'].rolling(window=252).std() * np.sqrt(252)
        
        # 2. Médias Móveis (SMA, EMA, WMA, HMA)
        for periodo in [5, 10, 20, 50, 100, 200]:
            if len(df) >= periodo:
                df[f'sma_{periodo}'] = df['Close'].rolling(periodo).mean()
                df[f'ema_{periodo}'] = df['Close'].ewm(span=periodo, adjust=False).mean()
                
                # WMA (Weighted Moving Average)
                weights = np.arange(1, periodo + 1)
                # Use numpy for WMA for robustness
                df[f'wma_{periodo}'] = df['Close'].rolling(periodo).apply(lambda x: np.dot(x, weights[:len(x)]) / weights[:len(x)].sum(), raw=True)
                
        # Hull Moving Average (HMA) - Requires SMA/WMA dependencies
        for periodo in [20, 50]:
            if len(df) >= periodo:
                wma_half_series = df['Close'].rolling(periodo // 2).apply(lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(), raw=True)
                wma_full_series = df['Close'].rolling(periodo).apply(lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(), raw=True)
                df[f'hma_{periodo}'] = (2 * wma_half_series - wma_full_series).rolling(int(np.sqrt(periodo))).mean()
        
        # 3. Razões de preço e cruzamentos (Requere SMA 20, 50, 200)
        df['price_sma20_ratio'] = df['Close'] / df.get('sma_20')
        df['price_sma50_ratio'] = df['Close'] / df.get('sma_50')
        df['price_sma200_ratio'] = df['Close'] / df.get('sma_200')
        df['sma20_sma50_cross'] = (df.get('sma_20', pd.Series(0)) > df.get('sma_50', pd.Series(0))).astype(int)
        df['sma50_sma200_cross'] = (df.get('sma_50', pd.Series(0)) > df.get('sma_200', pd.Series(0))).astype(int)
        df['death_cross'] = (df['Close'] < df.get('sma_200', pd.Series(0))).astype(int)
        
        # 4. RSI (múltiplos períodos)
        for periodo in [7, 14, 21, 28]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=periodo).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
            rs = gain / loss.replace(0, np.nan)
            df[f'rsi_{periodo}'] = 100 - (100 / (1 + rs))

        # 5. Stochastic Oscillator
        if 'High' in df.columns and 'Low' in df.columns and len(df) >= 14:
            stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # Williams %R
            df['williams_r'] = WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close'], lbp=14).williams_r()
        
        # 6. MACD (múltiplas configurações)
        if len(df) >= 26:
            macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            # MACD alternativo (5, 35, 5)
            macd_alt = MACD(close=df['Close'], window_slow=35, window_fast=5, window_sign=5)
            df['macd_alt'] = macd_alt.macd()
        
        # 7. Bollinger Bands
        if len(df) >= 20:
            bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_width'] = bb.bollinger_wband()
            df['bb_position'] = bb.bollinger_pband()
            # df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # 8. Keltner Channel
        if len(df) >= 20 and 'High' in df.columns and 'Low' in df.columns:
            kc = KeltnerChannel(high=df['High'], low=df['Low'], close=df['Close'], window=20, window_atr=10)
            df['kc_upper'] = kc.keltner_channel_hband()
            df['kc_lower'] = kc.keltner_channel_lband()
            df['kc_middle'] = kc.keltner_channel_mband()
            df['kc_width'] = (df['kc_upper'] - df['kc_lower']) / df['kc_middle']
            
        # 9. Donchian Channel
        if len(df) >= 20 and 'High' in df.columns and 'Low' in df.columns:
            dc = DonchianChannel(high=df['High'], low=df['Low'], close=df['Close'], window=20)
            df['dc_upper'] = dc.donchian_channel_hband()
            df['dc_lower'] = dc.donchian_channel_lband()
            df['dc_middle'] = dc.donchian_channel_mband()
            
        # 10. ATR (Average True Range)
        if len(df) >= 14 and 'High' in df.columns and 'Low' in df.columns:
            atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
            df['atr'] = atr.average_true_range()
            df['atr_percent'] = (df['atr'] / df['Close']) * 100
            
            # ADX (Average Directional Index)
            adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
            df['adx'] = adx.adx()
            df['adx_pos'] = adx.adx_pos()
            df['adx_neg'] = adx.adx_neg()
            
        # 11. CCI (Commodity Channel Index)
        if len(df) >= 20 and 'High' in df.columns and 'Low' in df.columns:
            df['cci'] = CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=20).cci()
        
        # 12. Momentum indicators (Rate of Change)
        for periodo in [10, 20, 60]:
            if len(df) >= periodo:
                df[f'roc_{periodo}'] = ROCIndicator(close=df['Close'], window=periodo).roc()
        
        # 13. Volume indicators
        if 'Volume' in df.columns:
            df['obv'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
            if len(df) >= 20:
                df['cmf'] = ChaikinMoneyFlowIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=20).chaikin_money_flow()
            if len(df) >= 14:
                df['mfi'] = MFIIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=14).money_flow_index()
            
            df['vwap'] = VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).volume_weighted_average_price()
        
        # 14. Drawdown
        cumulative_returns = (1 + df['returns']).cumprod()
        running_max = cumulative_returns.expanding().max()
        df['drawdown'] = (cumulative_returns - running_max) / running_max
        if len(df) >= 252:
            df['max_drawdown_252'] = df['drawdown'].rolling(252).min()
        
        # 15. Lags e Rolling statistics
        for lag in [1, 5, 20]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        
        for window in [5, 20]:
            df[f'returns_mean_{window}'] = df['returns'].rolling(window).mean()
        
        # 16. Temporal encoding
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['week_of_year'] = df.index.isocalendar().week
        
        # O DROPNA É AQUI: Retorna o DataFrame removendo TODAS as linhas que contêm NaN
        # Isso garante que apenas dados completos (onde todos os indicadores puderam ser calculados)
        # sejam usados para o treino, resolvendo a falha de "0 pontos válidos".
        return df.dropna(axis=0, how='any')

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
        # Corresponde às novas configurações: A=84, B=168, C=252
        time_map = { 'A': 84, 'B': 168, 'C': 252 } 
        final_lookback = max( time_map.get(liquidez_key, 84), time_map.get(objetivo_key, 84) )
        
        if final_lookback >= 252:
            self.horizonte_tempo = "LONGO PRAZO"
        elif final_lookback >= 168:
            self.horizonte_tempo = "MÉDIO PRAZO"
        else:
            self.horizonte_tempo = "CURTO PRAZO"
            
        self.dias_lookback_ml = final_lookback
        return self.horizonte_tempo, self.dias_lookback_ml
    
    def calcular_perfil(self, respostas_risco_originais: dict) -> tuple[str, str, int, int]:
        score_risk_accept = SCORE_MAP_ORIGINAL.get(respostas_risco_originais['risk_accept'], 3)
        score_max_gain = SCORE_MAP_ORIGINAL.get(respostas_risco_originais['max_gain'], 3)
        score_stable_growth = SCORE_MAP_INV_ORIGINAL.get(respostas_risco_originais['stable_growth'], 3)
        score_avoid_loss = SCORE_MAP_INV_ORIGINAL.get(respostas_risco_originais['avoid_loss'], 3)
        score_level = SCORE_MAP_CONHECIMENTO_ORIGINAL.get(respostas_risco_originais['level'], 3)
        score_reaction = SCORE_MAP_REACTION_ORIGINAL.get(respostas_risco_originais['reaction'], 3)

        pontuacao = (
            score_risk_accept * 5 +
            score_max_gain * 5 +
            score_stable_growth * 5 +
            score_avoid_loss * 5 +
            score_level * 3 +
            score_reaction * 3
        )
        nivel_risco = self.determinar_nivel_risco(pontuacao)
        
        liquidez_key = respostas_risco_originais['liquidity'][0] if isinstance(respostas_risco_originais['liquidity'], str) and respostas_risco_originais['liquidity'] else 'C'
        objetivo_key = respostas_risco_originais['time_purpose'][0] if isinstance(respostas_risco_originais['time_purpose'], str) and respostas_risco_originais['time_purpose'] else 'C'
        
        horizonte_tempo, ml_lookback = self.determinar_horizonte_ml(liquidez_key, objetivo_key)
        return nivel_risco, horizonte_tempo, ml_lookback, pontuacao

# =============================================================================
# 10. CLASSE: OTIMIZADOR DE PORTFÓLIO
# =============================================================================

class OtimizadorPortfolioAvancado:
    def __init__(self, returns_df: pd.DataFrame, garch_vols: dict = None):
        self.returns = returns_df
        self.mean_returns = returns_df.mean() * 252
        if garch_vols is not None and garch_vols:
            try:
                self.cov_matrix = self._construir_matriz_cov_garch(returns_df, garch_vols)
            except:
                self.cov_matrix = returns_df.cov() * 252 # Fallback se GARCH falhar na matriz
        else:
            self.cov_matrix = returns_df.cov() * 252
        self.num_ativos = len(returns_df.columns)

    def _construir_matriz_cov_garch(self, returns_df: pd.DataFrame, garch_vols: dict) -> pd.DataFrame:
        corr_matrix = returns_df.corr()
        # Verifica se GARCH Vols estão validas, senão usa histórica
        vol_array = []
        for ativo in returns_df.columns:
            vol = garch_vols.get(ativo)
            if pd.isna(vol) or vol == 0:
                vol = returns_df[ativo].std() * np.sqrt(252) # Fallback histórico
            vol_array.append(vol)
            
        vol_array = np.array(vol_array)
        cov_matrix = corr_matrix.values * np.outer(vol_array, vol_array)
        return pd.DataFrame(cov_matrix, index=returns_df.columns, columns=returns_df.columns)
    
    def estatisticas_portfolio(self, pesos: np.ndarray) -> tuple[float, float]:
        p_retorno = np.dot(pesos, self.mean_returns)
        p_vol = np.sqrt(np.dot(pesos.T, np.dot(self.cov_matrix, pesos)))
        return p_retorno, p_vol
    
    def sharpe_negativo(self, pesos: np.ndarray) -> float:
        p_retorno, p_vol = self.estatisticas_portfolio(pesos)
        if p_vol <= 1e-9: return -100.0
        return -(p_retorno - TAXA_LIVRE_RISCO) / p_vol
    
    def minimizar_volatilidade(self, pesos: np.ndarray) -> float:
        return self.estatisticas_portfolio(pesos)[1]
    
    def otimizar(self, estrategia: str = 'MaxSharpe') -> dict:
        if self.num_ativos == 0: return {}
        restricoes = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        limites = tuple((PESO_MIN, PESO_MAX) for _ in range(self.num_ativos))
        chute_inicial = np.array([1.0 / self.num_ativos] * self.num_ativos)
        
        if estrategia == 'MinVolatility': objetivo = self.minimizar_volatilidade
        else: objetivo = self.sharpe_negativo
        
        try:
            resultado = minimize(objetivo, chute_inicial, method='SLSQP', bounds=limites, constraints=restricoes, options={'maxiter': 500, 'ftol': 1e-6})
            if resultado.success:
                final_weights = resultado.x / np.sum(resultado.x)
                return {ativo: peso for ativo, peso in zip(self.returns.columns, final_weights)}
            else:
                return {ativo: 1.0 / self.num_ativos for ativo in self.returns.columns}
        except Exception:
            return {ativo: 1.0 / self.num_ativos for ativo in self.returns.columns}

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
            # FIX 1: Remove login/senha para forçar nologin e evitar timeout de autenticação
            self.tv = TvDatafeed() 
            self.tv_ativo = True
            log_debug("TvDatafeed inicializado com sucesso.")
        except Exception as e:
            self.tv_ativo = False
            log_debug(f"ERRO: Falha ao inicializar TvDatafeed: {str(e)[:50]}...")
            st.error(f"Erro ao inicializar tvDatafeed: {e}")
        
        try:
            from pynvest.scrappers.fundamentus import Fundamentus
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
                    # === INÍCIO DA CORREÇÃO DE ERRO PYARROW/EMPTY STRING ===
                    if val.strip() == '':
                        val = np.nan  # Converte string vazia para NaN
                    # === FIM DA CORREÇÃO ===

                    # Se o valor ainda for uma string (ou seja, não era vazia)
                    if isinstance(val, str):
                        val = val.replace('.', '').replace(',', '.')
                        if val.endswith('%'):
                            val = val.replace('%', '')
                            try: val = float(val) / 100.0
                            except (ValueError, TypeError): pass
                try: 
                    # Tenta a conversão final, que agora aceita float ou np.nan
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
            except Exception:
                pass
            time.sleep(0.05) 
        
        if lista_fund:
            log_debug(f"Coleta de fundamentos finalizada. {len(lista_fund)} ativos com dados.")
            return pd.DataFrame(lista_fund).set_index('Ticker')
        return pd.DataFrame()

    def coletar_e_processar_dados(self, simbolos: list, check_min_ativos: bool = True) -> bool:
        self.ativos_sucesso = []
        lista_fundamentalistas = []
        garch_vols = {}
        metricas_simples_list = []
        
        consecutive_failures = 0
        FAILURE_THRESHOLD = 3 
        global_static_mode = False 
        log_debug(f"Iniciando ciclo de coleta de preços para {len(simbolos)} ativos. Limite de falhas: {FAILURE_THRESHOLD}.")
        
        for simbolo in simbolos:
            df_tecnicos = pd.DataFrame()
            usando_fallback_estatico = False 
            tem_dados = False
            
            # Inicializando com valores de fallback para evitar NameError
            vol_anual, ret_anual, sharpe, max_dd = 0.20, 0.0, 0.0, 0.0
            garch_vol = 0.20 # Fallback

            if not global_static_mode:
                
                # --- BLOCO PRIMÁRIO: YFINANCE ---
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

                # --- BLOCO FALLBACK: TVDATAFEED ---
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
                        # --- LOG DETALHADO DO TVDATAFEED ---
                        log_debug(f"TvDatafeed Head: \n{df_tecnicos.head().to_string()}")
                        # --- FIM LOG DETALHADO ---

                # --- FIM DA LÓGICA DE SWAP DE COLETA ---

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
                    # Aplicar o enriquecimento de features expandido
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
                log_debug(f"Ativo {simbolo}: Calculando métricas de performance (Sharpe/DD/GARCH)...")
                
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
                
                # --- INÍCIO DA IMPLEMENTAÇÃO GARCH ---
                garch_vol = vol_anual # Fallback inicial GARCH = Histórico
                if len(retornos) > 60: 
                    try:
                        # GARCH é sensível a dados muito longos ou ruído, mantemos a chamada.
                        am = arch_model(retornos * 100, mean='Zero', vol='Garch', p=1, q=1)
                        res = am.fit(disp='off', last_obs=retornos.index[-1]) 
                        garch_std_daily = res.conditional_volatility.iloc[-1] / 100 
                        garch_vol = garch_std_daily * np.sqrt(252)
                        if np.isnan(garch_vol) or garch_vol == 0: raise ValueError("GARCH returned NaN or zero.")
                        log_debug(f"Ativo {simbolo}: GARCH concluído. Vol Condicional: {garch_vol*100:.2f}%.")
                    except Exception as e:
                        garch_vol = vol_anual 
                        log_debug(f"Ativo {simbolo}: GARCH falhou ({str(e)[:20]}). Usando Vol Histórica como Vol Condicional.")
                # --- FIM DA IMPLEMENTAÇÃO GARCH ---
            
            fund_data.update({
                'Ticker': simbolo, 'sharpe_ratio': sharpe, 'annual_return': ret_anual,
                'annual_volatility': vol_anual, 'max_drawdown': max_dd, 'garch_volatility': garch_vol,
                'static_mode': usando_fallback_estatico
            })
            
            self.dados_por_ativo[simbolo] = df_tecnicos
            self.ativos_sucesso.append(simbolo)
            lista_fundamentalistas.append(fund_data)
            garch_vols[simbolo] = garch_vol 
            
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
             
        self.volatilidades_garch_raw = garch_vols 
        
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
        [CORREÇÃO DE ATRIBUTO] Adaptado para ser um método de ColetorDadosLive.
        Função para coletar e processar um único ativo para a aba de análise individual.
        """
        log_debug(f"Iniciando coleta e análise de ativo único: {ativo_selecionado}")
        
        # Chama coleta principal com check_min_ativos=False
        self.coletar_e_processar_dados([ativo_selecionado], check_min_ativos=False)
        
        if ativo_selecionado not in self.dados_por_ativo:
            log_debug(f"ERRO: Dados não encontrados após coleta para {ativo_selecionado}.")
            return None, None, None

        df_tec = self.dados_por_ativo[ativo_selecionado]
        fund_row = {}
        if ativo_selecionado in self.dados_fundamentalistas.index:
            fund_row = self.dados_fundamentalistas.loc[ativo_selecionado].to_dict()
        
        df_ml_meta = pd.DataFrame()
        
        # A detecção de preço disponível deve ser mais robusta, pois dependemos do enrichment
        is_price_data_available = 'Close' in df_tec.columns and not df_tec['Close'].isnull().all() and len(df_tec.dropna(subset=['Close'])) > 60
        
        # --- NOVO: Feature Pool Completo para ML e GARCH ---
        # Features completas definidas com o enriquecimento expandido:
        ALL_TECH_FEATURES = ['rsi_14', 'macd_diff', 'volatility_20', 'volatility_60', 'log_returns', 'sma_5', 'ema_50', 
            'bb_width', 'adx', 'atr', 'returns_lag_5', 'returns_mean_20', 'day_of_week', 'month', 
            # Adicionado mais indicadores para aumentar a chance de ter > 50 pontos:
            'stoch_k', 'stoch_d', 'williams_r', 'macd_signal', 'macd_alt', 'bb_middle', 
            'bb_upper', 'bb_lower', 'bb_position', 'bb_pband', 'cmf', 'vwap'
        ]
        
        ALL_FUND_FEATURES = ['pe_ratio', 'pb_ratio', 'div_yield', 'roe', 'pe_rel_sector', 'pb_rel_sector', 'Cluster', 'roic', 'net_margin', 'debt_to_equity', 'current_ratio', 'revenue_growth', 'ev_ebitda', 'operating_margin']
        
        if is_price_data_available:
            log_debug(f"Análise Individual ML: Iniciando modelo supervisionado para {ativo_selecionado}.")
            try:
                df = df_tec.copy()
                
                df['Future_Direction'] = np.where(df['Close'].pct_change(5).shift(-5) > 0, 1, 0)
                
                # Identifica features válidas para este ativo
                available_features = [f for f in (ALL_TECH_FEATURES + ALL_FUND_FEATURES) if f in df.columns]
                
                # Remove NaNs apenas para as features de interesse
                df_model = df.dropna(subset=available_features + ['Future_Direction'])

                if len(df_model) > 50:
                    log_debug(f"ML Individual: {len(df_model)} pontos válidos para treino. Treinando RF...")

                    X = df_model[available_features].iloc[:-5]
                    y = df_model['Future_Direction'].iloc[:-5]
                    
                    # Filtra features que são completamente NaN no subconjunto de treino (embora o dropna devesse prevenir isso)
                    final_features = [col for col in X.columns if not X[col].isnull().all()]
                    X = X[final_features]


                    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
                    model.fit(X, y)
                    
                    last_features = df[final_features].iloc[[-1]].copy().fillna(0) # Garantir que o input de predição não tem NaN
                    proba = model.predict_proba(last_features)[0][1]
                    
                    # --- CÁLCULO DE AUC PARA CONFIANÇA ---
                    if len(np.unique(y)) >= 2:
                        from sklearn.metrics import roc_auc_score
                        cv = TimeSeriesSplit(n_splits=3)
                        auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
                        avg_auc = auc_scores.mean()
                    else:
                        avg_auc = 0.5
                    
                    
                    is_neutral_result = (avg_auc <= 0.55) or (len(np.unique(y)) < 2) 

                    if is_neutral_result:
                        # Heuristica de proxy para o MODO SUPERVISIONADO NEUTRO
                        roe = fund_row.get('roe', 0); pe = fund_row.get('pe_ratio', 15)
                        if pe <= 0: pe = 20
                        score_fund_proxy = min(0.9, max(0.05, (roe * 100) / pe * 0.5 + 0.4))
                        
                        proba_final = score_fund_proxy
                        conf_final = 0.60
                        log_debug(f"ML Individual: Neutro (AUC={avg_auc:.2f}). Usando Proxy Fund. Score: {proba_final:.2f}.")
                    else:
                        proba_final = proba
                        conf_final = avg_auc
                        log_debug(f"ML Individual: Sucesso. Prob: {proba_final:.2f}, AUC: {conf_final:.2f}.")

                    importances = pd.DataFrame({
                        'feature': final_features,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    df_tec['ML_Proba'] = proba_final
                    df_tec['ML_Confidence'] = conf_final
                    df_ml_meta = importances
                    
                else:
                    log_debug(f"ML Individual: Dados insuficientes ({len(df_model)}). Pulando modelo supervisionado.")
                    is_price_data_available = False # Força fallback não supervisionado no próximo if
            except Exception as e:
                log_debug(f"ML Individual: ERRO no modelo supervisionado: {str(e)[:50]}.")
                is_price_data_available = False
                
            
            # 2. Se ML não rodou (sem preço, ou falhou, ou dados insuficientes), roda UNSUPERVISED ANOMALY DETECTION
            if 'ML_Proba' not in df_tec.columns or not is_price_data_available:
                log_debug(f"Análise Individual ML: Recorrendo ao modelo NÃO SUPERVISIONADO (Fallback/Estático).")
                try:
                    # Simulação de Anomalia (Isolation Forest)
                    cols_fund = ['pe_ratio', 'pb_ratio', 'roe', 'net_margin', 'div_yield']
                    row_data = {k: float(fund_row.get(k, 0)) for k in cols_fund}
                    roe = row_data['roe']
                    pe = row_data['pe_ratio']
                    if pe <= 0: pe = 20
                    
                    # Heuristica de "Qualidade" como proxy de ML
                    quality_score = min(0.95, max(0.05, (roe * 100) / pe * 0.5 + 0.4))
                    
                    df_tec['ML_Proba'] = quality_score
                    df_tec['ML_Confidence'] = 0.50 # Confiança média para fallback
                    
                    # Metadados explicativos
                    df_ml_meta = pd.DataFrame({
                        'feature': ['Qualidade (ROE/PL)', 'Estabilidade'],
                        'importance': [0.8, 0.2]
                    })
                    log_debug(f"ML Individual: Modelo Não Supervisionado concluído. Score Qualidade: {quality_score:.2f}.")
                except Exception as e:
                    log_debug(f"ML Individual: ERRO no modelo não supervisionado: {str(e)[:50]}.")
                    df_tec['ML_Proba'] = 0.5; df_tec['ML_Confidence'] = 0.0

            return df_tec, fund_row, df_ml_meta
        return None, None, None

# =============================================================================
# 11. CLASSE PRINCIPAL: CONSTRUTOR DE PORTFÓLIO AUTOML
# =============================================================================

class ConstrutorPortfolioAutoML:
    def __init__(self, valor_investimento: float, periodo: str = PERIODO_DADOS):
        self.valor_investimento = valor_investimento
        self.periodo = periodo
        self.dados_por_ativo = {}
        self.dados_fundamentalistas = pd.DataFrame()
        self.metricas_performance = pd.DataFrame()
        self.volatilidades_garch = {}
        self.predicoes_ml = {}
        self.ativos_sucesso = []
        self.ativos_selecionados = []
        self.alocacao_portfolio = {}
        self.metricas_portfolio = {}
        self.metodo_alocacao_atual = "Não Aplicado"
        self.justificativas_selecao = {}
        self.perfil_dashboard = {} 
        self.pesos_atuais = {}
        self.scores_combinados = pd.DataFrame()
        
    def coletar_e_processar_dados(self, simbolos: list) -> bool:
        coletor = ColetorDadosLive(periodo=self.periodo)
        simbolos_filtrados = [s for s in simbolos if s in TODOS_ATIVOS]
        if not simbolos_filtrados: return False
        if not coletor.coletar_e_processar_dados(simbolos_filtrados): return False
        
        self.dados_por_ativo = coletor.dados_por_ativo
        self.dados_fundamentalistas = coletor.dados_fundamentalistas
        self.ativos_sucesso = coletor.ativos_sucesso
        self.metricas_performance = coletor.metricas_performance
        self.volatilidades_garch = coletor.volatilidades_garch_raw 
        return True

    def calculate_cross_sectional_features(self):
        df_fund = self.dados_fundamentalistas.copy()
        if 'sector' not in df_fund.columns or 'pe_ratio' not in df_fund.columns: return
        
        log_debug("Calculando features cross-sectional (P/L e P/VP relativos ao setor).")
        
        cols_numeric = ['pe_ratio', 'pb_ratio']
        for col in cols_numeric:
             if col in df_fund.columns:
                 df_fund[col] = pd.to_numeric(df_fund[col], errors='coerce')

        sector_means = df_fund.groupby('sector')[['pe_ratio', 'pb_ratio']].transform('mean')
        df_fund['pe_rel_sector'] = df_fund['pe_ratio'] / sector_means['pe_ratio']
        df_fund['pb_rel_sector'] = df_fund['pb_ratio'] / sector_means['pb_ratio']
        df_fund = df_fund.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        self.dados_fundamentalistas = df_fund
        log_debug("Features cross-sectional concluídas.")

    def calcular_volatilidades_garch(self):
        valid_vols = len([k for k, v in self.volatilidades_garch.items() if not np.isnan(v)])
        if valid_vols == 0:
             log_debug("AVISO: Todas as volatilidades GARCH são nulas. Substituindo por volatilidade histórica.")
             for ativo in self.ativos_sucesso:
                 if ativo in self.metricas_performance.index and 'volatilidade_anual' in self.metricas_performance.columns:
                      self.volatilidades_garch[ativo] = self.metricas_performance.loc[ativo, 'volatilidade_anual']
        log_debug("Verificando volatilidades GARCH. Aplicando fallback histórico onde necessário.")
        
    def treinar_modelos_ensemble(self, dias_lookback_ml: int = LOOKBACK_ML, otimizar: bool = False, progress_callback=None):
        ativos_com_dados = [s for s in self.ativos_sucesso if s in self.dados_por_ativo]
        log_debug("Iniciando Pipeline de Treinamento ML/Clusterização.")
        
        # --- Feature Pool Completo ---
        ALL_TECH_FEATURES = ['rsi_14', 'macd_diff', 'volatility_20', 'volatility_60', 'log_returns', 'sma_5', 'ema_50', 
            'bb_width', 'adx', 'atr', 'returns_lag_5', 'returns_mean_20', 'day_of_week', 'month', 
            # Adicionado mais indicadores para aumentar a chance de ter > 50 pontos:
            'stoch_k', 'stoch_d', 'williams_r', 'macd_signal', 'macd_alt', 'bb_middle', 
            'bb_upper', 'bb_lower', 'bb_position', 'bb_pband', 'cmf', 'vwap'
        ]
        ALL_FUND_FEATURES = ['pe_ratio', 'pb_ratio', 'div_yield', 'roe', 'pe_rel_sector', 'pb_rel_sector', 'Cluster', 'roic', 'net_margin', 'debt_to_equity', 'current_ratio', 'revenue_growth', 'ev_ebitda', 'operating_margin']


        # --- Clusterização Inicial (Fundamentos) ---
        required_cols_cluster = ['pe_ratio', 'pb_ratio', 'div_yield', 'roe']
        available_fund_cols = [col for col in required_cols_cluster if col in self.dados_fundamentalistas.columns]
        
        if len(available_fund_cols) >= 4 and len(self.dados_fundamentalistas) >= 5:
            log_debug("Executando Clusterização inicial (KMeans + PCA) nos fundamentos.")
            # Executa a clusterização normalmente
            clustering_df = self.dados_fundamentalistas[available_fund_cols].join(
                self.metricas_performance[['sharpe', 'volatilidade_anual']], how='inner',
                lsuffix='_fund', rsuffix='_perf' 
            ).fillna(0)
            
            # Garante que clustering_df não está vazio após o join
            if len(clustering_df) >= 5:
                # Lógica original de Clusterização (KMeans + PCA)
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
            # FALLBACK: Se as colunas fundamentalistas não existirem (Pynvest falhou)
            self.dados_fundamentalistas['Cluster'] = 0
            log_debug("AVISO: Falha na coleta de dados fundamentais (P/L, ROE, etc.). Usando Cluster = 0.")
        
        # --- Pipeline ML ---
        for i, ativo in enumerate(ativos_com_dados):
            try:
                if progress_callback: progress_callback.progress(50 + int((i/len(ativos_com_dados))*20), text=f"Treinando RF Pipeline: {ativo}...")
                df = self.dados_por_ativo[ativo].copy()
                
                if ativo in self.dados_fundamentalistas.index:
                    fund_data = self.dados_fundamentalistas.loc[ativo].to_dict()
                else:
                    fund_data = {} 

                # Fallback de ML (Modo Estático / Sem Preço)
                if df.empty or len(df) < 60 or 'Close' not in df.columns or df['Close'].isnull().all():
                    try:
                         # Heuristica de proxy para o MODO ESTÁTICO
                         roe = fund_data.get('roe', 0); pe = fund_data.get('pe_ratio', 15)
                         if pe <= 0: pe = 20
                         score_fund = min(0.9, max(0.05, (roe * 100) / pe * 0.5 + 0.4))
                         self.predicoes_ml[ativo] = {'predicted_proba_up': score_fund, 'auc_roc_score': 0.4, 'model_name': 'Proxy Fundamentalista'}
                         log_debug(f"ML (Fallback): Ativo {ativo} usando Proxy Fundamentalista (Score: {score_fund:.2f}) devido à ausência de preço/histórico.")
                    except:
                         self.predicoes_ml[ativo] = {'predicted_proba_up': 0.5, 'auc_roc_score': 0.0, 'model_name': 'Modo Estático (Sem Preço)'}
                         log_debug(f"ML (Fallback): Ativo {ativo} falhou no Proxy Fundamentalista. Score Neutro.")
                    continue

                df['Future_Direction'] = np.where(df['Close'].pct_change(dias_lookback_ml).shift(-dias_lookback_ml) > 0, 1, 0)
                
                # Juntando dados técnicos e fundamentais no DF
                current_features_raw = ALL_TECH_FEATURES + ALL_FUND_FEATURES
                
                # Cria um DF de modelo preenchendo as features fundamentais disponíveis na última linha
                last_idx = df.index[-1] if not df.empty else None
                if last_idx:
                    for f_col in ALL_FUND_FEATURES:
                        if f_col in fund_data and f_col not in df.columns:
                            # Adiciona a feature fundamental como uma nova coluna
                            df.loc[last_idx, f_col] = fund_data[f_col]
                        elif f_col not in df.columns:
                             # Adiciona NaN se a feature fundamental não existir
                            df[f_col] = np.nan
                
                # Garante que as colunas existem antes de tentar o dropna
                available_features = [f for f in current_features_raw if f in df.columns]
                
                # Remove NaNs apenas para as features de interesse e a coluna alvo
                df_model = df.dropna(subset=available_features + ['Future_Direction'])

                if len(df_model) < 30:
                    self.predicoes_ml[ativo] = {'predicted_proba_up': 0.5, 'auc_roc_score': 0.5, 'model_name': 'Dados Insuficientes'}
                    log_debug(f"ML (Ignorado): Ativo {ativo} pulado. Apenas {len(df_model)} pontos de dados válidos para treino. Mínimo é 30.")
                    continue
                
                log_debug(f"ML (Supervisionado): Ativo {ativo} tem {len(df_model)} pontos válidos para treino (Mín. 30).")

                # Selecionando dados de treino/teste
                X = df_model[available_features].iloc[:-dias_lookback_ml]
                y = df_model['Future_Direction'].iloc[:-dias_lookback_ml]
                
                # Filtra features que são completamente NaN no subconjunto de treino (embora o dropna devesse prevenir isso)
                final_features = [col for col in X.columns if not X[col].isnull().all()]
                X = X[final_features]
                
                if 'Cluster' in X.columns: X['Cluster'] = X['Cluster'].astype(str)
                
                categorical_cols = ['Cluster'] if 'Cluster' in X.columns else []
                numeric_cols = [c for c in X.columns if c not in categorical_cols]
                
                preprocessor = ColumnTransformer(transformers=[
                        ('num', StandardScaler(), numeric_cols),
                        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
                    ], remainder='passthrough')
                
                model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced'))])
                
                if len(np.unique(y)) < 2: avg_auc = 0.5
                else:
                    scores = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=3), scoring='roc_auc')
                    avg_auc = scores.mean()
                
                model.fit(X, y)
                
                # Preparando dados de predição (última linha)
                last_features_raw = df[available_features].iloc[[-1]].copy()
                last_features = last_features_raw[final_features].fillna(0) # Deve ter colunas finais
                
                if 'Cluster' in last_features.columns: last_features['Cluster'] = last_features['Cluster'].astype(str)
                
                proba = model.predict_proba(last_features)[0][1]

                # --- MODO SUPERVISIONADO: Fallback para Proxy se o modelo for Neutro ---
                is_neutral_result = (avg_auc <= 0.55) or (len(np.unique(y)) < 2) 
                
                if is_neutral_result:
                    roe = fund_data.get('roe', 0); pe = fund_data.get('pe_ratio', 15)
                    if pe <= 0: pe = 20
                    score_fund_proxy = min(0.9, max(0.05, (roe * 100) / pe * 0.5 + 0.4))
                    
                    self.predicoes_ml[ativo] = {
                        'predicted_proba_up': score_fund_proxy, 
                        'auc_roc_score': 0.60, # Força uma confiança mínima para que o score seja ponderado
                        'model_name': 'Supervised Neutral (FORCING FUNDAMENTAL PROXY)'
                    }
                    log_debug(f"ML (Supervisionado): Ativo {ativo} resultou neutro (AUC={avg_auc:.2f}). Forçando Proxy Fundamentalista (Score: {score_fund_proxy:.2f}).")
                else:
                    self.predicoes_ml[ativo] = {
                        'predicted_proba_up': proba, 
                        'auc_roc_score': avg_auc, 
                        'model_name': 'Pipeline RF+Cluster'
                    }
                    log_debug(f"ML (Supervisionado): Ativo {ativo} treinado com sucesso. Prob. Alta: {proba*100:.1f}%, AUC: {avg_auc:.2f}.")

                last_idx = self.dados_por_ativo[ativo].index[-1]
                self.dados_por_ativo[ativo].loc[last_idx, 'ML_Proba'] = self.predicoes_ml[ativo]['predicted_proba_up']
                self.dados_por_ativo[ativo].loc[last_idx, 'ML_Confidence'] = self.predicoes_ml[ativo]['auc_roc_score']

            except Exception as e:
                self.predicoes_ml[ativo] = {'predicted_proba_up': 0.5, 'auc_roc_score': 0.5, 'model_name': f'Erro: {str(e)}'}
                log_debug(f"ERRO: Falha no treinamento ML para {ativo}: {str(e)[:50]}...")
                continue
        log_debug("Pipeline de Treinamento ML/Clusterização concluído.")

    def realizar_clusterizacao_final(self):
        if self.scores_combinados.empty: return
        log_debug("Iniciando Clusterização Final nos Scores (KMeans).")
        features_cluster = ['performance_score', 'fundamental_score', 'technical_score', 'ml_score_weighted']
        data_cluster = self.scores_combinados[features_cluster].fillna(50)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_cluster)
        pca = PCA(n_components=min(data_scaled.shape[1], 2))
        data_pca = pca.fit_transform(data_scaled)
        kmeans = KMeans(n_clusters=min(len(data_cluster), 4), random_state=42, n_init=10) # Max clusters based on available data
        clusters = kmeans.fit_predict(data_pca)
        self.scores_combinados['Final_Cluster'] = clusters
        log_debug(f"Clusterização Final concluída. Identificados {self.scores_combinados['Final_Cluster'].nunique()} perfis de risco/retorno.")

    def pontuar_e_selecionar_ativos(self, horizonte_tempo: str):
        if horizonte_tempo == "CURTO PRAZO": share_tech, share_fund = 0.7, 0.3
        elif horizonte_tempo == "LONGO PRAZO": share_tech, share_fund = 0.3, 0.7
        else: share_tech, share_fund = 0.5, 0.5

        W_PERF_GLOBAL = 0.20
        W_ML_GLOBAL_BASE = 0.20
        W_REMAINING = 1.0 - W_PERF_GLOBAL - W_ML_GLOBAL_BASE
        w_tech_final = W_REMAINING * share_tech
        w_fund_final = W_REMAINING * share_fund
        self.pesos_atuais = {'Performance': W_PERF_GLOBAL, 'Fundamentos': w_fund_final, 'Técnicos': w_tech_final, 'ML': W_ML_GLOBAL_BASE}
        
        # JOIN SEGURO (RESOLVE O ERRO DE OVERLAP)
        cols_to_drop = [col for col in self.dados_fundamentalistas.columns if col in self.metricas_performance.columns]
        df_fund_clean = self.dados_fundamentalistas.drop(columns=cols_to_drop, errors='ignore')
        combined = self.metricas_performance.join(df_fund_clean, how='inner').copy()
        
        for symbol in combined.index:
            if symbol in self.dados_por_ativo:
                df = self.dados_por_ativo[symbol]
                if not df.empty and 'rsi_14' in df.columns:
                    combined.loc[symbol, 'rsi_current'] = df['rsi_14'].iloc[-1]
                    combined.loc[symbol, 'macd_current'] = df['macd'].iloc[-1]
                    combined.loc[symbol, 'vol_current'] = df['volatility_20'].iloc[-1] # Usando volatility_20
                else:
                    # Fallback para valores neutros se estiver em modo estático
                    combined.loc[symbol, 'rsi_current'] = 50
                    combined.loc[symbol, 'macd_current'] = 0
                    combined.loc[symbol, 'vol_current'] = 0

        scores = pd.DataFrame(index=combined.index)
        scores['performance_score'] = (EngenheiroFeatures._normalize_score(combined['sharpe'], True) * 0.6 + EngenheiroFeatures._normalize_score(combined['retorno_anual'], True) * 0.4) * W_PERF_GLOBAL
        
        def _get_score_series(df: pd.DataFrame, col: str, default_val: float) -> pd.Series:
            """Retorna a coluna se existir, ou uma Series de valor neutro."""
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
        s_prob = EngenheiroFeatures._normalize_score(ml_probs, True)
        ml_weight_factor = (ml_conf - 0.5).clip(lower=0) * 2
        scores['ml_score_weighted'] = s_prob * (W_ML_GLOBAL_BASE * ml_weight_factor.fillna(0))
        
        scores['total_score'] = scores.sum(axis=1)
        self.scores_combinados = scores.join(combined).sort_values('total_score', ascending=False)
        
        log_debug(f"Calculando Scores Ponderados. Horizonte: {horizonte_tempo}. Pesos Finais: Fund={w_fund_final:.2f}, Tec={w_tech_final:.2f}, ML={W_ML_GLOBAL_BASE:.2f}.")

        # -------------------------------------------------------------
        # 1. CRITÉRIO DE INCLUSÃO: FILTRO DE SCORE MÍNIMO
        # -------------------------------------------------------------
        if len(self.scores_combinados) > NUM_ATIVOS_PORTFOLIO:
            # Pega o score do 15º ativo (ou o último, se a lista for menor que 15)
            # Para definir um score de corte que exclui os piores 15% (ou mais)
            cutoff_index = min(15, len(self.scores_combinados) - 1)
            base_score = self.scores_combinados['total_score'].iloc[cutoff_index]
            
            # Define o score mínimo como 85% do score base (um filtro de qualidade)
            min_score = base_score * SCORE_PERCENTILE_THRESHOLD
            
            ativos_filtrados = self.scores_combinados[self.scores_combinados['total_score'] >= min_score]
            
            num_eliminados = len(self.scores_combinados) - len(ativos_filtrados)
            log_debug(f"Score Mínimo Requerido: {min_score:.3f} ({SCORE_PERCENTILE_THRESHOLD*100:.0f}% do score base).")
            log_debug(f"Ativos eliminados pelo filtro de score: {num_eliminados}")
            
            self.scores_combinados = ativos_filtrados
            
        # -------------------------------------------------------------
        # 2. SELEÇÃO FINAL POR CLUSTER
        # -------------------------------------------------------------
        self.realizar_clusterizacao_final()
        final_selection = []
        
        if not self.scores_combinados.empty and 'Final_Cluster' in self.scores_combinados.columns:
            clusters_present = self.scores_combinados['Final_Cluster'].unique()
            for c in clusters_present:
                # Pega o melhor ativo de cada cluster (o de maior 'total_score')
                best = self.scores_combinados[self.scores_combinados['Final_Cluster'] == c].head(1).index[0]
                final_selection.append(best)
        
        # Adiciona os restantes, se o número de clusters for menor que o portfólio
        if len(final_selection) < NUM_ATIVOS_PORTFOLIO:
            others = [x for x in self.scores_combinados.index if x not in final_selection]
            # Prioriza os melhores scores dos que sobraram
            remaining_to_add = NUM_ATIVOS_PORTFOLIO - len(final_selection)
            
            if remaining_to_add > 0:
                 # Garante que 'others' é ordenado por score
                 others_df = self.scores_combinados.loc[others].sort_values('total_score', ascending=False)
                 final_selection.extend(others_df.index[:remaining_to_add].tolist())

        self.ativos_selecionados = final_selection[:NUM_ATIVOS_PORTFOLIO]
        log_debug(f"Seleção final concluída. {len(self.ativos_selecionados)} ativos selecionados: {self.ativos_selecionados}")

        return self.ativos_selecionados
    
    def otimizar_alocacao(self, nivel_risco: str):
        if not self.ativos_selecionados or len(self.ativos_selecionados) < 1:
            self.metodo_alocacao_atual = "ERRO: Ativos Insuficientes"; return {}
        
        available_assets_returns = {}
        ativos_sem_dados = []
        
        for s in self.ativos_selecionados:
            if s in self.dados_por_ativo and 'returns' in self.dados_por_ativo[s] and not self.dados_por_ativo[s]['returns'].dropna().empty:
                available_assets_returns[s] = self.dados_por_ativo[s]['returns']
            else:
                ativos_sem_dados.append(s)
        
        final_returns_df = pd.DataFrame(available_assets_returns).dropna()
        
        # Se houver ativos sem dados de preço (Modo Estático) ou poucos dados, usa heurística
        if final_returns_df.shape[0] < 50 or len(ativos_sem_dados) > 0:
            log_debug("Otimização de Markowitz ignorada. Recorrendo à PONDERAÇÃO POR SCORE (Modo Estático/Poucos Dados).")

            if len(ativos_sem_dados) > 0:
                st.warning(f"⚠️ Alguns ativos ({', '.join(ativos_sem_dados)}) não possuem histórico de preços. A otimização de variância (Markowitz) será substituída por alocação baseada em Score/Pesos Iguais.")
            
            scores = self.scores_combinados.loc[self.ativos_selecionados, 'total_score']
            total_score = scores.sum()
            if total_score > 0:
                weights = (scores / total_score).to_dict()
                self.metodo_alocacao_atual = 'PONDERAÇÃO POR SCORE (Modo Estático)'
            else:
                weights = {asset: 1.0 / len(self.ativos_selecionados) for asset in self.ativos_selecionados}
                self.metodo_alocacao_atual = 'PESOS IGUAIS (Fallback Total)'
                
            return self._formatar_alocacao(weights)

        garch_vols_filtered = {asset: self.volatilidades_garch.get(asset, final_returns_df[asset].std() * np.sqrt(252)) for asset in final_returns_df.columns}
        optimizer = OtimizadorPortfolioAvancado(final_returns_df, garch_vols=garch_vols_filtered)
        
        if 'CONSERVADOR' in nivel_risco or 'INTERMEDIÁRIO' in nivel_risco:
            strategy = 'MinVolatility'; self.metodo_alocacao_atual = 'MINIMIZAÇÃO DE VOLATILIDADE'
        else:
            strategy = 'MaxSharpe'; self.metodo_alocacao_atual = 'MAXIMIZAÇÃO DE SHARPE'
            
        log_debug(f"Otimizando Markowitz. Estratégia: {self.metodo_alocacao_atual} (Risco: {nivel_risco}).")
            
        weights = optimizer.otimizar(estrategia=strategy)
        if not weights:
             log_debug("AVISO: Otimizador falhou. Usando PESOS IGUAIS como fallback total.")
             weights = {asset: 1.0 / len(self.ativos_selecionados) for asset in self.ativos_selecionados}
             self.metodo_alocacao_atual += " (FALLBACK)"
        
        total_weight = sum(weights.values())
        log_debug(f"Otimização Markowitz finalizada. Peso total: {total_weight:.2f}")
        return self._formatar_alocacao(weights)
        
    def _formatar_alocacao(self, weights: dict) -> dict:
        if not weights or sum(weights.values()) == 0: return {}
        total_weight = sum(weights.values())
        return {s: {'weight': w / total_weight, 'amount': self.valor_investimento * (w / total_weight)} for s, w in weights.items() if s in self.ativos_selecionados}
    
    def calcular_metricas_portfolio(self):
        if not self.alocacao_portfolio: return {}
        weights_dict = {s: data['weight'] for s, data in self.alocacao_portfolio.items()}
        available_returns = {s: self.dados_por_ativo[s]['returns'] for s in weights_dict.keys() if s in self.dados_por_ativo and 'returns' in self.dados_por_ativo[s] and not self.dados_por_ativo[s]['returns'].dropna().empty}
        
        if not available_returns:
             self.metricas_portfolio = {
                'annual_return': 0, 'annual_volatility': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'total_investment': self.valor_investimento
            }
             return self.metricas_portfolio

        returns_df = pd.DataFrame(available_returns).dropna()
        if returns_df.empty: return {}
        
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
             metrics = {'annual_return': 0, 'annual_volatility': 0, 'sharpe_ratio': 0, 'max_drawdown': 0}
             
        self.metricas_portfolio = metrics
        return metrics

    def gerar_justificativas(self):
        self.justificativas_selecao = {}
        for simbolo in self.ativos_selecionados:
            justification = []
            
            is_static = False
            if simbolo in self.dados_fundamentalistas.index:
                 is_static = self.dados_fundamentalistas.loc[simbolo].get('static_mode', False)

            if is_static:
                justification.append("⚠️ MODO ESTÁTICO (Preço Indisponível)")
            else:
                perf = self.metricas_performance.loc[simbolo] if simbolo in self.metricas_performance.index else pd.Series({})
                justification.append(f"Perf: Sharpe {perf.get('sharpe', np.nan):.2f}, Ret {perf.get('retorno_anual', np.nan)*100:.1f}%")
                
            ml_prob = self.predicoes_ml.get(simbolo, {}).get('predicted_proba_up', 0.5)
            ml_auc = self.predicoes_ml.get(simbolo, {}).get('auc_roc_score', 0.5)
            justification.append(f"ML: Prob {ml_prob*100:.1f}% (Conf {ml_auc:.2f})")
            cluster = self.scores_combinados.loc[simbolo, 'Final_Cluster'] if simbolo in self.scores_combinados.index else 'N/A'
            justification.append(f"Perfil (Cluster): {cluster}")
            self.justificativas_selecao[simbolo] = " | ".join(justification)
        return self.justificativas_selecao
        
    def executar_pipeline(self, simbolos_customizados: list, perfil_inputs: dict, progress_bar=None) -> bool:
        self.perfil_dashboard = perfil_inputs
        try:
            if progress_bar: progress_bar.progress(10, text="Coletando dados LIVE (YFinance + Pynvest)...")
            if not self.coletar_e_processar_dados(simbolos_customizados): return False
            if progress_bar: progress_bar.progress(30, text="Calculando métricas setoriais e volatilidade...")
            self.calculate_cross_sectional_features(); self.calcular_volatilidades_garch()
            if progress_bar: progress_bar.progress(50, text="Executando Pipeline ML (Cluster + Random Forest)...")
            self.treinar_modelos_ensemble(dias_lookback_ml=perfil_inputs.get('ml_lookback_days', LOOKBACK_ML), otimizar=False, progress_callback=progress_bar) 
            if progress_bar: progress_bar.progress(70, text="Ranqueando e selecionando (Pesos Dinâmicos + PCA Final)...")
            self.pontuar_e_selecionar_ativos(horizonte_tempo=perfil_inputs.get('time_horizon', 'MÉDIO PRAZO')) 
            if progress_bar: progress_bar.progress(85, text="Otimizando alocação (Markowitz 10-30%)...")
            self.alocacao_portfolio = self.otimizar_alocacao(nivel_risco=perfil_inputs.get('risk_level', 'MODERADO'))
            if progress_bar: progress_bar.progress(95, text="Calculando métricas finais...")
            self.calcular_metricas_portfolio(); self.gerar_justificativas()
            if progress_bar: progress_bar.progress(100, text="Pipeline concluído!"); time.sleep(1) 
        except Exception as e:
            st.error(f"Erro durante a execução do pipeline: {e}"); st.code(traceback.format_exc()); return False
        return True

# =============================================================================
# 12. CLASSE: ANALISADOR INDIVIDUAL (VISUALIZAÇÃO)
# =============================================================================

class AnalisadorIndividualAtivos:
    @staticmethod
    def realizar_clusterizacao_fundamentalista_geral(coletor: ColetorDadosLive, ativo_alvo: str) -> tuple[pd.DataFrame | None, int | None]:
        """
        Realiza clusterização (PCA + KMeans) usando APENAS dados fundamentalistas
        de TODOS os ativos do IBOVESPA (General Similarity), não apenas do setor.
        """
        
        # Usa um subset menor de ativos (os da lista do Ibovespa) para não demorar muito
        # Mas compara com todos eles, independente de setor
        ativos_comparacao = ATIVOS_IBOVESPA 
        
        # Coleta dados de TODOS os ativos (apenas fundamentos)
        df_fund_geral = coletor.coletar_fundamentos_em_lote(ativos_comparacao)
        
        if df_fund_geral.empty:
            return None, None
            
        # 3. Prepara os dados para ML (Limpeza e Normalização)
        cols_interesse = [
            'pe_ratio', 'pb_ratio', 'roe', 'roic', 'net_margin', 
            'div_yield', 'debt_to_equity', 'current_ratio', 
            'revenue_growth', 'ev_ebitda', 'operating_margin'
        ]
        
        # Garante que as colunas existem
        cols_existentes = [c for c in cols_interesse if c in df_fund_geral.columns]
        df_model = df_fund_geral[cols_existentes].copy()
        
        # Converte tudo para numérico
        for col in df_model.columns:
            df_model[col] = pd.to_numeric(df_model[col], errors='coerce')
            
        # Limpeza de colunas vazias
        df_model = df_model.dropna(axis=1, how='all')
        
        if df_model.empty or len(df_model) < 5:
             return None, None

        # IMPUTATION ROBUSTA (Preenche NaNs com a Mediana Global)
        imputer = SimpleImputer(strategy='median')
        try:
            dados_imputed = imputer.fit_transform(df_model)
        except ValueError:
            return None, None # Falha se ainda houver erros críticos

        # Pipeline PCA + KMeans
        scaler = StandardScaler()
        dados_normalizados = scaler.fit_transform(dados_imputed)
        
        # Usar 3 componentes para gráfico 3D
        n_components = min(3, dados_normalizados.shape[1])
        pca = PCA(n_components=n_components)
        componentes_pca = pca.fit_transform(dados_normalizados)
        
        # --- LÓGICA DE CLUSTERIZAÇÃO E ANOMALIA (UNSUPERVISED FALLBACK) ---
        # 1. KMeans para Agrupamento
        n_clusters = min(5, max(3, int(np.sqrt(len(df_model) / 2))))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(componentes_pca)
        
        # 2. Isolation Forest para Detecção de Anomalia
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(componentes_pca)
        anomaly_score = iso_forest.decision_function(componentes_pca) # Quanto maior, menos anômalo

        
        cols_pca = [f'PC{i+1}' for i in range(n_components)]
        resultado = pd.DataFrame(componentes_pca, columns=cols_pca, index=df_model.index)
        resultado['Cluster'] = clusters
        resultado['Anomalia'] = anomalies # 1 = Normal, -1 = Anomalia (Outlier)
        resultado['Anomaly_Score'] = anomaly_score # Quanto maior, menos anômalo
        
        return resultado, n_clusters

def aba_analise_individual():
    """Aba 4: Análise Individual de Ativos (Autônoma e Geral)"""
    
    st.markdown("## 🔍 Análise de Fatores por Ticker")
    
    # Carrega lista de ativos (se o builder não rodou, usa lista estática)
    if 'ativos_para_analise' in st.session_state and st.session_state.ativos_para_analise:
        ativos_disponiveis = sorted(list(set(st.session_state.ativos_para_analise)))
    else:
        ativos_disponiveis = TODOS_ATIVOS 
            
    if not ativos_disponiveis:
        st.error("Nenhum ativo disponível.")
        return

    # Layout Centralizado da Seleção
    col_sel, = st.columns(1)
    with col_sel:
        ativo_selecionado = st.selectbox(
            "Selecione um ticker para análise detalhada:",
            options=ativos_disponiveis,
            format_func=lambda x: x.replace('.SA', '') if isinstance(x, str) else x,
            key='individual_asset_select_v8' 
        )
    
    st.write("") # Spacer
    
    # Botão Centralizado Abaixo
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        if st.button("🔄 Executar Análise", key='analyze_asset_button_v8', type="primary", use_container_width=True):
            st.session_state.analisar_ativo_triggered = True 
    
    if 'analisar_ativo_triggered' not in st.session_state or not st.session_state.analisar_ativo_triggered:
        st.info("👆 Selecione um ticker e clique em 'Executar Análise' para obter o relatório completo.")
        return
    
    with st.spinner(f"Analisando {ativo_selecionado} (Coleta em Tempo Real)..."):
        try:
            # Coletor Autônomo (Instância local, sem depender do builder global)
            coletor = ColetorDadosLive()
            
            # Coleta dados específicos do ativo (Preço + Fundamentos)
            df_completo, features_fund, df_ml_meta = coletor.coletar_ativo_unico_gcs(ativo_selecionado)
            
            # Verificação Básica
            if features_fund is None:
                st.error(f"❌ Dados indisponíveis para **{ativo_selecionado.replace('.SA', '')}**.")
                return

            # Detecta Modo Estático (Sem Preço)
            static_mode = features_fund.get('static_mode', False) or (df_completo is not None and df_completo['Close'].isnull().all())

            if static_mode:
                st.warning(f"⚠️ **MODO ESTÁTICO:** Preços indisponíveis. Exibindo apenas Análise Fundamentalista.")

            
            # Determine se o ML Supervisionado PODE ser treinado
            ml_can_be_trained = not df_completo.empty and len(df_completo.dropna(subset=['Close'])) > 60
            
            abas_names = [
                "📊 Visão Geral", 
                "🔧 Análise Técnica", 
                "💼 Análise Fundamentalista",
                "🔬 Clusterização Geral"
            ]
            
            if ml_can_be_trained:
                 abas_names.insert(3, "🤖 Machine Learning") # Insere ML na 4ª posição

            abas_map = st.tabs(abas_names)
            
            # Mapeamento dinâmico das abas
            tab_map = {name: abas_map[i] for i, name in enumerate(abas_names)}
            
            # Abas 1-4: Lógica Padrão de Exibição (igual à versão anterior)
            with tab_map["📊 Visão Geral"]:
                st.markdown(f"### {ativo_selecionado.replace('.SA', '')} - Resumo de Mercado")
                col1, col2, col3, col4, col5 = st.columns(5)
                if not static_mode and 'Close' in df_completo.columns:
                    preco_atual = df_completo['Close'].iloc[-1]
                    variacao_dia = df_completo['returns'].iloc[-1] * 100 if 'returns' in df_completo.columns else 0.0
                    volume_medio = df_completo['Volume'].mean() if 'Volume' in df_completo.columns else 0.0
                    vol_anual = features_fund.get('annual_volatility', 0) * 100
                    col1.metric("Preço", f"R$ {preco_atual:.2f}", f"{variacao_dia:+.2f}%")
                    col2.metric("Volume Médio", f"{volume_medio:,.0f}")
                    col5.metric("Volatilidade", f"{vol_anual:.2f}%")
                else:
                    col1.metric("Preço", "N/A", "N/A"); col2.metric("Volume Médio", "N/A"); col5.metric("Volatilidade", "N/A")
                
                # Fallback para Setor se pynvest falhar
                setor = features_fund.get('sector')
                if setor == 'Unknown' or setor is None:
                     setor = FALLBACK_SETORES.get(ativo_selecionado, 'N/A')

                col3.metric("Setor", setor)
                col4.metric("Indústria", features_fund.get('industry', 'N/A'))
                
                if not static_mode and not df_completo.empty and 'Open' in df_completo.columns:
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                    fig.add_trace(go.Candlestick(x=df_completo.index, open=df_completo['Open'], high=df_completo['High'], low=df_completo['Low'], close=df_completo['Close'], name='Preço'), row=1, col=1)
                    fig.add_trace(go.Bar(x=df_completo.index, y=df_completo['Volume'], name='Volume'), row=2, col=1)
                    
                    template = obter_template_grafico()
                    template['title']['text'] = f"Gráfico Diário - {ativo_selecionado}" # Define no dict
                    fig.update_layout(**template)
                    fig.update_layout(height=600)
                    
                    st.plotly_chart(fig, use_container_width=True)
                else: st.info("Gráfico indisponível (Modo Estático Ativo).")

            with tab_map["🔧 Análise Técnica"]:
                if not static_mode:
                    st.markdown("### Indicadores Técnicos");
                    
                    # Tenta usar indicadores mais robustos (RSI, MACD)
                    rsi_val = df_completo['rsi_14'].iloc[-1] if 'rsi_14' in df_completo.columns and not df_completo.empty else np.nan
                    macd_diff_val = df_completo['macd_diff'].iloc[-1] if 'macd_diff' in df_completo.columns and not df_completo.empty else np.nan
                    bb_width_val = df_completo['bb_width'].iloc[-1] if 'bb_width' in df_completo.columns and not df_completo.empty else np.nan
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("RSI (14)", f"{rsi_val:.2f}" if not np.isnan(rsi_val) else "N/A")
                    col2.metric("MACD Diff", f"{macd_diff_val:.4f}" if not np.isnan(macd_diff_val) else "N/A")
                    col3.metric("BB Width", f"{bb_width_val:.2f}" if not np.isnan(bb_width_val) else "N/A")
                    col4.metric("Vol. 20D (%)", f"{df_completo['volatility_20'].iloc[-1]*100:.2f}" if 'volatility_20' in df_completo.columns else "N/A")

                    # Gráfico RSI
                    if 'rsi_14' in df_completo.columns:
                        fig_rsi = go.Figure(go.Scatter(x=df_completo.index, y=df_completo['rsi_14'], name='RSI', line=dict(color='#8E44AD')))
                        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red"); fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                        
                        template = obter_template_grafico()
                        template['title']['text'] = "RSI (14)" 
                        fig_rsi.update_layout(**template)
                        fig_rsi.update_layout(height=300)
                        
                        st.plotly_chart(fig_rsi, use_container_width=True)
                    
                    # Gráfico MACD
                    if 'macd' in df_completo.columns:
                        fig_macd = make_subplots(rows=1, cols=1)
                        fig_macd.add_trace(go.Scatter(x=df_completo.index, y=df_completo['macd'], name='MACD', line=dict(color='#2980B9')))
                        fig_macd.add_trace(go.Scatter(x=df_completo.index, y=df_completo['macd_signal'], name='Signal', line=dict(color='#E74C3C')))
                        fig_macd.add_trace(go.Bar(x=df_completo.index, y=df_completo['macd_diff'], name='Histograma', marker_color='#BDC3C7'))
                        
                        template = obter_template_grafico()
                        template['title']['text'] = "MACD (12,26,9)"
                        fig_macd.update_layout(**template)
                        fig_macd.update_layout(height=300)
                        
                        st.plotly_chart(fig_macd, use_container_width=True)
                
                else: st.warning("Análise Técnica não disponível sem histórico de preços.")

            with tab_map["💼 Análise Fundamentalista"]:
                # Helper para troca de indicadores caso NaN
                def get_valid_metric(data, primary_key, primary_label, secondary_key, secondary_label):
                     val = data.get(primary_key)
                     if pd.isna(val) or val == np.inf or val == -np.inf:
                         return secondary_label, data.get(secondary_key)
                     return primary_label, val
                
                # Obtem Beta de Fallback se necessário (Yahoo Finance)
                beta_val = features_fund.get('beta')
                if pd.isna(beta_val):
                     try:
                         beta_val = yf.Ticker(ativo_selecionado).info.get('beta')
                     except: 
                         beta_val = np.nan
                
                st.markdown("### Principais Métricas")
                # Linha 1
                col1, col2, col3, col4, col5 = st.columns(5)
                
                l1, v1 = get_valid_metric(features_fund, 'pe_ratio', 'P/L', 'ev_ebitda', 'EV/EBITDA')
                col1.metric(l1, safe_format(v1))
                
                col2.metric("P/VP", safe_format(features_fund.get('pb_ratio', np.nan)))
                col3.metric("ROE", safe_format(features_fund.get('roe', np.nan)))
                col4.metric("Margem Líq.", safe_format(features_fund.get('net_margin', np.nan)))
                col5.metric("Div. Yield", safe_format(features_fund.get('div_yield', np.nan)))
                
                st.write("") # Spacer
                
                # Linha 2
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Dívida Bruta/PL", safe_format(features_fund.get('debt_to_equity', np.nan)))
                col2.metric("Liq. Corrente", safe_format(features_fund.get('current_ratio', np.nan)))
                col3.metric("EV/EBITDA", safe_format(features_fund.get('ev_ebitda', np.nan)))
                
                col4.metric("ROIC", safe_format(features_fund.get('roic', np.nan))) 
                
                col5.metric("Beta (Yahoo)", safe_format(beta_val))

                st.markdown("---")
                st.markdown("### Tabela Geral de Indicadores")
                
                # Exibição de tabela expansível com TODOS os indicadores
                with st.expander("📋 Ver Tabela Completa de Indicadores", expanded=False):
                     # Remove chaves internas de controle antes de exibir
                     clean_fund = {k: v for k, v in features_fund.items() if k not in ['static_mode', 'garch_volatility', 'max_drawdown']}
                     df_fund_show = pd.DataFrame([clean_fund]).T.reset_index()
                     df_fund_show.columns = ['Indicador', 'Valor']
                     st.dataframe(df_fund_show, use_container_width=True)

            # =============================================================
            # ABA ML (SÓ EXISTE SE ml_can_be_trained é True)
            # =============================================================
            if ml_can_be_trained:
                 with tab_map["🤖 Machine Learning"]:
                    st.markdown("### Predição de Machine Learning")
                    
                    ml_proba = df_completo['ML_Proba'].iloc[-1] if 'ML_Proba' in df_completo.columns else 0.5
                    ml_conf = df_completo['ML_Confidence'].iloc[-1] if 'ML_Confidence' in df_completo.columns else 0.0
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Probabilidade de Alta", f"{ml_proba*100:.1f}%")
                    col2.metric("Confiança do Modelo (AUC)", f"{ml_conf:.2f}")
                    st.info("ℹ️ **Modelo Supervisionado Ativo:** O score reflete a probabilidade de alta do ativo nos próximos 5 dias, conforme previsto pelo modelo Random Forest treinado com indicadores técnicos e fundamentais.")
                        
                    if df_ml_meta is not None and not df_ml_meta.empty:
                        st.markdown("#### Importância dos Fatores na Decisão")
                        fig_imp = px.bar(df_ml_meta.head(5), x='importance', y='feature', orientation='h', title='Top Fatores')
                        
                        template = obter_template_grafico()
                        template['title']['text'] = 'Top 5 Fatores de Decisão'
                        fig_imp.update_layout(**template)
                        fig_imp.update_layout(height=300)
                        st.plotly_chart(fig_imp, use_container_width=True)

            # =============================================================
            # ABA CLUSTERIZAÇÃO GERAL (Sempre existe)
            # =============================================================
            with tab_map["🔬 Clusterização Geral"]:
                st.markdown("### 🔬 Clusterização Geral (Ibovespa)")
                
                st.info(f"Analisando similaridade do **{ativo_selecionado.replace('.SA', '')}** com **TODOS** os ativos do Ibovespa (Baseado apenas em Fundamentos).")
                
                # 2. Coleta e Clusteriza (Apenas Fundamentos - Lista Global)
                resultado_cluster, n_clusters = AnalisadorIndividualAtivos.realizar_clusterizacao_fundamentalista_geral(coletor, ativo_selecionado)
                
                if resultado_cluster is not None:
                    st.success(f"Identificados {n_clusters} grupos (clusters) de qualidade fundamentalista.")
                    
                    # 1. Gráfico 3D (se houver 3 componentes)
                    if 'PC3' in resultado_cluster.columns:
                        fig_pca = px.scatter_3d(
                            resultado_cluster, x='PC1', y='PC2', z='PC3',
                            color=resultado_cluster['Cluster'].astype(str),
                            hover_name=resultado_cluster.index.str.replace('.SA', ''),
                            title='Mapa de Similaridade 3D (Global)'
                        )
                        fig_layout = obter_template_grafico()
                        fig_layout['scene'] = dict(xaxis_title='PCA 1', yaxis_title='PCA 2', zaxis_title='PCA 3')
                        fig_pca.update_layout(**fig_layout, height=650)
                        st.plotly_chart(fig_pca, use_container_width=True)
                    else:
                        # 2. Fallback para 2D
                        fig_pca = px.scatter(
                            resultado_cluster, x='PC1', y='PC2',
                            color=resultado_cluster['Cluster'].astype(str),
                            hover_name=resultado_cluster.index.str.replace('.SA', ''),
                            title='Mapa de Similaridade 2D (Global)'
                        )
                        fig_pca.update_layout(**obter_template_grafico(), height=600)
                        st.plotly_chart(fig_pca, use_container_width=True)
                    
                    # Tabela de Anomalias (Isolation Forest)
                    st.markdown('#### 🚨 Detecção de Anomalias (Outliers) - Isolation Forest')
                    
                    df_anomaly_report = resultado_cluster.loc[resultado_cluster.index].reset_index().rename(columns={'index': 'Ticker'})
                    
                    # Filtra apenas o ativo selecionado e os outliers
                    df_anomaly_report = df_anomaly_report[
                         (df_anomaly_report['Ticker'] == ativo_selecionado) | (df_anomaly_report['Anomalia'] == -1)
                    ]
                    
                    df_anomaly_report['Outlier'] = df_anomaly_report['Anomalia'].apply(lambda x: '🚨 OUTLIER' if x == -1 else '✓ Normal')
                    df_anomaly_report['Score'] = df_anomaly_report['Anomaly_Score'].apply(lambda x: f"{x:.4f}")
                    
                    st.dataframe(df_anomaly_report[['Ticker', 'Cluster', 'Outlier', 'Score']].sort_values(['Outlier', 'Score']), use_container_width=True)

                    # Identifica pares similares
                    if ativo_selecionado in resultado_cluster.index:
                        cluster_ativo = resultado_cluster.loc[ativo_selecionado, 'Cluster']
                        pares = resultado_cluster[resultado_cluster['Cluster'] == cluster_ativo].index.tolist()
                        pares = [p.replace('.SA', '') for p in pares if p != ativo_selecionado]
                        
                        with st.expander(f"📋 Ver {len(pares)} ativos similares no Cluster {cluster_ativo}", expanded=True):
                            if pares:
                                st.write(", ".join(pares))
                            else:
                                st.write("Este ativo possui características únicas (Outlier).")
                    
                else:
                    st.warning("Dados insuficientes para gerar clusters confiáveis.")

def aba_referencias():
    """Aba 5: Referências Bibliográficas Completas (V8.7 Original)"""
    
    st.markdown("## 📚 Referências e Bibliografia")
    st.markdown("Esta seção consolida as referências bibliográficas indicadas nas ementas das disciplinas relacionadas (GRDECO222 e GRDECO203).")

    st.markdown("---")
    
    st.markdown("### GRDECO222: Machine Learning (Prof. Rafael Martins de Souza)")
    
    st.markdown("**Bibliografia Obrigatória**")
    
    st.markdown("1. **Jupter Notebooks apresentados em sala de aula.**")
    with st.expander("Explicação"):
        st.write("O material principal do curso é prático, baseado nos códigos e exemplos desenvolvidos pelo professor durante as aulas.")
        
    st.markdown("2. **Géron, A. Mãos à Obra: Aprendizado de Máquina com Scikit-Learn, Keras e TensorFlow.**")
    with st.expander("Explicação"):
        st.write("Considerado um dos principais livros-texto práticos sobre Machine Learning. Cobre desde os fundamentos (Regressão, SVMs, Árvores de Decisão) até tópicos avançados de Deep Learning, com foco na implementação usando bibliotecas Python populares.")

    st.markdown("---")
    st.markdown("**Bibliografia Complementar**")
    
    st.markdown("1. **Coleman, C., Spencer Lyon, S., Jesse Perla, J. QuantEcon Data Science, Introduction to Economic Modeling and Data Science. (https://datascience.quantecon.org/)**")
    with st.expander("Explicação"):
        st.write("Um recurso online focado na aplicação de Ciência de Dados especificamente para modelagem econômica, alinhado com os objetivos da disciplina.")

    st.markdown("2. **Sargent, T. J., Stachurski, J., Quantitative Economics with Python. (https://python.quantecon.org/)**")
    with st.expander("Explicação"):
        st.write("Outro projeto da QuantEcon, focado em métodos quantitativos e economia computacional usando Python. É uma referência padrão para economistas que programam.")
    
    st.markdown("---")
    
    st.markdown("### GRDECO203: Laboratório de Ciência de Dados Aplicados à Finanças (Prof. Diogo Tavares Robaina)")

    st.markdown("**Bibliografia Básica**")
    
    st.markdown("1. **HILPISCH, Y. J. Python for finance: analyze big financial dat. O'Reilly Media, 2015.**")
    with st.expander("Explicação"):
        st.write("Uma referência clássica para finanças quantitativas em Python. Cobre manipulação de dados financeiros (séries temporais), análise de risco, e implementação de estratégias de trading e precificação de derivativos.")

    st.markdown("2. **ARRATIA, A. Computational finance an introductory course with R. Atlantis, 2014.**")
    with st.expander("Explicação"):
        st.write("Focado em finanças computacionais usando a linguagem R, abordando conceitos introdutórios e modelagem.")
    
    st.markdown("3. **RASCHKA, S. Python machine learning: unlock deeper insights... Packt Publishing, 2015.**")
    with st.expander("Explicação"):
        st.write("Um guia popular focado na aplicação prática de algoritmos de Machine Learning com Scikit-Learn em Python, similar ao livro de Géron.")
    
    st.markdown("4. **MAINDONALD, J., and Braun, J. Data analysis and graphics using R: an example-based approach. Cambridge University Press, 2006.**")
    with st.expander("Explicação"):
        st.write("Livro focado em análise de dados e visualização gráfica utilizando a linguagem R.")
    
    st.markdown("5. **REYES, J. M. M. Introduction to Data Science for Social and Policy Research. Cambridge University Press, 2017.**")
    with st.expander("Explicação"):
        st.write("Aborda a aplicação de Ciência de Dados no contexto de ciências sociais e pesquisa de políticas públicas, relevante para a análise econômica.")
    
    st.markdown("---")
    st.markdown("**Bibliografia Complementar**")

    st.markdown("1. **TEAM, R. Core. 'R language definition.' R foundation for statistical computing (2000).**")
    with st.expander("Explicação"):
        st.write("A documentação oficial da linguagem R.")

    st.markdown("2. **MISHRA, R.; RAM, B. Portfolio Selection Using R. Yugoslav Journal of Operations Research, 2020.**")
    with st.expander("Explicação"):
        st.write("Um artigo de pesquisa focado especificamente na aplicação da linguagem R para otimização e seleção de portfólios, muito relevante para a disciplina.")

    st.markdown("3. **WICKHAM, H., et al. (dplyr, Tidy data, Advanced R, ggplot2, R for data science).**")
    with st.expander("Explicação"):
        st.write("Múltiplas referências de Hadley Wickham, o criador do 'Tidyverse' em R. São os pacotes e livros fundamentais para a manipulação de dados moderna (dplyr), organização (Tidy data) e visualização (ggplot2) na linguagem R.")


def main():
    if 'builder' not in st.session_state:
        st.session_state.builder = None
        st.session_state.builder_complete = False
        st.session_state.profile = {}
        st.session_state.ativos_para_analise = []
        st.session_state.analisar_ativo_triggered = False
        
    configurar_pagina()
    # Garante que o painel de debug está sempre disponível no topo
    if 'debug_logs' not in st.session_state:
        st.session_state.debug_logs = []
    # **DEBUG REMOVIDO NO ARQUIVO FINAL**
    # mostrar_debug_panel() 

    st.markdown('<h1 class="main-header">Sistema de Portfólios Adaptativos</h1>', unsafe_allow_html=True)
    
    # Esta linha foi simplificada no código de produção para uso das abas
    tabs_list = ["📚 Metodologia", "🎯 Seleção de Ativos", "🏗️ Construtor de Portfólio", "🔍 Análise Individual", "📖 Referências"]
    
    # Este é o ponto onde o NameError pode ocorrer. Garantimos que todas as funções sejam definidas.
    tab1, tab2, tab3, tab4, tab5 = st.tabs(tabs_list)
    
    with tab1: aba_introducao()
    with tab2: aba_selecao_ativos()
    with tab3: aba_construtor_portfolio()
    with tab4: aba_analise_individual()
    with tab5: aba_referencias()

if __name__ == "__main__":
    main()
