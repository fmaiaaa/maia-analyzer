# -*- coding: utf-8 -*-
"""
=============================================================================
SISTEMA DE OTIMIZAÇÃO QUANTITATIVA
=============================================================================

Modelo de Alocação de Ativos com Métodos Adaptativos.
- Preços: Estratégia Linear com Fail-Fast (YFinance -> TvDatafeed -> Estático Global). 
- Fundamentos: Coleta Exaustiva Pynvest (50+ indicadores).
- Lógica de Construção (V9.4): Pesos Dinâmicos + Seleção por Clusterização.
- Modelagem (V9.43): ML Restaurado para Estabilidade (Lógica 6.0.9) + GARCH Removido.

Versão: 9.32.50 (Final Build: Estabilidade de Escopo e Indexação)
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
import math 

# --- 3. STREAMLIT, DATA ACQUISITION, & PLOTTING ---\
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests

# --- IMPORTAÇÕES PARA COLETA LIVE (HÍBRIDO) ---\
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

# --- 4. FEATURE ENGINEERING / TECHNICAL ANALYSIS (ML) ---
from sklearn.ensemble import IsolationForest, RandomForestClassifier, VotingClassifier 
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer 
from tqdm import tqdm # Necessário para a função GARCH avançada
from sklearn.linear_model import LogisticRegression 
from sklearn.linear_model import ElasticNetCV 

# NOVO: Adicionando LightGBM e XGBoost (para o modo FULL)
try:
    import lightgbm as lgb
except ImportError:
    lgb = None 
try:
    import xgboost as xgb
except ImportError:
    xgb = None 

try:
    import hdbscan
except ImportError:
    pass # Fallback to KMeans if hdbscan not installed

# --- 7. SPECIALIZED TIME SERIES & ECONOMETRICS ---
# GARCH REMOVIDO PARA ESTABILIDADE
arch_model = None
tabulate_arch_output = None


# =============================================================================
# 1. CONFIGURAÇÕES E CONSTANTES GLOBAIS
# =============================================================================

PERIODO_DADOS = 'max' 
MIN_DIAS_HISTORICO = 252
NUM_ATIVOS_PORTFOLIO = 5
TAXA_LIVRE_RISCO = 0.1075
LOOKBACK_ML = 30 # Usado como base para os horizontes
SCORE_PERCENTILE_THRESHOLD = 0.85 

# Pesos de alocação (Markowitz - Lógica Analyzer)
PESO_MIN = 0.10
PESO_MAX = 0.30

# NOVO: Limite mínimo de dias para o treinamento ML (AJUSTADO PARA 120)
MIN_TRAIN_DAYS_ML = 120 

# NOVO: Horizontes ML baseados no lookback adaptativo
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

# FEATURES DE ML (LÓGICA DO ARQUIVO PORTFOLIO_ANALYZER.PY - RENOMEADA PARA CONSISTÊNCIA)
ML_FEATURES = [
    'rsi_14', 'macd_diff', 'vol_20d', 'momentum_10d', 'sma_50d', 'sma_200d', # Technical (Renomeado)
    'pe_ratio', 'pb_ratio', 'div_yield', 'roe', 'pe_rel_sector', 'pb_rel_sector', # Fundamental
    'Cluster' # Categorical
]
ML_CATEGORICAL_FEATURES = ['Cluster']
# CORREÇÃO: LGBM_FEATURES definida globalmente para uso no ColetorDadosLive.coletar_ativo_unico_gcs
LGBM_FEATURES = ["ret", "vol20", "ma20", "z20", "trend", "volrel"]


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
    'Petróleo, Gás e Biocombustíveis': ['ENEV3.SA', 'PETR3.SA', 'PETR4.SA', 'PRIO3.SA', 'RAIZ4.SA', 'RECV3.SA', 'UGPA3.SA', 'VBBR3.SA'],
    'Saúde': ['FLRY3.SA', 'HAPV3.SA', 'RADL3.SA'],
    'Tecnologia da Informação': ['TOTS3.SA'],
    'Telecomunicações': ['TIMS3.SA', 'VIVT3.SA'],
    'Utilidade Pública': ['AESB3.SA', 'AURE3.SA', 'BRAV3.SA', 'CMIG4.SA', 'CPLE6.SA', 'CPFE3.SA', 'EGIE3.SA', 'ELET3.SA', 'ELET6.SA', 'ENGI11.SA', 'EQTL3.SA', 'ISAE4.SA', 'RAIL3.SA', 'SBSP3.SA', 'TAEE11.SA']
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
    timestamp = datetime.now().strftime("%H:%M:%S")
    # Imprime no console (para Streamlit Cloud logs) e armazena na sessão
    print(f"DEBUG {timestamp} | {message}") 
    if 'debug_logs' in st.session_state:
        st.session_state.debug_logs.append(f"[{timestamp}] {message}")

# REMOVIDO: A função mostrar_debug_panel() foi removida conforme solicitado.

def obter_template_grafico() -> dict:
    """Retorna o template padrão de cores e estilo para gráficos Plotly."""
    corporate_colors = ['#2E86C1', '#D35400', '#27AE60', '#8E44AD', '#C0392B', '#16A085', '#F39C12', '#34495E']
    return {
        # Fundo transparente para gráficos
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'font': {'family': 'Inter, sans-serif', 'size': 12, 'color': '#343a40'},
        # TÍTULOS DE EIXO E ETC. MANTÊM-SE, APENAS O TÍTULO PRINCIPAL É REMOVIDO NAS ABAS TÉCNICAS
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
    """
    CLASSE REFEITA PARA USAR AS FEATURES LIGHTGBM (fast_features)
    E MANTER RETORNOS/VOLATILIDADE E MÉTRICAS PARA SCORING TÉCNICO.
    """
    @staticmethod
    def enriquecer_dados_tecnicos(df_ativo: pd.DataFrame) -> pd.DataFrame:
        if df_ativo.empty: return df_ativo
        df = df_ativo.sort_index().copy()
        
        # Renomeia colunas para o padrão do modelo LightGBM
        df.rename(columns={'Close': 'close', 'High': 'high', 'Low': 'low', 'Open': 'open', 'Volume': 'volume'}, inplace=True)

        df['returns'] = df['close'].pct_change()
        
        # === fast_features (EXATAMENTE COMO O CÓDIGO FORNECIDO) ===
        df["ret"] = np.log(df["close"] / df["close"].shift(1))
        df["vol20"] = df["ret"].rolling(20).std()
        df["ma20"] = df["close"].rolling(20).mean()
        df["z20"] = (df["close"] - df["ma20"]) / (df["close"].rolling(20).std() + 1e-9)
        df["trend"] = df["close"].diff(5)
        df["volrel"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-9)
        # ===========================================================
        
        # --- MÉTRICAS DE SCORING / DISPLAY TRADICIONAL (RSI, MACD, BBands) ---
        
        # 1. Vol Anualizada (usada no score de performance/vol_current)
        df['vol_20d'] = df["ret"].rolling(20).std() * np.sqrt(252) 
        
        # 2. MACD Diff (Proxy para score técnico e display)
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26 # Necessário para o plot (linha macd)
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean() # Necessário para o plot (linha signal)
        df['macd_diff'] = df['macd'] - df['macd_signal'] # Usado no score técnico e plot (histograma)

        # 3. RSI (Proxy para score técnico e display)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi_14'] = 100 - (100 / (1 + rs)) 

        # 4. Momentum (10 períodos) - Adicionado para ML (Lógica Portfolio Analyzer)
        df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
        
        # 5. Médias Móveis - Adicionado para ML (Lógica Portfolio Analyzer)
        df['sma_50d'] = df['close'].rolling(window=50).mean()
        df['sma_200d'] = df['close'].rolling(window=200).mean()

        # 6. Bollinger Bands (Para o plot de análise individual)
        rolling_mean_20 = df['close'].rolling(window=20).mean()
        rolling_std_20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = rolling_mean_20 + (rolling_std_20 * 2)
        df['bb_lower'] = rolling_mean_20 - (rolling_std_20 * 2)


        # Renomeia de volta para o padrão Streamlit
        df.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low', 'open': 'Open', 'volume': 'Volume'}, inplace=True)
        
        # Remove colunas duplicadas ou temporárias não essenciais (Média móvel e outros features antigos)
        cols_to_keep = ['Close', 'High', 'Low', 'Open', 'Volume', 'returns', 
                        'ret', 'vol20', 'ma20', 'z20', 'trend', 'volrel', # LGBM Features
                        'vol_20d', 'macd_diff', 'rsi_14', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
                        'momentum_10d', 'sma_50d', 'sma_200d'] # Adicionado ML Features
        
        cols_to_drop = [c for c in df.columns if c not in cols_to_keep and c not in ['Close', 'High', 'Low', 'Open', 'Volume', 'returns']]
        df.drop(columns=cols_to_drop, errors='ignore', inplace=True)

        return df.dropna(subset=['Close']).fillna(0) 

# =============================================================================
# 6. CLASSE: ANALISADOR DE PERFIL DO INVESTIDOR
# =============================================================================
# (Manter inalterado)

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
        
        # NOVO: Define o lookback a partir dos ML_HORIZONS que mais se aproxima, 
        # mas mantemos o lookback de 5 dias do modelo LightGBM para predição.
        # Aqui, apenas mapeamos o nome do horizonte (Longo/Medio/Curto)
        
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
        
        # CORREÇÃO: Trata a chave ausente ou None/vazia
        liquidez_val = respostas_risco_originais.get('liquidity')
        objetivo_val = respostas_risco_originais.get('time_purpose')

        liquidez_key = liquidez_val[0] if isinstance(liquidez_val, str) and liquidez_val else 'C'
        objetivo_key = objetivo_val[0] if isinstance(objetivo_val, str) and objetivo_val else 'C'
        
        # Mantemos o lookback do perfil, mas o ML usará o ML_HORIZONS
        horizonte_tempo, ml_lookback = self.determinar_horizonte_ml(liquidez_key, objetivo_key)
        return nivel_risco, horizonte_tempo, ml_lookback, pontuacao

# =============================================================================
# 10. CLASSE: OTIMIZADOR DE PORTFÓLIO
# =============================================================================
# (Manter inalterado)

class OtimizadorPortfolioAvancado:
    def __init__(self, returns_df: pd.DataFrame, garch_vols: dict = None):
        self.returns = returns_df
        self.mean_returns = returns_df.mean() * 252
        
        # NOVO: Filtra garch_vols garantindo que apenas valores válidos (não zero/NaN) sejam usados
        valid_garch_vols = {k: v for k, v in garch_vols.items() if not np.isnan(v) and v > 0} if garch_vols else {}

        if valid_garch_vols:
            try:
                self.cov_matrix = self._construir_matriz_cov_garch(returns_df, valid_garch_vols)
            except Exception:
                # Fallback total para matriz de covariância histórica
                self.cov_matrix = returns_df.cov() * 252 
        else:
            self.cov_matrix = returns_df.cov() * 252
            
        self.num_ativos = len(returns_df.columns)

    def _construir_matriz_cov_garch(self, returns_df: pd.DataFrame, garch_vols: dict) -> pd.DataFrame:
        # A matriz de correlação é baseada em retornos históricos
        corr_matrix = returns_df.corr()
        
        vol_array = []
        for ativo in returns_df.columns:
            # Usa GARCH vol se disponível e válida, senão cai para vol histórica
            vol = garch_vols.get(ativo)
            if pd.isna(vol) or vol <= 0:
                vol = returns_df[ativo].std() * np.sqrt(252) # Fallback histórico
            vol_array.append(vol)
            
        vol_array = np.array(vol_array)
        # Reconstroi a matriz de covariância usando correlação histórica e volatilidade condicional/histórica
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
# (Função Auto GARCH integrada)

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
            from pynvest.scrappers.fundamentus import Fundamentus # Re-importa dentro do try/except
            self.pynvest_scrapper = Fundamentus()
            self.pynvest_ativo = True
            log_debug("Pynvest (Fundamentus) inicializado com sucesso.")
        except Exception:
            self.pynvest_ativo = False
            log_debug("AVISO: Pynvest falhou ao inicializar. Coleta de fundamentos desativada.")
            st.warning("Biblioteca pynvest não inicializada corretamente.")

    def _garch_auto_search(self, returns: pd.Series, symbol: str) -> tuple[float, str]:
        """Implementa a lógica de grid search massivo (Auto GARCH)."""
        
        returns_percent = returns * 100
        
        vol_types = ["Garch", "EGARCH", "GJR-GARCH", "HARCH", "APARCH"]
        dists = ["normal", "t", "ged"]
        p_range = range(1, 3)     
        q_range = range(1, 3)     
        o_range = [0, 1]          

        results = []
        fitted_models = {}
        
        log_debug(f"Iniciando Auto GARCH (Grid Search) para {symbol}. ")

        for vol in vol_types:
            for dist in dists:
                for p in p_range:
                    for q in q_range:
                        for o in o_range:
                            if vol not in ["GJR-GARCH", "APARCH"] and o > 0:
                                continue 
                            
                            if vol in ["HARCH", "APARCH"] and p > 1:
                                continue 

                            try:
                                model = arch_model(
                                    returns_percent,
                                    mean="Zero",
                                    vol=vol,
                                    p=p,
                                    q=q,
                                    o=o,
                                    dist=dist,
                                )

                                fit = model.fit(disp="off")
                                
                                model_name = f"{vol}(p={p},q={q},o={o}), dist={dist}"
                                results.append({"model": model_name, "bic": fit.bic, "aic": fit.aic})
                                fitted_models[model_name] = fit

                            except:
                                continue
        
        results_df = pd.DataFrame(results)
        if results_df.empty:
             log_debug(f"AVISO: Auto GARCH falhou em todas as combinações para {symbol}.")
             return np.nan, "GARCH(1,1) (Fallback: Auto-Search Failed)"

        # Selecionar melhor modelo por BIC
        best_row = results_df.loc[results_df["bic"].idxmin()]
        best_model_name = best_row["model"]
        best_fit = fitted_models[best_model_name]
        
        final_vol_daily = best_fit.conditional_volatility.iloc[-1] / 100
        final_vol_annualized = final_vol_daily * np.sqrt(252)

        log_debug(f"Auto GARCH para {symbol} concluído. Melhor Modelo: {best_model_name}")
        
        return final_vol_annualized, best_model_name


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
        garch_vols = {}
        metricas_simples_list = []
        
        # Obtém o modo GARCH configurado (padrão Garch(1,1))
        garch_mode = st.session_state.get('garch_mode', 'GARCH(1,1)')
        
        consecutive_failures = 0
        FAILURE_THRESHOLD = 3 
        global_static_mode = False 
        log_debug(f"Iniciando ciclo de coleta de preços para {len(simbolos)} ativos. Limite de falhas: {FAILURE_THRESHOLD}. Modo GARCH: {garch_mode}")
        
        for simbolo in simbolos:
            df_tecnicos = pd.DataFrame()
            usando_fallback_estatico = False 
            tem_dados = False
            
            # Inicializando com valores de fallback para evitar NameError
            vol_anual, ret_anual, sharpe, max_dd = 0.20, 0.0, 0.0, 0.0
            garch_vol = 0.20 # Fallback
            garch_model_name = "N/A"

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
                    # Não filtramos por NaNs aqui, deixamos o CalculadoraTecnica tratar.
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
                garch_model_name = "Vol. Histórica"
                
                if len(retornos) > 60: 
                    try:
                        if 'Auto-Search GARCH' in garch_mode:
                            garch_vol, garch_model_name = self._garch_auto_search(retornos, simbolo)
                        else:
                            # MODO RÁPIDO: GARCH(1,1) padrão
                            am = arch_model(retornos * 100, mean='Zero', vol='Garch', p=1, q=1)
                            res = am.fit(disp='off', last_obs=retornos.index[-1]) 
                            garch_std_daily = res.conditional_volatility.iloc[-1] / 100 
                            temp_garch_vol = garch_std_daily * np.sqrt(252)

                            if np.isnan(temp_garch_vol) or temp_garch_vol == 0 or temp_garch_vol > 1.0: 
                                raise ValueError("GARCH returned invalid value or nan.")
                            
                            garch_vol = temp_garch_vol
                            garch_model_name = "GARCH(1,1) (Rápido)"
                            log_debug(f"Ativo {simbolo}: GARCH(1,1) concluído. Vol Condicional: {garch_vol*100:.2f}%.")
                            
                    except Exception as e:
                        garch_vol = vol_anual 
                        garch_model_name = "Vol. Histórica (GARCH Failed)"
                        log_debug(f"Ativo {simbolo}: GARCH falhou ({str(e)[:20]}). Usando Vol Histórica como Vol Condicional.")
                # --- FIM DA IMPLEMENTAÇÃO GARCH ---
            
            fund_data.update({
                'Ticker': simbolo, 'sharpe_ratio': sharpe, 'annual_return': ret_anual,
                'annual_volatility': vol_anual, 'max_drawdown': max_dd, 'garch_volatility': garch_vol,
                'static_mode': usando_fallback_estatico, 'garch_model': garch_model_name
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
        Função para coletar e processar um único ativo para a aba de análise individual,
        incluindo o pipeline LightGBM.
        """
        log_debug(f"Iniciando coleta e análise de ativo único: {ativo_selecionado}")
        
        # Define o modo GARCH para a coleta individual (usamos o modo fast como padrão)
        st.session_state['garch_mode'] = st.session_state.get('individual_garch_mode', 'GARCH(1,1)')
        
        self.coletar_e_processar_dados([ativo_selecionado], check_min_ativos=False)
        
        if ativo_selecionado not in self.dados_por_ativo:
            log_debug(f"ERRO: Dados não encontrados após coleta para {ativo_selecionado}.")
            return None, None, None

        df_tec = self.dados_por_ativo[ativo_selecionado]
        fund_row = {}
        if ativo_selecionado in self.dados_fundamentalistas.index:
            fund_row = self.dados_fundamentalistas.loc[ativo_selecionado].to_dict()
        
        df_ml_meta = pd.DataFrame()
        
        # Features do modelo
        LGBM_FEATURES = ["ret", "vol20", "ma20", "z20", "trend", "volrel"]
        ALL_FUND_FEATURES = ['pe_ratio', 'pb_ratio', 'div_yield', 'roe', 'pe_rel_sector', 'pb_rel_sector', 'Cluster', 'roic', 'net_margin', 'debt_to_equity', 'current_ratio', 'revenue_growth', 'ev_ebitda', 'operating_margin']
        
        is_price_data_available = 'Close' in df_tec.columns and not df_tec['Close'].isnull().all() and len(df_tec.dropna(subset=['Close'])) > 60
        
        # Assume modo FAST para análise individual se a seleção não foi feita
        ml_mode_for_individual = st.session_state.get('individual_ml_mode', 'fast') 

        # Configura o Classificador e Features baseado no modo selecionado
        if ml_mode_for_individual == 'fast':
            MODEL_FEATURES = LGBM_FEATURES
            CLASSIFIER = LogisticRegression # Mudado para LogReg (Modelo mais rápido)
            MODEL_PARAMS = dict(penalty='l2', solver='liblinear', class_weight='balanced', random_state=42)
            MODEL_NAME = 'Regressão Logística Rápida'
        else:
            MODEL_FEATURES = ["ret", "vol20", "ma20", "z20", "trend", "volrel", 'rsi_14', 'macd_diff', 'vol_20d'] # Usamos features descorrelacionados
            CLASSIFIER = RandomForestClassifier
            MODEL_PARAMS = dict(n_estimators=150, max_depth=7, random_state=42, class_weight='balanced', n_jobs=-1)
            MODEL_NAME = 'Full Ensemble (RF/XGB)'

        # CORREÇÃO CRÍTICA: Inicializa is_ml_trained antes do bloco try/except
        is_ml_trained = False
        
        if is_price_data_available and CLASSIFIER is not None:
            log_debug(f"Análise Individual ML: Iniciando modelo {MODEL_NAME} para {ativo_selecionado}.")
            try:
                df = df_tec.copy()
                
                # Obtendo os Horizons adaptativos (embora o lookback do perfil não seja fornecido aqui, usamos um padrão)
                # Tenta usar a seleção da UI, senão usa o padrão do perfil (252)
                if st.session_state.get('individual_horizon_selection') == 'Curto Prazo (CP)':
                     ml_lookback_days = 84
                elif st.session_state.get('individual_horizon_selection') == 'Médio Prazo (MP)':
                     ml_lookback_days = 168
                else: # Longo (LP) ou Fallback
                     ml_lookback_days = st.session_state.profile.get('ml_lookback_days', 252) 
                     
                ML_HORIZONS_IND = get_ml_horizons(ml_lookback_days)
                
                # REFORÇANDO: Garante que os features fundamentais estão na última linha (para a predição)
                last_idx = df.index[-1] if not df.empty else None
                if last_idx:
                    for f_col in ALL_FUND_FEATURES:
                        if f_col in fund_row and f_col not in df.columns:
                            df.loc[last_idx, f_col] = fund_row[f_col]
                        elif f_col not in df.columns:
                            df[f_col] = np.nan
                            
                # Targets Futuros (make_targets logic)
                for d in ML_HORIZONS_IND:
                    df[f"t_{d}"] = (df["Close"].shift(-d) > df["Close"]).astype(int)

                # Remove NaNs da parte de treino e predição
                df_model = df.dropna(subset=MODEL_FEATURES + [f"t_{ML_HORIZONS_IND[-1]}"]) # Usa o maior horizonte para filtrar NaNs
                
                if len(df_model) > 200: # Mínimo de pontos para treino (70% de 200 = 140)
                    X_full = df_model[MODEL_FEATURES]
                    
                    # Split para treino (70%)
                    split_idx = int(len(X_full) * 0.7)
                    X_train = X_full.iloc[:split_idx]
                    
                    probabilities = []
                    auc_scores = []
                    
                    # --- TREINAMENTO PARA CADA HORIZONTE ---
                    for tgt_d in ML_HORIZONS_IND:
                        tgt = f"t_{tgt_d}"
                        y = df_model[tgt].values
                        y_train = y[:split_idx]
                        X_test = X_full.iloc[split_idx:]
                        y_test = y[split_idx:] 
                        
                        model = CLASSIFIER(**MODEL_PARAMS)
                        
                        # Fix para classes desbalanceadas ou únicas no treino
                        if len(np.unique(y_train)) < 2:
                             log_debug(f"ML Individual: {ativo_selecionado} - Target {tgt} tem apenas uma classe no treino. Pulando Target.")
                             continue
                             
                        # Aplica Scaling para modelos lineares/LogReg
                        if CLASSIFIER is LogisticRegression:
                             scaler = StandardScaler().fit(X_train)
                             X_train_scaled = scaler.transform(X_train)
                             X_test_scaled = scaler.transform(X_test)
                             X_predict_scaled = scaler.transform(X_full.iloc[[-1]].copy())
                        else:
                             X_train_scaled = X_train
                             X_test_scaled = X_test
                             X_predict_scaled = X_full.iloc[[-1]].copy()

                        model.fit(X_train_scaled, y_train)
                        
                        # --- VERIFICAÇÃO PARA PREDICAO ---
                        
                        if not X_full.iloc[[-1]].isnull().any().any():
                            prob_now = model.predict_proba(X_predict_scaled)[0, 1]
                            probabilities.append(prob_now)

                        # Cálculo de AUC no conjunto de teste (para confiança)
                        if len(y_test) > 0 and len(np.unique(y_test)) >= 2:
                             prob_test = model.predict_proba(X_test_scaled)[:, 1]
                             auc_scores.append(roc_auc_score(y_test, prob_test))
                             
                    # Score final ML = Média das 3 probabilidades
                    ensemble_proba = np.mean(probabilities) if probabilities else 0.5
                    
                    # Confiança final = Média dos AUCs de teste
                    conf_final = np.mean(auc_scores) if auc_scores else 0.5
                    
                    log_debug(f"ML Individual: Sucesso {MODEL_NAME}. Prob Média: {ensemble_proba:.2f}, AUC Teste Média: {conf_final:.2f}.")

                    # Importância das features
                    try:
                         if CLASSIFIER is LogisticRegression:
                             # Usa coeficientes para LogReg
                             importances_data = np.abs(model.coef_[0])
                         else:
                             # Usa feature_importances_ para RF/XGB
                             importances_data = model.feature_importances_

                         importances = pd.DataFrame({
                            'feature': MODEL_FEATURES,
                            'importance': importances_data
                         }).sort_values('importance', ascending=False)
                    except:
                         importances = pd.DataFrame({'feature': MODEL_FEATURES, 'importance': [1/len(MODEL_FEATURES)]*len(MODEL_FEATURES)})


                    df_tec['ML_Proba'] = ensemble_proba
                    df_tec['ML_Confidence'] = conf_final
                    df_ml_meta = importances
                    is_ml_trained = True # SUCESSO NO TREINAMENTO
                    
                else:
                    log_debug(f"ML Individual: Dados insuficientes ({len(df_model)}). Pulando modelo supervisionado.")
                    
            except Exception as e:
                log_debug(f"ML Individual: ERRO no modelo {MODEL_NAME}: {str(e)[:50]}. {traceback.format_exc()[:100]}")
                
            
        # 2. Fallback: Se ML falhou no cálculo ou não foi treinado
        if not is_ml_trained:
            log_debug("ML Individual: Modelo supervisionado não foi treinado. Excluindo ML_Proba/Confidence.")
            
            # NOVO: Apenas remove as colunas se existirem para que o Fallback de exibição funcione.
            if 'ML_Proba' in df_tec.columns:
                df_tec.drop(columns=['ML_Proba', 'ML_Confidence'], errors='ignore', inplace=True)
            
            # Gera uma tabela de importância de fallback se a original não foi gerada
            if df_ml_meta.empty:
                df_ml_meta = pd.DataFrame({
                    'feature': ['Qualidade (ROE/PL)', 'Estabilidade'],
                    'importance': [0.8, 0.2]
                })
            
        return df_tec, fund_row, df_ml_meta
        return None, None, None

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
    # Garante que st.session_state.debug_logs está inicializado para log_debug()
    if 'debug_logs' not in st.session_state:
        st.session_state.debug_logs = []
    # REMOVIDO: A chamada mostrar_debug_panel() foi removida conforme solicitado.

    st.markdown('<h1 class="main-header">Sistema de Otimização Quantitativa</h1>', unsafe_allow_html=True)
    
    # Esta linha foi simplificada no código de produção para uso das abas
    tabs_list = ["📚 Metodologia", "🎯 Seleção de Ativos", "🏗️ Construtor de Portfólio", "🔍 Análise Individual", "📖 Referências"]
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(tabs_list)
    
    with tab1: aba_introducao()
    with tab2: aba_selecao_ativos()
    with tab3: aba_construtor_portfolio()
    with tab4: aba_analise_individual()
    with tab5: aba_referencias()

if __name__ == "__main__":
    main()
