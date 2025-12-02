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

Versão: 9.32.47 (Final Build: Debug no Topo da Página)
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
        'title': {'font': {'family': 'Inter', 'size': 16, 'color': '#212529', 'weight': 'bold'}, 'x': 0.5, 'xanchor': 'center'},
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
    CLASSE REFEITA PARA USAR AS FEATURES DO MODELO ESTÁVEL (LÓGICA 6.0.9)
    """
    @staticmethod
    def enriquecer_dados_tecnicos(df_ativo: pd.DataFrame) -> pd.DataFrame:
        if df_ativo.empty: return df_ativo
        df = df_ativo.sort_index().copy()
        
        # Renomeia colunas para o padrão
        df.rename(columns={'Close': 'close', 'High': 'high', 'Low': 'low', 'Open': 'open', 'Volume': 'volume'}, inplace=True)

        df['returns'] = df['close'].pct_change()
        
        # === FEATURES ADICIONAIS PARA ML/ANÁLISE TÉCNICA (RESTAURO - LÓGICA 6.0.9) ===
        
        # 1. Vol Anualizada (usada no score de performance/vol_current)
        df['vol_20d'] = df["returns"].rolling(20).std() * np.sqrt(252) 
        
        # 2. MACD Diff (Proxy para score técnico e display)
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26 
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean() 
        df['macd_diff'] = df['macd'] - df['macd_signal'] 
        
        # 3. RSI (Proxy para score técnico e display)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi_14'] = 100 - (100 / (1 + rs)) 

        # 4. Momentum (10 períodos)
        df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1

        # 5. Médias Móveis
        df['sma_50d'] = df['close'].rolling(window=50).mean()
        df['sma_200d'] = df['close'].rolling(window=200).mean()
        
        # 6. Bollinger Bands (Para o plot de análise individual)
        rolling_mean_20 = df['close'].rolling(window=20).mean()
        rolling_std_20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = rolling_mean_20 + (rolling_std_20 * 2)
        df['bb_lower'] = rolling_mean_20 - (rolling_std_20 * 2)


        # Renomeia de volta para o padrão Streamlit
        df.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low', 'open': 'Open', 'volume': 'Volume'}, inplace=True)
        
        # Garante que as colunas essenciais para o ML estão presentes
        cols_to_keep = ['Close', 'High', 'Low', 'Open', 'Volume', 'returns', 
                        'vol_20d', 'macd_diff', 'rsi_14', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
                        'momentum_10d', 'sma_50d', 'sma_200d'] 
        
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

class OtimizadorPortfolioAvancado:
    def __init__(self, returns_df: pd.DataFrame, garch_vols: dict = None):
        self.returns = returns_df
        self.mean_returns = returns_df.mean() * 252
        
        # ALTERAÇÃO: Otimizador usa apenas a matriz de covariância histórica (Vol Histórica)
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
            # Aumento o maxiter para maior robustez na minimização numérica (Correção)
            resultado = minimize(objetivo, chute_inicial, method='SLSQP', bounds=limites, constraints=restricoes, options={'maxiter': 1000, 'ftol': 1e-6})
            if resultado.success:
                final_weights = resultado.x / np.sum(resultado.x)
                return {ativo: peso for ativo, peso in zip(self.returns.columns, final_weights)}
            else:
                # Se a otimização falhar, retorna um dicionário vazio para forçar o fallback de score
                return {} 
        except Exception:
            # Se ocorrer um erro de cálculo (ex: singular matrix), retorna vazio
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
        self.volatilidades_garch_raw = {} # Mantido para armazenar Vol. Histórica
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
        except ImportError:
            self.pynvest_ativo = False
            log_debug("AVISO: Pynvest falhou ao inicializar. Coleta de fundamentos desativada.")
            st.warning("Biblioteca pynvest não inicializada corretamente.")
        except Exception:
             self.pynvest_ativo = False
             log_debug("AVISO: Pynvest falhou ao inicializar. Coleta de fundamentos desativada.")
             st.warning("Biblioteca pynvest não inicializada corretamente.")


    # GARCH Logic: Removida a seleção por HQIC e a previsão GARCH para estabilidade (Correção)
    
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
        metricas_simples_list = []
        garch_vols = {} 
        
        garch_mode = 'Vol. Histórica' # GARCH Removido
        
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
            garch_vol = 0.20 
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
                
                # --- GARCH REMOVIDO: Usando Vol Histórica como Vol Condicional ---
                garch_vol = vol_anual 
                garch_model_name = "Vol. Histórica"
                # --- FIM DA IMPLEMENTAÇÃO GARCH ---
            
            fund_data.update({
                'Ticker': simbolo, 'sharpe_ratio': sharpe, 'annual_return': ret_anual,
                'annual_volatility': vol_anual, 'max_drawdown': max_dd, 'garch_volatility': garch_vol, # Armazena Vol GARCH/Histórica
                'static_mode': usando_fallback_estatico, 'garch_model': garch_model_name
            })
            
            self.dados_por_ativo[simbolo] = df_tecnicos
            self.ativos_sucesso.append(simbolo)
            lista_fundamentalistas.append(fund_data)
            
            metricas_simples_list.append({
                'Ticker': simbolo, 'sharpe': sharpe, 'retorno_anual': ret_anual,
                'volatilidade_anual': vol_anual, 'max_drawdown': max_dd,
            })
            
            garch_vols[simbolo] = garch_vol # Armazena a vol condicional/histórica para otimizador

            if not global_static_mode:
                time.sleep(0.1) 

        self.dados_fundamentalistas = pd.DataFrame(lista_fundamentalistas)
        if not self.dados_fundamentalistas.empty:
             self.dados_fundamentalistas = self.dados_fundamentalistas.set_index('Ticker')
             
        self.metricas_performance = pd.DataFrame(metricas_simples_list)
        if not self.metricas_performance.empty:
             self.metricas_performance = self.metricas_performance.set_index('Ticker')
        
        # ALTERAÇÃO: Inicializa volatilidades_garch com a volatilidade histórica (pois GARCH foi removido)
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

    # CORREÇÃO: Método restaurado para resolver o AttributeError
    def coletar_ativo_unico_gcs(self, ativo_selecionado: str):
        """
        Função para coletar e processar um único ativo para a aba de análise individual,
        incluindo o pipeline ML.
        """
        log_debug(f"Iniciando coleta e análise de ativo único: {ativo_selecionado}")
        
        # Define o modo GARCH para a coleta individual (usamos o modo fast como padrão)
        st.session_state['garch_mode'] = 'GARCH(1,1)' # Fixado para Vol Histórica
        
        self.coletar_e_processar_dados([ativo_selecionado], check_min_ativos=False)
        
        if ativo_selecionado not in self.dados_por_ativo:
            log_debug(f"ERRO: Dados não encontrados após coleta para {ativo_selecionado}.")
            return None, None, None

        df_tec = self.dados_por_ativo[ativo_selecionado]
        fund_row = {}
        if ativo_selecionado in self.dados_fundamentalistas.index:
            fund_row = self.dados_fundamentalistas.loc[ativo_selecionado].to_dict()
        
        df_ml_meta = pd.DataFrame()
        
        # FEATURES AGORA SÃO BASEADAS NO MODELO ESTÁVEL (PORTFOLIO_ANALYZER.PY)
        # CORREÇÃO: A lista ML_FEATURES já é globalmente definida
        current_ml_features = ML_FEATURES
        ALL_FUND_FEATURES = ['pe_ratio', 'pb_ratio', 'div_yield', 'roe', 'pe_rel_sector', 'pb_rel_sector', 'Cluster', 'roic', 'net_margin', 'debt_to_equity', 'current_ratio', 'revenue_growth', 'ev_ebitda', 'operating_margin']
        
        is_price_data_available = 'Close' in df_tec.columns and not df_tec['Close'].isnull().all() and len(df_tec.dropna(subset=['Close'])) > 60
        
        ml_mode_for_individual = st.session_state.get('individual_ml_mode', 'fast') 

        # Configura o Classificador e Features baseado no modo selecionado (Simplificado)
        if ml_mode_for_individual == 'fast':
            CLASSIFIER = LogisticRegression 
            MODEL_PARAMS = dict(penalty='l2', solver='liblinear', class_weight='balanced', random_state=42)
            MODEL_NAME = 'Regressão Logística'
        else:
            CLASSIFIER = RandomForestClassifier
            MODEL_PARAMS = dict(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced', n_jobs=-1)
            MODEL_NAME = 'Random Forest'


        is_ml_trained = False
        
        if is_price_data_available and CLASSIFIER is not None:
            log_debug(f"Análise Individual ML: Iniciando modelo {MODEL_NAME} para {ativo_selecionado}.")
            try:
                df = df_tec.copy()
                
                # Adiciona features fundamentais para o ML
                last_idx = df.index[-1] if not df.empty else None
                if last_idx:
                    for f_col in ALL_FUND_FEATURES:
                        if f_col in fund_row:
                            df[f_col] = fund_row[f_col] # Adiciona como coluna constante no histórico
                    
                    df['Cluster'] = fund_row.get('Cluster', 0)
                
                
                # Targets Futuros (make_targets logic)
                ml_lookback_days = 5 # Usando lookback fixo para a Análise Individual
                
                # Targets Futuros (Lógica do Portfolio_Analyzer: Future_Direction)
                df['Future_Direction'] = np.where(
                    df['Close'].pct_change(ml_lookback_days).shift(-ml_lookback_days) > 0,
                    1,
                    0
                )

                # Prepara Features para o ML
                current_features = [f for f in ML_FEATURES if f in df.columns]

                # Adiciona Cluster para o preprocessor
                X_cols = [f for f in current_features if f not in ML_CATEGORICAL_FEATURES] + ML_CATEGORICAL_FEATURES
                
                df_model = df.dropna(subset=X_cols + ['Future_Direction']).copy()
                
                if len(df_model) > MIN_TRAIN_DAYS_ML:
                    
                    # Preprocessador (Mantendo a estrutura do pipeline estável - Mapeando colunas)
                    numeric_cols = [f for f in X_cols if df_model[f].dtype in [np.float64, np.int64] and f not in ML_CATEGORICAL_FEATURES]
                    categorical_cols = [f for f in X_cols if f in ML_CATEGORICAL_FEATURES]
                    
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', StandardScaler(), numeric_cols),
                            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
                        ],
                        remainder='passthrough'
                    )
                    
                    # Cria pipeline de modelo
                    model_pipeline = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('classifier', CLASSIFIER(**MODEL_PARAMS))
                    ])

                    X = df_model[X_cols].iloc[:-ml_lookback_days]
                    y = df_model['Future_Direction'].iloc[:-ml_lookback_days]
                    
                    # Ajusta X para o scaler: Cluster precisa ser string
                    if 'Cluster' in X.columns:
                        X['Cluster'] = X['Cluster'].astype(str)

                    # Split treino/teste (simulado por TimeSeriesSplit)
                    tscv = TimeSeriesSplit(n_splits=5)
                    
                    # Treinamento e Cross-Validation (Simplificado para o último horizonte)
                    auc_scores = cross_val_score(model_pipeline, X, y, cv=tscv, scoring='roc_auc', n_jobs=-1)
                    conf_final = auc_scores.mean()

                    # Treinamento final para predição
                    model_pipeline.fit(X, y)
                    
                    # Predição da última linha
                    last_features = df_tec[X_cols].iloc[[-1]].copy()
                    if 'Cluster' in last_features.columns:
                        last_features['Cluster'] = last_features['Cluster'].astype(str)

                    proba = model_pipeline.predict_proba(last_features)[:, 1][0]
                    ensemble_proba = proba
                    
                    df_tec.loc[df_tec.index[-1], 'ML_Proba'] = ensemble_proba
                    df_tec.loc[df_tec.index[-1], 'ML_Confidence'] = conf_final
                    is_ml_trained = True
                    log_debug(f"ML Individual: Sucesso {MODEL_NAME}. Prob Média: {ensemble_proba:.2f}, AUC Teste Média: {conf_final:.2f}.")

                    # Calculando Feature Importance para o display
                    try:
                        if hasattr(model_pipeline['classifier'], 'feature_importances_'):
                            importances = model_pipeline['classifier'].feature_importances_
                        elif hasattr(model_pipeline['classifier'], 'coef_'):
                            importances = np.abs(model_pipeline['classifier'].coef_[0])
                        
                        # Mapeando os nomes das colunas de volta
                        feature_names = model_pipeline['preprocessor'].get_feature_names_out(X.columns)
                        df_ml_meta = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
                    except:
                        df_ml_meta = pd.DataFrame({'feature': current_features, 'importance': [1/len(current_features)]*len(current_features)})
                    
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

    def calculate_cross_sectional_features(self):
        df_fund = self.dados_fundamentalistas.copy()
        if 'sector' not in df_fund.columns or 'pe_ratio' not in df_fund.columns: return
        
        log_debug("Calculando features cross-sectional (P/L e P/VP relativos ao setor).")
        
        cols_numeric = ['pe_ratio', 'pb_ratio']
        for col in cols_numeric:
             if col in df_fund.columns:
                 df_fund[col] = pd.to_numeric(df_fund[col], errors='coerce')

        # Substitui a média setorial por 1.0 se a média for zero ou NaN para evitar divisão por zero
        sector_means = df_fund.groupby('sector')[['pe_ratio', 'pb_ratio']].transform('mean')
        
        valid_pe_mean = sector_means['pe_ratio'].replace(0, np.nan).fillna(1.0)
        valid_pb_mean = sector_means['pb_ratio'].replace(0, np.nan).fillna(1.0)

        df_fund['pe_rel_sector'] = df_fund['pe_ratio'] / valid_pe_mean
        df_fund['pb_rel_sector'] = df_fund['pb_ratio'] / valid_pb_mean
        
        df_fund = df_fund.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        self.dados_fundamentalistas = df_fund
        log_debug("Features cross-sectional concluídas.")

    def calcular_volatilidades_garch(self):
        # ALTERAÇÃO: A função agora apenas registra o uso de Volatilidade Histórica
        log_debug("Verificando volatilidades (Utilizando Volatilidade Histórica - GARCH Removido).")
        
    def treinar_modelos_ensemble(self, ml_mode: str = 'fast', progress_callback=None):
        ativos_com_dados = [s for s in self.ativos_sucesso if s in self.dados_por_ativo]
        log_debug(f"Iniciando Pipeline de Treinamento ML/Clusterização (Modo: {ml_mode}).")
        
        # --- DEFINIÇÃO DE FEATURES (RESTAURO) ---
        ML_FEATURES = ML_FEATURES # Mantendo a lista de features do arquivo estável
        ALL_FUND_FEATURES = ['pe_ratio', 'pb_ratio', 'div_yield', 'roe', 'pe_rel_sector', 'pb_rel_sector', 'Cluster', 'roic', 'net_margin', 'debt_to_equity', 'current_ratio', 'revenue_growth', 'ev_ebitda', 'operating_margin']
        
        # --- Seleção de Classificador (Restaurado e Simplificado) ---
        if ml_mode == 'fast':
            CLASSIFIER = LogisticRegression
            MODEL_PARAMS = dict(penalty='l2', solver='liblinear', class_weight='balanced', random_state=42)
            MODEL_NAME = 'Regressão Logística'
        else: 
            CLASSIFIER = RandomForestClassifier
            MODEL_PARAMS = dict(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced', n_jobs=-1)
            MODEL_NAME = 'Random Forest'
            
        
        # --- Clusterização Inicial (Fundamentos) ---
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
            log_debug("AVISO: Falha na coleta de dados fundamentais (P/L, ROE, etc.). Usando Cluster = 0.")
        
        # --- Pipeline ML ---
        ml_lookback_days = self.perfil_dashboard.get('ml_lookback_days', 252) 
        ML_HORIZONS_CONST = get_ml_horizons(ml_lookback_days)
        
        all_ml_results = {}
        total_ml_success = 0
        
        for i, ativo in enumerate(ativos_com_dados):
            result_for_ativo = {'predicted_proba_up': 0.5, 'auc_roc_score': 0.0, 'model_name': 'Not Run/Data Error'}
            
            try:
                if progress_callback: progress_bar.progress(50 + int((i/len(ativos_com_dados))*20), text=f"Treinando {MODEL_NAME}: {ativo}...")
                df = self.dados_por_ativo[ativo].copy()
                
                # 1. Adiciona Features Fundamentais e Cluster como colunas constantes no histórico
                if ativo in self.dados_fundamentalistas.index:
                    fund_row = self.dados_fundamentalistas.loc[ativo].to_dict()
                    for f_col in ALL_FUND_FEATURES:
                        if f_col in fund_row and f_col not in df.columns:
                            df[f_col] = fund_row[f_col]
                    df['Cluster'] = fund_row.get('Cluster', 0)
                else:
                    for f_col in ALL_FUND_FEATURES: df[f_col] = np.nan
                    df['Cluster'] = 0

                # 2. Targets Futuros (usando a lógica do Portfolio_Analyzer)
                # Targets Futuros (Lógica do Portfolio_Analyzer: Future_Direction)
                df['Future_Direction'] = np.where(
                    df['Close'].pct_change(ml_lookback_days).shift(-ml_lookback_days) > 0,
                    1,
                    0
                )

                # 3. Prepara Features: Combinação de features técnicos e fundamentais
                current_features = [f for f in ML_FEATURES if f in df.columns]

                # Adiciona Cluster para o preprocessor
                X_cols = [f for f in current_features if f not in ML_CATEGORICAL_FEATURES] + ML_CATEGORICAL_FEATURES
                
                df_model = df.dropna(subset=X_cols + ['Future_Direction']).copy()
                
                # Apenas segue se houver dados suficientes após limpeza
                if len(df_model) < MIN_TRAIN_DAYS_ML: 
                    raise ValueError(f"Apenas {len(df_model)} pontos válidos para treino (Requerido: {MIN_TRAIN_DAYS_ML}).")
                
                # 4. Configura Pipeline de ML
                numeric_cols = [f for f in X_cols if df_model[f].dtype in [np.float64, np.int64] and f not in ML_CATEGORICAL_FEATURES]
                categorical_cols = [f for f in X_cols if f in ML_CATEGORICAL_FEATURES]

                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', StandardScaler(), numeric_cols),
                        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
                    ],
                    remainder='passthrough'
                )

                model_pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('classifier', CLASSIFIER(**MODEL_PARAMS))
                ])

                X_full = df_model[X_cols].iloc[:-ml_lookback_days]
                y = df_model['Future_Direction'].iloc[:-ml_lookback_days]
                
                probabilities = []
                auc_scores = []
                
                # Para maior estabilidade, treinamos apenas no último horizonte (Future_Direction)
                # que já foi calculado para o lookback do perfil.
                
                # Ajusta X para o scaler/encoder: Cluster precisa ser string
                if 'Cluster' in X_full.columns:
                    X_full['Cluster'] = X_full['Cluster'].astype(str)

                # Split treino/teste (simulado por TimeSeriesSplit)
                tscv = TimeSeriesSplit(n_splits=5)
                
                # Treinamento e Cross-Validation
                auc_scores = cross_val_score(model_pipeline, X_full, y, cv=tscv, scoring='roc_auc', n_jobs=-1)
                conf_final = auc_scores.mean()

                # Treinamento final para predição
                model_pipeline.fit(X_full, y)
                
                # Predição da última linha (usando os últimos dados conhecidos - lookback dias antes do final)
                last_features = df_model[X_cols].iloc[[-1]].copy()
                if 'Cluster' in last_features.columns:
                    last_features['Cluster'] = last_features['Cluster'].astype(str)
                
                prob_now = model_pipeline.predict_proba(last_features)[:, 1][0]
                
                ensemble_proba = prob_now
                
                # Armazenamento dos resultados na última linha do DF
                self.dados_por_ativo[ativo].loc[self.dados_por_ativo[ativo].index[-1], 'ML_Proba'] = ensemble_proba
                self.dados_por_ativo[ativo].loc[self.dados_por_ativo[ativo].index[-1], 'ML_Confidence'] = conf_final
                log_debug(f"ML (Supervisionado): Ativo {ativo} sucesso. Prob: {ensemble_proba:.2f}, AUC: {conf_final:.2f}.")
                total_ml_success += 1

            except Exception as e:
                log_debug(f"ML (Fallback): Ativo {ativo} falhou no treinamento ({str(e)[:20]}). Não aplicando Score Fallback.")
                
                if ativo in self.dados_por_ativo and not self.dados_por_ativo[ativo].empty:
                    df_local = self.dados_por_ativo[ativo]
                    df_local.drop(columns=['ML_Proba', 'ML_Confidence'], errors='ignore', inplace=True)
                
                result_for_ativo = {'predicted_proba_up': 0.5, 'auc_roc_score': 0.0, 'model_name': 'Training Failed'}
                
            all_ml_results[ativo] = result_for_ativo


        if total_ml_success == 0 and len(ativos_com_dados) > 0:
            log_debug("AVISO: Falha total no ML supervisionado. Score ML será desabilitado/neutro.")
            for ativo in ativos_com_dados:
                all_ml_results[ativo]['auc_roc_score'] = 0.0
                all_ml_results[ativo]['model_name'] = 'Total Fallback'
        
        self.predicoes_ml = all_ml_results
        log_debug("Pipeline de Treinamento ML/Clusterização concluído.")


    def realizar_clusterizacao_final(self):
        if self.scores_combinados.empty: return
        log_debug("Iniciando Clusterização Final nos Scores (KMeans).")
        # ALTERAÇÃO 5: Removido 'performance_score' do features_cluster
        features_cluster = ['fundamental_score', 'technical_score', 'ml_score_weighted']
        data_cluster = self.scores_combinados[features_cluster].fillna(50)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_cluster)
        pca = PCA(n_components=min(data_scaled.shape[1], 2))
        data_pca = pca.fit_transform(data_scaled)
        kmeans = KMeans(n_clusters=min(len(data_pca), 4), random_state=42, n_init=10)
        clusters = kmeans.fit_predict(data_pca)
        self.scores_combinados['Final_Cluster'] = clusters
        log_debug(f"Clusterização Final concluída. Identificados {self.scores_combinados['Final_Cluster'].nunique()} perfis de risco/retorno.")

    def pontuar_e_selecionar_ativos(self, horizonte_tempo: str):
        if horizonte_tempo == "CURTO PRAZO": share_tech, share_fund = 0.7, 0.3
        elif horizonte_tempo == "LONGO PRAZO": share_tech, share_fund = 0.3, 0.7
        else: share_tech, share_fund = 0.5, 0.5

        # ALTERAÇÃO 5: Removido W_PERF_GLOBAL e peso de performance
        W_ML_GLOBAL_BASE = 0.20 # Mantido 20%
        W_REMAINING = 1.0 - W_ML_GLOBAL_BASE # AGORA É 0.80
        w_tech_final = W_REMAINING * share_tech
        w_fund_final = W_REMAINING * share_fund
        self.pesos_atuais = {'Fundamentos': w_fund_final, 'Técnicos': w_tech_final, 'ML': W_ML_GLOBAL_BASE} # ALTERADO
        
        # JOIN SEGURO (RESOLVE O ERRO DE OVERLAP)
        cols_to_drop = [col for col in self.dados_fundamentalistas.columns if col in self.metricas_performance.columns]
        df_fund_clean = self.dados_fundamentalistas.drop(columns=cols_to_drop, errors='ignore')
        combined = self.metricas_performance.join(df_fund_clean, how='inner').copy()
        
        for symbol in combined.index:
            if symbol in self.dados_por_ativo:
                df = self.dados_por_ativo[symbol]
                
                if not df.empty and 'rsi_14' in df.columns:
                    # Usamos 'rsi_current', 'macd_current', 'vol_current' para o scoring
                    combined.loc[symbol, 'rsi_current'] = df['rsi_14'].iloc[-1]
                    combined.loc[symbol, 'macd_current'] = df['macd_diff'].iloc[-1] # Usa macd_diff para o score
                    combined.loc[symbol, 'vol_current'] = df['vol_20d'].iloc[-1]
                    
                    # Adiciona as colunas técnicas usadas no ranqueamento para garantir que existam
                    combined.loc[symbol, 'macd_diff'] = df['macd_diff'].iloc[-1]
                    combined.loc[symbol, 'rsi_14'] = df['rsi_14'].iloc[-1]
                    combined.loc[symbol, 'vol_20d'] = df['vol_20d'].iloc[-1]
                    
                else:
                    # Fallback para valores neutros se estiver em modo estático
                    combined.loc[symbol, 'rsi_current'] = 50
                    combined.loc[symbol, 'macd_current'] = 0
                    combined.loc[symbol, 'vol_current'] = 0
                    
                    # Garante que as colunas de métricas individuais cruas existam no combined para a tabela de ranking
                    combined.loc[symbol, 'macd_diff'] = np.nan
                    combined.loc[symbol, 'rsi_14'] = np.nan
                    combined.loc[symbol, 'vol_20d'] = np.nan
                    

        scores = pd.DataFrame(index=combined.index)
        
        # ALTERAÇÃO 5: Cria o score de performance, mas não o inclui na soma final
        scores['raw_performance_score'] = (EngenheiroFeatures._normalize_score(combined['sharpe'], True) * 0.6 + EngenheiroFeatures._normalize_score(combined['retorno_anual'], True) * 0.4)
        
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
        
        # ALTERAÇÃO: Nova fórmula de ponderação do Score ML (AUC * Prob * Peso)
        # Score ML Ponderado = AUC * Probabilidade de Alta * Peso Base * 100
        scores['ml_score_weighted'] = ml_conf.fillna(0) * ml_probs.fillna(0) * W_ML_GLOBAL_BASE * 100 

        # ALTERAÇÃO 5: Total score é a soma dos 3 pilares
        scores['total_score'] = scores['fundamental_score'] + scores['technical_score'] + scores['ml_score_weighted']
        
        # Junta os scores com todos os dados fundamentais e métricas técnicas
        self.combined_scores = scores.join(combined).sort_values('total_score', ascending=False)
        
        log_debug(f"Calculando Scores Ponderados. Horizonte: {horizonte_tempo}. Pesos Finais: Fund={w_fund_final:.2f}, Tec={w_tech_final:.2f}, ML={W_ML_GLOBAL_BASE:.2f}. (Performance Removida).")

        # -------------------------------------------------------------
        # 1. CRITÉRIO DE INCLUSÃO: FILTRO DE SCORE MÍNIMO
        # -------------------------------------------------------------
        if len(self.scores_combinados) > NUM_ATIVOS_PORTFOLIO:
            cutoff_index = min(15, len(self.scores_combinados) - 1)
            base_score = self.scores_combinados['total_score'].iloc[cutoff_index]
            
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
                best = self.scores_combinados[self.scores_combinados['Final_Cluster'] == c].head(1).index[0]
                final_selection.append(best)
        
        if len(final_selection) < NUM_ATIVOS_PORTFOLIO:
            others = [x for x in self.scores_combinados.index if x not in final_selection]
            remaining_to_add = NUM_ATIVOS_PORTFOLIO - len(final_selection)
            
            if remaining_to_add > 0:
                 others_df = self.scores_combinados.loc[others].sort_values('total_score', ascending=False)
                 final_selection.extend(others_df.index[:remaining_to_add].tolist())

        self.ativos_selecionados = final_selection[:NUM_ATIVOS_PORTFOLIO]
        log_debug(f"Seleção final concluída. {len(self.ativos_selecionados)} ativos selecionados: {self.ativos_selecionados}")

        return self.ativos_selecionados
    
    def otimizar_alocacao(self, nivel_risco: str):
        if not self.ativos_selecionados or len(self.ativos_selecionados) < 1:
            self.metodo_alocacao_atual = "ERRO: Ativos Insuficientes"; return {}
        
        # --- FILTRAGEM DE ATIVOS COM DADOS DE PREÇO VÁLIDOS ---
        available_assets_returns = {}
        ativos_sem_dados = []
        ativos_static_mode = [] # Novo: Rastrear ativos em modo estático

        for s in self.ativos_selecionados:
            is_static = self.dados_fundamentalistas.loc[s, 'static_mode'] if s in self.dados_fundamentalistas.index else True
            
            if is_static:
                 ativos_static_mode.append(s)
            
            if s in self.dados_por_ativo and 'returns' in self.dados_por_ativo[s] and not self.dados_por_ativo[s]['returns'].dropna().empty:
                available_assets_returns[s] = self.dados_por_ativo[s]['returns']
            else:
                ativos_sem_dados.append(s)

        final_returns_df = pd.DataFrame(available_assets_returns).dropna()
        
        # DECISÃO: USAR MARKOWITZ APENAS SE TIVER DADOS SUFICIENTES E NÃO ESTIVER EM MODO FUNDAMENTALISTA FORÇADO
        # E SE TODOS OS ATIVOS TIVEREM PREÇO (não estático).
        
        is_markowitz_possible = (
            final_returns_df.shape[0] >= 50 and 
            len(final_returns_df.columns) == len(self.ativos_selecionados) and
            not any(self.dados_fundamentalistas.loc[s, 'static_mode'] for s in final_returns_df.columns if s in self.dados_fundamentalistas.index)
        )
        
        if not is_markowitz_possible or 'PESOS_IGUAIS' in nivel_risco:
            log_debug(f"Markowitz Indisponível: Forçando Ponderação por Score/Fallback. Motivo: Dados Retorno: {final_returns_df.shape[0]}, Ativos com Retorno: {len(final_returns_df.columns)}, Estáticos: {len(ativos_static_mode)}.")

            # Esta lógica garante que todos os ativos *selecionados* (mesmo os estáticos/sem retorno)
            # recebam um peso baseado no Score Total calculado anteriormente.
            
            valid_selection = [a for a in self.ativos_selecionados if a in self.scores_combinados.index]
            
            if valid_selection:
                 scores = self.scores_combinados.loc[valid_selection, 'total_score']
                 total_score = scores.sum()
                 
                 # Cria um dicionário de pesos para *todos* os ativos selecionados (incluindo os problemáticos)
                 if total_score > 0:
                     weights = (scores / total_score).to_dict()
                     self.metodo_alocacao_atual = 'PONDERAÇÃO POR SCORE (Modo Estático/Dados Insuficientes)'
                 else:
                     weights = {asset: 1.0 / len(valid_selection) for asset in valid_selection}
                     self.metodo_alocacao_atual = 'PESOS IGUAIS (Fallback Total)'
            else:
                 # Fallback de segurança se nem o score foi calculado corretamente
                 weights = {asset: 1.0 / len(self.ativos_selecionados) for asset in self.ativos_selecionados}
                 self.metodo_alocacao_atual = 'PESOS IGUAIS (Fallback Total)'
                 
            return self._formatar_alocacao(weights)

        # --- EXECUÇÃO MARKOWITZ (Se a condição for True) ---
        
        # Filtra as volatilidades GARCH apenas para os ativos com dados de retorno
        # ALTERAÇÃO: Não usa GARCH, usa apenas Volatilidade Histórica (que está em volatilidades_garch)
        garch_vols_filtered = {asset: self.volatilidades_garch_raw.get(asset, final_returns_df[asset].std() * np.sqrt(252)) for asset in final_returns_df.columns}
        optimizer = OtimizadorPortfolioAvancado(final_returns_df, garch_vols=garch_vols_filtered)
        
        if 'CONSERVADOR' in nivel_risco or 'INTERMEDIÁRIO' in nivel_risco:
            strategy = 'MinVolatility'; self.metodo_alocacao_atual = 'MINIMIZAÇÃO DE VOLATILIDADE'
        else:
            strategy = 'MaxSharpe'; self.metodo_alocacao_atual = 'MAXIMIZAÇÃO DE SHARPE'
            
        log_debug(f"Otimizando Markowitz. Estratégia: {self.metodo_alocacao_atual} (Risco: {nivel_risco}).")
            
        weights = optimizer.otimizar(estrategia=strategy)
        
        if not weights:
             # CORREÇÃO: Se a otimização falhou (retornou {}), usa o fallback de PONDERAÇÃO POR SCORE (Linha 1391)
             log_debug("AVISO: Otimizador Markowitz falhou. Usando PONDERAÇÃO POR SCORE como fallback.")
             
             valid_selection = [a for a in self.ativos_selecionados if a in self.scores_combinados.index]
             if valid_selection:
                 scores = self.scores_combinados.loc[valid_selection, 'total_score']
                 total_score = scores.sum()
                 if total_score > 0:
                     weights = (scores / total_score).to_dict()
                     self.metodo_alocacao_atual = 'PONDERAÇÃO POR SCORE (Fallback Markowitz)'
                 else:
                     weights = {asset: 1.0 / len(valid_selection) for asset in valid_selection}
                     self.metodo_alocacao_atual = 'PESOS IGUAIS (Fallback Total)'
             else:
                 weights = {asset: 1.0 / len(self.ativos_selecionados) for asset in self.ativos_selecionados}
                 self.metodo_alocacao_atual = 'PESOS IGUAIS (Fallback Total)'
        
        total_weight = sum(weights.values())
        log_debug(f"Otimização Markowitz finalizada. Peso total: {total_weight:.2f}")
        return self._formatar_alocacao(weights)
        
    def _formatar_alocacao(self, weights: dict) -> dict:
        if not weights or sum(weights.values()) == 0: 
            return {s: {'weight': 0.0, 'amount': 0.0} for s in self.ativos_selecionados} # Retorna estrutura vazia segura

        total_weight = sum(weights.values())
        # Filtra e normaliza os pesos, garantindo que a saída seja um dicionário com todos os ativos
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
        
        # Re-normaliza se houver ativos filtrados por NaN/dados insuficientes
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
        
        # NOVO: Garante que self.ativos_selecionados é usado como base, mesmo que scores_combinados esteja vazio
        if self.scores_combinados.empty and self.ativos_selecionados:
             for simbolo in self.ativos_selecionados:
                 self.justificativas_selecao[simbolo] = "Dados de ranqueamento indisponíveis. Usando Fallback (Pesos Iguais/Score)."
             return self.justificativas_selecao
        elif self.scores_combinados.empty:
             return self.justificativas_selecao


        for simbolo in self.ativos_selecionados:
            justification = []
            
            # Buscando scores individuais
            score_row = self.scores_combinados.loc[simbolo] if simbolo in self.scores_combinados.index else {}
            
            ml_data = self.predicoes_ml.get(simbolo, {})
            # ML Fallback: Verifica se o AUC é 0.0 (falha total)
            is_ml_failed = ml_data.get('auc_roc_score', 0.0) == 0.0
            
            is_static = False
            if simbolo in self.dados_fundamentalistas.index:
                 is_static = self.dados_fundamentalistas.loc[simbolo].get('static_mode', False)

            if is_static:
                justification.append("⚠️ MODO ESTÁTICO (Preço Indisponível)")
            
            # Adiciona os scores de cada vertente
            
            justification.append(f"Score Fund: {score_row.get('fundamental_score', 0.0):.3f}")
            justification.append(f"Score Téc: {score_row.get('technical_score', 0.0):.3f}")
            # justification.append(f"Score Perf: {score_row.get('performance_score', 0.0):.3f}") # REMOVIDO (Alteração 5)
            
            if is_ml_failed:
                justification.append("Score ML: N/A (Falha de Treinamento)")
                justification.append("✅ Selecionado por Fundamentos (ML não disponível)")
            else:
                ml_prob = ml_data.get('predicted_proba_up', np.nan)
                ml_auc = ml_data.get('auc_roc_score', np.nan)
                justification.append(f"Score ML: {score_row.get('ml_score_weighted', 0.0):.3f} (Prob {ml_prob*100:.1f}%, Conf {ml_auc:.2f})")
            
            # Adiciona Cluster e Setor
            cluster = score_row.get('Final_Cluster', 'N/A')
            sector = self.dados_fundamentalistas.loc[simbolo, 'sector'] if simbolo in self.dados_fundamentalistas.index else 'N/A'
            justification.append(f"Cluster: {cluster}")
            justification.append(f"Setor: {sector}")
            
            self.justificativas_selecao[simbolo] = " | ".join(justification)
        return self.justificativas_selecao
        
    def executar_pipeline(self, simbolos_customizados: list, perfil_inputs: dict, ml_mode: str, pipeline_mode: str, progress_bar=None) -> bool:
        self.perfil_dashboard = perfil_inputs
        
        # ALTERAÇÃO: O modo fundamentalista só é um fallback, não uma opção de UI
        if pipeline_mode == 'fundamentalista':
             log_debug("Modo de Pipeline: FUNDAMENTALISTA (ML e Markowitz ignorados).")
             ml_mode = 'fallback' 

        try:
            if progress_bar: progress_bar.progress(10, text="Coletando dados LIVE (YFinance + Pynvest)...")
            
            if not self.coletar_e_processar_dados(simbolos_customizados):
                log_debug("AVISO: Coleta inicial falhou parcialmente/totalmente. Prosseguindo com os dados de fallback disponíveis.")
            
            if not self.ativos_sucesso: 
                 st.error("Falha na aquisição: Nenhum ativo pôde ser processado.")
                 return False

            if progress_bar: progress_bar.progress(30, text="Calculando métricas setoriais e volatilidade...")
            self.calculate_cross_sectional_features(); self.calcular_volatilidades_garch()
            
            if pipeline_mode == 'general':
                if progress_bar: progress_bar.progress(50, text=f"Executando Pipeline ML ({ml_mode.upper()})...")
                self.treinar_modelos_ensemble(ml_mode=ml_mode, progress_callback=progress_bar)
            else:
                 # Simula o resultado de ML para que o score ML seja desativado, mas o processo continue.
                 for ativo in self.ativos_sucesso:
                    self.predicoes_ml[ativo] = {'predicted_proba_up': 0.5, 'auc_roc_score': 0.0, 'model_name': 'Fundamental Mode Forced'}
                 if progress_bar: progress_bar.progress(60, text="Pulando ML (Modo Fundamentalista Ativo)...")
            
            if progress_bar: progress_bar.progress(70, text="Ranqueando e selecionando (Pesos Dinâmicos + PCA Final)...")
            self.pontuar_e_selecionar_ativos(horizonte_tempo=perfil_inputs.get('time_horizon', 'MÉDIO PRAZO')) 
            
            if pipeline_mode == 'general':
                if progress_bar: progress_bar.progress(85, text="Otimizando alocação (Markowitz 10-30%)...")
                self.alocacao_portfolio = self.otimizar_alocacao(nivel_risco=perfil_inputs.get('risk_level', 'MODERADO'))
            else:
                # Alocação neutra por score no modo Fundamentalista, se houver dados
                if progress_bar: progress_bar.progress(85, text="Alocação por Score (Modo Fundamentalista)...")
                self.alocacao_portfolio = self.otimizar_alocacao(nivel_risco='PESOS_IGUAIS') # Força fallback interno por score/pesos iguais

            if progress_bar: progress_bar.progress(95, text="Calculando métricas finais...")
            self.calcular_metricas_portfolio(); self.gerar_justificativas()
            if progress_bar: progress_bar.progress(100, text="Pipeline concluído!"); time.sleep(1) 
        
            # DEBUG COMPLEXO (ADICIONADO)
            with st.expander("🐛 LOG DE DEBUG AVANÇADO (Entradas, Scores e Pesos)", expanded=False):
                st.markdown("##### 1. Inputs do Perfil")
                st.json(profile_inputs)
                st.markdown("##### 2. Pesos Finais Utilizados na Pontuação")
                st.json(self.pesos_atuais)
                st.markdown("##### 3. Ranqueamento e Scores Combinados (Head)")
                # Novo: Exibir colunas relevantes para o debug (scores crus e ml_weighted)
                debug_cols = ['total_score', 'fundamental_score', 'technical_score', 'ml_score_weighted', 'raw_performance_score', 'sharpe', 'retorno_anual']
                debug_df = self.scores_combinados[[c for c in debug_cols if c in self.scores_combinados.columns]]
                st.dataframe(debug_df.head(10).style.format('{:.4f}'), use_container_width=True)
                st.markdown("##### 4. Resultados da Otimização Markowitz/Alocação")
                st.json({
                    "Método": self.metodo_alocacao_atual,
                    "Métricas Portfólio": self.metricas_portfolio,
                    "Alocação Final": {k: f"{v['weight']:.4f}" for k, v in self.alocacao_portfolio.items()}
                })
                st.markdown("##### 5. Predições ML por Ativo")
                st.dataframe(pd.DataFrame(self.predicoes_ml).T.reset_index().rename(columns={'index': 'Ticker'}), use_container_width=True)


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
        pca = PCA(n_components=n_comp)
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

def safe_format(value):
    """Formata valor float para string com 2 casas, tratando strings e NaNs."""
    if pd.isna(value):
        return "N/A"
    try:
        float_val = float(value)
        return f"{float_val:.2f}"
    except (ValueError, TypeError):
        return str(value)

# =============================================================================
# 13. INTERFACE STREAMLIT - CONFIGURAÇÃO E CSS ORIGINAL (V8.7)
# =============================================================================

def configurar_pagina():
    st.set_page_config(page_title="Sistema de Otimização Quantitativa", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
        <style>
        :root { --primary-color: #000000; --secondary-color: #6c757d; --background-light: #f8f9fa; --background-dark: #ffffff; --text-color: #212529; --text-color-light: #ffffff; --border-color: #dee2e6; }
        
        /* Modern Font */
        body, .stApp { font-family: 'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important; background-color: var(--background-light); color: var(--text-color); }
        
        /* Main Header - Mantido Centralizado */
        .main-header { 
            font-family: 'Inter', sans-serif; 
            color: #111; 
            text-align: center; 
            padding: 2rem 0; 
            font-size: 2.5rem !important; 
            font-weight: 700; 
            letter-spacing: -1px;
        }
        
        /* Cards */
        .info-box { 
            background-color: #ffffff; 
            border: 1px solid #e0e0e0; 
            padding: 20px; 
            border-radius: 12px; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.05); 
            margin-bottom: 20px;
        }
        
        /* Regras de centralização RESTAURADAS E APRIMORADAS PARA LARGURA TOTAL */
        
        /* H2 e H3 CENTRALIZADOS */
        h2, h3 {
            text-align: center !important;
        }

        /* Metrics (centralizados) */
        [data-testid="stMetric"] {
            text-align: center;
            margin: auto;
        }
        [data-testid="stMetricLabel"] {
            justify-content: center;
            text-align: center;
            width: 100%;
        }

        /* Metrics (FORÇANDO ALTURA MÍNIMA para prevenir CLS por quebra de linha) */
        [data-testid="stMetricValue"] {
            min-height: 2.2rem; /* Garante que o valor da métrica tenha espaço */
        }


        /* Tabs (centralizadas) */
        .stTabs [data-baseweb="tab-list"] { 
            border-bottom: 1px solid #eee; 
            gap: 10px;
            justify-content: center;
        }
        .stTabs [data-baseweb="tab"] { 
            font-weight: 600; 
            color: #666; 
            padding: 10px 20px;
            border-radius: 6px 6px 0 0;
        }
        .stTabs [aria-selected="true"] { 
            color: #111; 
            background-color: white; 
            border-bottom: 2px solid #111;
        }
        
        /* Botões CENTRALIZADOS FORÇADOS (usando flexbox no container pai) */
        /* Esta regra centraliza o container que segura o botão */
        .stButton > div {
             justify-content: center !important;
        }
        
        /* Garante que tabelas e gráficos ocupem a largura do container principal */
        /* Usamos use_container_width=True nos elementos Streamlit, e esta regra CSS garante que os containers internos se estiquem */
        div[data-testid="stPlotlyChart"] {
            width: 100% !important;
        }
        
        div.stDataFrame {
            width: 100% !important;
            /* Garante que o container da tabela se estique */
        }
        
        /* Outros estilos básicos */
        .stMetric { 
            background-color: #ffffff; 
            border: 1px solid #eee; 
            border-radius: 10px; 
            padding: 15px; 
            box-shadow: 0 2px 5px rgba(0,0,0,0.03); 
        }
        .stMetric label { font-weight: 600; color: #555; font-size: 0.9rem; }
        .stMetric div[data-testid="stMetricValue"] { font-weight: 700; color: #111; font-size: 1.6rem; }
        
        .stButton button { 
            border-radius: 8px; 
            font-weight: 600; 
            border: 1px solid #333; 
            color: #333;
            transition: all 0.2s;
        }
        .stButton button[kind="primary"] { 
            background-color: #111; 
            color: white; 
            border: none; 
        }
        .dataframe { font-size: 0.9rem; }
        .streamlit-expanderHeader { 
            font-weight: 600; 
            color: #333; 
            background-color: #fff; 
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

def aba_introducao():
    """Aba 1: Introdução Metodológica Didática e Exaustiva (Estilo Manual Completo)"""
    
    st.markdown("## 📚 Visão Geral do Sistema de Otimização Quantitativa")
    
    st.markdown("""
    <div class="info-box" style="text-align: center;">
    <h3>📈 Modelo de Alocação de Ativos Adaptativo</h3>
    <p>Este sistema utiliza uma metodologia quantitativa e híbrida para construir portfólios otimizados para o mercado brasileiro (B3). O objetivo é maximizar o retorno ajustado ao risco (Sharpe Ratio) e garantir a diversificação estatística, baseando-se no perfil de risco e horizonte de tempo do investidor.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("1. Arquitetura do Motor de Decisão Multicritério")
    st.write("A avaliação de cada ativo é realizada sob quatro pilares independentes. A ponderação de cada pilar é ajustada dinamicamente com base nas respostas do questionário de perfil (Ex: Curto Prazo prioriza Fatores Técnicos e ML; Longo Prazo prioriza Fundamentos).")

    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        st.markdown("#### Fatores de Decisão")
        # ALTERAÇÃO 5: Removido o pilar "Performance" da tabela
        st.markdown("""
        | Pilar | Peso Base | Foco Principal |
        | :--- | :--- | :--- |
        | **Machine Learning** | **20%** | Probabilidade de Movimento Direcional Futuro. |
        | **Fundamentos** | **Varia (máx 56%)** | Qualidade e Saúde Financeira (P/L, ROE, Dívida). |
        | **Técnicos** | **Varia (máx 56%)** | Momentum e Tendência (RSI, MACD, Volatilidade). |
        """)

    with col_p2:
        st.markdown("#### 🧠 Lógica de Ponderação (Exemplo)")
        st.markdown("A alocação final combina a otimização de portfólio (Markowitz) com a pontuação multicritério, garantindo que o portfólio seja eficiente e alinhado ao risco. O peso é distribuído entre Fundamentos e Técnicos com base no Horizonte de Tempo.")
        st.dataframe(pd.DataFrame({
            "Pilar": ["Fundamentalista", "Técnico", "ML (Base)", "Peso Total"],
            # ALTERAÇÃO 5: Ajustado os pesos com Performance removida (Total 80%)
            "CP (4 Meses)": ["24%", "56%", "20%", "100%"], # 0.7 * 0.8 = 0.56; 0.3 * 0.8 = 0.24
            "LP (12 Meses)": ["56%", "24%", "20%", "100%"] # 0.3 * 0.8 = 0.24; 0.7 * 0.8 = 0.56
        }).set_index('Pilar'), use_container_width=True)
        
    with col_p3:
        st.markdown("#### 🛡️ Gestão de Risco e Modelagem")
        # ALTERAÇÃO: Descrição de modelos simplificada para refletir as mudanças
        st.markdown("O sistema oferece diferentes níveis de sofisticação para estimar o risco (Volatilidade Condicional) e a previsão:")
        st.markdown("""
        * **Volatilidade:** Utiliza o modelo **Volatilidade Histórica Anualizada** para o cálculo de risco. (O GARCH foi removido para maior estabilidade).
        * **Previsão (ML - Simples):** **Regressão Logística** com regularização (Elastic Net/L2).
        * **Previsão (ML - Complexa):** **Random Forest** com profundidade limitada e balanceamento de classes.
        """)


    st.markdown("---")
    st.subheader("2. Detalhamento do Pipeline Quantitativo")
    st.markdown("O sistema executa um fluxo de trabalho em etapas, garantindo que o portfólio final seja robusto e estatisticamente fundamentado.")

    with st.expander("2.1. Etapa de Coleta e Enriquecimento de Dados"):
        st.markdown("##### 📥 Módulos de Aquisição e Transformação")
        st.markdown("""
        * **Aquisição Híbrida:** Priorizamos fontes robustas (`YFinance`, `TvDatafeed`). O sistema executa um *fail-over* automático e, se falhar consecutivamente, ativa o **Modo Estático Global**.
        * **Indicadores Técnicos:** São calculados: RSI (14), MACD (12/26/9), Momentum (10d), e Médias Móveis (50/200d).
        * **Indicadores Fundamentalistas:** Mais de 50 métricas são coletadas (P/L, ROE, Dívida/PL, etc.) via `Pynvest` e tratadas.
        """)
        
        st.markdown("##### Exemplo de Feature Engineering")
        st.code("""
# Cálculo de Momentum (10 dias)
df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
# Volatilidade Anualizada (20 dias)
df['vol_20d'] = df["returns"].rolling(20).std() * np.sqrt(252) 
        """)

    with st.expander("2.2. Etapa de Machine Learning e Ranqueamento"):
        st.markdown("##### 🧠 Modelagem de Previsão e Score")
        st.markdown("""
        * **Target:** A variável alvo (`Future_Direction`) é definida como a direção do preço (alta/baixa) nos próximos N dias (dias ajustados pelo perfil do usuário).
        * **Feature Set:** O modelo utiliza uma combinação de **Indicadores Técnicos** e **Métricas Fundamentalistas** (P/L, ROE, etc.) como *features*.
        * **Regressão Logística (Rápido):** Utiliza penalidade L2 (`LogisticRegression(penalty='l2')`) para seleção de features e regularização, garantindo velocidade e estabilidade.
        * **Random Forest (Complexo):** Utiliza *ensemble* de árvores com profundidade limitada (`max_depth=5`) e balanceamento de classes para máxima precisão e mitigação de *overfitting* em séries temporais.
        
        * **Score ML Final:** É uma ponderação direta da **Probabilidade de Alta** e da **Confiança do Modelo (AUC-ROC)**, garantindo que previsões de modelos não confiáveis sejam neutralizadas.
        """)
        
        st.markdown("##### Exemplo de Pipeline ML")
        st.code("""
# Preprocessador com normalização e One-Hot Encoding para o Cluster
preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), numeric_cols),
                  ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)])

# Pipeline final para treino e validação
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(...))
])
        """)


    with st.expander("2.3. Etapa de Otimização e Alocação Final"):
        st.markdown("##### 💼 Markowitz e Gestão de Capital")
        st.markdown("""
        * **Seleção por Score:** Os ativos são ranqueados pelo `Score Total` (soma ponderada dos pilares Fundamentos, Técnicos e ML). Apenas os ativos com score acima de um limite percentual (geralmente 85% do 15º melhor ativo) são considerados.
        * **Clusterização Final:** O K-Means + PCA é aplicado nos scores finais para garantir a diversificação entre perfis de risco/retorno (clusters).
        * **Otimização Markowitz:** Utiliza a matriz de covariância histórica para calcular a **Fronteira Eficiente**. A alocação final pode ser ajustada para:
            1. **Maximização de Sharpe (MaxSharpe):** Para perfis mais arrojados.
            2. **Minimização de Volatilidade (MinVolatility):** Para perfis mais conservadores.
        * **Cálculo de Compra (Integridade Financeira):** O sistema calcula a quantidade exata de ações inteiras a serem compradas com o capital fornecido (`math.floor`), retornando o valor residual (*sobra*).
        """)

def aba_selecao_ativos():
    """Aba 2: Seleção de Ativos (Design Original Restaurado)"""
    
    st.markdown("## 🎯 Definição do Universo de Análise")
    
    st.markdown("""
    <div class="info-box">
    <p>O universo de análise está restrito ao <b>Índice Ibovespa</b>. O sistema utiliza todos os ativos selecionados para realizar o ranqueamento multi-fatorial e otimizar a carteira.</p>
    </div>
    """, unsafe_allow_html=True)
    
    modo_selecao = st.radio(
        "**Modo de Seleção:**",
        [
            "📊 Índice de Referência (Todos do Ibovespa)",
            "🏢 Seleção Setorial",
            "✍️ Seleção Individual"
        ],
        index=0,
        key='selection_mode_radio_v8'
    )
    
    ativos_selecionados = []
    
    if "Índice de Referência" in modo_selecao:
        ativos_selecionados = TODOS_ATIVOS.copy()
        st.success(f"✔️ **{len(ativos_selecionados)} ativos** (Ibovespa completo) definidos para análise.")
        
        with st.expander("📋 Visualizar Tickers"):
            st.write(", ".join([a.replace('.SA', '') for a in ativos_selecionados]))
    
    elif "Seleção Setorial" in modo_selecao:
        st.markdown("### 🏢 Seleção por Setor")
        setores_disponiveis = sorted(list(ATIVOS_POR_SETOR.keys()))
        
        # BARRA DE SELEÇÃO SETORIAL
        setores_selecionados = st.multiselect(
            "Escolha um ou mais setores:",
            options=setores_disponiveis,
            default=setores_disponiveis[:3] if setores_disponiveis else [],
            key='setores_multiselect_v8'
        )
        
        if setores_selecionados:
            for setor in setores_selecionados: ativos_selecionados.extend(ATIVOS_POR_SETOR[setor])
            ativos_selecionados = list(set(ativos_selecionados))
            
            # NOVO: Centraliza e expande as métricas abaixo do multiselect (lateralidade total)
            st.markdown("#### Setores e Ativos Selecionados")
            # Ajuste de Layout: Usar 3 colunas para ocupar a lateralidade
            col_metrics_s = st.columns(3) 
            
            with col_metrics_s[0]:
                st.metric("Setores Selecionados", len(setores_selecionados))
            with col_metrics_s[1]:
                st.metric("Total de Ativos", len(ativos_selecionados))
            with col_metrics_s[2]:
                 # Placeholder para manter o layout lateralizado (pode ser ajustado se houver mais métricas)
                 st.metric("Tickers/Setor (Visual)", "OK") 
            
            with st.expander("📋 Visualizar Ativos por Setor"):
                for setor in setores_selecionados:
                    # CORRIGIDO: O erro "един" foi corrigido para "join" (Correção)
                    ativos_do_setor = ATIVOS_POR_SETOR.get(setor, []) 
                    st.markdown(f"**{setor}** ({len(ativos_do_setor)} ativos)")
                    st.write(", ".join([a.replace('.SA', '') for a in ativos_do_setor]))
        else:
            st.warning("⚠️ Selecione pelo menos um setor.")
    
    elif "Seleção Individual" in modo_selecao:
        st.markdown("### ✍️ Seleção Individual de Tickers")
        
        ativos_com_setor = {}
        for setor, ativos in ATIVOS_POR_SETOR.items():
            for ativo in ativos: ativos_com_setor[ativo] = setor
        
        todos_tickers_ibov = sorted(list(ativos_com_setor.keys()))
        
        # BARRA DE SELECÇÃO INDIVIDUAL
        st.markdown("#### 📝 Selecione Tickers (Ibovespa)")
        ativos_selecionados = st.multiselect(
            "Pesquise e selecione os tickers:",
            options=todos_tickers_ibov,
            format_func=lambda x: f"{x.replace('.SA', '')} - {ativos_com_setor.get(x, 'Desconhecido')}",
            key='ativos_individuais_multiselect_v8'
        )
        
        # NOVO: Centraliza e expande a métrica abaixo do multiselect (lateralidade total)
        st.markdown("#### Tickers Selecionados")
        # Ajuste de Layout: Usar 3 colunas para ocupar a lateralidade
        col_metrics_i = st.columns(3)
        with col_metrics_i[1]:
            st.metric("Tickers Selecionados", len(ativos_selecionados))

        if not ativos_selecionados:
            st.warning("⚠️ Nenhum ativo definido.")
    
    if ativos_selecionados:
        st.session_state.ativos_para_analise = ativos_selecionados
        st.markdown("---")
        
        # Quadrados de resumo (Mantidos para a próxima tela)
        col1, col2, col3 = st.columns(3)
        col1.metric("Tickers Definidos", len(ativos_selecionados))
        col2.metric("Para Ranqueamento", len(ativos_selecionados))
        col3.metric("Carteira Final", NUM_ATIVOS_PORTFOLIO)
        
        st.success("✔️ Definição concluída. Prossiga para a aba **'Construtor de Portfólio'**.")
    else:
        st.warning("⚠️ O universo de análise está vazio.")

def aba_construtor_portfolio():
    """Aba 3: Construtor de Portfólio (Design Original Restaurado)"""
    
    if 'ativos_para_analise' not in st.session_state or not st.session_state.ativos_para_analise:
        st.warning("⚠️ Por favor, defina o universo de análise na aba **'Seleção de Ativos'** primeiro.")
        return
    
    if 'builder' not in st.session_state: st.session_state.builder = None
    if 'profile' not in st.session_state: st.session_state.profile = {}
    if 'builder_complete' not in st.session_state: st.session_state.builder_complete = False
    
    # Exibe o debug avançado no topo da aba (CORREÇÃO DE POSICIONAMENTO)
    if st.session_state.builder_complete:
        builder = st.session_state.builder
        with st.expander("🐛 LOG DE DEBUG AVANÇADO (Entradas, Scores e Pesos)", expanded=False):
            st.markdown("##### 1. Inputs do Perfil")
            st.json(st.session_state.profile)
            st.markdown("##### 2. Pesos Finais Utilizados na Pontuação")
            st.json(builder.pesos_atuais)
            st.markdown("##### 3. Ranqueamento e Scores Combinados (Head)")
            debug_cols = ['total_score', 'fundamental_score', 'technical_score', 'ml_score_weighted', 'raw_performance_score', 'sharpe', 'retorno_anual']
            debug_df = builder.scores_combinados[[c for c in debug_cols if c in builder.scores_combinados.columns]]
            st.dataframe(debug_df.head(10).style.format('{:.4f}'), use_container_width=True)
            st.markdown("##### 4. Resultados da Otimização Markowitz/Alocação")
            st.json({
                "Método": builder.metodo_alocacao_atual,
                "Métricas Portfólio": builder.metricas_portfolio,
                "Alocação Final": {k: f"{v['weight']:.4f}" for k, v in builder.alocacao_portfolio.items()}
            })
            st.markdown("##### 5. Predições ML por Ativo")
            st.dataframe(pd.DataFrame(builder.predicoes_ml).T.reset_index().rename(columns={'index': 'Ticker'}), use_container_width=True)


    if not st.session_state.builder_complete:
        st.markdown('## 📋 Calibração do Perfil de Risco')
        
        st.info(f"✔️ **{len(st.session_state.ativos_para_analise)} ativos** prontos. Responda o questionário para calibrar a otimização.")
        
        col_question1, col_question2 = st.columns(2)
        
        with st.form("investor_profile_form_v8_6", clear_on_submit=False): 
            
            with col_question1:
                st.markdown("#### Tolerância ao Risco")
                
                p2_risk_desc = st.radio(
                    "**1. Tolerância à Volatilidade:** Como você se sente sobre flutuações significativas (ex: quedas de 15-20%) no valor do seu portfólio em um único ano?", 
                    options=OPTIONS_CONCORDA, index=2, key='risk_accept_radio_v8_q1'
                )
                
                p3_gain_desc = st.radio(
                    "**2. Foco em Retorno Máximo:** Meu objetivo principal é maximizar o retorno, mesmo que isso signifique assumir riscos substancialmente maiores.", 
                    options=OPTIONS_CONCORDA, index=2, key='max_gain_radio_v8_q2'
                )
                
                p4_stable_desc = st.radio(
                    "**3. Prioridade de Estabilidade:** Priorizo a estabilidade e a preservação do meu capital acima do potencial de grandes ganhos.", 
                    options=OPTIONS_DISCORDA, index=2, key='stable_growth_radio_v8_q3'
                )
                
                p5_loss_desc = st.radio(
                    "**4. Aversão à Perda:** A prevenção de perdas de curto prazo é mais crítica para mim do que a busca por crescimento acelerado no longo prazo.", 
                    options=OPTIONS_DISCORDA, index=2, key='avoid_loss_radio_v8_q4'
                )
                
                p511_reaction_desc = st.radio(
                    "**5. Reação a Queda de 10%:** Se um ativo em sua carteira caísse 10% rapidamente, qual seria sua reação mais provável?", 
                    options=OPTIONS_REACTION_DETALHADA, index=1, key='reaction_radio_v8_q5'
                )
                
                p_level_desc = st.radio(
                    "**6. Nível de Conhecimento:** Qual seu nível de conhecimento sobre o mercado financeiro e tipos de investimento?", 
                    options=OPTIONS_CONHECIMENTO_DETALHADA, index=1, key='level_radio_v8_q6'
                )
            
            with col_question2:
                st.markdown("#### Horizonte e Capital")
                
                p211_time_desc = st.radio(
                    "**7. Horizonte de Investimento:** Por quanto tempo você pretende manter este investimento antes de precisar de uma reavaliação estratégica ou do capital?", 
                    options=OPTIONS_TIME_HORIZON_DETALHADA, index=2, key='time_purpose_radio_v8_q7'
                )
                
                p311_liquid_desc = st.radio(
                    "**8. Necessidade de Liquidez:** Qual é o prazo mínimo que você pode garantir que *não* precisará resgatar este capital?", 
                    options=OPTIONS_LIQUIDEZ_DETALHADA, index=2, key='liquidity_radio_v8_q8'
                )
                
                st.markdown("---")
                investment = st.number_input(
                    "Capital Total a ser Alocado (R$)",
                    min_value=1000, max_value=10000000, value=10000, step=1000, key='investment_amount_input_v8'
                )

                st.markdown("---")
                st.markdown("#### Modo de Execução do Pipeline")

                # ALTERAÇÃO: Modo Fundamentalista removido como opção de UI, forçando o modo geral
                pipeline_mode = "Modo Geral (ML + Otimização Markowitz)"
                st.info("Modo de Construção: **Geral (ML + Otimização Markowitz)** (O modo Fundamentalista é usado apenas como fallback automático).")

                ml_mode = 'fast'
                if pipeline_mode == 'Modo Geral (ML + Otimização Markowitz)':
                    # ALTERAÇÃO: Ajuste na lista e no format_func para refletir a simplificação
                    ml_mode = st.selectbox(
                        "**Seleção de Modelo ML:**",
                        [
                            'fast', 
                            'full'
                        ],
                        format_func=lambda x: "Rápido (Regressão Logística c/ ElasticNet)" if x == 'fast' else "Complexo (Ensemble RF+XGB c/ ElasticNet)",
                        index=0,
                        key='ml_model_mode_select'
                    )
            
            # NOVO: Centralização do botão e barra de loading
            st.markdown("---")
            col_btn_start, col_btn_center, col_btn_end = st.columns([1, 2, 1])
            with col_btn_center:
                submitted = st.form_submit_button("🚀 Gerar Alocação Otimizada", type="primary", use_container_width=True)
            
            progress_bar_placeholder = st.empty() # Placeholder para o loading
            
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
                    'liquidez': p311_liquid_desc,
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
                    return

                # NOVO: Barra de progresso visível logo após o submit
                progress_widget = progress_bar_placeholder.progress(0, text=f"Iniciando pipeline para PERFIL {risk_level}...")
                
                success = builder_local.executar_pipeline(
                    simbolos_customizados=st.session_state.ativos_para_analise,
                    perfil_inputs=st.session_state.profile,
                    ml_mode=ml_mode,
                    pipeline_mode='general', # HARDCODED para o modo geral
                    progress_bar=progress_widget
                )
                
                progress_widget.empty() # Limpa após o sucesso
                    
                if not success:
                    st.error("Falha na aquisição ou processamento dos dados. Resultados baseados em Fallback.")
                    st.session_state.builder_complete = True # Mantém True para mostrar a aba, mesmo com falha
                    # st.session_state.builder = None; st.session_state.profile = {}; Removido reset para manter dados de Fallback
                    st.rerun() 
                
                st.session_state.builder_complete = True
                st.rerun()
    
    else:
        builder = st.session_state.builder
        if builder is None: st.error("Objeto construtor não encontrado. Recomece a análise."); st.session_state.builder_complete = False; return
            
        profile = st.session_state.profile
        assets = builder.ativos_selecionados
        allocation = builder.alocacao_portfolio
        
        st.markdown('## ✅ Relatório de Alocação Otimizada')
        
        # --- 5 BOXES ALINHADOS EM UMA LINHA ---
        col1, col2, col3, col4, col5 = st.columns(5)
        
        col1.metric("Perfil Identificado", profile.get('risk_level', 'N/A'))
        col2.metric("Score Risco", profile.get('risk_score', 'N/A'))
        col3.metric("Horizonte Estratégico", profile.get('time_horizon', 'N/A'))
        
        # VERIFICAÇÃO ROBUSTA PARA SHARPE ANTES DE FORMATAR
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
        
        # Verifica se temos dados de preço para decidir o que mostrar
        has_price_data = not builder.metricas_performance.empty and 'volatilidade_anual' in builder.metricas_performance.columns and builder.metricas_performance['volatilidade_anual'].sum() != 0
        
        # FIX 4: Verifica se o GARCH foi minimamente bem sucedido (soma da vol GARCH > 0)
        has_garch_data = False # ALTERAÇÃO: GARCH removido

        # FIX 2: A aba ML só é exibida se houver resultados de ML utilizáveis (soma do score ponderado > 0)
        # ALTERAÇÃO 3: Adicionado 'assets and' para prevenir IndexError
        is_ml_actually_trained = assets and builder.predicoes_ml.get(assets[0], {}).get('auc_roc_score', 0.0) > 0.0
        
        # --- NOVO: Verificação de Redundância GARCH ---
        is_garch_redundant = False
        
        # Define as abas (agora consolidada)
        tabs_list = ["📊 Alocação de Capital", "📈 Performance e Retornos", "🔬 Análise de Fatores e Clusterização"]
        
        # O GARCH(1,1) é a única opção, então a aba é mostrada se houver dados de preço
        if has_price_data:
             tabs_list.append("📉 Fator Volatilidade") # Nome alterado
             
        tabs_list.append("❓ Justificativas e Ranqueamento")

        # Mapeia os índices das abas
        tabs_map = st.tabs(tabs_list)
        tab1 = tabs_map[0] # Alocação
        tab2 = tabs_map[1] # Performance/Retornos
        tab_fator_cluster = tabs_map[2] # Aba consolidada de ML/Clusters
        
        # Atribuição dinâmica das últimas abas
        if "📉 Fator Volatilidade" in tabs_list:
            tab_garch = tabs_map[3]
            tab_justificativas = tabs_map[4]
        else:
            tab_garch = None
            tab_justificativas = tabs_map[3]

        
        with tab1:
            st.markdown('#### Distribuição do Capital')
            col_alloc, col_table = st.columns([1, 2])
            
            with col_alloc:
                # Garante que não existem NaNs nos pesos antes de plotar
                weights_clean = {k: v['weight'] for k, v in allocation.items() if not pd.isna(v['weight'])}
                
                alloc_data = pd.DataFrame([
                    {'Ativo': a.replace('.SA', ''), 'Peso (%)': w * 100}
                    for a, w in weights_clean.items()
                ])
                
                if not alloc_data.empty:
                    fig_alloc = px.pie(alloc_data, values='Peso (%)', names='Ativo', hole=0.4)
                    template = obter_template_grafico()
                    fig_alloc.update_layout(**template)
                    fig_alloc.update_layout(title_text="Distribuição Otimizada por Ativo")
                    
                    st.plotly_chart(fig_alloc, use_container_width=True)
                else:
                    st.warning("Nenhuma alocação significativa para exibir. Otimização não retornou pesos.")
            
            with col_table:
                st.markdown('#### Detalhamento da Alocação Financeira')
                
                # Para evitar circular references ou NaN prices:
                total_investido_real = 0.0
                total_investimento_desejado = builder.valor_investimento
                
                alloc_table = []
                for asset in assets:
                    if asset in allocation and allocation[asset]['weight'] > 0:
                        weight = allocation[asset]['weight']
                        amount = allocation[asset]['amount']
                        
                        score_row = builder.scores_combinados.loc[asset] if asset in builder.scores_combinados.index else {}
                        total_score = score_row.get('total_score', np.nan)
                        cluster_id = score_row.get('Final_Cluster', 'N/A')
                        
                        # Safe Sector Access
                        try:
                             sector = builder.dados_fundamentalistas.loc[asset, 'sector']
                        except:
                             sector = "Unknown"
                        
                        # --- CÁLCULO PRINCIPAL (Alteração 2) ---
                        qtd_comprar = 0
                        preco_atual = np.nan
                        
                        if asset in builder.dados_por_ativo and 'Close' in builder.dados_por_ativo[asset].columns and not builder.dados_por_ativo[asset]['Close'].empty:
                             # Garantindo que o preço é o da última observação
                             preco_atual = builder.dados_por_ativo[asset]['Close'].iloc[-1]
                        
                        if not np.isnan(preco_atual) and preco_atual > 0:
                             # qtd = math.floor(peso * valor de investimento total / valor do ativo)
                             qtd_comprar = math.floor(amount / preco_atual)
                             custo_real = qtd_comprar * preco_atual
                             total_investido_real += custo_real
                        # --- FIM CÁLCULO PRINCIPAL ---
                             
                        alloc_table.append({
                            'Ticker': asset.replace('.SA', ''), 
                            'Peso (%)': f"{weight * 100:.2f}",
                            'Valor Desejado (R$)': f"R$ {amount:,.2f}",
                            'Preço Atual (R$)': f"R$ {preco_atual:,.2f}" if not np.isnan(preco_atual) else 'N/A',
                            'Qtd. Comprar': qtd_comprar, # NOVO CAMPO
                            'Score Total': f"{total_score:.3f}" if not np.isnan(total_score) else 'N/A', 
                            'Setor': sector,                                                           
                            'Cluster': str(cluster_id),                                                
                        })
                
                df_alloc = pd.DataFrame(alloc_table)
                st.dataframe(df_alloc, use_container_width=True)
                
                # --- EXIBIÇÃO DA SOBRA (Alteração 2) ---
                sobra_total = total_investimento_desejado - total_investido_real
                
                st.markdown("---")
                st.markdown("##### Resumo da Execução de Compra")
                col_res1, col_res2, col_res3 = st.columns(3)
                col_res1.metric("Capital Desejado (R$)", f"R$ {total_investimento_desejado:,.2f}")
                col_res2.metric("Capital Investido (R$)", f"R$ {total_investido_real:,.2f}")
                col_res3.metric("Capital Residual/Sobra (R$)", f"R$ {sobra_total:,.2f}")
                # --- FIM EXIBIÇÃO DA SOBRA ---
        
        with tab2:
            st.markdown('#### Métricas Chave do Portfólio (Histórico Recente)')
            
            col1, col2, col3, col4 = st.columns(4)
            # Safe Formatting for Portfolio Metrics
            m = builder.metricas_portfolio
            col1.metric("Retorno Anualizado", safe_format(m.get('annual_return', 0)*100) + "%")
            col2.metric("Volatilidade Anualizada", safe_format(m.get('annual_volatility', 0)*100) + "%")
            col3.metric("Sharpe Ratio", safe_format(m.get('sharpe_ratio', 0)))
            col4.metric("Máximo Drawdown", safe_format(m.get('max_drawdown', 0)*100) + "%")
            
            st.markdown("---")
            st.markdown('#### Trajetória de Retornos Cumulativos')
            
            fig_cum = go.Figure()
            has_data_plot = False
            
            for asset in assets:
                if asset in builder.dados_por_ativo and 'returns' in builder.dados_por_ativo[asset]:
                    returns = builder.dados_por_ativo[asset]['returns']
                    # Verifica se tem dados reais
                    if not returns.empty and not returns.isna().all():
                        cum_returns = (1 + returns).cumprod()
                        fig_cum.add_trace(go.Scatter(
                            x=cum_returns.index, y=cum_returns.values, name=asset.replace('.SA', ''), mode='lines'
                        ))
                        has_data_plot = True
            
            if has_data_plot:
                template = obter_template_grafico()
                fig_cum.update_layout(**template)
                fig_cum.update_layout(title_text="Retorno Acumulado dos Tickers Selecionados", yaxis_title="Retorno Acumulado (Base 1)", xaxis_title="Data", height=500)
                st.plotly_chart(fig_cum, use_container_width=True)
            else:
                st.info("Gráfico de retorno indisponível (Modo Estático Ativo - Sem histórico de preços).")
        
        with tab_fator_cluster:
            st.markdown('#### 🔬 Análise Consolidada de Fatores e Clusterização')

            # --- PARTE 1: FATOR ML/FUNDAMENTOS (Conteúdo da Antiga tab3) ---
            
            # Condição para mostrar resultados ML: Se o score ML ponderado for > 0 ou se o modo geral foi forçado
            # A variável is_ml_actually_trained já foi corrigida (Alteração 3)
            # ALTERAÇÃO: Força a exibição da seção ML se houver dados de preço (has_price_data)
            if has_price_data:
                 
                 if is_ml_actually_trained:
                     # ALTERAÇÃO 4: Nome do modelo atualizado para refletir o LogReg/RandomForest
                     st.markdown(f"##### 🤖 Predição de Movimento Direcional ({builder.predicoes_ml.get(assets[0], {}).get('model_name', 'Modelo ML')})")
                     st.markdown("O modelo utiliza histórico de preços para prever a probabilidade de alta no curto prazo.")
                     title_text_plot = "Probabilidade de Alta (0-100%)"
                     
                     ml_data = []
                     for asset in assets:
                        ml_info = builder.predicoes_ml.get(asset, {})
                        ml_data.append({
                            'Ticker': asset.replace('.SA', ''),
                            'Score/Prob.': ml_info.get('predicted_proba_up', np.nan) * 100,
                            'Confiança': ml_info.get('auc_roc_score', np.nan),
                            'Modelo': ml_info.get('model_name', 'N/A')
                        })
                     
                     df_ml = pd.DataFrame(ml_data).dropna(subset=['Confiança']) # Dropa se não houve treino ML
                
                     if not df_ml.empty:
                        fig_ml = go.Figure()
                        plot_df_ml = df_ml.sort_values('Score/Prob.', ascending=False)
                        
                        fig_ml.add_trace(go.Bar(
                            x=plot_df_ml['Ticker'],
                            y=plot_df_ml['Score/Prob.'],
                            marker=dict(
                                color=plot_df_ml['Score/Prob.'],
                                colorscale='Viridis', # Cor profissional
                                showscale=True,
                                colorbar=dict(title="Score")
                            ),
                            text=plot_df_ml['Score/Prob.'].round(1),
                                textposition='outside'
                        ))
                        
                        template = obter_template_grafico()
                        fig_ml.update_layout(**template)
                        
                        fig_ml.update_layout(title_text=title_text_plot, yaxis_title="Score", xaxis_title="Ticker", height=400)
                        
                        st.plotly_chart(fig_ml, use_container_width=True)
                        
                        st.markdown("---")
                        st.markdown('##### Detalhamento Predição ML')
                        df_ml_display = df_ml.copy()
                        df_ml_display['Score/Prob.'] = df_ml_display['Score/Prob.'].round(2)
                        df_ml_display['Confiança'] = df_ml_display['Confiança'].apply(lambda x: safe_format(x))
                        st.dataframe(df_ml_display, use_container_width=True, hide_index=True)
                     else:
                         st.info("ℹ️ **Modelo ML Não Treinado:** A pipeline de Machine Learning falhou para todos os ativos. A classificação se baseia puramente nos fatores Fundamentais e Técnicos.")
                 else:
                      st.info("ℹ️ **Modelo ML Não Treinado:** A pipeline de Machine Learning falhou para todos os ativos. A classificação se baseia puramente nos fatores Fundamentais e Técnicos.")


            # Sempre mostra a análise de Fundamentos/Cluster (que é a base)
            st.markdown("---")
            st.markdown('##### 🔬 Análise de Qualidade Fundamentalista (Unsupervised Learning)')
            
            # ALTERAÇÃO: Modo Fundamentalista não é mais opção de UI
            # if st.session_state.get('pipeline_mode_radio', '') == 'Modo Fundamentalista (Cluster/Anomalias)':
            #      st.info("ℹ️ **Modo Fundamentalista Ativo:** A classificação se baseia EXCLUSIVAMENTE nos fatores Fundamentais e Clusterização.")
                 
            st.markdown("###### Score Fundamentalista e Cluster por Ativo")
            
            if not builder.scores_combinados.empty:
                 # ALTERAÇÃO 5: Removido 'performance_score' do subset
                 df_cluster_display = builder.scores_combinados[['fundamental_score', 'Final_Cluster', 'pe_ratio', 'roe']].copy()
                 df_cluster_display.rename(columns={'fundamental_score': 'Score Fund.', 'Final_Cluster': 'Cluster', 'pe_ratio': 'P/L', 'roe': 'ROE'}, inplace=True)
                 
                 st.dataframe(df_cluster_display.style.format({
                     'Score Fund.': '{:.3f}', 'P/L': '{:.2f}', 'ROE': '{:.2f}'
                 }).background_gradient(cmap='Blues', subset=['Score Fund.']), use_container_width=True)
            else:
                 st.warning("Dados de fundamentos insuficientes para exibir clusters.")
            
            st.markdown("---")
            # --- PARTE 2: ANÁLISE DE CLUSTERS (Conteúdo da Antiga tab_portfolio_cluster) ---
            st.markdown('#### 🔭 Visualização da Diversificação e Clusters')
            st.info("Esta análise utiliza PCA sobre os scores para visualizar a distribuição dos ativos selecionados no 'espaço de risco/retorno' e confirmar a diversificação entre os clusters.")
            
            if 'Final_Cluster' in builder.scores_combinados.columns and len(builder.scores_combinados) >= 2:
                assets = builder.ativos_selecionados # Garante que 'assets' está definido
                df_viz = builder.scores_combinados.loc[assets].copy().reset_index().rename(columns={'index': 'Ticker'})
                
                # Prepara dados para PCA (apenas scores)
                # ALTERAÇÃO 5: Removido 'performance_score' do features_for_pca
                features_for_pca = ['fundamental_score', 'technical_score', 'ml_score_weighted']
                data_pca_input = df_viz[features_for_pca].fillna(50)
                
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data_pca_input)
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(data_scaled)
                
                df_viz['PC1'] = pca_result[:, 0]
                df_viz['PC2'] = pca_result[:, 1]
                
                # Gráfico de Dispersão 2D dos Clusters
                fig_cluster_scatter = px.scatter(
                    df_viz, 
                    x='PC1', 
                    y='PC2', 
                    color=df_viz['Final_Cluster'].astype(str),
                    size=df_viz['total_score'] / df_viz['total_score'].max() * 20, # Tamanho pela pontuação
                    hover_data={'Ticker': True, 'total_score': ':.2f', 'Final_Cluster': True},
                    text=df_viz['Ticker'].str.replace('.SA', ''),
                    title="Distribuição do Portfólio no Espaço PCA (Scores Finais)"
                )
                
                template = obter_template_grafico()
                fig_cluster_scatter.update_layout(**template)
                fig_cluster_scatter.update_traces(textposition='top center')
                fig_cluster_scatter.update_layout(height=600)
                
                st.plotly_chart(fig_cluster_scatter, use_container_width=True)
                
                st.markdown("##### Distribuição Setorial (Confirmação de Diversificação)")
                df_sector = df_viz.groupby('sector')['Ticker'].count().reset_index()
                df_sector.columns = ['Setor', 'Contagem']
                fig_sector = px.bar(df_sector, x='Setor', y='Contagem', title='Contagem de Ativos por Setor')
                
                # NOVO: Aplicando o template para garantir o fundo transparente
                fig_sector.update_layout(**obter_template_grafico())
                
                st.plotly_chart(fig_sector, use_container_width=True)
                
            else:
                 st.warning("Dados de scores insuficientes para análise de Clusterização do Portfólio.")
        
        # O bloco tab_garch só existe se has_price_data for True (GARCH foi removido, mas Volatilidade Histórica é essencial)
        if tab_garch is not None:
            with tab_garch:
                st.markdown('#### Volatilidade Histórica Anualizada')
                
                # IF ROBUSTO: SÓ MOSTRA SE TIVER PREÇO
                if has_price_data:
                    dados_vol = []
                    for ativo in assets:
                        if ativo in builder.metricas_performance.index:
                            perf = builder.metricas_performance.loc[ativo]
                            vol_hist = perf.get('volatilidade_anual', np.nan)
                            
                            if vol_hist is not None and not np.isnan(vol_hist):
                                status = '✓ Volatilidade Histórica'
                                vol_display = vol_hist
                            else:
                                status = '❌ Indisponível'
                                vol_display = np.nan
                            
                            dados_vol.append({
                                'Ticker': ativo.replace('.SA', ''),
                                'Vol. Anualizada (%)': vol_hist * 100 if not np.isnan(vol_hist) else 'N/A',
                                'Status de Cálculo': status
                            })
                    
                    df_vol = pd.DataFrame(dados_vol)
                    
                    if not df_vol.empty:
                        fig_vol = go.Figure()
                        
                        plot_df_vol = df_vol[df_vol['Vol. Anualizada (%)'] != 'N/A'].copy()
                        if not plot_df_vol.empty:
                            plot_df_vol['Vol. Anualizada (%)'] = plot_df_vol['Vol. Anualizada (%)'].astype(float)
                            
                            template_colors = obter_template_grafico()['colorway']
                            
                            fig_vol.add_trace(go.Bar(name='Volatilidade Histórica', x=plot_df_vol['Ticker'], y=plot_df_vol['Vol. Anualizada (%)'], marker=dict(color=template_colors[0]))) 
                            
                            template = obter_template_grafico()
                            fig_vol.update_layout(**template)
                            fig_vol.update_layout(title_text="Volatilidade Anualizada Histórica", yaxis_title="Volatilidade Anual (%)", height=400)
                            
                            st.plotly_chart(fig_vol, use_container_width=True)
                        else:
                            st.info("Dados de volatilidade insuficientes para gráfico.")

                        st.dataframe(df_vol, use_container_width=True, hide_index=True)
                    else:
                        st.warning("Não há dados de volatilidade para exibir.")
                else:
                     st.warning("⚠️ Dados de preços insuficientes para calcular volatilidade histórica.")
            
            with tab_justificativas:
                st.markdown('#### Ranqueamento Final e Justificativas Detalhadas')
                
                # ALTERAÇÃO 5: Removido Performance da exibição de pesos
                
                # CORREÇÃO: Adicionada verificação de chave antes de exibir
                if builder.pesos_atuais and all(key in builder.pesos_atuais for key in ['Fundamentos', 'Técnicos', 'ML']):
                     st.markdown(f"**Pesos Adaptativos Usados:** Fundamentos: {builder.pesos_atuais['Fundamentos']:.2f} | Técnicos: {builder.pesos_atuais['Técnicos']:.2f} | ML: {builder.pesos_atuais['ML']:.2f}")
                else:
                     st.warning("Pesos adaptativos não calculados ou ausentes.")

                st.markdown("---")
                
                # Safe Rename Logic (Check columns existence first)
                rename_map = {
                    'total_score': 'Score Total', 
                    'raw_performance_score': 'Score Perf.', # Score de performance original mantido para exibição no ranking (Alteração 5)
                    'performance_score': 'Score Perf. (Antigo)', 
                    'fundamental_score': 'Score Fund.', 
                    'technical_score': 'Score Téc.', 
                    'ml_score_weighted': 'Score ML', 
                    'sharpe': 'Sharpe',                 # USAR SHARPE CRU DA performance_metrics
                    'retorno_anual': 'Retorno Anual (%)', # USAR RETORNO CRU DA performance_metrics
                    'annual_volatility': 'Vol. Hist. (%)', # USAR VOLATILIDADE CRUA DA performance_metrics
                    'pe_ratio': 'P/L', 
                    'pb_ratio': 'P/VP',
                    'div_yield': 'Div. Yield (%)',
                    'roe': 'ROE (%)', 
                    'roic': 'ROIC (%)',
                    'net_margin': 'Margem Líq. (%)',
                    'rsi_14': 'RSI 14', 
                    'macd_diff': 'MACD Hist.', 
                    'ML_Proba': 'Prob. Alta ML',
                    # NOVO: Features LGBM (adicionados na pontuação para exibição)
                    'ret': 'Ret. Diário',
                    'vol20': 'Vol. 20d',
                    'ma20': 'Média 20d',
                    'z20': 'Z-Score 20d',
                    'trend': 'Trend 5d',
                    'volrel': 'Vol. Relativa'
                }
                
                if not builder.scores_combinados.empty:
                    # Adicionando colunas de ML/RSI/MACD se existirem no DataFrame completo dos ativos analisados
                    df_full_data = builder.scores_combinados.copy()
                    
                    # Tenta extrair a última linha de dados técnicos e ML para juntar
                    data_at_last_idx = {}
                    for ticker in df_full_data.index:
                        df_tec = builder.dados_por_ativo.get(ticker)
                        if df_tec is not None and not df_tec.empty:
                            last_row = df_tec.iloc[-1]
                            data_at_last_idx[ticker] = {
                                # CORRIGIDO o problema de sobreposição. Apenas colunas com nome de exibição exclusivo
                                # são adicionadas aqui. O resto já está em df_full_data.
                                'Preço Fechamento': last_row.get('Close'),
                                'rsi_14_raw': last_row.get('rsi_14'), # Usamos um nome distinto para o valor raw
                                'macd_diff_raw': last_row.get('macd_diff'), # Usamos um nome distinto para o valor raw
                                
                                # Adiciona os recursos do LGBM, pois eles só existem no DF de retorno do coletor (df_tec)
                                'ret': last_row.get('ret'),
                                'vol20': last_row.get('vol20'),
                                'ma20': last_row.get('ma20'),
                                'z20': last_row.get('z20'),
                                'trend': last_row.get('trend'),
                                'volrel': last_row.get('volrel'),
                            }
                    
                    df_last_data_clean = pd.DataFrame.from_dict(data_at_last_idx, orient='index')

                    # CORREÇÃO FINAL DO VALUEERROR: O join é feito aqui.
                    # As colunas duplicadas que causaram o erro (como 'rsi_14') foram renomeadas ou tratadas no loop acima.
                    df_scores_display = df_full_data.join(df_last_data_clean, how='left')
                    
                    # Mapeamento de colunas para exibição final (incluindo as renomeadas)
                    final_rename_map = {
                        'total_score': 'Score Total', 'raw_performance_score': 'Score Perf.', 
                        'fundamental_score': 'Score Fund.', 'technical_score': 'Score Téc.', 
                        'ml_score_weighted': 'Score ML', 'sharpe': 'Sharpe',
                        'retorno_anual': 'Retorno Anual (%)', 'annual_volatility': 'Vol. Hist. (%)', 
                        'pe_ratio': 'P/L', 'pb_ratio': 'P/VP', 'div_yield': 'Div. Yield (%)',
                        'roe': 'ROE (%)', 'roic': 'ROIC (%)', 'net_margin': 'Margem Líq. (%)',
                        'rsi_14_raw': 'RSI 14', 'macd_diff_raw': 'MACD Hist.',
                        'ret': 'Ret. Diário', 'vol20': 'Vol. 20d', 'ma20': 'Média 20d',
                        'z20': 'Z-Score 20d', 'trend': 'Trend 5d', 'volrel': 'Vol. Relativa',
                        'Preço Fechamento': 'Preço Fechamento'
                    }

                    # Filtra colunas para exibição (apenas as que existem após o join)
                    cols_to_display = [col for col in final_rename_map.keys() if col in df_scores_display.columns]

                    df_scores_display = df_scores_display[cols_to_display].copy()
                    df_scores_display.rename(columns=final_rename_map, inplace=True)
                    
                    # Multiplicando por 100 para percentual (apenas colunas que existem)
                    if 'ROE (%)' in df_scores_display.columns: df_scores_display['ROE (%)'] = df_scores_display['ROE (%)'] * 100
                    if 'Retorno Anual (%)' in df_scores_display.columns: df_scores_display['Retorno Anual (%)'] = df_scores_display['Retorno Anual (%)'] * 100
                    if 'Vol. Hist. (%)' in df_scores_display.columns: df_scores_display['Vol. Hist. (%)'] = df_scores_display['Vol. Hist. (%)'] * 100
                    if 'Div. Yield (%)' in df_scores_display.columns: df_scores_display['Div. Yield (%)'] = df_scores_display['Div. Yield (%)'] * 100
                    if 'ROIC (%)' in df_scores_display.columns: df_scores_display['ROIC (%)'] = df_scores_display['ROIC (%)'] * 100
                    if 'Margem Líq. (%)' in df_scores_display.columns: df_scores_display['Margem Líq. (%)'] = df_scores_display['Margem Líq. (%)'] * 100
                         
                    df_scores_display = df_scores_display.iloc[:20] # Exibe um top 20, para ser mais útil

                    
                    st.markdown("##### Ranqueamento Ponderado Multi-Fatorial (Top 20 Tickers do Universo Analisado)")
                    
                    # Definindo o dicionário de formatação de forma robusta
                    format_dict = {}
                    for col in df_scores_display.columns:
                        if 'Score' in col: format_dict[col] = '{:.3f}'
                        elif 'Sharpe' in col: format_dict[col] = '{:.3f}'
                        elif any(pct in col for pct in ['%']): format_dict[col] = '{:.2f}%'
                        elif 'P/L' in col or 'P/VP' in col or 'MACD' in col or 'Ret. Diário' in col or 'Z-Score 20d' in col or 'Vol. 20d' in col: format_dict[col] = '{:.2f}'
                        elif 'RSI' in col: format_dict[col] = '{:.2f}'
                        elif 'Prob' in col: format_dict[col] = '{:.2f}'
                        elif 'Média 20d' in col or 'Trend 5d' in col or 'Preço Fechamento' in col: format_dict[col] = '{:.2f}'
                        elif 'Vol. Relativa' in col: format_dict[col] = '{:.2f}'
                        else: format_dict[col] = '{}'
                        
                    st.dataframe(df_scores_display.style.format(
                        format_dict
                    ).background_gradient(cmap='Greys', subset=['Score Total'] if 'Score Total' in df_scores_display.columns else None), use_container_width=True)
                else:
                    st.warning("Tabela de scores indisponível (falha no processamento de dados).")
                
                st.markdown("---")
                st.markdown('##### Resumo da Seleção de Ativos (Portfólio Final)')
                
                if not builder.justificativas_selecao:
                    st.warning("Nenhuma justificativa gerada.")
                else:
                    for asset, justification in builder.justificativas_selecao.items():
                        # NOVO: Garante que o peso seja acessível para exibição (mesmo que 0 no fallback)
                        weight = builder.alocacao_portfolio.get(asset, {}).get('weight', 0)
                        
                        st.markdown(f"""
                        <div class="info-box">
                        <h4>{asset.replace('.SA', '')} ({weight*100:.2f}%)</h4>
                        <p><strong>Fatores-Chave:</strong> {justification}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Botão Recalibrar Perfil Centralizado no Final
            st.markdown("---")
            col_space1, col_btn, col_space2 = st.columns([1, 1, 1])
            with col_btn:
                if st.button("🔄 Recalibrar Perfil e Otimizar Novamente", type="primary", use_container_width=True):
                    st.session_state.builder_complete = False
                    st.session_state.builder = None
                    st.session_state.profile = {}
                    st.rerun()

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
    
    # --- NOVO: Seleção de Horizonte ML para Análise Individual (Alteração 1) ---
    st.markdown("#### Prazos de Predição (Dias Úteis Futuros)")
    
    # Mapeia as opções para os lookback days (que serão usados por get_ml_horizons)
    horizon_options = ['Longo Prazo (LP)', 'Médio Prazo (MP)', 'Curto Prazo (CP)']
    horizon_map_individual = {
        'Curto Prazo (CP)': 84,
        'Médio Prazo (MP)': 168,
        'Longo Prazo (LP)': 252
    }

    # Usa st.radio (horizontal) para simular os bullets/botões (Alteração 1)
    # Usa o estado salvo como valor padrão, senão usa 'Longo Prazo (LP)'
    default_horizon = st.session_state.get('individual_horizon_selection', 'Longo Prazo (LP)')
    default_index = horizon_options.index(default_horizon) if default_horizon in horizon_options else 0
    
    selected_horizon = st.radio(
        "Selecione o horizonte de predição:",
        options=horizon_options,
        index=default_index,
        key='individual_horizon_selection_radio',
        label_visibility="collapsed",
        horizontal=True # Para exibir em linha
    )
    
    # Atualiza o estado da sessão com a seleção do radio
    st.session_state['individual_horizon_selection'] = selected_horizon
    st.session_state.profile['ml_lookback_days'] = horizon_map_individual.get(selected_horizon, 252)


    st.write("") # Spacer
    
    # NOVO: Seleção de Modos de GARCH e ML para a Análise Individual (ocupando a lateralidade)
    st.markdown("#### Seleção de Modelos Quantitativos")
    col_modes = st.columns(2) # Reduzido para 2 colunas para ML e GARCH
    
    with col_modes[0]:
        st.markdown("##### Volatilidade (Risco):") # Nome alterado
        # ALTERAÇÃO: Apenas GARCH(1,1) é permitido (para manter o layout, mas é apenas vol histórica)
        # REMOVIDO o radio button de seleção de volatilidade (Correção)
        st.info("Modelo de Risco: Volatilidade Histórica Anualizada")
        st.session_state['individual_garch_mode'] = 'GARCH(1,1)' 
        
    with col_modes[1]:
        st.markdown("##### Modelo ML:")
        # ALTERAÇÃO: Refletindo as simplificações de ML (LogReg e Random Forest)
        ml_mode_select = st.radio(
            "Selecione o Modelo de Predição:",
            ['fast', 'full'],
            key='individual_ml_mode_radio',
            index=0,
            format_func=lambda x: "Rápido (Regressão Logística)" if x == 'fast' else "Lento (Random Forest)",
            label_visibility="collapsed"
        )
        st.session_state['individual_ml_mode'] = ml_mode_select
    
    
    # NOVO: Botões Centralizados (Executar Análise e Limpar Análise)
    col_btn_start, col_btn_center, col_btn_end = st.columns([1, 2, 1])
    with col_btn_center:
        col_exec, col_clear = st.columns(2)
        with col_exec:
            if st.button("🔄 Gerar Resultados", key='analyze_asset_button_v8', type="primary", use_container_width=True):
                st.session_state.analisar_ativo_triggered = True 
        with col_clear:
            if st.button("🗑️ Limpar Análise", key='clear_asset_analysis_button_v8', type="secondary", use_container_width=True):
                 st.session_state.analisar_ativo_triggered = False # Resetar o trigger
                 st.rerun()
    
    if 'analisar_ativo_triggered' not in st.session_state or not st.session_state.analisar_ativo_triggered:
        st.info("👆 Selecione um ticker, defina o Horizonte e os modelos para gerar os resultados.")
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
            
            # Define has_price_data no escopo local (CORREÇÃO)
            has_price_data = not static_mode

            # Verifica se o ML supervisionado foi executado com sucesso (AUC > 0.0)
            is_ml_trained = 'ML_Proba' in df_completo.columns and not static_mode and df_completo.get('ML_Confidence', 0.0).iloc[-1] > 0.0

            if static_mode:
                st.warning(f"⚠️ **MODO ESTÁTICO:** Preços indisponíveis. Exibindo apenas Análise Fundamentalista.")
                
            # Define a lista de abas, incluindo ML se foi treinado
            tabs_list_individual = ["📊 Visão Geral", "💼 Fundamentos", "🔧 Análise Técnica", "🔬 Clusterização Geral"]
            # ALTERAÇÃO: Mantém a aba ML sempre que houver dados de preço, para consistência da UI
            if has_price_data and not static_mode: tabs_list_individual.insert(3, "🤖 Machine Learning") 

            tabs_map = st.tabs(tabs_list_individual)
            
            tab_map_index = lambda title: tabs_list_individual.index(title)
            
            tab1 = tabs_map[tab_map_index("📊 Visão Geral")]
            tab2 = tabs_map[tab_map_index("💼 Fundamentos")]
            tab3 = tabs_map[tab_map_index("🔧 Análise Técnica")]
            tab5 = tabs_map[tab_map_index("🔬 Clusterização Geral")]
            
            # CORREÇÃO: Define tab_ml condicionalmente para evitar UnboundLocalError
            tab_ml = tabs_map[tab_map_index("🤖 Machine Learning")] if "🤖 Machine Learning" in tabs_list_individual else None
            
            # Abas 1-4: Lógica Padrão de Exibição (igual à versão anterior)
            with tab1:
                st.markdown(f"### {ativo_selecionado.replace('.SA', '')} - Resumo de Mercado")
                # CORREÇÃO: Reorganizando as colunas para evitar duplicação e alinhar
                col1, col2, col3, col4, col5 = st.columns(5)
                
                # Seção de Preço (sempre na coluna 1)
                if not static_mode and 'Close' in df_completo.columns:
                    preco_atual = df_completo['Close'].iloc[-1]
                    variacao_dia = df_completo['returns'].iloc[-1] * 100 if 'returns' in df_completo.columns else 0.0
                    volume_medio = df_completo['Volume'].mean() if 'Volume' in df_completo.columns else 0.0
                    vol_anual = features_fund.get('annual_volatility', 0) * 100
                    garch_model_name = features_fund.get('garch_model', "Vol. Histórica") # Nome Hardcoded
                    
                    # Exibição de Métricas (Ajuste para usar colunas 1, 2, 3, 4, 5)
                    col1.metric("Preço", f"R$ {preco_atual:.2f}", f"{variacao_dia:+.2f}%")
                    col2.metric("Volume Médio", f"{volume_medio:,.0f}")

                    # CORREÇÃO: Removido duplicação. Usando colunas 3 e 4
                    col3.metric("Setor", features_fund.get('sector', 'N/A'))
                    col4.metric("Indústria", features_fund.get('industry', 'N/A'))
                    
                    # Coluna 5: Volatilidade (Vol Histórica)
                    col5.metric(f"Vol. Anualizada", f"{vol_anual:.2f}%")

                else:
                    col1.metric("Preço", "N/A", "N/A"); col2.metric("Volume Médio", "N/A"); 
                    col3.metric("Setor", features_fund.get('sector', 'N/A'));
                    col4.metric("Indústria", features_fund.get('industry', 'N/A'));
                    col5.metric("Volatilidade", "N/A")
                
                
                if not static_mode and not df_completo.empty and 'Open' in df_completo.columns:
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                    fig.add_trace(go.Candlestick(x=df_completo.index, open=df_completo['Open'], high=df_completo['High'], low=df_completo['Low'], close=df_completo['Close'], name='Preço'), row=1, col=1)
                    fig.add_trace(go.Bar(x=df_completo.index, y=df_completo['Volume'], name='Volume'), row=2, col=1)
                    
                    template = obter_template_grafico()
                    template['title']['text'] = f"Gráfico Diário - {ativo_selecionado.replace('.SA', '')}" # Define no dict
                    fig.update_layout(**template)
                    fig.update_layout(height=600)
                    
                    st.plotly_chart(fig, use_container_width=True)
                else: st.info("Gráfico indisponível (Modo Estático Ativo).")

            with tab2:
                # --- TROCA DE ORDEM: PRIMEIRO INDICADORES PRINCIPAIS, DEPOIS TABELA GERAL ---
                st.markdown("### Principais Métricas")
                
                # Helper para troca de indicadores caso NaN
                def get_valid_metric(data, primary_key, primary_label, secondary_key, secondary_label):
                     val = data.get(primary_key)
                     if pd.isna(val):
                         return secondary_label, data.get(secondary_key)
                     return primary_label, val
                
                # Obtem Beta de Fallback se necessário (Yahoo Finance)
                beta_val = features_fund.get('beta')
                if pd.isna(beta_val):
                     try:
                         beta_val = yf.Ticker(ativo_selecionado).info.get('beta')
                     except: 
                         beta_val = np.nan
                
                # Linha 1
                col1, col2, col3, col4, col5 = st.columns(5)
                
                # Se P/L for NaN, tenta mostrar EV/EBITDA no lugar
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
                
                # FIX 5: Substitui 'Cresc. Receita (5a)' que estava dando N/A por ROIC (Return on Invested Capital)
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
                     st.dataframe(df_fund_show, use_container_width=True, hide_index=True)

            with tab3:
                # LÓGICA DE EXIBIÇÃO: SÓ MOSTRA SE NÃO ESTIVER EM MODO ESTÁTICO
                if not static_mode:
                    st.markdown("### Indicadores Técnicos"); col1, col2, col3 = st.columns(3)
                    
                    # Usa os novos features se existirem, senão usa os antigos/NA
                    rsi_display = f"{df_completo['rsi_14'].iloc[-1]:.2f}" if 'rsi_14' in df_completo else "N/A"
                    macd_display = f"{df_completo['macd_diff'].iloc[-1]:.4f}" if 'macd_diff' in df_completo else "N/A"
                    vol20_display = f"{df_completo['vol_20d'].iloc[-1]:.4f}" if 'vol_20d' in df_completo else "N/A"

                    col1.metric("RSI (14)", rsi_display)
                    col2.metric("MACD Diff", macd_display)
                    col3.metric("Vol. Anualizada (20d)", vol20_display)
                    
                    # --- Gráfico de Bandas de Bollinger (Reintroduzido) ---
                    if 'bb_upper' in df_completo.columns:
                        st.markdown('#### Bandas de Bollinger (20, 2)')
                        fig_bb = go.Figure()
                        
                        fig_bb.add_trace(go.Scatter(x=df_completo.index, y=df_completo['bb_upper'], name='Upper Band', line=dict(color='#95A5A6'), showlegend=False))
                        fig_bb.add_trace(go.Scatter(x=df_completo.index, y=df_completo['bb_lower'], name='Lower Band', line=dict(color='#95A5A6'), fill='tonexty', fillcolor='rgba(149, 165, 166, 0.1)', showlegend=False))
                        fig_bb.add_trace(go.Scatter(x=df_completo.index, y=df_completo['Close'], name='Close', line=dict(color='#2C3E50')))
                        
                        template = obter_template_grafico()
                        fig_bb.update_layout(**template)
                        # Remove título do template para evitar duplicação
                        fig_bb.update_layout(title_text='Bandas de Bollinger (20, 2)', height=400)
                        
                        st.plotly_chart(fig_bb, use_container_width=True)
                    
                    # --- Gráfico RSI ---
                    st.markdown('#### Índice de Força Relativa (RSI)')
                    fig_rsi = go.Figure(go.Scatter(x=df_completo.index, y=df_completo['rsi_14'], name='RSI', line=dict(color='#8E44AD')))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1); 
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
                    
                    template = obter_template_grafico()
                    fig_rsi.update_layout(**template)
                    # Remove título do template para evitar duplicação
                    fig_rsi.update_layout(title_text='Índice de Força Relativa (RSI)', height=300)
                    
                    st.plotly_chart(fig_rsi, use_container_width=True)
                    
                    # --- Gráfico MACD ---
                    st.markdown('#### Convergência/Divergência de Média Móvel (MACD)')
                    fig_macd = make_subplots(rows=1, cols=1)
                    # Certifique-se que 'macd' e 'macd_signal' estão no DF
                    if 'macd' in df_completo.columns and 'macd_signal' in df_completo.columns:
                        # CORREÇÃO: Usando df_completo em vez de df (causa do NameError)
                        fig_macd.add_trace(go.Scatter(x=df_completo.index, y=df_completo['macd'], name='MACD', line=dict(color='#2980B9')))
                        fig_macd.add_trace(go.Scatter(x=df_completo.index, y=df_completo['macd_signal'], name='Signal', line=dict(color='#E74C3C')))
                        fig_macd.add_trace(go.Bar(x=df_completo.index, y=df_completo['macd_diff'], name='Histograma', marker_color='#BDC3C7'))
                    
                        template = obter_template_grafico()
                        fig_macd.update_layout(**template)
                        # Remove título do template para evitar duplicação
                        fig_macd.update_layout(title_text='Converência/Divergência de Média Móvel (MACD)', height=300)
                        
                        st.plotly_chart(fig_macd, use_container_width=True)
                    else:
                         st.info("Dados de MACD insuficientes para plotar.")

                    
                else: st.warning("Análise Técnica não disponível sem histórico de preços.")

            # CORREÇÃO: Utiliza a variável tab_ml para o bloco condicional
            if tab_ml is not None:
                with tab_ml:
                    st.markdown("### Predição de Machine Learning")
                    
                    # Se veio do ML Real (com preço) ou do Fallback (com proxy fundamentalista)
                    ml_proba = df_completo['ML_Proba'].iloc[-1] if 'ML_Proba' in df_completo.columns else 0.5
                    ml_conf = df_completo['ML_Confidence'].iloc[-1] if 'ML_Confidence' in df_completo.columns else 0.0
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Probabilidade Média de Alta", f"{ml_proba*100:.1f}%")
                    
                    # NOVO: Apenas se AUC > 0.0, mostra que foi treinado
                    if ml_conf > 0.0:
                        col2.metric("Confiança do Modelo (AUC)", f"{ml_conf:.2f}")
                        st.info(f"ℹ️ **Modelo Supervisionado Ativo:** O score reflete a **MÉDIA** da probabilidade de alta do ativo nos {len(get_ml_horizons(st.session_state.profile.get('ml_lookback_days', 252)))} horizontes, conforme previsto pelo modelo. Confiança validada via AUC de teste.")
                    else:
                         col2.metric("Confiança do Modelo (AUC)", "N/A (Falha de Treinamento)")
                         st.warning("⚠️ **Modelo ML Falhou:** Não foi possível treinar o modelo supervisionado (dados insuficientes ou classes desbalanceadas). A predição não está disponível.")
                        
                    if df_ml_meta is not None and not df_ml_meta.empty:
                        st.markdown("#### Importância dos Fatores na Decisão")
                        fig_imp = px.bar(df_ml_meta.head(6), x='importance', y='feature', orientation='h', title='Top Fatores')
                        
                        template = obter_template_grafico()
                        fig_imp.update_layout(**template)
                        fig_imp.update_layout(height=300)
                        st.plotly_chart(fig_imp, use_container_width=True)

            with tab5: 
                st.markdown("### 🔬 Clusterização Geral (Ibovespa)")
                
                st.info(f"Analisando similaridade do **{ativo_selecionado.replace('.SA', '')}** com **TODOS** os ativos do Ibovespa (Baseado apenas em Fundamentos).")
                
                # 2. Coleta e Clusteriza (Apenas Fundamentos - Lista Global)
                resultado_cluster, n_clusters = AnalisadorIndividualAtivos.realizar_clusterizacao_fundamentalista_geral(coletor, ativo_selecionado)
                
                if resultado_cluster is not None:
                    st.success(f"Identificados {n_clusters} grupos (clusters) de qualidade fundamentalista.")
                    
                    # --- NOVO: Gráfico 3D (Com formatação de cor/símbolo) ---
                    df_plot = resultado_cluster.copy().reset_index().rename(columns={'index': 'Ticker'})
                    
                    # 1. Determina a cor: Ticker Selecionado (preto) ou Cluster
                    df_plot['Cor'] = df_plot['Cluster'].astype(str)
                    df_plot.loc[df_plot['Ticker'] == ativo_selecionado, 'Cor'] = 'Ativo Selecionado'
                    
                    # 2. Determina o formato: Anomalia (Losango) ou Normal (Círculo)
                    df_plot['Formato'] = df_plot['Anomalia'].astype(str).replace({'1': 'Normal', '-1': 'Anomalia'})
                    
                    # Mapeamento de cores para manter consistência, com o ativo selecionado em preto
                    cluster_colors = obter_template_grafico()['colorway']
                    color_map = {str(i): cluster_colors[i % len(cluster_colors)] for i in range(n_clusters)}
                    color_map['Ativo Selecionado'] = 'black'
                    
                    # Mapeamento de formato
                    symbol_map = {'Normal': 'circle', 'Anomalia': 'diamond'}
                    
                    # Garante que PC1, PC2 e PC3 existem
                    if 'PC3' in df_plot.columns: # Verifica se PCA 3D é possível
                        
                        fig_combined = px.scatter_3d(
                            df_plot, 
                            x='PC1', 
                            y='PC2', 
                            z='PC3',
                            color='Cor',
                            symbol='Formato',
                            hover_name=df_plot['Ticker'].str.replace('.SA', ''), 
                            color_discrete_map=color_map,
                            symbol_map=symbol_map,
                            opacity=0.8,
                            title="Mapa de Similaridade Fundamentalista (PCA 3D: Cluster e Anomalia)"
                        )
                        
                        template = obter_template_grafico()
                        fig_combined.update_layout(
                            title=template['title'],
                            paper_bgcolor=template['paper_bgcolor'],
                            plot_bgcolor=template['plot_bgcolor'],
                            font=template['font'],
                            scene=dict(xaxis_title='PCA 1', yaxis_title='PCA 2', zaxis_title='PCA 3'),
                            margin=dict(l=0, r=0, b=0, t=40),
                            height=600
                        )

                        # Aplica o formato e tamanho para todos os pontos
                        fig_combined.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
                        
                        # Aplica o formato específico para o ativo selecionado (cor preta forçada)
                        fig_combined.update_traces(selector=dict(name='Ativo Selecionado'), 
                                                   marker=dict(size=14, line=dict(width=2), 
                                                               color='black', symbol='circle'))
                        
                        st.plotly_chart(fig_combined, use_container_width=True)
                        
                    else:
                        st.warning("Dados PCA insuficientes (menos de 3 componentes) para gerar o gráfico 3D. Exibindo 2D.")
                        # Se 3D falhar, mostra o 2D (fallback)
                        if 'PC2' in df_plot.columns:
                            st.markdown('#### Detecção de Anomalias (Visualização 2D - Fallback)')
                            fig_anomaly_2d = px.scatter(
                                df_plot, x='PC1', y='PC2',
                                color='Cor',
                                symbol='Formato',
                                hover_name=df_plot['Ticker'].str.replace('.SA', ''), 
                                color_discrete_map=color_map,
                                symbol_map=symbol_map,
                                title="Detecção de Anomalias (Isolation Forest) no Espaço PCA 2D"
                            )
                            template = obter_template_grafico()
                            fig_anomaly_2d.update_layout(**template)
                            fig_anomaly_2d.update_layout(height=500)
                            fig_anomaly_2d.update_traces(marker=dict(size=10))
                            fig_anomaly_2d.update_traces(selector=dict(name='Ativo Selecionado'), marker=dict(size=14))
                            st.plotly_chart(fig_anomaly_2d, use_container_width=True)
                        else:
                             st.warning("Dados PCA insuficientes (menos de 2 componentes) para qualquer gráfico de dispersão.")

                    
                    # Tabela de Anomalia
                    st.markdown('##### Tabela de Scores de Anomalia (Quanto maior, menos anômalo)')
                    df_anomaly_show = resultado_cluster[['Anomaly_Score', 'Anomalia', 'Cluster']].copy()
                    df_anomaly_show.rename(columns={'Anomaly_Score': 'Score Anomalia', 'Anomalia': 'Status', 'Cluster': 'Grupo Cluster'}, inplace=True)
                    df_anomaly_show['Status'] = df_anomaly_show['Status'].replace({1: 'Normal', -1: 'Outlier/Anomalia'})
                    
                    st.dataframe(df_anomaly_show.sort_values('Score Anomalia', ascending=False), use_container_width=True)
                    st.markdown("---")
                    
                    # Identifica pares
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
                        st.warning("Ativo não encontrado no mapa de clusters (provavelmente sem dados suficientes).")
                else:
                    st.warning("Dados insuficientes para gerar clusters confiáveis.")
        
        except Exception as e:
            st.error(f"Erro ao analisar o ticker {ativo_selecionado}: {str(e)}")
            st.code(traceback.format_exc())

def aba_referencias():
    """Aba 5: Referências Bibliográficas Completas (V8.7 Original)"""
    
    st.markdown("## 📚 Referências e Bibliografia")
    st.markdown("Esta seção consolida as referências bibliográficas indicadas nas ementas das disciplinas relacionadas (GRDECO222 e GRDECO203).")

    st.markdown("---")
    
    st.markdown("### GRDECO222: Machine Learning (Prof. Rafael Martins de Souza)")
    
    st.markdown("**Bibliografia Obrigatória**")
    
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
