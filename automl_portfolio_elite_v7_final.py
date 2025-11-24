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

Versão: 9.32.27 (Update: WIDE SELECTION METRICS, DETAILED RANKING, CLUSTER VISUALIZATION UNIFIED)
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
from sklearn.metrics import silhouette_score, roc_auc_score
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
        df['returns'] = df['Close'].pct_change()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        # Evita divisão por zero
        rs = gain / loss.replace(0, np.nan)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # Volatilidade e Momentum
        df['vol_20d'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        df['momentum_60'] = df['Close'].pct_change(60)
        
        # Bollinger Bands Width
        rolling_mean = df['Close'].rolling(window=20).mean()
        rolling_std = df['Close'].rolling(window=20).std()
        upper = rolling_mean + (rolling_std * 2)
        lower = rolling_mean - (rolling_std * 2)
        # Evita divisão por zero/NaN
        df['bb_width'] = (upper - lower) / rolling_mean.replace(0, np.nan)
        
        return df

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
        ALL_TECH_FEATURES = ['rsi_14', 'macd_diff', 'vol_20d', 'momentum_10', 'sma_50', 'momentum_60', 'bb_width']
        ALL_FUND_FEATURES = ['pe_ratio', 'pb_ratio', 'div_yield', 'roe', 'pe_rel_sector', 'pb_rel_sector', 'Cluster', 'roic', 'net_margin', 'debt_to_equity', 'current_ratio', 'revenue_growth', 'ev_ebitda', 'operating_margin']
        
        if is_price_data_available:
            log_debug(f"Análise Individual ML: Iniciando modelo supervisionado para {ativo_selecionado}.")
            try:
                df = df_tec.copy()
                # O enrichment já foi feito no coletor, mas garantimos que as colunas existem
                
                df['Future_Direction'] = np.where(df['Close'].pct_change(5).shift(-5) > 0, 1, 0)
                
                # Identifica features válidas para este ativo
                available_features = [f for f in (ALL_TECH_FEATURES + ALL_FUND_FEATURES) if f in df.columns]
                
                # Remove NaNs apenas para as features de interesse
                df_model = df.dropna(subset=available_features + ['Future_Direction'])

                if len(df_model) > 50:
                    log_debug(f"ML Individual: {len(df_model)} pontos válidos para treino. Treinando RF...")

                    X = df_model[available_features].iloc[:-5]
                    y = df_model['Future_Direction'].iloc[:-5]
                    
                    # Filtra features que ainda são todas NaN após o dropna (impossível se dropna for bem sucedido, mas seguro)
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
        ALL_TECH_FEATURES = ['rsi_14', 'macd_diff', 'vol_20d', 'momentum_10', 'sma_50', 'momentum_60', 'bb_width']
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
            # FALLBACK: Se as colunas fundamentais não existirem (Pynvest falhou)
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
        kmeans = KMeans(n_clusters=min(len(data_pca), 4), random_state=42, n_init=10)
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
                    combined.loc[symbol, 'vol_current'] = df['vol_20d'].iloc[-1]
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
    st.set_page_config(page_title="Sistema de Portfólios Adaptativos", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
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
            background-color: #f8f9fa; 
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
    
    st.markdown("## 📚 Manual Completo e Metodologia do Sistema")
    
    st.markdown("""
    <div class="info-box" style="text-align: center;">
    <h3>🎯 O Que é este Sistema?</h3>
    <p>Este é um <b>Robo-Advisor Quantitativo Híbrido</b> projetado para o mercado brasileiro (B3). Ele utiliza uma abordagem sistemática para eliminar o viés emocional da seleção de ativos.</p>
    <p><b>Objetivo Principal:</b> Construir uma carteira de investimentos otimizada (geralmente 5 ativos) que busque a melhor relação risco-retorno, adaptando-se dinamicamente ao perfil do investidor e às condições de mercado em tempo real.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("1. O 'Cérebro' do Sistema (Motor de Decisão)")
    st.write("O sistema utiliza uma abordagem de **Decisão Multicritério**. Ele avalia cada ativo sob 4 pilares fundamentais simultaneamente. Dependendo da disponibilidade de dados em tempo real, ele alterna automaticamente entre dois modos de operação para garantir robustez:")

    with st.expander("🧠 MODO A: Tradicional (Análise Completa Híbrida) - O Ideal"):
        st.markdown("""
        Este é o modo padrão ativado quando o sistema consegue acessar **todos** os dados necessários (Preços Históricos + Balanços das Empresas).
        
        A nota final (Score) de cada ativo é composta por:

        | Pilar | Peso | O que é analisado? | Racional Econômico |
        | :--- | :--- | :--- | :--- |
        | **1. Performance** | **20% (Fixo)** | **Índice Sharpe** e **Retorno Anual**. | Identifica ativos que historicamente entregaram retorno excedente ajustado ao risco. |
        | **2. Machine Learning** | **20% (Fixo)** | **Probabilidade Estatística**. | Um modelo *Random Forest* (Ensemble) processa padrões não-lineares para estimar a probabilidade de alta futura. |
        | **3. Fundamentos** | **Varia (30% dos 60% restantes)** | **Saúde Financeira** (P/L, ROE, Margens). | Avalia a qualidade intrínseca do negócio. |
        | **4. Técnicos** | **Varia (30% dos 60% restantes)** | **Momentum** (RSI, MACD, Volatilidade). | Avalia o timing e a tendência de curto prazo. |
        
        *Os pesos dos pilares Fundamentalista e Técnico variam sobre os 60% restantes da pontuação, ajustados dinamicamente com base no horizonte de investimento selecionado pelo usuário.*
        """)

    with st.expander("🛡️ MODO B: Fallback (Segurança Fundamentalista) - Alta Resiliência"):
        st.markdown("""
        A coleta de dados financeiros em tempo real via APIs públicas pode apresentar instabilidades momentâneas.
        
        **Mecanismo de Fail-Fast:** Se o sistema detectar falhas consecutivas na obtenção de preços, ele ativa automaticamente o **Modo de Segurança**.
        
        Neste modo:
        1.  **Exclusão de Métricas de Preço:** Gráficos, volatilidade e modelos de ML (que dependem de séries temporais) são desativados para evitar dados corrompidos.
        2.  **Foco em Qualidade (Quality Investing):** O Score do ativo passa a ser **100% derivado dos Fundamentos** (Balanço Patrimonial, DRE, Fluxo de Caixa), obtidos de fontes alternativas.
        3.  **ML Não-Supervisionado:** Utilizamos algoritmos de Clusterização e Detecção de Anomalias (Isolation Forest) para identificar ativos "fora da curva" (Outliers positivos) sem depender de previsão de preço.
        
        *Este mecanismo garante que o usuário sempre receba uma recomendação de investimento válida baseada em dados concretos, mesmo em cenários de falha de API.*
        """)

    st.markdown("---")
    st.subheader("2. O Processo de Seleção (Funil de Investimento)")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="info-box" style="text-align: center;">
        <h4>1️⃣ Coleta de Dados</h4>
        <p style="font-size: 0.9rem;">Mineração em tempo real de ~90 ativos do Ibovespa.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-box" style="text-align: center;">
        <h4>2️⃣ Scoring & Filtro</h4>
        <p style="font-size: 0.9rem;">Cálculo de indicadores e eliminação de ativos fracos.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="info-box" style="text-align: center;">
        <h4>3️⃣ Otimização Final</h4>
        <p style="font-size: 0.9rem;">Combinação matemática para diversificação eficiente.</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🔍 Detalhe Técnico: A Clusterização (K-Means)"):
        st.write("""
        Para garantir uma diversificação real, o sistema não seleciona apenas os "Top 5" melhores Scores. Isso poderia resultar em uma carteira concentrada (ex: 5 bancos).
        
        **O Método:**
        Utilizamos um algoritmo de aprendizado não-supervisionado (**K-Means Clustering**) aplicado sobre os componentes principais (**PCA**) dos indicadores das empresas.
        
        1.  O algoritmo agrupa as empresas em "Clusters" (famílias) com comportamento estatístico e fundamentalista semelhante.
        2.  Na seleção final, o sistema é forçado a escolher o melhor ativo de **clusters diferentes**.
        
        **Resultado:** Uma carteira estatisticamente descorrelacionada, reduzindo o risco sistêmico setorial.
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
        st.success(f"✓ **{len(ativos_selecionados)} ativos** (Ibovespa completo) definidos para análise.")
        
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
            col_metrics_s = st.columns(len(setores_selecionados) if len(setores_selecionados) > 0 else 2) # Cria colunas dinâmicas para expandir
            
            # Garante que temos pelo menos duas colunas para Tickers e Setores
            num_cols = 2 if len(setores_selecionados) == 0 else 2
            col_metrics_s = st.columns(num_cols)
            
            with col_metrics_s[0]:
                st.metric("Setores Selecionados", len(setores_selecionados))
            with col_metrics_s[-1]:
                st.metric("Total de Ativos", len(ativos_selecionados))
            
            with st.expander("📋 Visualizar Ativos por Setor"):
                for setor in setores_selecionados:
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
        
        # BARRA DE SELEÇÃO INDIVIDUAL
        st.markdown("#### 📝 Selecione Tickers (Ibovespa)")
        ativos_selecionados = st.multiselect(
            "Pesquise e selecione os tickers:",
            options=todos_tickers_ibov,
            format_func=lambda x: f"{x.replace('.SA', '')} - {ativos_com_setor.get(x, 'Desconhecido')}",
            key='ativos_individuais_multiselect_v8'
        )
        
        # NOVO: Centraliza e expande a métrica abaixo do multiselect (lateralidade total)
        st.markdown("#### Tickers Selecionados")
        col_metrics_i1, col_metrics_i2, col_metrics_i3 = st.columns([1, 2, 1]) # Usamos 1, 2, 1 para centralizar e expandir
        with col_metrics_i2:
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
        
        st.success("✓ Definição concluída. Prossiga para a aba **'Construtor de Portfólio'**.")
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
    
    # REMOVIDO: progress_bar_placeholder = st.empty()
    
    if not st.session_state.builder_complete:
        st.markdown('## 📋 Calibração do Perfil de Risco')
        
        st.info(f"✓ **{len(st.session_state.ativos_para_analise)} ativos** prontos. Responda o questionário para calibrar a otimização.")
        
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
                    return

                # NOVO: Barra de progresso visível logo após o submit
                progress_widget = progress_bar_placeholder.progress(0, text=f"Iniciando pipeline para PERFIL {risk_level}...")
                
                success = builder_local.executar_pipeline(
                    simbolos_customizados=st.session_state.ativos_para_analise,
                    perfil_inputs=st.session_state.profile,
                    progress_bar=progress_widget
                )
                
                progress_widget.empty() # Limpa após o sucesso
                    
                if not success:
                    st.error("Falha na aquisição ou processamento dos dados.")
                    st.session_state.builder = None; st.session_state.profile = {}; return
                
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
        has_garch_data = builder.volatilidades_garch.values() and any(v > 0.05 for v in builder.volatilidades_garch.values()) 
        
        # FIX 2: A aba ML só é exibida se houver resultados de ML utilizáveis (soma do score ponderado > 0)
        has_usable_ml = builder.scores_combinados['ml_score_weighted'].sum() > 0 if not builder.scores_combinados.empty else False
        
        # --- NOVO: Verificação de Redundância GARCH ---
        is_garch_redundant = False
        if has_price_data:
            distinct_garch_count = 0
            for ativo in assets:
                if ativo in builder.metricas_performance.index and ativo in builder.volatilidades_garch:
                    vol_hist = builder.metricas_performance.loc[ativo].get('volatilidade_anual', np.nan)
                    vol_garch = builder.volatilidades_garch.get(ativo)
                    # Checa se a diferença é maior que 1 ponto percentual
                    if not np.isnan(vol_hist) and not np.isnan(vol_garch) and abs(vol_garch - vol_hist) > 0.01:
                        distinct_garch_count += 1
            
            if distinct_garch_count == 0:
                is_garch_redundant = True
                log_debug("GARCH: Ocultando aba GARCH. Nenhuma diferença significativa em relação à Volatilidade Histórica (Redundante).")
        # --- FIM NOVO ---
        
        # Define as abas (agora consolidada)
        tabs_list = ["📊 Alocação de Capital", "📈 Performance e Retornos", "🔬 Análise de Fatores e Clusterização"]
        
        if has_price_data and has_garch_data and not is_garch_redundant:
             tabs_list.append("📉 Fator Volatilidade GARCH")
             
        tabs_list.append("❓ Justificativas e Ranqueamento")

        # Mapeia os índices das abas
        tabs_map = st.tabs(tabs_list)
        tab1 = tabs_map[0] # Alocação
        tab2 = tabs_map[1] # Performance/Retornos
        tab_fator_cluster = tabs_map[2] # Aba consolidada de ML/Clusters
        
        # Atribuição dinâmica das últimas abas
        if "📉 Fator Volatilidade GARCH" in tabs_list:
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
                             
                        alloc_table.append({
                            'Ticker': asset.replace('.SA', ''), 
                            'Peso (%)': f"{weight * 100:.2f}",
                            'Valor (R$)': f"R$ {amount:,.2f}",
                            'Score Total': f"{total_score:.3f}" if not np.isnan(total_score) else 'N/A', # Adicionado
                            'Setor': sector,                                                           # Adicionado
                            'Cluster': str(cluster_id),                                                # Adicionado
                        })
                
                df_alloc = pd.DataFrame(alloc_table)
                st.dataframe(df_alloc, use_container_width=True)
        
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
            if has_price_data and has_usable_ml:
                 st.markdown('##### 🤖 Predição de Movimento Direcional (Random Forest)')
                 st.markdown("O modelo utiliza histórico de preços para prever a probabilidade de alta no curto prazo.")
                 title_text_plot = "Probabilidade de Alta (0-100%)"
                 
                 # Exibe os gráficos de barra de ML 
                 ml_data = []
                 for asset in assets:
                    if asset in builder.predicoes_ml:
                        ml_info = builder.predicoes_ml[asset]
                        ml_data.append({
                            'Ticker': asset.replace('.SA', ''),
                            'Score/Prob.': ml_info.get('predicted_proba_up', 0.5) * 100,
                            'Confiança': ml_info.get('auc_roc_score', np.nan),
                            'Modelo': ml_info.get('model_name', 'N/A')
                        })
                 
                 df_ml = pd.DataFrame(ml_data)
            
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
                     st.warning("Não há dados de ML para exibir.")

            else:
                 # Exibe a tabela de clusters e scores se o modo fallback estiver ativo
                 st.markdown('##### 🔬 Análise de Qualidade Fundamentalista (Unsupervised Learning)')
                 st.info("ℹ️ **Modo Fallback Ativo:** O modelo de predição supervisionada não encontrou padrões significativos ou o histórico de preços é limitado. O sistema está utilizando a **Clusterização Fundamentalista** (qualidade) como fator ML dominante.")
                 
                 st.markdown("###### Score Fundamentalista e Cluster por Ativo")
                 if not builder.scores_combinados.empty:
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
                df_viz = builder.scores_combinados.loc[assets].copy().reset_index().rename(columns={'index': 'Ticker'})
                
                # Prepara dados para PCA (apenas scores)
                features_for_pca = ['performance_score', 'fundamental_score', 'technical_score', 'ml_score_weighted']
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
        
        # O bloco tab_garch só existe se has_price_data and has_garch_data and not is_garch_redundant for True
        if tab_garch is not None:
            with tab_garch:
                st.markdown('#### Volatilidade Condicional (GARCH) e Histórica')
                
                # IF ROBUSTO: SÓ MOSTRA SE TIVER PREÇO
                if has_price_data:
                    dados_garch = []
                    for ativo in assets:
                        if ativo in builder.metricas_performance.index and ativo in builder.volatilidades_garch:
                            perf = builder.metricas_performance.loc[ativo]
                            vol_hist = perf.get('volatilidade_anual', np.nan)
                            vol_garch = builder.volatilidades_garch.get(ativo)
                            
                            if vol_garch is not None and not np.isnan(vol_garch):
                                status = '✓ GARCH Ajustado (Previsão de Risco)'
                                vol_display = vol_garch
                            elif vol_hist is not None and not np.isnan(vol_hist): 
                                status = '⚠️ Histórica (Fallback)'
                                vol_display = vol_hist
                            else:
                                status = '❌ Indisponível'
                                vol_display = np.nan
                            
                            dados_garch.append({
                                'Ticker': ativo.replace('.SA', ''),
                                'Vol. Histórica (%)': vol_hist * 100 if not np.isnan(vol_hist) else 'N/A',
                                'Vol. Condicional (%)': vol_display * 100 if vol_display is not None and not np.isnan(vol_display) else 'N/A',
                                'Status de Cálculo': status
                            })
                    
                    df_garch = pd.DataFrame(dados_garch)
                    
                    if not df_garch.empty:
                        fig_garch = go.Figure()
                        
                        # Safe filtering for plotting
                        plot_df_garch = df_garch[df_garch['Vol. Condicional (%)'] != 'N/A'].copy()
                        if not plot_df_garch.empty:
                            plot_df_garch['Vol. Condicional (%)'] = plot_df_garch['Vol. Condicional (%)'].astype(float)
                            plot_df_garch['Vol. Histórica (%)'] = plot_df_garch['Vol. Histórica (%)'].apply(lambda x: float(x) if x != 'N/A' else np.nan)

                            template_colors = obter_template_grafico()['colorway']
                            
                            fig_garch.add_trace(go.Bar(name='Volatilidade Histórica', x=plot_df_garch['Ticker'], y=plot_df_garch['Vol. Histórica (%)'], marker=dict(color=template_colors[2]), opacity=0.7)) 
                            fig_garch.add_trace(go.Bar(name='Volatilidade Condicional', x=plot_df_garch['Ticker'], y=plot_df_garch['Vol. Condicional (%)'], marker=dict(color=template_colors[0]))) 
                            
                            template = obter_template_grafico()
                            fig_garch.update_layout(**template)
                            fig_garch.update_layout(title_text="Volatilidade Anualizada: Histórica vs. Condicional (GARCH)", yaxis_title="Volatilidade Anual (%)", barmode='group', height=400)
                            
                            st.plotly_chart(fig_garch, use_container_width=True)
                        else:
                            st.info("Dados de volatilidade insuficientes para gráfico.")

                        st.dataframe(df_garch, use_container_width=True, hide_index=True)
                    else:
                        st.warning("Não há dados de volatilidade para exibir.")
                else:
                     st.warning("⚠️ Dados de preços insuficientes para calcular volatilidade histórica ou GARCH.")
            
            with tab_justificativas:
                st.markdown('#### Ranqueamento Final e Justificativas Detalhadas')
                
                st.markdown(f"**Pesos Adaptativos Usados:** Performance: {builder.pesos_atuais['Performance']:.2f} | Fundamentos: {builder.pesos_atuais['Fundamentos']:.2f} | Técnicos: {builder.pesos_atuais['Técnicos']:.2f} | ML: {builder.pesos_atuais['ML']:.2f}")
                st.markdown("---")
                
                # Safe Rename Logic (Check columns existence first)
                rename_map = {
                    'total_score': 'Score Total', 
                    'performance_score': 'Score Perf.', 
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
                    'ML_Proba': 'Prob. Alta ML'
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
                                'rsi_14': last_row.get('rsi_14'),
                                'macd_diff': last_row.get('macd_diff'),
                                'ML_Proba': last_row.get('ML_Proba'),
                            }
                    df_last_data = pd.DataFrame.from_dict(data_at_last_idx, orient='index')
                    
                    df_scores_display = df_full_data.join(df_last_data, how='left')

                    # Adicionando/Renomeando colunas
                    cols_to_display = list(rename_map.keys())
                    cols_to_display = [col for col in cols_to_display if col in df_scores_display.columns]

                    df_scores_display = df_scores_display[cols_to_display].copy()
                    df_scores_display.rename(columns=rename_map, inplace=True)
                    
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
                        elif 'P/L' in col or 'P/VP' in col or 'MACD' in col: format_dict[col] = '{:.2f}'
                        elif 'RSI' in col: format_dict[col] = '{:.2f}'
                        elif 'Prob' in col: format_dict[col] = '{:.2f}'
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
                        weight = builder.alocacao_portfolio.get(asset, {}).get('weight', 0)
                        st.markdown(f"""
                        <div class="info-box">
                        <h4>{asset.replace('.SA', '')} ({weight*100:.2f}%)</h4>
                        <p><strong>Fatores-Chave:</strong> {justification}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Botão Recalibrar Centralizado no Final
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
    
    # NOVO: Botões Centralizados (Executar Análise e Limpar Análise)
    col_btn_start, col_btn_center, col_btn_end = st.columns([1, 2, 1])
    with col_btn_center:
        col_exec, col_clear = st.columns(2)
        with col_exec:
            if st.button("🔄 Executar Análise", key='analyze_asset_button_v8', type="primary", use_container_width=True):
                st.session_state.analisar_ativo_triggered = True 
        with col_clear:
            if st.button("🗑️ Limpar Análise", key='clear_asset_analysis_button_v8', type="secondary", use_container_width=True):
                 st.session_state.analisar_ativo_triggered = False # Resetar o trigger
                 st.rerun()
    
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
            
            # Verifica se o ML supervisionado foi ignorado
            # A confidence de 0.55 é o limite neutro/baixo, se for maior, consideramos treinado
            is_ml_trained = 'ML_Proba' in df_completo.columns and not static_mode and df_completo['ML_Confidence'].iloc[-1] > 0.55

            if static_mode:
                st.warning(f"⚠️ **MODO ESTÁTICO:** Preços indisponíveis. Exibindo apenas Análise Fundamentalista.")
                
            # Define a lista de abas, excluindo ML se não foi treinado
            tabs_list_individual = ["📊 Visão Geral", "💼 Fundamentos", "🔧 Análise Técnica", "🔬 Clusterização Geral"]
            if is_ml_trained:
                tabs_list_individual.insert(3, "🤖 Machine Learning") # Insere a aba ML se treinou

            tabs_map = st.tabs(tabs_list_individual)
            
            tab_map_index = lambda title: tabs_list_individual.index(title)
            
            tab1 = tabs_map[tab_map_index("📊 Visão Geral")]
            tab2 = tabs_map[tab_map_index("💼 Fundamentos")]
            tab3 = tabs_map[tab_map_index("🔧 Análise Técnica")]
            tab5 = tabs_map[tab_map_index("🔬 Clusterização Geral")]
            
            # CORREÇÃO: Define tab_ml condicionalmente para evitar UnboundLocalError
            tab_ml = tabs_map[tab_map_index("🤖 Machine Learning")] if is_ml_trained else None
            
            # Abas 1-4: Lógica Padrão de Exibição (igual à versão anterior)
            with tab1:
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
                    col1.metric("RSI (14)", f"{df_completo['rsi_14'].iloc[-1]:.2f}" if 'rsi_14' in df_completo else "N/A")
                    col2.metric("MACD Diff", f"{df_completo['macd_diff'].iloc[-1]:.4f}" if 'macd_diff' in df_completo else "N/A")
                    col3.metric("BB Width", f"{df_completo['bb_width'].iloc[-1]:.2f}" if 'bb_width' in df_completo else "N/A")
                    
                    # Gráfico RSI
                    fig_rsi = go.Figure(go.Scatter(x=df_completo.index, y=df_completo['rsi_14'], name='RSI', line=dict(color='#8E44AD')))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red"); fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                    
                    template = obter_template_grafico()
                    template['title']['text'] = "RSI (14)" 
                    fig_rsi.update_layout(**template)
                    fig_rsi.update_layout(height=300)
                    
                    st.plotly_chart(fig_rsi, use_container_width=True)
                    
                    # Gráfico MACD
                    fig_macd = make_subplots(rows=1, cols=1)
                    fig_macd.add_trace(go.Scatter(x=df_completo.index, y=df_completo['macd'], name='MACD', line=dict(color='#2980B9')))
                    fig_macd.add_trace(go.Scatter(x=df_completo.index, y=df_completo['macd_signal'], name='Signal', line=dict(color='#E74C3C')))
                    fig_macd.add_trace(go.Bar(x=df_completo.index, y=df_completo['macd_diff'], name='Histograma', marker_color='#BDC3C7'))
                    
                    template = obter_template_grafico()
                    template['title']['text'] = "MACD (12,26,9)"
                    fig_macd.update_layout(**template)
                    fig_macd.update_layout(height=300)
                    
                    st.plotly_chart(fig_macd, use_container_width=True)
                    
                    # Gráfico BB
                    rolling_mean = df_completo['Close'].rolling(window=20).mean()
                    rolling_std = df_completo['Close'].rolling(window=20).std()
                    upper_band = rolling_mean + (rolling_std * 2)
                    lower_band = rolling_mean - (rolling_std * 2)
                    
                    fig_bb = go.Figure()
                    fig_bb.add_trace(go.Scatter(x=df_completo.index, y=upper_band, name='Upper Band', line=dict(color='#95A5A6'), showlegend=False))
                    fig_bb.add_trace(go.Scatter(x=df_completo.index, y=lower_band, name='Lower Band', line=dict(color='#95A5A6'), fill='tonexty', fillcolor='rgba(149, 165, 166, 0.1)', showlegend=False))
                    fig_bb.add_trace(go.Scatter(x=df_completo.index, y=df_completo['Close'], name='Close', line=dict(color='#2C3E50')))
                    
                    template = obter_template_grafico()
                    template['title']['text'] = "Bandas de Bollinger (20, 2)"
                    fig_bb.update_layout(**template)
                    fig_bb.update_layout(height=400)
                    
                    st.plotly_chart(fig_bb, use_container_width=True)
                    
                else: st.warning("Análise Técnica não disponível sem histórico de preços.")

            # CORREÇÃO: Utiliza a variável tab_ml para o bloco condicional
            if tab_ml is not None:
                with tab_ml:
                    st.markdown("### Predição de Machine Learning")
                    
                    # Se veio do ML Real (com preço) ou do Fallback (com proxy fundamentalista)
                    ml_proba = df_completo['ML_Proba'].iloc[-1] if 'ML_Proba' in df_completo.columns else 0.5
                    ml_conf = df_completo['ML_Confidence'].iloc[-1] if 'ML_Confidence' in df_completo.columns else 0.0
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Probabilidade de Alta" if not static_mode else "Score de Anomalia Positiva", f"{ml_proba*100:.1f}%")
                    
                    if static_mode:
                        col2.metric("Status", "Fallback Não-Supervisionado")
                        st.info("ℹ️ **Modo Fallback Ativo:** O modelo não encontrou histórico de preços. Este score reflete uma análise de anomalia (Isolation Forest) baseada nos fundamentos. Um score alto indica que o ativo se destaca positivamente em relação aos seus pares.")
                    elif ml_conf <= 0.55:
                         col2.metric("Confiança do Modelo (AUC)", f"{ml_conf:.2f}")
                         st.info("ℹ️ **Modelo Supervisionado Neutro:** A confiança do modelo é baixa (AUC ≤ 0.55). O score de probabilidade exibido foi ajustado para um **Proxy Fundamentalista** para garantir relevância na classificação.")
                    else:
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

            with tab5: 
                st.markdown("### 🔬 Clusterização Geral (Ibovespa)")
                
                st.info(f"Analisando similaridade do **{ativo_selecionado.replace('.SA', '')}** com **TODOS** os ativos do Ibovespa (Baseado apenas em Fundamentos).")
                
                # 2. Coleta e Clusteriza (Apenas Fundamentos - Lista Global)
                resultado_cluster, n_clusters = AnalisadorIndividualAtivos.realizar_clusterizacao_fundamentalista_geral(coletor, ativo_selecionado)
                
                if resultado_cluster is not None:
                    st.success(f"Identificados {n_clusters} grupos (clusters) de qualidade fundamentalista.")
                    
                    # --- NOVO: Gráfico ÚNICO (PCA 2D com Formato/Cor Condicional) ---
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
                    
                    # Garante que PC1 e PC2 existem
                    if 'PC1' in df_plot.columns and 'PC2' in df_plot.columns:
                        
                        fig_combined = px.scatter(
                            df_plot, 
                            x='PC1', 
                            y='PC2',
                            color='Cor',
                            symbol='Formato',
                            hover_name=df_plot['Ticker'].str.replace('.SA', ''), 
                            color_discrete_map=color_map,
                            symbol_map=symbol_map,
                            title="Mapa de Similaridade Fundamentalista (PCA 2D: Cluster e Anomalia)"
                        )
                        
                        template = obter_template_grafico()
                        fig_combined.update_layout(**template)
                        fig_combined.update_layout(height=600)
                        
                        # Aumenta o tamanho do ponto do ativo selecionado
                        fig_combined.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
                        fig_combined.update_traces(selector=dict(name='Ativo Selecionado'), marker=dict(size=14, line=dict(width=2)))
                        
                        st.plotly_chart(fig_combined, use_container_width=True)
                        
                    else:
                        st.warning("Dados PCA insuficientes para gerar o gráfico 2D.")

                    
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

    st.markdown('<h1 class="main-header">Sistema de Portfólios Adaptativos</h1>', unsafe_allow_html=True)
    
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
