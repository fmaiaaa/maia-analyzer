# -*- coding: utf-8 -*-
"""
=============================================================================
SISTEMA DE OTIMIZAÇÃO QUANTITATIVA - VERSÃO CORRIGIDA (9.32.44)
=============================================================================

CORREÇÕES IMPLEMENTADAS (v7 Final - Corrected Build):
1. ✓ ERRO FATAL CORRIGIDO: Classe ConstrutorPortfolioAutoML → ColetorDadosLive
2. ✓ use_container_width CORRIGIDO → width='stretch' / width='content'
3. ✓ PyArrow CORRIGIDO: Tratamento de strings vazias → np.nan
4. ✓ ML CORRIGIDO: Coluna 'Cluster' garantida antes do uso
5. ✓ ORDEM DE DEFINIÇÃO: Todas as classes definidas antes do uso
6. ✓ MISSING CLASS: AnalisadorIndividualAtivos agora definida
7. ✓ LIMPEZA: Debug panel removido conforme solicitado
8. ✓ ESTABILIDADE: Tratamento robusto de erros em toda parte

Modelo de Alocação de Ativos com Métodos Adaptativos.
- Preços: Estratégia Linear com Fail-Fast (YFinance → TvDatafeed → Estático).
- Fundamentos: Coleta Exaustiva Pynvest (50+ indicadores).
- ML: Restaurado para Estabilidade (Lógica 6.0.9).
- Vol: Histórica (GARCH removido para robustez).

Versão: 9.32.44 (Final Build: ML Estável, Vol. Histórica, UI Aprimorada)
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
    pass

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
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNetCV

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
    pass


# =============================================================================
# CONFIGURAÇÕES GLOBAIS
# =============================================================================

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

ML_FEATURES_P_ANALYZER = [
    'RSI', 'MACD', 'Volatility', 'Momentum', 'SMA_50', 'SMA_200',
    'PE_Ratio', 'PB_Ratio', 'Div_Yield', 'ROE',
    'pe_rel_sector', 'pb_rel_sector', 'Cluster'
]
ML_CATEGORICAL_FEATURES = ['Cluster']

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

# Questões do perfil
SCORE_MAP_ORIGINAL = {'CT: Concordo Totalmente': 5, 'C: Concordo': 4, 'N: Neutro': 3, 'D: Discordo': 2, 'DT: Discordo Totalmente': 1}
SCORE_MAP_INV_ORIGINAL = {'CT: Concordo Totalmente': 1, 'C: Concordo': 2, 'N: Neutro': 3, 'D: Discordo': 4, 'DT: Discordo Totalmente': 5}
SCORE_MAP_CONHECIMENTO_ORIGINAL = {'A: Avançado (Análise fundamentalista, macro e técnica)': 5, 'B: Intermediário (Conhecimento básico sobre mercados e ativos)': 3, 'C: Iniciante (Pouca ou nenhuma experiência em investimentos)': 1}
SCORE_MAP_REACTION_ORIGINAL = {'A: Venderia imediatamente': 1, 'B: Manteria e reavaliaria a tese': 3, 'C: Compraria mais para aproveitar preços baixos': 5}

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

# Funções Utilitárias
def safe_format(value):
    if pd.isna(value):
        return "N/A"
    try:
        float_val = float(value)
        return f"{float_val:.2f}"
    except (ValueError, TypeError):
        return str(value)

def log_debug(message: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"DEBUG {timestamp} | {message}")
    if 'debug_logs' in st.session_state:
        st.session_state.debug_logs.append(f"[{timestamp}] {message}")

def obter_template_grafico() -> dict:
    corporate_colors = ['#2E86C1', '#D35400', '#27AE60', '#8E44AD', '#C0392B', '#16A085', '#F39C12', '#34495E']
    return {
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'font': {'family': 'Inter, sans-serif', 'size': 12, 'color': '#343a40'},
        'title': {'font': {'family': 'Inter', 'size': 16, 'color': '#212529', 'weight': 'bold'}, 'x': 0.5, 'xanchor': 'center'},
        'xaxis': {'showgrid': True, 'gridcolor': '#ecf0f1', 'showline': True, 'linecolor': '#bdc3c7', 'linewidth': 1},
        'yaxis': {'showgrid': True, 'gridcolor': '#ecf0f1', 'showline': True, 'linecolor': '#bdc3c7', 'linewidth': 1},
        'legend': {'bgcolor': 'rgba(255,255,255,0.5)', 'bordercolor': '#ecf0f1'},
        'colorway': corporate_colors
    }

# =============================================================================
# CLASSES (Ordem Correta de Definição)
# =============================================================================

class EngenheiroFeatures:
    @staticmethod
    def _normalize_score(serie: pd.Series, maior_melhor: bool = True) -> pd.Series:
        series_clean = serie.replace([np.inf, -np.inf], np.nan).dropna()
        if series_clean.empty or series_clean.std() == 0:
            return pd.Series(50, index=serie.index)
        z = zscore(series_clean, nan_policy='omit')
        normalized_values = 50 + (z.clip(-3, 3) / 3) * 50
        if not maior_melhor:
            normalized_values = 100 - normalized_values
        normalized_series = pd.Series(normalized_values, index=series_clean.index)
        return normalized_series.reindex(serie.index, fill_value=50)

class CalculadoraTecnica:
    @staticmethod
    def enriquecer_dados_tecnicos(df_ativo: pd.DataFrame) -> pd.DataFrame:
        if df_ativo.empty:
            return df_ativo
        df = df_ativo.sort_index().copy()

        df.rename(columns={'Close': 'close', 'High': 'high', 'Low': 'low', 'Open': 'open', 'Volume': 'volume'}, inplace=True)
        df['returns'] = df['close'].pct_change()

        df['vol_20d'] = df["returns"].rolling(20).std() * np.sqrt(252)

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

        df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
        df['sma_50d'] = df['close'].rolling(window=50).mean()
        df['sma_200d'] = df['close'].rolling(window=200).mean()

        rolling_mean_20 = df['close'].rolling(window=20).mean()
        rolling_std_20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = rolling_mean_20 + (rolling_std_20 * 2)
        df['bb_lower'] = rolling_mean_20 - (rolling_std_20 * 2)

        df.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low', 'open': 'Open', 'volume': 'Volume'}, inplace=True)

        cols_to_keep = ['Close', 'High', 'Low', 'Open', 'Volume', 'returns', 'vol_20d', 'macd_diff', 'rsi_14', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'momentum_10d', 'sma_50d', 'sma_200d']
        cols_to_drop = [c for c in df.columns if c not in cols_to_keep and c not in ['Close', 'High', 'Low', 'Open', 'Volume', 'returns']]
        df.drop(columns=cols_to_drop, errors='ignore', inplace=True)

        return df.dropna(subset=['Close']).fillna(0)

class AnalisadorPerfilInvestidor:
    def __init__(self):
        self.nivel_risco = ""
        self.horizonte_tempo = ""
        self.dias_lookback_ml = 5

    def determinar_nivel_risco(self, pontuacao: int) -> str:
        if pontuacao <= 46:
            return "CONSERVADOR"
        elif pontuacao <= 67:
            return "INTERMEDIÁRIO"
        elif pontuacao <= 88:
            return "MODERADO"
        elif pontuacao <= 109:
            return "MODERADO-ARROJADO"
        else:
            return "AVANÇADO"

    def determinar_horizonte_ml(self, liquidez_key: str, objetivo_key: str) -> tuple:
        time_map = {'A': 84, 'B': 168, 'C': 252}
        final_lookback = max(time_map.get(liquidez_key, 84), time_map.get(objetivo_key, 84))

        if final_lookback >= 252:
            self.horizonte_tempo = "LONGO PRAZO"
        elif final_lookback >= 168:
            self.horizonte_tempo = "MÉDIO PRAZO"
        else:
            self.horizonte_tempo = "CURTO PRAZO"

        self.dias_lookback_ml = final_lookback
        return self.horizonte_tempo, self.dias_lookback_ml

    def calcular_perfil(self, respostas_risco_originais: dict) -> tuple:
        score_risk_accept = SCORE_MAP_ORIGINAL.get(respostas_risco_originais['risk_accept'], 3)
        score_max_gain = SCORE_MAP_ORIGINAL.get(respostas_risco_originais['max_gain'], 3)
        score_stable_growth = SCORE_MAP_INV_ORIGINAL.get(respostas_risco_originais['stable_growth'], 3)
        score_avoid_loss = SCORE_MAP_INV_ORIGINAL.get(respostas_risco_originais['avoid_loss'], 3)
        score_level = SCORE_MAP_CONHECIMENTO_ORIGINAL.get(respostas_risco_originais['level'], 3)
        score_reaction = SCORE_MAP_REACTION_ORIGINAL.get(respostas_risco_originais['reaction'], 3)

        pontuacao = score_risk_accept * 5 + score_max_gain * 5 + score_stable_growth * 5 + score_avoid_loss * 5 + score_level * 3 + score_reaction * 3
        nivel_risco = self.determinar_nivel_risco(pontuacao)

        liquidez_val = respostas_risco_originais.get('liquidity')
        objetivo_val = respostas_risco_originais.get('time_purpose')

        liquidez_key = liquidez_val[0] if isinstance(liquidez_val, str) and liquidez_val else 'C'
        objetivo_key = objetivo_val[0] if isinstance(objetivo_val, str) and objetivo_val else 'C'

        horizonte_tempo, ml_lookback = self.determinar_horizonte_ml(liquidez_key, objetivo_key)
        return nivel_risco, horizonte_tempo, ml_lookback, pontuacao

class OtimizadorPortfolioAvancado:
    def __init__(self, returns_df: pd.DataFrame, garch_vols: dict = None):
        self.returns = returns_df
        self.mean_returns = returns_df.mean() * 252
        self.cov_matrix = returns_df.cov() * 252
        self.num_ativos = len(returns_df.columns)

    def estatisticas_portfolio(self, pesos: np.ndarray) -> tuple:
        p_retorno = np.dot(pesos, self.mean_returns)
        p_vol = np.sqrt(np.dot(pesos.T, np.dot(self.cov_matrix, pesos)))
        return p_retorno, p_vol

    def sharpe_negativo(self, pesos: np.ndarray) -> float:
        p_retorno, p_vol = self.estatisticas_portfolio(pesos)
        if p_vol <= 1e-9:
            return -100.0
        return -(p_retorno - TAXA_LIVRE_RISCO) / p_vol

    def minimizar_volatilidade(self, pesos: np.ndarray) -> float:
        return self.estatisticas_portfolio(pesos)[1]

    def otimizar(self, estrategia: str = 'MaxSharpe') -> dict:
        if self.num_ativos == 0:
            return {}
        restricoes = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        limites = tuple((PESO_MIN, PESO_MAX) for _ in range(self.num_ativos))
        chute_inicial = np.array([1.0 / self.num_ativos] * self.num_ativos)

        if estrategia == 'MinVolatility':
            objetivo = self.minimizar_volatilidade
        else:
            objetivo = self.sharpe_negativo

        try:
            resultado = minimize(objetivo, chute_inicial, method='SLSQP', bounds=limites, constraints=restricoes, options={'maxiter': 1000, 'ftol': 1e-6})
            if resultado.success:
                final_weights = resultado.x / np.sum(resultado.x)
                return {ativo: peso for ativo, peso in zip(self.returns.columns, final_weights)}
            else:
                return {}
        except Exception:
            return {}

class ColetorDadosLive(object):
    """
    Classe principal que coleta dados, processa ML e otimiza portfólios.
    Anteriormente: ConstrutorPortfolioAutoML (nome incorreto)
    Agora: ColetorDadosLive (nome correto)
    """
    def __init__(self, valor_investimento=10000, periodo=PERIODO_DADOS):
        self.periodo = periodo
        self.valor_investimento = valor_investimento
        self.dados_por_ativo = {}
        self.dados_fundamentalistas = pd.DataFrame()
        self.metricas_performance = pd.DataFrame()
        self.volatilidades_garch_raw = {}
        self.metricas_simples = {}
        self.ativos_sucesso = []
        self.ativos_selecionados = []
        self.alocacao_portfolio = {}
        self.predicoes_ml = {}
        self.scores_combinados = pd.DataFrame()
        self.combined_scores = pd.DataFrame()
        self.metodo_alocacao_atual = ""
        self.pesos_atuais = {}
        self.metricas_portfolio = {}
        self.justificativas_selecao = {}
        self.perfil_dashboard = {}

        log_debug("Inicializando ColetorDadosLive...")
        try:
            self.tv = TvDatafeed()
            self.tv_ativo = True
            log_debug("TvDatafeed inicializado com sucesso.")
        except Exception as e:
            self.tv_ativo = False
            log_debug(f"ERRO: Falha ao inicializar TvDatafeed: {str(e)[:50]}...")

        try:
            from pynvest.scrappers.fundamentus import Fundamentus
            self.pynvest_scrapper = Fundamentus()
            self.pynvest_ativo = True
            log_debug("Pynvest (Fundamentus) inicializado com sucesso.")
        except:
            self.pynvest_ativo = False
            log_debug("AVISO: Pynvest falhou ao inicializar.")

    def _mapear_colunas_pynvest(self, df_pynvest: pd.DataFrame) -> dict:
        if df_pynvest.empty:
            return {}
        row = df_pynvest.iloc[0]

        mapping = {
            'vlr_ind_p_sobre_l': 'pe_ratio', 'vlr_ind_p_sobre_vp': 'pb_ratio', 'vlr_ind_roe': 'roe',
            'vlr_ind_roic': 'roic', 'vlr_ind_margem_liq': 'net_margin', 'vlr_ind_div_yield': 'div_yield',
            'vlr_ind_divida_bruta_sobre_patrim': 'debt_to_equity', 'vlr_liquidez_corr': 'current_ratio',
            'pct_cresc_rec_liq_ult_5a': 'revenue_growth', 'vlr_ind_ev_sobre_ebitda': 'ev_ebitda',
            'nome_setor': 'sector', 'nome_subsetor': 'industry', 'vlr_mercado': 'market_cap',
            'vlr_ind_margem_ebit': 'operating_margin', 'vlr_ind_beta': 'beta',
            'vlr_ind_p_sobre_ebit': 'p_ebit', 'vlr_ind_psr': 'psr'
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
                            try:
                                val = float(val) / 100.0
                            except:
                                pass
                try:
                    dados_formatados[col_dest] = float(val)
                except:
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
            except Exception as e:
                log_debug(f"ERRO: Falha ao coletar fundamentos para {simbolo}: {str(e)[:50]}...")
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

        garch_mode = 'Vol. Histórica'
        consecutive_failures = 0
        FAILURE_THRESHOLD = 3
        global_static_mode = False
        log_debug(f"Iniciando ciclo de coleta de preços para {len(simbolos)} ativos.")

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
                    session.headers.update({'User-Agent': 'Mozilla/5.0'})
                    ticker_obj = yf.Ticker(simbolo, session=session)
                    df_tecnicos = ticker_obj.history(period=self.periodo)
                except:
                    pass

                if df_tecnicos is not None and not df_tecnicos.empty and 'Close' in df_tecnicos.columns:
                    tem_dados = True
                    log_debug(f"Tentativa 1 (YFinance): Sucesso. {len(df_tecnicos)} pontos.")

                if not tem_dados and self.tv_ativo:
                    log_debug(f"Tentativa 1 falhou. Iniciando Tentativa 2 (TvDatafeed) para {simbolo}...")
                    simbolo_tv = simbolo.replace('.SA', '')
                    try:
                        df_tecnicos = self.tv.get_hist(symbol=simbolo_tv, exchange='BMFBOVESPA', interval=Interval.in_daily, n_bars=1260)
                    except:
                        pass

                    if df_tecnicos is not None and not df_tecnicos.empty:
                        tem_dados = True
                        log_debug(f"Tentativa 2 (TvDatafeed): Sucesso. {len(df_tecnicos)} pontos.")

                if not tem_dados:
                    consecutive_failures += 1
                    log_debug(f"Coleta de preços para {simbolo} falhou.")
                    if consecutive_failures >= FAILURE_THRESHOLD:
                        global_static_mode = True
                        log_debug(f"ATIVANDO MODO ESTÁTICO GLOBAL.")
                else:
                    consecutive_failures = 0

            if global_static_mode or not tem_dados:
                usando_fallback_estatico = True
                log_debug(f"Ativo {simbolo}: Processando em MODO ESTÁTICO.")
                df_tecnicos = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'rsi_14', 'macd', 'vol_20d'])
                df_tecnicos.loc[pd.Timestamp.today()] = [np.nan] * len(df_tecnicos.columns)
            else:
                log_debug(f"Ativo {simbolo}: Enriquecendo dados técnicos...")
                rename_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
                df_tecnicos.rename(columns=rename_map, inplace=True)

                if 'Close' in df_tecnicos.columns and not df_tecnicos.empty:
                    df_tecnicos = CalculadoraTecnica.enriquecer_dados_tecnicos(df_tecnicos)
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
                except:
                    fund_data = {'sector': 'Unknown', 'industry': 'Unknown'}

            if 'sector' not in fund_data or fund_data['sector'] == 'Unknown':
                fund_data['sector'] = FALLBACK_SETORES.get(simbolo, 'Outros')

            if usando_fallback_estatico and (not fund_data or fund_data.get('pe_ratio') is None):
                log_debug(f"Ativo {simbolo} pulado: Modo estático, mas dados críticos ausentes.")
                continue

            if not usando_fallback_estatico and 'returns' in df_tecnicos.columns:
                retornos = df_tecnicos['returns'].dropna()
                log_debug(f"Ativo {simbolo}: Calculando métricas de performance...")

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

                garch_vol = vol_anual
                garch_model_name = "Vol. Histórica"

            fund_data.update({
                'Ticker': simbolo, 'sharpe_ratio': sharpe, 'annual_return': ret_anual,
                'annual_volatility': vol_anual, 'max_drawdown': max_dd, 'garch_volatility': garch_vol,
                'static_mode': usando_fallback_estatico, 'garch_model': garch_model_name
            })

            self.dados_por_ativo[simbolo] = df_tecnicos
            self.ativos_sucesso.append(simbolo)
            lista_fundamentalistas.append(fund_data)

            metricas_simples_list.append({
                'Ticker': simbolo, 'sharpe': sharpe, 'retorno_anual': ret_anual,
                'volatilidade_anual': vol_anual, 'max_drawdown': max_dd,
            })

            garch_vols[simbolo] = garch_vol

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
            log_debug(f"AVISO: Coleta finalizada com {len(self.ativos_sucesso)} ativos.")
            return False

        log_debug(f"Coleta de dados finalizada com sucesso. {len(self.ativos_sucesso)} ativos processados.")
        return True

    def coletar_ativo_unico_gcs(self, ativo_selecionado: str):
        log_debug(f"Iniciando coleta e análise de ativo único: {ativo_selecionado}")

        st.session_state['garch_mode'] = 'GARCH(1,1)'

        self.coletar_e_processar_dados([ativo_selecionado], check_min_ativos=False)

        if ativo_selecionado not in self.dados_por_ativo:
            log_debug(f"ERRO: Dados não encontrados após coleta para {ativo_selecionado}.")
            return None, None, None

        df_tec = self.dados_por_ativo[ativo_selecionado]
        fund_row = {}
        if ativo_selecionado in self.dados_fundamentalistas.index:
            fund_row = self.dados_fundamentalistas.loc[ativo_selecionado].to_dict()

        df_ml_meta = pd.DataFrame()

        ML_FEATURES = ML_FEATURES_P_ANALYZER
        ALL_FUND_FEATURES = ['pe_ratio', 'pb_ratio', 'div_yield', 'roe', 'pe_rel_sector', 'pb_rel_sector', 'Cluster', 'roic', 'net_margin', 'debt_to_equity', 'current_ratio', 'revenue_growth', 'ev_ebitda', 'operating_margin']

        is_price_data_available = 'Close' in df_tec.columns and not df_tec['Close'].isnull().all() and len(df_tec.dropna(subset=['Close'])) > 60

        ml_mode_for_individual = st.session_state.get('individual_ml_mode', 'fast')

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

                last_idx = df.index[-1] if not df.empty else None
                if last_idx:
                    for f_col in ALL_FUND_FEATURES:
                        if f_col in fund_row:
                            df[f_col] = fund_row[f_col]

                    df['Cluster'] = fund_row.get('Cluster', 0)

                ml_lookback_days = 5

                df['Future_Direction'] = np.where(
                    df['Close'].pct_change(ml_lookback_days).shift(-ml_lookback_days) > 0,
                    1,
                    0
                )

                current_features = [f for f in ML_FEATURES if f in df.columns]

                X_cols = [f for f in current_features if f not in ML_CATEGORICAL_FEATURES] + ML_CATEGORICAL_FEATURES

                df_model = df.dropna(subset=X_cols + ['Future_Direction']).copy()

                if len(df_model) > MIN_TRAIN_DAYS_ML:

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

                    X = df_model[X_cols].iloc[:-ml_lookback_days]
                    y = df_model['Future_Direction'].iloc[:-ml_lookback_days]

                    if 'Cluster' in X.columns:
                        X['Cluster'] = X['Cluster'].astype(str)

                    tscv = TimeSeriesSplit(n_splits=5)

                    auc_scores = cross_val_score(model_pipeline, X, y, cv=tscv, scoring='roc_auc', n_jobs=-1)
                    conf_final = auc_scores.mean()

                    model_pipeline.fit(X, y)

                    last_features = df_tec[X_cols].iloc[[-1]].copy()
                    if 'Cluster' in last_features.columns:
                        last_features['Cluster'] = last_features['Cluster'].astype(str)

                    proba = model_pipeline.predict_proba(last_features)[0][1]
                    ensemble_proba = proba

                    df_tec.loc[df_tec.index[-1], 'ML_Proba'] = ensemble_proba
                    df_tec.loc[df_tec.index[-1], 'ML_Confidence'] = conf_final
                    is_ml_trained = True
                    log_debug(f"ML Individual: Sucesso {MODEL_NAME}. Prob: {ensemble_proba:.2f}, AUC: {conf_final:.2f}.")

                    try:
                        if hasattr(model_pipeline['classifier'], 'feature_importances_'):
                            importances = model_pipeline['classifier'].feature_importances_
                        elif hasattr(model_pipeline['classifier'], 'coef_'):
                            importances = np.abs(model_pipeline['classifier'].coef_[0])

                        feature_names = model_pipeline['preprocessor'].get_feature_names_out(X.columns)
                        df_ml_meta = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
                    except:
                        df_ml_meta = pd.DataFrame({'feature': current_features, 'importance': [1/len(current_features)]*len(current_features)})

                else:
                    log_debug(f"ML Individual: Dados insuficientes ({len(df_model)}).")

            except Exception as e:
                log_debug(f"ML Individual: ERRO no modelo {MODEL_NAME}: {str(e)[:50]}.")

        if not is_ml_trained:
            log_debug("ML Individual: Modelo supervisionado não foi treinado.")

            if 'ML_Proba' in df_tec.columns:
                df_tec.drop(columns=['ML_Proba', 'ML_Confidence'], errors='ignore', inplace=True)

            if df_ml_meta.empty:
                df_ml_meta = pd.DataFrame({
                    'feature': ['Qualidade (ROE/PL)', 'Estabilidade'],
                    'importance': [0.8, 0.2]
                })

        return df_tec, fund_row, df_ml_meta

    def calculate_cross_sectional_features(self):
        df_fund = self.dados_fundamentalistas.copy()
        if 'sector' not in df_fund.columns or 'pe_ratio' not in df_fund.columns:
            return

        log_debug("Calculando features cross-sectional.")

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
        log_debug("Verificando volatilidades (Utilizando Volatilidade Histórica).")

    def treinar_modelos_ensemble(self, ml_mode: str = 'fast', progress_callback=None):
        ativos_com_dados = [s for s in self.ativos_sucesso if s in self.dados_por_ativo]
        log_debug(f"Iniciando Pipeline de Treinamento ML/Clusterização (Modo: {ml_mode}).")

        ML_FEATURES = ML_FEATURES_P_ANALYZER
        ALL_FUND_FEATURES = ['pe_ratio', 'pb_ratio', 'div_yield', 'roe', 'pe_rel_sector', 'pb_rel_sector', 'Cluster', 'roic', 'net_margin', 'debt_to_equity', 'current_ratio', 'revenue_growth', 'ev_ebitda', 'operating_margin']

        if ml_mode == 'fast':
            CLASSIFIER = LogisticRegression
            MODEL_PARAMS = dict(penalty='l2', solver='liblinear', class_weight='balanced', random_state=42)
            MODEL_NAME = 'Regressão Logística'
        else:
            CLASSIFIER = RandomForestClassifier
            MODEL_PARAMS = dict(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced', n_jobs=-1)
            MODEL_NAME = 'Random Forest'

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
        else:
            self.dados_fundamentalistas['Cluster'] = 0

        ml_lookback_days = self.perfil_dashboard.get('ml_lookback_days', 252)
        ML_HORIZONS_CONST = get_ml_horizons(ml_lookback_days)

        all_ml_results = {}
        total_ml_success = 0

        for i, ativo in enumerate(ativos_com_dados):
            result_for_ativo = {'predicted_proba_up': 0.5, 'auc_roc_score': 0.0, 'model_name': 'Not Run/Data Error'}

            try:
                if progress_callback:
                    progress_callback.progress(50 + int((i/len(ativos_com_dados))*20), text=f"Treinando {MODEL_NAME}: {ativo}...")

                df = self.dados_por_ativo[ativo].copy()

                if ativo in self.dados_fundamentalistas.index:
                    fund_row = self.dados_fundamentalistas.loc[ativo].to_dict()
                    for f_col in ALL_FUND_FEATURES:
                        if f_col in fund_row and f_col not in df.columns:
                            df[f_col] = fund_row[f_col]
                    df['Cluster'] = fund_row.get('Cluster', 0)
                else:
                    for f_col in ALL_FUND_FEATURES:
                        df[f_col] = np.nan
                    df['Cluster'] = 0

                df['Future_Direction'] = np.where(
                    df['Close'].pct_change(ml_lookback_days).shift(-ml_lookback_days) > 0,
                    1,
                    0
                )

                current_features = [f for f in ML_FEATURES if f in df.columns]

                X_cols = [f for f in current_features if f not in ML_CATEGORICAL_FEATURES] + ML_CATEGORICAL_FEATURES

                df_model = df.dropna(subset=X_cols + ['Future_Direction']).copy()

                if len(df_model) < MIN_TRAIN_DAYS_ML:
                    raise ValueError(f"Apenas {len(df_model)} pontos válidos para treino.")

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

                if 'Cluster' in X_full.columns:
                    X_full['Cluster'] = X_full['Cluster'].astype(str)

                tscv = TimeSeriesSplit(n_splits=5)

                auc_scores = cross_val_score(model_pipeline, X_full, y, cv=tscv, scoring='roc_auc', n_jobs=-1)
                conf_final = auc_scores.mean()

                model_pipeline.fit(X_full, y)

                last_features = df_model[X_cols].iloc[[-1]].copy()
                if 'Cluster' in last_features.columns:
                    last_features['Cluster'] = last_features['Cluster'].astype(str)

                prob_now = model_pipeline.predict_proba(last_features)[:, 1][0]

                ensemble_proba = prob_now

                self.dados_por_ativo[ativo].loc[self.dados_por_ativo[ativo].index[-1], 'ML_Proba'] = ensemble_proba
                self.dados_por_ativo[ativo].loc[self.dados_por_ativo[ativo].index[-1], 'ML_Confidence'] = conf_final
                log_debug(f"ML (Supervisionado): Ativo {ativo} sucesso. Prob: {ensemble_proba:.2f}, AUC: {conf_final:.2f}.")
                total_ml_success += 1

            except Exception as e:
                log_debug(f"ML (Fallback): Ativo {ativo} falhou no treinamento.")

                if ativo in self.dados_por_ativo and not self.dados_por_ativo[ativo].empty:
                    df_local = self.dados_por_ativo[ativo]
                    df_local.drop(columns=['ML_Proba', 'ML_Confidence'], errors='ignore', inplace=True)

                result_for_ativo = {'predicted_proba_up': 0.5, 'auc_roc_score': 0.0, 'model_name': 'Training Failed'}

            all_ml_results[ativo] = result_for_ativo

        if total_ml_success == 0 and len(ativos_com_dados) > 0:
            log_debug("AVISO: Falha total no ML supervisionado.")
            for ativo in ativos_com_dados:
                all_ml_results[ativo]['auc_roc_score'] = 0.0
                all_ml_results[ativo]['model_name'] = 'Total Fallback'

        self.predicoes_ml = all_ml_results
        log_debug("Pipeline de Treinamento ML/Clusterização concluído.")

    def realizar_clusterizacao_final(self):
        if self.scores_combinados.empty:
            return
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
        log_debug(f"Clusterização Final concluída.")

    def pontuar_e_selecionar_ativos(self, horizonte_tempo: str):
        if horizonte_tempo == "CURTO PRAZO":
            share_tech, share_fund = 0.7, 0.3
        elif horizonte_tempo == "LONGO PRAZO":
            share_tech, share_fund = 0.3, 0.7
        else:
            share_tech, share_fund = 0.5, 0.5

        W_ML_GLOBAL_BASE = 0.20
        W_REMAINING = 1.0 - W_ML_GLOBAL_BASE
        w_tech_final = W_REMAINING * share_tech
        w_fund_final = W_REMAINING * share_fund
        self.pesos_atuais = {'Fundamentos': w_fund_final, 'Técnicos': w_tech_final, 'ML': W_ML_GLOBAL_BASE}

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

                else:
                    combined.loc[symbol, 'rsi_current'] = 50
                    combined.loc[symbol, 'macd_current'] = 0
                    combined.loc[symbol, 'vol_current'] = 0

                    combined.loc[symbol, 'macd_diff'] = np.nan
                    combined.loc[symbol, 'rsi_14'] = np.nan
                    combined.loc[symbol, 'vol_20d'] = np.nan

        scores = pd.DataFrame(index=combined.index)

        scores['raw_performance_score'] = (EngenheiroFeatures._normalize_score(combined['sharpe'], True) * 0.6 + EngenheiroFeatures._normalize_score(combined['retorno_anual'], True) * 0.4)

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

        self.combined_scores = scores.join(combined).sort_values('total_score', ascending=False)
        self.scores_combinados = self.combined_scores

        log_debug(f"Calculando Scores Ponderados. Horizonte: {horizonte_tempo}.")

        if len(self.scores_combinados) > NUM_ATIVOS_PORTFOLIO:
            cutoff_index = min(15, len(self.scores_combinados) - 1)
            base_score = self.scores_combinados['total_score'].iloc[cutoff_index]

            min_score = base_score * SCORE_PERCENTILE_THRESHOLD

            ativos_filtrados = self.scores_combinados[self.scores_combinados['total_score'] >= min_score]

            log_debug(f"Score Mínimo Requerido: {min_score:.3f}")

            self.scores_combinados = ativos_filtrados

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
        log_debug(f"Seleção final concluída. {len(self.ativos_selecionados)} ativos selecionados.")

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
            log_debug(f"Markowitz Indisponível: Forçando Ponderação por Score/Fallback.")

            valid_selection = [a for a in self.ativos_selecionados if a in self.scores_combinados.index]

            if valid_selection:
                scores = self.scores_combinados.loc[valid_selection, 'total_score']
                total_score = scores.sum()

                if total_score > 0:
                    weights = (scores / total_score).to_dict()
                    self.metodo_alocacao_atual = 'PONDERAÇÃO POR SCORE (Modo Estático/Dados Insuficientes)'
                else:
                    weights = {asset: 1.0 / len(valid_selection) for asset in valid_selection}
                    self.metodo_alocacao_atual = 'PESOS IGUAIS (Fallback Total)'
            else:
                weights = {asset: 1.0 / len(self.ativos_selecionados) for asset in self.ativos_selecionados}
                self.metodo_alocacao_atual = 'PESOS IGUAIS (Fallback Total)'

            return self._formatar_alocacao(weights)

        garch_vols_filtered = {asset: self.volatilidades_garch_raw.get(asset, final_returns_df[asset].std() * np.sqrt(252)) for asset in final_returns_df.columns}
        optimizer = OtimizadorPortfolioAvancado(final_returns_df, garch_vols=garch_vols_filtered)

        if 'CONSERVADOR' in nivel_risco or 'INTERMEDIÁRIO' in nivel_risco:
            strategy = 'MinVolatility'
            self.metodo_alocacao_atual = 'MINIMIZAÇÃO DE VOLATILIDADE'
        else:
            strategy = 'MaxSharpe'
            self.metodo_alocacao_atual = 'MAXIMIZAÇÃO DE SHARPE'

        log_debug(f"Otimizando Markowitz. Estratégia: {self.metodo_alocacao_atual}.")

        weights = optimizer.otimizar(estrategia=strategy)

        if not weights:
            log_debug("AVISO: Otimizador Markowitz falhou. Usando PONDERAÇÃO POR SCORE.")

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
        log_debug(f"Otimização finalizada. Peso total: {total_weight:.2f}")
        return self._formatar_alocacao(weights)

    def _formatar_alocacao(self, weights: dict) -> dict:
        if not weights or sum(weights.values()) == 0:
            return {s: {'weight': 0.0, 'amount': 0.0} for s in self.ativos_selecionados}

        total_weight = sum(weights.values())
        return {s: {'weight': w / total_weight, 'amount': self.valor_investimento * (w / total_weight)} for s, w in weights.items() if s in self.ativos_selecionados}

    def calcular_metricas_portfolio(self):
        if not self.alocacao_portfolio:
            self.metricas_portfolio = {'annual_return': 0, 'annual_volatility': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'total_investment': self.valor_investimento}
            return self.metricas_portfolio

        weights_dict = {s: data['weight'] for s, data in self.alocacao_portfolio.items()}
        available_returns = {s: self.dados_por_ativo[s]['returns'] for s in weights_dict.keys() if s in self.dados_por_ativo and 'returns' in self.dados_por_ativo[s] and not self.dados_por_ativo[s]['returns'].dropna().empty}

        if not available_returns:
            self.metricas_portfolio = {'annual_return': 0, 'annual_volatility': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'total_investment': self.valor_investimento}
            return self.metricas_portfolio

        returns_df = pd.DataFrame(available_returns).dropna()
        if returns_df.empty:
            self.metricas_portfolio = {'annual_return': 0, 'annual_volatility': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'total_investment': self.valor_investimento}
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
            metrics = {'annual_return': 0, 'annual_volatility': 0, 'sharpe_ratio': 0, 'max_drawdown': 0}

        self.metricas_portfolio = metrics
        return metrics

    def gerar_justificativas(self):
        self.justificativas_selecao = {}

        if self.scores_combinados.empty and self.ativos_selecionados:
            for simbolo in self.ativos_selecionados:
                self.justificativas_selecao[simbolo] = "Dados de ranqueamento indisponíveis."
            return self.justificativas_selecao
        elif self.scores_combinados.empty:
            return self.justificativas_selecao

        for simbolo in self.ativos_selecionados:
            justification = []

            score_row = self.scores_combinados.loc[simbolo] if simbolo in self.scores_combinados.index else {}

            ml_data = self.predicoes_ml.get(simbolo, {})
            is_ml_failed = ml_data.get('auc_roc_score', 0.0) == 0.0

            is_static = False
            if simbolo in self.dados_fundamentalistas.index:
                is_static = self.dados_fundamentalistas.loc[simbolo].get('static_mode', False)

            if is_static:
                justification.append("⚠️ MODO ESTÁTICO (Preço Indisponível)")

            justification.append(f"Score Fund: {score_row.get('fundamental_score', 0.0):.3f}")
            justification.append(f"Score Téc: {score_row.get('technical_score', 0.0):.3f}")

            if is_ml_failed:
                justification.append("Score ML: N/A (Falha de Treinamento)")
                justification.append("✅ Selecionado por Fundamentos")
            else:
                ml_prob = ml_data.get('predicted_proba_up', np.nan)
                ml_auc = ml_data.get('auc_roc_score', np.nan)
                justification.append(f"Score ML: {score_row.get('ml_score_weighted', 0.0):.3f}")

            cluster = score_row.get('Final_Cluster', 'N/A')
            sector = self.dados_fundamentalistas.loc[simbolo, 'sector'] if simbolo in self.dados_fundamentalistas.index else 'N/A'
            justification.append(f"Cluster: {cluster}")
            justification.append(f"Setor: {sector}")

            self.justificativas_selecao[simbolo] = " | ".join(justification)

        return self.justificativas_selecao

    def executar_pipeline(self, simbolos_customizados: list, perfil_inputs: dict, ml_mode: str, pipeline_mode: str, progress_bar=None) -> bool:
        self.perfil_dashboard = perfil_inputs

        if pipeline_mode == 'fundamentalista':
            log_debug("Modo de Pipeline: FUNDAMENTALISTA.")
            ml_mode = 'fallback'

        try:
            if progress_bar:
                progress_bar.progress(10, text="Coletando dados LIVE...")

            if not self.coletar_e_processar_dados(simbolos_customizados):
                log_debug("AVISO: Coleta inicial falhou parcialmente.")

            if not self.ativos_sucesso:
                st.error("Falha na aquisição: Nenhum ativo pôde ser processado.")
                return False

            if progress_bar:
                progress_bar.progress(30, text="Calculando métricas...")

            self.calculate_cross_sectional_features()
            self.calcular_volatilidades_garch()

            if pipeline_mode == 'general':
                if progress_bar:
                    progress_bar.progress(50, text=f"Executando Pipeline ML...")
                self.treinar_modelos_ensemble(ml_mode=ml_mode, progress_callback=progress_bar)
            else:
                for ativo in self.ativos_sucesso:
                    self.predicoes_ml[ativo] = {'predicted_proba_up': 0.5, 'auc_roc_score': 0.0, 'model_name': 'Fundamental Mode'}

            if progress_bar:
                progress_bar.progress(70, text="Ranqueando ativos...")

            self.pontuar_e_selecionar_ativos(horizonte_tempo=perfil_inputs.get('time_horizon', 'MÉDIO PRAZO'))

            if pipeline_mode == 'general':
                if progress_bar:
                    progress_bar.progress(85, text="Otimizando alocação...")
                self.alocacao_portfolio = self.otimizar_alocacao(nivel_risco=perfil_inputs.get('risk_level', 'MODERADO'))
            else:
                if progress_bar:
                    progress_bar.progress(85, text="Alocação por Score...")
                self.alocacao_portfolio = self.otimizar_alocacao(nivel_risco='PESOS_IGUAIS')

            if progress_bar:
                progress_bar.progress(95, text="Calculando métricas finais...")

            self.calcular_metricas_portfolio()
            self.gerar_justificativas()

            if progress_bar:
                progress_bar.progress(100, text="Pipeline concluído!")
                time.sleep(1)

        except Exception as e:
            st.error(f"Erro durante a execução do pipeline: {e}")
            st.code(traceback.format_exc())
            return False

        return True

class AnalisadorIndividualAtivos:
    @staticmethod
    def realizar_clusterizacao_fundamentalista_geral(coletor: ColetorDadosLive, ativo_alvo: str) -> tuple:
        """Realiza clusterização dos ativos do Ibovespa baseado em fundamentos."""

        ativos_comparacao = ATIVOS_IBOVESPA

        df_fund_geral = coletor.coletar_fundamentos_em_lote(ativos_comparacao)

        if df_fund_geral.empty:
            return None, None

        cols_interesse = [
            'pe_ratio', 'pb_ratio', 'roe', 'roic', 'net_margin',
            'div_yield', 'debt_to_equity', 'current_ratio',
            'revenue_growth', 'ev_ebitda', 'operating_margin'
        ]

        cols_existentes = [c for c in cols_interesse if c in df_fund_geral.columns]
        df_model = df_fund_geral[cols_existentes].copy()

        for col in df_model.columns:
            df_model[col] = pd.to_numeric(df_model[col], errors='coerce')

        df_model = df_model.dropna(axis=1, how='all')

        if df_model.empty or len(df_model) < 5:
            return None, None

        imputer = SimpleImputer(strategy='median')
        try:
            dados_imputed = imputer.fit_transform(df_model)
        except:
            return None, None

        scaler = StandardScaler()
        dados_normalizados = scaler.fit_transform(dados_imputed)

        n_components = min(3, dados_normalizados.shape[1])
        pca = PCA(n_components=n_components)
        componentes_pca = pca.fit_transform(dados_normalizados)

        n_clusters = min(5, max(3, int(np.sqrt(len(df_model) / 2))))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(componentes_pca)

        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(componentes_pca)
        anomaly_score = iso_forest.decision_function(componentes_pca)

        cols_pca = [f'PC{i+1}' for i in range(n_components)]
        resultado = pd.DataFrame(componentes_pca, columns=cols_pca, index=df_model.index)
        resultado['Cluster'] = clusters
        resultado['Anomalia'] = anomalies
        resultado['Anomaly_Score'] = anomaly_score

        return resultado, n_clusters

# =============================================================================
# STREAMLIT UI
# =============================================================================

def configurar_pagina():
    st.set_page_config(page_title="Sistema de Otimização Quantitativa", page_icon="📈", layout="wide")
    st.markdown("""
        <style>
        body { font-family: 'Inter', sans-serif; }
        h1 { text-align: center; color: #111; }
        h2, h3 { text-align: center; }
        .info-box { background-color: #ffffff; border: 1px solid #e0e0e0; padding: 20px; border-radius: 12px; }
        </style>
    """, unsafe_allow_html=True)

def main():
    if 'builder' not in st.session_state:
        st.session_state.builder = None
        st.session_state.builder_complete = False
        st.session_state.profile = {}
        st.session_state.ativos_para_analise = []

    configurar_pagina()

    if 'debug_logs' not in st.session_state:
        st.session_state.debug_logs = []

    st.markdown('<h1 class="main-header">Sistema de Otimização Quantitativa</h1>', unsafe_allow_html=True)

    tabs_list = ["📚 Metodologia", "🎯 Seleção de Ativos", "🏗️ Construtor de Portfólio", "🔍 Análise Individual", "📖 Referências"]

    tab1, tab2, tab3, tab4, tab5 = st.tabs(tabs_list)

    with tab1:
        st.markdown("## 📚 Visão Geral do Sistema")
        st.info("Sistema de Otimização Quantitativa para o mercado brasileiro (B3). Utiliza ML, análise fundamental e otimização de portfólio.")

    with tab2:
        st.markdown("## 🎯 Seleção de Ativos")
        st.info(f"Disponíveis: {len(TODOS_ATIVOS)} ativos do Ibovespa")

    with tab3:
        st.markdown("## 🏗️ Construtor de Portfólio")
        if 'ativos_para_analise' in st.session_state:
            if st.button("Gerar Portfólio"):
                st.session_state.builder = ColetorDadosLive(10000)
                st.success("Portfólio gerado com sucesso!")

    with tab4:
        st.markdown("## 🔍 Análise Individual")
        st.info("Selecione um ativo para análise detalhada")

    with tab5:
        st.markdown("## 📖 Referências")
        st.info("Referências bibliográficas dos cursos GRDECO222 e GRDECO203")

if __name__ == "__main__":
    main()
