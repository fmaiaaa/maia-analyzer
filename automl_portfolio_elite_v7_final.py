# -*- coding: utf-8 -*-
"""
=============================================================================
SISTEMA DE PORTFÓLIOS ADAPTATIVOS - OTIMIZAÇÃO QUANTITATIVA
=============================================================================

Adaptação do Sistema AutoML para coleta em TEMPO REAL (Live Data).
- Preços via yfinance.
- Fundamentos via Pynvest (Fundamentus).
- Lógica de Construção (V9.4): Pesos Dinâmicos + Seleção por Clusterização.
- Design (V8.7): Estritamente alinhado ao original (Textos Exaustivos).

Versão: 9.5.0 (Logic V9.4 + Design V8.7 Strict + Textos Originais)
=============================================================================
"""

# --- 1. CORE LIBRARIES & UTILITIES ---
import warnings
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

# --- 3. STREAMLIT, DATA ACQUISITION, & PLOTTING ---
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests # Adicionado para gerenciar sessão do yfinance

# --- NOVAS IMPORTAÇÕES PARA COLETA LIVE ---
import yfinance as yf
try:
    from pynvest.scrappers.fundamentus import Fundamentus
except ImportError:
    pass

# --- 4. FEATURE ENGINEERING / TECHNICAL ANALYSIS (TA) ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- 7. SPECIALIZED TIME SERIES & ECONOMETRICS ---
from arch import arch_model

# --- 8. CONFIGURATION ---
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURAÇÕES E CONSTANTES GLOBAIS
# =============================================================================

PERIODO_DADOS = '5y'
MIN_DIAS_HISTORICO = 252
NUM_ATIVOS_PORTFOLIO = 5
TAXA_LIVRE_RISCO = 0.1075
LOOKBACK_ML = 30
SCORE_PERCENTILE_THRESHOLD = 0.85 

# Pesos de alocação (Markowitz - Lógica Analyzer)
PESO_MIN = 0.10
PESO_MAX = 0.30

LOOKBACK_ML_DAYS_MAP = {
    'curto_prazo': 5,
    'medio_prazo': 20,
    'longo_prazo': 30
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
    'Consumo Cíclico': ['AZZA3.SA', 'ALOS3.SA', 'CEAB3.SA', 'COGN3.SA', 'CURY3.SA', 'CVCB3.SA', 'CYRE3.SA', 'DIRR3.SA', 'LREN3.SA', 'MGLU3.SA', 'MRVE3.SA', 'RENT3.SA', 'YDUQ3.SA'],
    'Consumo não Cíclico': ['BEEF3.SA', 'NATU3.SA', 'PCAR3.SA', 'VIVA3.SA'], 
    'Financeiro': ['B3SA3.SA', 'BBSE3.SA', 'BBDC3.SA', 'BBDC4.SA', 'BBAS3.SA', 'BPAC11.SA', 'CXSE3.SA', 'HYPE3.SA', 'IGTI11.SA', 'IRBR3.SA', 'ITSA4.SA', 'ITUB4.SA', 'MULT3.SA', 'PSSA3.SA', 'RDOR3.SA', 'SANB11.SA'],
    'Materiais Básicos': ['BRAP4.SA', 'BRKM5.SA', 'CSNA3.SA', 'GGBR4.SA', 'GOAU4.SA', 'KLBN11.SA', 'POMO4.SA', 'SUZB3.SA', 'USIM5.SA', 'VALE3.SA'],
    'Petróleo, Gás e Biocombustíveis': ['ENEV3.SA', 'PETR3.SA', 'PETR4.SA', 'PRIO3.SA', 'RAIZ4.SA', 'RECV3.SA', 'UGPA3.SA', 'VBBR3.SA'],
    'Saúde': ['FLRY3.SA', 'HAPV3.SA', 'RADL3.SA'],
    'Tecnologia da Informação': ['TOTS3.SA'],
    'Telecomunicações': ['TIMS3.SA', 'VIVT3.SA'],
    'Utilidade Pública': ['AESB3.SA', 'AURE3.SA', 'BRAV3.SA', 'CMIG4.SA', 'CPLE6.SA', 'CPFE3.SA', 'EGIE3.SA', 'ELET3.SA', 'ELET6.SA', 'ENGI11.SA', 'EQTL3.SA', 'ISAE4.SA', 'RAIL3.SA', 'SBSP3.SA', 'TAEE11.SA']
}

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

# =============================================================================
# 6. CLASSE: ANALISADOR DE PERFIL DO INVESTIDOR
# =============================================================================

class AnalisadorPerfilInvestidor:
    """Analisa perfil de risco e horizonte temporal do investidor."""
    
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
        time_map = { 'A': 5, 'B': 20, 'C': 30 } 
        final_lookback = max( time_map.get(liquidez_key, 5), time_map.get(objetivo_key, 5) )
        
        if final_lookback >= 30:
            self.horizonte_tempo = "LONGO PRAZO"
        elif final_lookback >= 20:
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
# 7. FUNÇÕES DE ESTILO E VISUALIZAÇÃO (Design Original)
# =============================================================================

def obter_template_grafico() -> dict:
    return {
        'plot_bgcolor': '#f8f9fa', 
        'paper_bgcolor': 'white',
        'font': {'family': 'Arial, sans-serif', 'size': 12, 'color': '#343a40'},
        'title': {'font': {'family': 'Arial, sans-serif', 'size': 16, 'color': '#212529', 'weight': 'bold'}, 'x': 0.5, 'xanchor': 'center'},
        'xaxis': {'showgrid': True, 'gridcolor': '#e9ecef', 'showline': True, 'linecolor': '#ced4da', 'linewidth': 1, 'tickfont': {'family': 'Arial, sans-serif', 'color': '#343a40'}, 'title': {'font': {'family': 'Arial, sans-serif', 'color': '#343a40'}}, 'zeroline': False},
        'yaxis': {'showgrid': True, 'gridcolor': '#e9ecef', 'showline': True, 'linecolor': '#ced4da', 'linewidth': 1, 'tickfont': {'family': 'Arial, sans-serif', 'color': '#343a40'}, 'title': {'font': {'family': 'Arial, sans-serif', 'color': '#343a40'}}, 'zeroline': False},
        'legend': {'font': {'family': 'Arial, sans-serif', 'color': '#343a40'}, 'bgcolor': 'rgba(255, 255, 255, 0.8)', 'bordercolor': '#e9ecef', 'borderwidth': 1},
        'colorway': ['#212529', '#495057', '#6c757d', '#adb5bd', '#ced4da']
    }

# =============================================================================
# 8. CLASSE: ENGENHEIRO DE FEATURES
# =============================================================================

class EngenheiroFeatures:
    """Funções utilitárias de features e normalização."""

    @staticmethod
    def _normalize_score(serie: pd.Series, maior_melhor: bool = True) -> pd.Series:
        """
        Normaliza usando Z-score robusto.
        """
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
# 8.1. CLASSE: CALCULADORA TÉCNICA (Live Data)
# =============================================================================

class CalculadoraTecnica:
    """Calcula indicadores técnicos."""
    
    @staticmethod
    def enriquecer_dados_tecnicos(df_ativo: pd.DataFrame) -> pd.DataFrame:
        if df_ativo.empty: return df_ativo
        
        df = df_ativo.sort_index().copy()
        df['returns'] = df['Close'].pct_change()
        
        # RSI (14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['rsi_14'] = 100 - (100 / (1 + (gain / loss)))
        
        # MACD (12, 26, 9)
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # Volatilidade anualizada (20 dias)
        df['vol_20d'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Momentum (10 períodos)
        df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        
        # Médias móveis simples
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        df['sma_200'] = df['Close'].rolling(window=200).mean()
        
        # BBands Width
        rolling_mean = df['Close'].rolling(window=20).mean()
        rolling_std = df['Close'].rolling(window=20).std()
        upper = rolling_mean + (rolling_std * 2)
        lower = rolling_mean - (rolling_std * 2)
        df['bb_width'] = (upper - lower) / rolling_mean
        
        # Momentum 60 (para compatibilidade com visualização antiga se necessário)
        df['momentum_60'] = df['Close'].pct_change(60)
        
        return df

# =============================================================================
# 9. FUNÇÕES DE COLETA DE DADOS LIVE (YFINANCE + PYNVEST)
# =============================================================================

class ColetorDadosLive(object):
    def __init__(self, periodo=PERIODO_DADOS):
        self.periodo = periodo
        self.dados_por_ativo = {} 
        self.dados_fundamentalistas = pd.DataFrame() 
        self.metricas_performance = pd.DataFrame() 
        self.volatilidades_garch_raw = {}
        self.metricas_simples = {}
        
        try:
            self.pynvest_scrapper = Fundamentus()
            self.pynvest_ativo = True
        except Exception:
            self.pynvest_ativo = False
            st.warning("Biblioteca pynvest não inicializada corretamente.")

    def _mapear_colunas_pynvest(self, df_pynvest: pd.DataFrame) -> dict:
        if df_pynvest.empty: return {}
        row = df_pynvest.iloc[0]
        mapping = {
            'vlr_ind_p_sobre_l': 'pe_ratio', 'vlr_ind_p_sobre_vp': 'pb_ratio', 'vlr_ind_roe': 'roe',
            'vlr_ind_roic': 'roic', 'vlr_ind_margem_liq': 'net_margin', 'vlr_ind_div_yield': 'div_yield',
            'vlr_ind_divida_bruta_sobre_patrim': 'debt_to_equity', 'vlr_liquidez_corr': 'current_ratio',
            'pct_cresc_rec_liq_ult_5a': 'revenue_growth', 'vlr_ind_ev_sobre_ebitda': 'ev_ebitda',
            'nome_setor': 'sector', 'nome_subsetor': 'industry', 'vlr_mercado': 'market_cap',
            'vlr_ind_margem_ebit': 'operating_margin', # Adicionado para visualização detalhada
            'vlr_ind_beta': 'beta' # Adicionado se disponível
        }
        dados_formatados = {}
        for col_orig, col_dest in mapping.items():
            if col_orig in row:
                val = row[col_orig]
                try: dados_formatados[col_dest] = float(val)
                except (ValueError, TypeError): dados_formatados[col_dest] = val
        return dados_formatados

    def coletar_e_processar_dados(self, simbolos: list) -> bool:
        self.ativos_sucesso = []
        lista_fundamentalistas = []
        garch_vols = {}
        metricas_simples_list = []

        # --- CORREÇÃO: Sessão com User-Agent para evitar bloqueio do Yahoo ---
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

        dados_yf = pd.DataFrame()

        try:
            # Tenta passar a sessão para o yfinance
            # threads=False é CRÍTICO aqui para evitar rate limiting ou problemas de lock em versões recentes quando há falha
            try:
                dados_yf = yf.download(
                    simbolos, 
                    period=self.periodo, 
                    group_by='ticker', 
                    progress=False, 
                    threads=False, 
                    session=session
                )
            except TypeError:
                # Fallback para versões antigas do yfinance que não aceitam session no download
                dados_yf = yf.download(
                    simbolos, 
                    period=self.periodo, 
                    group_by='ticker', 
                    progress=False, 
                    threads=False
                )
        except Exception as e:
            st.warning(f"Aviso no download em lote (será tentado individualmente): {e}")

        for simbolo in simbolos:
            df_tecnicos = pd.DataFrame()

            # 1. Tenta pegar os dados do download em lote
            try:
                if not dados_yf.empty:
                    if len(simbolos) > 1:
                        if simbolo in dados_yf.columns.get_level_values(0): # Verifica se o ticker existe nas colunas
                             df_tecnicos = dados_yf[simbolo].copy()
                    else:
                        df_tecnicos = dados_yf.copy()
            except KeyError:
                pass

            # 2. FALLBACK: Se o lote falhou ou veio vazio para este ativo específico, tenta baixar individualmente
            if df_tecnicos.empty or 'Close' not in df_tecnicos.columns or df_tecnicos['Close'].dropna().empty:
                try:
                    # Tenta baixar individualmente usando yf.Ticker (endpoint diferente, muitas vezes funciona quando o bulk falha)
                    ticker_obj = yf.Ticker(simbolo, session=session)
                    df_tecnicos = ticker_obj.history(period=self.periodo)
                except Exception:
                     # Última tentativa sem sessão explícita
                    try:
                        ticker_obj = yf.Ticker(simbolo)
                        df_tecnicos = ticker_obj.history(period=self.periodo)
                    except Exception:
                        pass

            if df_tecnicos.empty or 'Close' not in df_tecnicos.columns:
                continue
            
            df_tecnicos = df_tecnicos.dropna(subset=['Close'])
            df_tecnicos = CalculadoraTecnica.enriquecer_dados_tecnicos(df_tecnicos)
            
            fund_data = {}
            if self.pynvest_ativo:
                try:
                    ticker_pynvest = simbolo.replace('.SA', '').lower()
                    df_fund_raw = self.pynvest_scrapper.coleta_indicadores_de_ativo(ticker_pynvest)
                    if df_fund_raw is not None and not df_fund_raw.empty:
                        fund_data = self._mapear_colunas_pynvest(df_fund_raw)
                    else:
                        fund_data = {'sector': 'Unknown', 'industry': 'Unknown'}
                except Exception as e:
                    fund_data = {'sector': 'Unknown', 'industry': 'Unknown'}
            
            retornos = df_tecnicos['returns'].dropna()
            if len(retornos) > 30:
                vol_anual = retornos.std() * np.sqrt(252)
                ret_anual = retornos.mean() * 252
                sharpe = (ret_anual - TAXA_LIVRE_RISCO) / vol_anual if vol_anual > 0 else 0
                cum_prod = (1 + retornos).cumprod()
                peak = cum_prod.expanding(min_periods=1).max()
                dd = (cum_prod - peak) / peak
                max_dd = dd.min()
            else:
                vol_anual, ret_anual, sharpe, max_dd = np.nan, np.nan, np.nan, np.nan

            fund_data.update({
                'Ticker': simbolo, 'sharpe_ratio': sharpe, 'annual_return': ret_anual,
                'annual_volatility': vol_anual, 'max_drawdown': max_dd, 'garch_volatility': vol_anual
            })
            
            self.dados_por_ativo[simbolo] = df_tecnicos
            self.ativos_sucesso.append(simbolo)
            lista_fundamentalistas.append(fund_data)
            garch_vols[simbolo] = vol_anual
            
            metricas_simples_list.append({
                'Ticker': simbolo, 'sharpe': sharpe, 'retorno_anual': ret_anual,
                'volatilidade_anual': vol_anual, 'max_drawdown': max_dd,
            })

        if len(self.ativos_sucesso) < NUM_ATIVOS_PORTFOLIO: return False
            
        self.dados_fundamentalistas = pd.DataFrame(lista_fundamentalistas).set_index('Ticker')
        self.metricas_performance = pd.DataFrame(metricas_simples_list).set_index('Ticker')
        self.volatilidades_garch_raw = garch_vols 
        
        for simbolo in self.ativos_sucesso:
            last_idx = self.dados_por_ativo[simbolo].index[-1]
            for k, v in self.dados_fundamentalistas.loc[simbolo].items():
                 if k not in self.dados_por_ativo[simbolo].columns:
                      self.dados_por_ativo[simbolo].loc[last_idx, k] = v

        return True

    def coletar_ativo_unico_gcs(self, ativo_selecionado: str):
        """Adaptado para usar a coleta Live, mas retornando formato compatível para análise individual."""
        sucesso = self.coletar_e_processar_dados([ativo_selecionado])
        if sucesso:
             df_tec = self.dados_por_ativo[ativo_selecionado]
             fund_row = self.dados_fundamentalistas.loc[ativo_selecionado].to_dict()
             
             # Mock de metadados de ML para compatibilidade com visualização
             df_ml_meta = pd.DataFrame(index=pd.MultiIndex.from_tuples([(ativo_selecionado, 'curto_prazo')], names=['ticker', 'target_name']))
             df_ml_meta['target_days'] = 5
             
             return df_tec, fund_row, df_ml_meta
        return None, None, None

# =============================================================================
# 10. CLASSE: OTIMIZADOR DE PORTFÓLIO
# =============================================================================

class OtimizadorPortfolioAvancado:
    def __init__(self, returns_df: pd.DataFrame, garch_vols: dict = None):
        self.returns = returns_df
        self.mean_returns = returns_df.mean() * 252
        if garch_vols is not None and garch_vols:
            self.cov_matrix = self._construir_matriz_cov_garch(returns_df, garch_vols)
        else:
            self.cov_matrix = returns_df.cov() * 252
        self.num_ativos = len(returns_df.columns)

    def _construir_matriz_cov_garch(self, returns_df: pd.DataFrame, garch_vols: dict) -> pd.DataFrame:
        corr_matrix = returns_df.corr()
        vol_array = np.array([garch_vols.get(ativo, returns_df[ativo].std() * np.sqrt(252)) for ativo in returns_df.columns])
        if np.isnan(vol_array).all() or np.all(vol_array <= 1e-9): return returns_df.cov() * 252
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
        sector_means = df_fund.groupby('sector')[['pe_ratio', 'pb_ratio']].transform('mean')
        df_fund['pe_rel_sector'] = df_fund['pe_ratio'] / sector_means['pe_ratio']
        df_fund['pb_rel_sector'] = df_fund['pb_ratio'] / sector_means['pb_ratio']
        df_fund = df_fund.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        self.dados_fundamentalistas = df_fund

    def calcular_volatilidades_garch(self):
        valid_vols = len([k for k, v in self.volatilidades_garch.items() if not np.isnan(v)])
        if valid_vols == 0:
             for ativo in self.ativos_sucesso:
                 if ativo in self.metricas_performance.index and 'volatilidade_anual' in self.metricas_performance.columns:
                      self.volatilidades_garch[ativo] = self.metricas_performance.loc[ativo, 'volatilidade_anual']
        
    def treinar_modelos_ensemble(self, dias_lookback_ml: int = LOOKBACK_ML, otimizar: bool = False, progress_callback=None):
        ativos_com_dados = [s for s in self.ativos_sucesso if s in self.dados_por_ativo]
        clustering_df = self.dados_fundamentalistas[['pe_ratio', 'pb_ratio', 'div_yield', 'roe']].join(
            self.metricas_performance[['sharpe', 'volatilidade_anual']], how='inner'
        ).fillna(0)
        
        if len(clustering_df) >= 5:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(clustering_df)
            pca = PCA(n_components=min(data_scaled.shape[1], 3))
            data_pca = pca.fit_transform(data_scaled)
            kmeans = KMeans(n_clusters=min(len(data_pca), 5), random_state=42, n_init=10)
            clusters = kmeans.fit_predict(data_pca)
            self.dados_fundamentalistas['Cluster'] = pd.Series(clusters, index=clustering_df.index).fillna(-1).astype(int)
        else:
            self.dados_fundamentalistas['Cluster'] = 0

        features_ml = ['rsi_14', 'macd_diff', 'vol_20d', 'momentum_10', 'sma_50', 'sma_200',
            'pe_ratio', 'pb_ratio', 'div_yield', 'roe', 'pe_rel_sector', 'pb_rel_sector', 'Cluster']
        
        total_ativos = len(ativos_com_dados)
        for i, ativo in enumerate(ativos_com_dados):
            try:
                if progress_callback: progress_callback.progress(50 + int((i/total_ativos)*20), text=f"Treinando RF Pipeline: {ativo}...")
                df = self.dados_por_ativo[ativo].copy()
                if ativo in self.dados_fundamentalistas.index:
                    fund_data = self.dados_fundamentalistas.loc[ativo].to_dict()
                    for col in [f for f in features_ml if f not in df.columns]:
                        if col in fund_data: df[col] = fund_data[col]
                
                df['Future_Direction'] = np.where(df['Close'].pct_change(dias_lookback_ml).shift(-dias_lookback_ml) > 0, 1, 0)
                current_features = [f for f in features_ml if f in df.columns]
                df_model = df.dropna(subset=current_features + ['Future_Direction'])
                
                if len(df_model) < 60:
                    self.predicoes_ml[ativo] = {'predicted_proba_up': 0.5, 'auc_roc_score': 0.5, 'model_name': 'Dados Insuficientes'}
                    continue
                
                X = df_model[current_features].iloc[:-dias_lookback_ml]
                y = df_model['Future_Direction'].iloc[:-dias_lookback_ml]
                if 'Cluster' in X.columns: X['Cluster'] = X['Cluster'].astype(str)
                
                numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
                categorical_cols = ['Cluster'] if 'Cluster' in X.columns else []
                
                preprocessor = ColumnTransformer(transformers=[
                        ('num', StandardScaler(), [f for f in numeric_cols if 'rel_sector' not in f]),
                        ('rel', 'passthrough', [f for f in numeric_cols if 'rel_sector' in f]),
                        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
                    ], remainder='passthrough')
                
                model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced'))])
                
                if len(np.unique(y)) < 2: avg_auc = 0.5
                else:
                    scores = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=3), scoring='roc_auc')
                    avg_auc = scores.mean()
                
                model.fit(X, y)
                last_features = df[current_features].iloc[[-1]].copy()
                if 'Cluster' in last_features.columns: last_features['Cluster'] = last_features['Cluster'].astype(str)
                proba = model.predict_proba(last_features)[0][1]
                
                self.predicoes_ml[ativo] = {'predicted_proba_up': proba, 'auc_roc_score': avg_auc, 'model_name': 'Pipeline RF+Cluster'}
                last_idx = self.dados_por_ativo[ativo].index[-1]
                self.dados_por_ativo[ativo].loc[last_idx, 'ML_Proba'] = proba
                self.dados_por_ativo[ativo].loc[last_idx, 'ML_Confidence'] = avg_auc

            except Exception as e:
                self.predicoes_ml[ativo] = {'predicted_proba_up': 0.5, 'auc_roc_score': 0.5, 'model_name': f'Erro: {str(e)}'}
                continue

    def realizar_clusterizacao_final(self):
        if self.scores_combinados.empty: return
        features_cluster = ['performance_score', 'fundamental_score', 'technical_score', 'ml_score_weighted']
        data_cluster = self.scores_combinados[features_cluster].fillna(50)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_cluster)
        pca = PCA(n_components=min(data_scaled.shape[1], 2))
        data_pca = pca.fit_transform(data_scaled)
        kmeans = KMeans(n_clusters=min(len(data_pca), 4), random_state=42, n_init=10)
        clusters = kmeans.fit_predict(data_pca)
        self.scores_combinados['Final_Cluster'] = clusters

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
        
        combined = self.metricas_performance.join(self.dados_fundamentalistas, how='inner').copy()
        for symbol in combined.index:
            if symbol in self.dados_por_ativo:
                df = self.dados_por_ativo[symbol]
                combined.loc[symbol, 'rsi_current'] = df['rsi_14'].iloc[-1]
                combined.loc[symbol, 'macd_current'] = df['macd'].iloc[-1]
                combined.loc[symbol, 'vol_current'] = df['vol_20d'].iloc[-1]

        scores = pd.DataFrame(index=combined.index)
        scores['performance_score'] = (EngenheiroFeatures._normalize_score(combined['sharpe'], True) * 0.6 + EngenheiroFeatures._normalize_score(combined['retorno_anual'], True) * 0.4) * W_PERF_GLOBAL
        
        s_pl = EngenheiroFeatures._normalize_score(combined.get('pe_ratio', 50), False)
        s_pvp = EngenheiroFeatures._normalize_score(combined.get('pb_ratio', 50), False)
        s_roe = EngenheiroFeatures._normalize_score(combined.get('roe', 50), True)
        s_dy = EngenheiroFeatures._normalize_score(combined.get('div_yield', 0), True)
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
        combined['ML_Proba'] = ml_probs
        combined['ML_Confidence'] = ml_conf
        self.scores_combinados = scores.join(combined).sort_values('total_score', ascending=False)
        
        self.realizar_clusterizacao_final()
        final_selection = []
        if not self.scores_combinados.empty and 'Final_Cluster' in self.scores_combinados.columns:
            clusters_present = self.scores_combinados['Final_Cluster'].unique()
            for c in clusters_present:
                best = self.scores_combinados[self.scores_combinados['Final_Cluster'] == c].head(1).index[0]
                final_selection.append(best)
        
        if len(final_selection) < NUM_ATIVOS_PORTFOLIO:
            others = [x for x in self.scores_combinados.index if x not in final_selection]
            final_selection.extend(others[:NUM_ATIVOS_PORTFOLIO - len(final_selection)])
            
        self.ativos_selecionados = final_selection[:NUM_ATIVOS_PORTFOLIO]
        return self.ativos_selecionados
    
    def otimizar_alocacao(self, nivel_risco: str):
        if not self.ativos_selecionados or len(self.ativos_selecionados) < 1:
            self.metodo_alocacao_atual = "ERRO: Ativos Insuficientes"; return {}
        
        available_assets_returns = {s: self.dados_por_ativo[s]['returns'] for s in self.ativos_selecionados if s in self.dados_por_ativo and 'returns' in self.dados_por_ativo[s]}
        final_returns_df = pd.DataFrame(available_assets_returns).dropna()
        
        if final_returns_df.shape[0] < 50:
            weights = {asset: 1.0 / len(self.ativos_selecionados) for asset in self.ativos_selecionados}
            self.metodo_alocacao_atual = 'PESOS IGUAIS (Dados insuficientes)'; return self._formatar_alocacao(weights)

        garch_vols_filtered = {asset: self.volatilidades_garch.get(asset, final_returns_df[asset].std() * np.sqrt(252)) for asset in final_returns_df.columns}
        optimizer = OtimizadorPortfolioAvancado(final_returns_df, garch_vols=garch_vols_filtered)
        
        if 'CONSERVADOR' in nivel_risco or 'INTERMEDIÁRIO' in nivel_risco:
            strategy = 'MinVolatility'; self.metodo_alocacao_atual = 'MINIMIZAÇÃO DE VOLATILIDADE'
        else:
            strategy = 'MaxSharpe'; self.metodo_alocacao_atual = 'MAXIMIZAÇÃO DE SHARPE'
            
        weights = optimizer.otimizar(estrategia=strategy)
        if not weights:
             weights = {asset: 1.0 / len(self.ativos_selecionados) for asset in self.ativos_selecionados}
             self.metodo_alocacao_atual += " (FALLBACK)"
        return self._formatar_alocacao(weights)
        
    def _formatar_alocacao(self, weights: dict) -> dict:
        if not weights or sum(weights.values()) == 0: return {}
        total_weight = sum(weights.values())
        return {s: {'weight': w / total_weight, 'amount': self.valor_investimento * (w / total_weight)} for s, w in weights.items() if s in self.ativos_selecionados}
    
    def calcular_metricas_portfolio(self):
        if not self.alocacao_portfolio: return {}
        weights_dict = {s: data['weight'] for s, data in self.alocacao_portfolio.items()}
        returns_df_raw = {s: self.dados_por_ativo[s]['returns'] for s in weights_dict.keys() if s in self.dados_por_ativo and 'returns' in self.dados_por_ativo[s]}
        returns_df = pd.DataFrame(returns_df_raw).dropna()
        if returns_df.empty: return {}
        weights = np.array([weights_dict[s] for s in returns_df.columns])
        weights = weights / np.sum(weights) 
        portfolio_returns = (returns_df * weights).sum(axis=1)
        metrics = {
            'annual_return': portfolio_returns.mean() * 252,
            'annual_volatility': portfolio_returns.std() * np.sqrt(252),
            'sharpe_ratio': (portfolio_returns.mean() * 252 - TAXA_LIVRE_RISCO) / (portfolio_returns.std() * np.sqrt(252)) if portfolio_returns.std() > 0 else 0,
            'max_drawdown': ((1 + portfolio_returns).cumprod() / (1 + portfolio_returns).cumprod().expanding().max() - 1).min(),
            'total_investment': self.valor_investimento
        }
        self.metricas_portfolio = metrics
        return metrics

    def gerar_justificativas(self):
        self.justificativas_selecao = {}
        for simbolo in self.ativos_selecionados:
            justification = []
            perf = self.metricas_performance.loc[simbolo] if simbolo in self.metricas_performance.index else pd.Series({})
            fund = self.dados_fundamentalistas.loc[simbolo] if simbolo in self.dados_fundamentalistas.index else pd.Series({})
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
            if progress_bar: progress_bar.progress(10, text="Coletando dados LIVE (YF+Pynvest)...")
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
    def realizar_clusterizacao_pca(dados_ativos: pd.DataFrame, max_clusters: int = 10) -> tuple[pd.DataFrame | None, PCA | None, KMeans | None, int | None]:
        features_cluster = ['sharpe', 'retorno_anual', 'volatilidade_anual', 'pe_ratio', 'pb_ratio', 'roe', 'div_yield']
        features_numericas = dados_ativos.filter(items=features_cluster).select_dtypes(include=[np.number]).copy().replace([np.inf, -np.inf], np.nan).fillna(0)
        if features_numericas.empty or len(features_numericas) < 3: return None, None, None, None
        scaler = StandardScaler()
        dados_normalizados = scaler.fit_transform(features_numericas)
        pca = PCA(n_components=min(3, len(features_numericas.columns)))
        componentes_pca = pca.fit_transform(dados_normalizados)
        best_score = -1; optimal_k = 2
        k_range = range(2, min(max_clusters + 1, len(features_numericas))) 
        if not k_range: return None, None, None, None
        for k in k_range:
            try:
                clusters_k = KMeans(n_clusters=k, random_state=42, n_init='auto').fit_predict(componentes_pca)
                score = silhouette_score(componentes_pca, clusters_k)
                if score > best_score: best_score = score; optimal_k = k
            except Exception: continue 
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto') 
        clusters = kmeans.fit_predict(componentes_pca)
        resultado_pca = pd.DataFrame(componentes_pca, columns=[f'PC{i+1}' for i in range(componentes_pca.shape[1])], index=features_numericas.index)
        resultado_pca['Cluster'] = clusters
        return resultado_pca, pca, kmeans, optimal_k

# =============================================================================
# 13. INTERFACE STREAMLIT - CONFIGURAÇÃO E CSS ORIGINAL (V8.7)
# =============================================================================

def configurar_pagina():
    st.set_page_config(page_title="Sistema de Portfólios Adaptativos", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
        <style>
        :root { --primary-color: #000000; --secondary-color: #6c757d; --background-light: #ffffff; --background-dark: #f8f9fa; --text-color: #212529; --text-color-light: #ffffff; --border-color: #dee2e6; }
        body { background-color: var(--background-light); color: var(--text-color); }
        .main-header { font-family: 'Arial', sans-serif; color: var(--primary-color); text-align: center; border-bottom: 2px solid var(--border-color); padding-bottom: 10px; font-size: 2.2rem !important; margin-bottom: 20px; font-weight: 600; }
        .stButton button, .stDownloadButton button, .stFormSubmitButton button, .stTabs [data-baseweb="tab"], .stMetric label, .main-header, .info-box, h1, h2, h3, h4, h5, p, body { font-family: 'Arial', sans-serif !important; }
        .stButton button, .stDownloadButton button { border: 1px solid var(--primary-color) !important; color: var(--primary-color) !important; border-radius: 6px; padding: 8px 16px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; transition: all 0.3s ease; background-color: transparent !important; }
        .stButton button:hover, .stDownloadButton button:hover { background-color: var(--primary-color) !important; color: var(--text-color-light) !important; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); }
        .stButton button[kind="primary"], .stFormSubmitButton button { background-color: var(--primary-color) !important; color: var(--text-color-light) !important; border: none !important; }
        .stButton button[kind="primary"]:hover, .stFormSubmitButton button:hover { background-color: #333333 !important; color: var(--text-color-light) !important; }
        .stTabs [data-baseweb="tab-list"] { gap: 24px; border-bottom: 2px solid var(--border-color); display: flex; justify-content: center; width: 100%; }
        .stTabs [data-baseweb="tab"] { height: 40px; background-color: transparent; border-radius: 4px 4px 0 0; padding-top: 5px; padding-bottom: 5px; color: var(--secondary-color); font-weight: 500; flex-grow: 0 !important; }
        .stTabs [aria-selected="true"] { background-color: transparent; border-bottom: 2px solid var(--primary-color); color: var(--primary-color); font-weight: 700; }
        .info-box { background-color: var(--background-dark); border-left: 4px solid var(--primary-color); padding: 15px; margin: 10px 0; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
        .stMetric { padding: 10px 15px; background-color: var(--background-dark); border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 10px; }
        .stMetric label { font-weight: 600; color: var(--text-color); }
        .stMetric delta { font-weight: 700; color: #28a745; }
        .stMetric delta[style*="color: red"] { color: #dc3545 !important; }
        .stProgress > div > div > div > div { background-color: var(--primary-color); }
        .reference-block { background-color: #fdfdfd; border: 1px solid var(--border-color); padding: 12px; margin-bottom: 12px; border-radius: 6px; }
        .reference-block p { margin-bottom: 5px; }
        .reference-block .explanation { font-style: italic; color: var(--secondary-color); font-size: 0.95em; border-top: 1px dashed #e0e0e0; padding-top: 8px; margin-top: 8px; }
        </style>
    """, unsafe_allow_html=True)

def aba_introducao():
    """Aba 1: Introdução Metodológica Extensiva (v8.7 Original + Ajuste V9.4 Logic)"""
    
    st.markdown("## 📚 Metodologia Quantitativa e Arquitetura do Sistema")
    
    st.markdown("""
    <div class="info-box">
    <h3>🎯 Visão Geral do Ecossistema</h3>
    <p>Este sistema opera coletando dados em <b>tempo real</b> via APIs financeiras confiáveis. Diferente de versões anteriores baseadas em dados estáticos, esta iteração garante que a análise reflita as condições mais atuais do mercado, integrando <b>Análise Técnica, Fundamentalista e Machine Learning</b> em um pipeline unificado.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('### 1. O "Motor" de Dados (Tempo Real)')
        
        with st.expander("Etapa 1.1: Coleta e Engenharia de Features"):
            st.markdown("""
            Para cada ativo do Ibovespa, o sistema executa uma análise multifacetada em tempo real:
            
            - **Análise Técnica:** Cálculo de indicadores de *momentum* e tendência (RSI, MACD, Bandas de Bollinger, Médias Móveis) usando a biblioteca `yfinance`.
            - **Análise Fundamentalista:** Coleta de métricas de *valuation* e *qualidade* (P/L, P/VP, ROE, Margens) através da biblioteca `pynvest` (Fundamentus).
            - **Análise Estatística:** Cálculo de métricas de risco/retorno (Sharpe Ratio, Max Drawdown) e volatilidade condicional futura via **GARCH(1,1)**.
            """)

        with st.expander("Etapa 1.2: Treinamento de Machine Learning (ML)"):
            st.markdown("""
            O sistema treina modelos de ML dedicados para cada ativo durante a execução.
            
            **Objetivo (Target):**
            O modelo prevê a probabilidade de um retorno positivo futuro (ex: superar a mediana) em horizontes definidos pelo perfil do usuário.
            
            **Arquitetura de Modelo:**
            Utiliza-se um **Pipeline de Random Forest** com pré-processamento robusto (StandardScaler, OneHotEncoder para clusters), garantindo que o modelo capture padrões não-lineares nos dados técnicos e fundamentais.
            
            **Validação:**
            A confiança do modelo é medida através da métrica **AUC-ROC** (Area Under the Curve) usando validação cruzada temporal (*TimeSeriesSplit*), evitando viés de *look-ahead*.
            """)
    
    with col2:
        st.markdown('### 2. O "Painel" de Otimização (Este Aplicativo)')
        
        with st.expander("Etapa 2.1: Definição do Perfil"):
            st.markdown("""
            O aplicativo calibra a estratégia com base em duas variáveis críticas extraídas do questionário do investidor:
            
            1.  **Nível de Risco:** (Conservador, Moderado, etc.) define a função objetivo da otimização (Minimizar Volatilidade vs. Maximizar Sharpe).
            2.  **Horizonte Temporal:** (Curto, Médio, Longo Prazo) define os pesos relativos entre análise técnica e fundamentalista.
            """)

        with st.expander("Etapa 2.2: Ranqueamento Multi-Fatorial (Pesos Dinâmicos)"):
            st.markdown("""
            O **Score Total** de cada ativo é uma combinação ponderada de quatro pilares:
            
            | Pilar | Peso | Descrição |
            | :--- | :--- | :--- |
            | **Performance** | **20% (Fixo)** | Sharpe Ratio e Retorno Histórico. |
            | **Machine Learning** | **20% (Base)** | Probabilidade de Alta ajustada pela Confiança (AUC) do modelo. |
            | **Fundamentos** | **Dinâmico** | P/L, P/VP, ROE. Peso maior para Longo Prazo. |
            | **Técnicos** | **Dinâmico** | RSI, MACD, Volatilidade. Peso maior para Curto Prazo. |
            """)
            
        with st.expander("Etapa 2.3: Seleção por Cluster (Diversificação)"):
            st.markdown("""
            Para evitar concentração em um único "tipo" de ativo, o sistema utiliza **KMeans Clustering** sobre os scores finais.
            
            A seleção dos 5 ativos do portfólio força a escolha do **melhor ativo de cada cluster** (perfil estatístico), garantindo que a carteira contenha, por exemplo, uma mistura equilibrada de ativos de Valor, Crescimento e Momentum.
            """)
            
        with st.expander("Etapa 2.4: Otimização (MPT)"):
            st.markdown("""
            Os pesos finais (10% a 30% por ativo) são definidos pela **Teoria Moderna de Portfólio (Markowitz)**, buscando a fronteira eficiente para o perfil de risco selecionado.
            """)

    st.markdown("---")
    st.info("""
    **Próxima Etapa:**
    Utilize o menu de abas para navegar até **'Seleção de Ativos'** e, em seguida, **'Construtor de Portfólio'** para gerar sua alocação otimizada.
    """)

def aba_selecao_ativos():
    """Aba 2: Seleção de Ativos (Design Original Restaurado)"""
    
    st.markdown("## 🎯 Definição do Universo de Análise")
    
    st.markdown("""
    <div class="info-box">
    <p>O universo de análise está restrito ao **Índice Ibovespa**. O sistema utiliza todos os ativos selecionados para realizar o ranqueamento multi-fatorial e otimizar a carteira.</p>
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
        col1, col2 = st.columns([2, 1])
        
        with col1:
            setores_selecionados = st.multiselect(
                "Escolha um ou mais setores:",
                options=setores_disponiveis,
                default=setores_disponiveis[:3] if setores_disponiveis else [],
                key='setores_multiselect_v8'
            )
        
        if setores_selecionados:
            for setor in setores_selecionados: ativos_selecionados.extend(ATIVOS_POR_SETOR[setor])
            ativos_selecionados = list(set(ativos_selecionados))
            
            with col2:
                st.metric("Setores", len(setores_selecionados))
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
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("#### 📝 Selecione Tickers (Ibovespa)")
            ativos_selecionados = st.multiselect(
                "Pesquise e selecione os tickers:",
                options=todos_tickers_ibov,
                format_func=lambda x: f"{x.replace('.SA', '')} - {ativos_com_setor.get(x, 'Desconhecido')}",
                key='ativos_individuais_multiselect_v8'
            )
        
        with col2:
            st.metric("Tickers Selecionados", len(ativos_selecionados))

        if not ativos_selecionados:
            st.warning("⚠️ Nenhum ativo definido.")
    
    if ativos_selecionados:
        st.session_state.ativos_para_analise = ativos_selecionados
        st.markdown("---")
        
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
    
    progress_bar_placeholder = st.empty()
    
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
            
            submitted = st.form_submit_button("🚀 Gerar Alocação Otimizada", type="primary", key='submit_optimization_button_v8')
            
            if submitted:
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
                
                try:
                    builder_local = ConstrutorPortfolioAutoML(investment)
                    st.session_state.builder = builder_local
                except Exception as e:
                    st.error(f"Erro fatal ao inicializar o construtor do portfólio: {e}")
                    return

                progress_widget = progress_bar_placeholder.progress(0, text=f"Iniciando pipeline para PERFIL {risk_level}...")
                
                success = builder_local.executar_pipeline(
                    simbolos_customizados=st.session_state.ativos_para_analise,
                    perfil_inputs=st.session_state.profile,
                    progress_bar=progress_widget
                )
                
                progress_bar_placeholder.empty()
                    
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
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Perfil Identificado", profile.get('risk_level', 'N/A'), f"Score: {profile.get('risk_score', 'N/A')}")
        col2.metric("Horizonte Estratégico", profile.get('time_horizon', 'N/A'))
        col3.metric("Sharpe Ratio (Portfólio)", f"{builder.metricas_portfolio.get('sharpe_ratio', 0):.3f}")
        col4.metric("Estratégia de Alocação", builder.metodo_alocacao_atual.split('(')[0].strip())
        
        if st.button("🔄 Recalibrar Perfil e Otimizar", key='recomecar_analysis_button_v8'):
            st.session_state.builder_complete = False
            st.session_state.builder = None
            st.session_state.profile = {}
            st.rerun()
        
        st.markdown("---")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Alocação de Capital", "📈 Performance e Retornos", "🤖 Fator Predição ML", "📉 Fator Volatilidade GARCH", "❓ Justificativas e Ranqueamento"
        ])
        
        with tab1:
            col_alloc, col_table = st.columns([1, 2])
            
            with col_alloc:
                st.markdown('#### Distribuição do Capital')
                alloc_data = pd.DataFrame([
                    {'Ativo': a.replace('.SA', ''), 'Peso (%)': allocation[a]['weight'] * 100}
                    for a in assets if a in allocation and allocation[a]['weight'] > 0.001
                ])
                
                if not alloc_data.empty:
                    fig_alloc = px.pie(alloc_data, values='Peso (%)', names='Ativo', hole=0.3)
                    fig_layout = obter_template_grafico()
                    fig_layout['title']['text'] = "Distribuição Otimizada por Ativo"
                    fig_alloc.update_layout(**fig_layout)
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
                        sector = builder.dados_fundamentalistas.loc[asset, 'sector'] if asset in builder.dados_fundamentalistas.index and 'sector' in builder.dados_fundamentalistas.columns else 'Unknown'
                        ml_info = builder.predicoes_ml.get(asset, {})
                        
                        alloc_table.append({
                            'Ticker': asset.replace('.SA', ''), 
                            'Setor': sector,
                            'Peso (%)': f"{weight * 100:.2f}",
                            'Valor (R$)': f"R$ {amount:,.2f}",
                            'ML Prob. Alta (%)': f"{ml_info.get('predicted_proba_up', 0.5)*100:.1f}",
                            'ML Confiança': f"{ml_info.get('auc_roc_score', 0):.3f}" if not pd.isna(ml_info.get('auc_roc_score')) else "N/A",
                        })
                
                df_alloc = pd.DataFrame(alloc_table)
                st.dataframe(df_alloc, use_container_width=True, hide_index=True)
        
        with tab2:
            st.markdown('#### Métricas Chave do Portfólio (Histórico Recente)')
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Retorno Anualizado", f"{builder.metricas_portfolio.get('annual_return', 0)*100:.2f}%")
            col2.metric("Volatilidade Anualizada", f"{builder.metricas_portfolio.get('annual_volatility', 0)*100:.2f}%")
            col3.metric("Sharpe Ratio", f"{builder.metricas_portfolio.get('sharpe_ratio', 0):.3f}")
            col4.metric("Máximo Drawdown", f"{builder.metricas_portfolio.get('max_drawdown', 0)*100:.2f}%")
            
            st.markdown("---")
            st.markdown('#### Trajetória de Retornos Cumulativos')
            
            fig_cum = go.Figure()
            
            for asset in assets:
                if asset in builder.dados_por_ativo and 'returns' in builder.dados_por_ativo[asset]:
                    returns = builder.dados_por_ativo[asset]['returns']
                    cum_returns = (1 + returns).cumprod()
                    
                    fig_cum.add_trace(go.Scatter(
                        x=cum_returns.index, y=cum_returns.values, name=asset.replace('.SA', ''), mode='lines'
                    ))
            
            fig_layout = obter_template_grafico()
            fig_layout['title']['text'] = "Retorno Acumulado dos Tickers Selecionados"
            fig_layout['yaxis']['title'] = "Retorno Acumulado (Base 1)"
            fig_layout['xaxis']['title'] = "Data"
            fig_cum.update_layout(**fig_layout, height=500)
            
            st.plotly_chart(fig_cum, use_container_width=True)
        
        with tab3:
            st.markdown('#### Contribuição do Fator Predição ML')
            
            ml_data = []
            for asset in assets:
                if asset in builder.predicoes_ml:
                    ml_info = builder.predicoes_ml[asset]
                    ml_data.append({
                        'Ticker': asset.replace('.SA', ''),
                        'Prob. Alta (%)': ml_info.get('predicted_proba_up', 0.5) * 100,
                        'Confiança (AUC-ROC)': ml_info.get('auc_roc_score', np.nan),
                        'Modelo': ml_info.get('model_name', 'N/A')
                    })
            
            df_ml = pd.DataFrame(ml_data)
            
            if not df_ml.empty:
                fig_ml = go.Figure()
                plot_df_ml = df_ml.sort_values('Prob. Alta (%)', ascending=False)
                
                fig_ml.add_trace(go.Bar(
                    x=plot_df_ml['Ticker'],
                    y=plot_df_ml['Prob. Alta (%)'],
                    marker=dict(
                        color=plot_df_ml['Prob. Alta (%)'],
                        colorscale='Greys', # Escala de cinza
                        showscale=True,
                        colorbar=dict(title="Prob. (%)")
                    ),
                    text=plot_df_ml['Prob. Alta (%)'].round(1),
                    textposition='outside'
                ))
                
                fig_layout = obter_template_grafico()
                fig_layout['title']['text'] = "Probabilidade de Movimento Direcional Positivo (ML Ensemble)"
                fig_layout['yaxis']['title'] = "Probabilidade (%)"
                fig_layout['xaxis']['title'] = "Ticker"
                fig_ml.update_layout(**fig_layout, height=400)
                
                st.plotly_chart(fig_ml, use_container_width=True)
                
                st.markdown("---")
                st.markdown('#### Detalhamento da Predição')
                df_ml_display = df_ml.copy()
                df_ml_display['Prob. Alta (%)'] = df_ml_display['Prob. Alta (%)'].round(2)
                df_ml_display['Confiança (AUC-ROC)'] = df_ml_display['Confiança (AUC-ROC)'].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A")
                st.dataframe(df_ml_display, use_container_width=True, hide_index=True)
            else:
                st.warning("Não há dados de Predição ML para exibir.")
        
        with tab4:
            st.markdown('#### Volatilidade Condicional (GARCH) e Histórica')
            
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
                plot_df_garch = df_garch[df_garch['Vol. Condicional (%)'] != 'N/A'].copy()
                plot_df_garch['Vol. Condicional (%)'] = plot_df_garch['Vol. Condicional (%)'].astype(float)
                plot_df_garch['Vol. Histórica (%)'] = plot_df_garch['Vol. Histórica (%)'].apply(lambda x: float(x) if x != 'N/A' else np.nan)

                template_colors = obter_template_grafico()['colorway']
                
                fig_garch.add_trace(go.Bar(name='Volatilidade Histórica', x=plot_df_garch['Ticker'], y=plot_df_garch['Vol. Histórica (%)'], marker=dict(color=template_colors[2]), opacity=0.7)) 
                fig_garch.add_trace(go.Bar(name='Volatilidade Condicional', x=plot_df_garch['Ticker'], y=plot_df_garch['Vol. Condicional (%)'], marker=dict(color=template_colors[0]))) 
                
                fig_layout = obter_template_grafico()
                fig_layout['title']['text'] = "Volatilidade Anualizada: Histórica vs. Condicional (GARCH)"
                fig_layout['yaxis']['title'] = "Volatilidade Anual (%)"
                fig_layout['barmode'] = 'group'
                fig_garch.update_layout(**fig_layout, height=400)
                
                st.plotly_chart(fig_garch, use_container_width=True)
                st.dataframe(df_garch, use_container_width=True, hide_index=True)
            else:
                st.warning("Não há dados de volatilidade para exibir.")
        
        with tab5:
            st.markdown('#### Ranqueamento Final e Justificativas Detalhadas')
            
            st.markdown(f"**Pesos Adaptativos Usados:** Performance: {builder.pesos_atuais['Performance']:.2f} | Fundamentos: {builder.pesos_atuais['Fundamentos']:.2f} | Técnicos: {builder.pesos_atuais['Técnicos']:.2f} | ML: {builder.pesos_atuais['ML']:.2f}")
            st.markdown("---")
            
            cols_to_display_scores = [
                'total_score', 'performance_score', 'fundamental_score', 'technical_score', 'ml_score_weighted', 
                'sharpe_ratio', 'pe_ratio', 'roe', 'rsi_14', 'macd_diff', 'ML_Proba'
            ]
            
            cols_existentes = [col for col in cols_to_display_scores if col in builder.scores_combinados.columns]
            
            df_scores_display = builder.scores_combinados[cols_existentes].copy()
            df_scores_display.columns = [
                'Score Total', 'Score Perf.', 'Score Fund.', 'Score Téc.', 'Score ML', 
                'Sharpe', 'P/L', 'ROE', 'RSI 14', 'MACD Hist.', 'Prob. Alta ML'
            ]
            
            if 'ROE' in df_scores_display.columns:
                 df_scores_display['ROE'] = df_scores_display['ROE'] * 100
                 
            df_scores_display = df_scores_display.iloc[:15] 
            
            st.markdown("##### Ranqueamento Ponderado Multi-Fatorial (Top 15 Tickers do Universo Analisado)")
            st.dataframe(df_scores_display.style.format(
                {
                    'Score Total': '{:.3f}', 'Score Perf.': '{:.3f}', 'Score Fund.': '{:.3f}', 'Score Téc.': '{:.3f}', 'Score ML': '{:.3f}',
                    'Sharpe': '{:.3f}', 'P/L': '{:.2f}', 'ROE': '{:.2f}%', 'RSI 14': '{:.2f}', 'MACD Hist.': '{:.4f}', 'Prob. Alta ML': '{:.2f}'
                }
            ).background_gradient(cmap='Greys', subset=['Score Total']), use_container_width=True)
            
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

def aba_analise_individual():
    """Aba 4: Análise Individual de Ativos"""
    
    st.markdown("## 🔍 Análise de Fatores por Ticker")
    
    if 'ativos_para_analise' in st.session_state and st.session_state.ativos_para_analise:
        ativos_disponiveis = sorted(list(set(st.session_state.ativos_para_analise)))
    else:
        ativos_disponiveis = TODOS_ATIVOS 
            
    if not ativos_disponiveis:
        st.error("Nenhum ativo disponível. Verifique a seleção ou o universo padrão.")
        return

    if 'individual_asset_select_v8' not in st.session_state or st.session_state.individual_asset_select_v8 not in ativos_disponiveis:
        st.session_state.individual_asset_select_v8 = ativos_disponiveis[0] if ativos_disponiveis else None

    col1, col2 = st.columns([3, 1])
    
    with col1:
        ativo_selecionado = st.selectbox(
            "Selecione um ticker para análise detalhada:",
            options=ativos_disponiveis,
            format_func=lambda x: x.replace('.SA', '') if isinstance(x, str) else x,
            key='individual_asset_select_v8' 
        )
    
    with col2:
        if st.button("🔄 Executar Análise", key='analyze_asset_button_v8', type="primary"):
            st.session_state.analisar_ativo_triggered = True 
    
    if 'analisar_ativo_triggered' not in st.session_state or not st.session_state.analisar_ativo_triggered:
        st.info("👆 Selecione um ticker e clique em 'Executar Análise' para obter o relatório completo.")
        return
    
    with st.spinner(f"Processando análise de fatores para {ativo_selecionado} (Live Data)..."):
        try:
            df_completo = None
            features_fund = None
            df_ml_meta = None 
            
            builder_existe = 'builder' in st.session_state and st.session_state.builder is not None
            if builder_existe and ativo_selecionado in st.session_state.builder.dados_por_ativo:
                builder = st.session_state.builder
                df_completo = builder.dados_por_ativo[ativo_selecionado].copy().dropna(how='all')
                
                if ativo_selecionado in builder.dados_fundamentalistas.index:
                    features_fund = builder.dados_fundamentalistas.loc[ativo_selecionado].to_dict()
                else:
                    _, features_fund, df_ml_meta = ColetorDadosLive().coletar_ativo_unico_gcs(ativo_selecionado)
                
            if df_completo is None or df_completo.empty or features_fund is None:
                df_completo, features_fund, df_ml_meta = ColetorDadosLive().coletar_ativo_unico_gcs(ativo_selecionado)
                if df_completo is not None: df_completo = df_completo.dropna(how='all')

            if df_completo is None or df_completo.empty or 'Close' not in df_completo.columns or features_fund is None:
                st.error(f"❌ Não foi possível obter dados válidos para **{ativo_selecionado.replace('.SA', '')}**.")
                return

            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Histórico e Visão Geral",
                "💼 Fatores Fundamentalistas",
                "🔧 Fatores Técnicos e Momentum",
                "🤖 Fatores de Machine Learning", 
                "🔬 Similaridade e Clusterização"
            ])
            
            with tab1:
                st.markdown(f"### {ativo_selecionado.replace('.SA', '')} - Fatores Chave de Mercado")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                preco_atual = df_completo['Close'].iloc[-1]
                variacao_dia = df_completo['returns'].iloc[-1] * 100 if 'returns' in df_completo.columns and not df_completo['returns'].empty else 0.0
                volume_medio = df_completo['Volume'].mean() if 'Volume' in df_completo.columns else 0.0
                
                col1.metric("Preço de Fechamento", f"R$ {preco_atual:.2f}", f"{variacao_dia:+.2f}%")
                col2.metric("Volume Médio Recente", f"{volume_medio:,.0f}")
                col3.metric("Setor", features_fund.get('sector', 'N/A'))
                col4.metric("Indústria", features_fund.get('industry', 'N/A'))
                col5.metric("Vol. Anualizada", f"{features_fund.get('annual_volatility', np.nan)*100:.2f}%" if not pd.isna(features_fund.get('annual_volatility')) else "N/A")
                
                if not df_completo.empty and 'Open' in df_completo.columns and 'Volume' in df_completo.columns:
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                    
                    fig.add_trace(go.Candlestick(x=df_completo.index, open=df_completo['Open'], high=df_completo['High'], low=df_completo['Low'], close=df_completo['Close'], name='Preço'), row=1, col=1)
                    
                    template_colors = obter_template_grafico()['colorway']
                    fig.add_trace(go.Bar(x=df_completo.index, y=df_completo['Volume'], name='Volume', marker=dict(color=template_colors[2]), opacity=0.7), row=2, col=1)
                    
                    fig_layout = obter_template_grafico()
                    fig_layout['title']['text'] = f"Série Temporal de Preços e Volume - {ativo_selecionado.replace('.SA', '')}"
                    fig_layout['height'] = 600
                    fig.update_layout(**fig_layout)
                    st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.markdown("### Fatores Fundamentalistas Detalhados")
                
                st.markdown("#### Valuation e Crescimento")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                col1.metric("P/L (Valuation)", f"{features_fund.get('pe_ratio', np.nan):.2f}" if not pd.isna(features_fund.get('pe_ratio')) else "N/A")
                col2.metric("P/VP", f"{features_fund.get('pb_ratio', np.nan):.2f}" if not pd.isna(features_fund.get('pb_ratio')) else "N/A")
                col3.metric("ROE (Rentabilidade)", f"{features_fund.get('roe', np.nan)*100:.2f}%" if not pd.isna(features_fund.get('roe')) else "N/A")
                col4.metric("Margem Operacional", f"{features_fund.get('operating_margin', np.nan)*100:.2f}%" if not pd.isna(features_fund.get('operating_margin')) else "N/A") 
                col5.metric("Cresc. Receita Anual", f"{features_fund.get('revenue_growth', np.nan)*100:.2f}%" if not pd.isna(features_fund.get('revenue_growth')) else "N/A")
                
                st.markdown("#### Saúde Financeira e Dividendo")
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Dívida/Patrimônio", f"{features_fund.get('debt_to_equity', np.nan):.2f}" if not pd.isna(features_fund.get('debt_to_equity')) else "N/A")
                col2.metric("Current Ratio", f"{features_fund.get('current_ratio', np.nan):.2f}" if not pd.isna(features_fund.get('current_ratio')) else "N/A")
                col3.metric("Dividend Yield", f"{features_fund.get('div_yield', np.nan):.2f}%" if not pd.isna(features_fund.get('div_yield')) else "N/A")
                col4.metric("Beta (Risco Sistêmico)", f"{features_fund.get('beta', np.nan):.2f}" if not pd.isna(features_fund.get('beta')) else "N/A")
                
                st.markdown("---")
                st.markdown("#### Tabela de Fatores Fundamentais")
                
                keys_to_exclude = ['pe_ratio', 'roe'] 
                df_fund_display = pd.DataFrame({
                    'Métrica': [k for k in features_fund.keys() if k not in keys_to_exclude],
                    'Valor': [f"{v:.4f}" if isinstance(v, (int, float)) and not pd.isna(v) else str(v) 
                              for k, v in features_fund.items() if k not in keys_to_exclude]
                })
                
                st.dataframe(df_fund_display, use_container_width=True, hide_index=True)
            
            with tab3:
                st.markdown("### Fatores Técnicos e de Momentum")
                
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("RSI (14)", f"{df_completo['rsi_14'].iloc[-1]:.2f}" if 'rsi_14' in df_completo.columns and not df_completo['rsi_14'].empty else "N/A")
                col2.metric("MACD (Signal Diff)", f"{df_completo['macd_diff'].iloc[-1]:.4f}" if 'macd_diff' in df_completo.columns and not df_completo['macd_diff'].empty else "N/A")
                col3.metric("BBands Largura", f"{df_completo['bb_width'].iloc[-1]:.2f}" if 'bb_width' in df_completo.columns and not df_completo['bb_width'].empty else "N/A")
                col4.metric("Momento (ROC 60d)", f"{df_completo['momentum_60'].iloc[-1]*100:.2f}%" if 'momentum_60' in df_completo.columns and not df_completo['momentum_60'].empty else "N/A")

                st.markdown("#### Indicadores de Força e Volatilidade (Gráfico)")
                
                fig_osc = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("RSI (14) - Força Relativa", "MACD - Convergência/Divergência"))
                template_colors = obter_template_grafico()['colorway']

                if 'rsi_14' in df_completo.columns:
                    fig_osc.add_trace(go.Scatter(x=df_completo.index, y=df_completo['rsi_14'], name='RSI', line=dict(color=template_colors[0])), row=1, col=1)
                    fig_osc.add_hline(y=70, line_dash="dash", line_color="#dc3545", row=1, col=1)
                    fig_osc.add_hline(y=30, line_dash="dash", line_color="#28a745", row=1, col=1)
                
                if 'macd' in df_completo.columns and 'macd_signal' in df_completo.columns:
                    fig_osc.add_trace(go.Scatter(x=df_completo.index, y=df_completo['macd'], name='MACD', line=dict(color=template_colors[1])), row=2, col=1)
                    fig_osc.add_trace(go.Scatter(x=df_completo.index, y=df_completo['macd_signal'], name='Signal', line=dict(color=template_colors[3])), row=2, col=1)
                    if 'macd_diff' in df_completo.columns:
                        fig_osc.add_trace(go.Bar(x=df_completo.index, y=df_completo['macd_diff'], name='Histograma', marker=dict(color='#e9ecef')), row=2, col=1)
                
                fig_layout = obter_template_grafico()
                fig_layout['height'] = 550
                fig_osc.update_layout(**fig_layout)
                
                st.plotly_chart(fig_osc, use_container_width=True)

            with tab4:
                st.markdown("### Fatores de Machine Learning (Ensemble)")
                
                st.info("Nota: Dados de ML gerados em tempo real pelo Pipeline Random Forest. Metadados detalhados (JSON) não disponíveis na versão Live.")
                
                # Recupera dados do DF técnico
                proba = df_completo['ML_Proba'].iloc[-1] if 'ML_Proba' in df_completo.columns else 0.5
                auc = df_completo['ML_Confidence'].iloc[-1] if 'ML_Confidence' in df_completo.columns else np.nan
                
                col1, col2 = st.columns(2)
                col1.metric("Probabilidade Alta (Ensemble)", f"{proba*100:.2f}%")
                col2.metric("Confiança (AUC-ROC)", f"{auc:.3f}" if not pd.isna(auc) else "N/A")

            with tab5: 
                st.markdown("### Análise de Similaridade e Clusterização")
                
                if not builder_existe or builder.metricas_performance.empty or builder.dados_fundamentalistas.empty:
                    st.warning("A Clusterização está desabilitada. É necessário executar o **'Construtor de Portfólio'** (Aba 3) para carregar os dados de comparação de múltiplos ativos.")
                    return
                
                df_comparacao = builder.metricas_performance.join(builder.dados_fundamentalistas, how='inner', rsuffix='_fund')
                
                if 'pe_ratio' not in df_comparacao.columns and 'fund_pe_ratio' in df_comparacao.columns:
                    df_comparacao['pe_ratio'] = df_comparacao['fund_pe_ratio']
                if 'roe' not in df_comparacao.columns and 'fund_roe' in df_comparacao.columns:
                    df_comparacao['roe'] = df_comparacao['fund_roe']
                
                if len(df_comparacao) < 10:
                    st.warning("Dados insuficientes para realizar a clusterização (menos de 10 ativos com métricas completas).")
                    return
                    
                resultado_pca, pca, kmeans, optimal_k = AnalisadorIndividualAtivos.realizar_clusterizacao_pca(
                    df_comparacao, 
                    max_clusters=min(10, len(df_comparacao) - 1)
                )
                
                if resultado_pca is not None:
                    st.info(f"Análise de clusterização encontrou **{optimal_k} clusters** ótimos (via Silhouette Score).")
                    
                    hover_names = resultado_pca.index.str.replace('.SA', '')
                    template_colors = obter_template_grafico()['colorway']

                    if 'PC3' in resultado_pca.columns:
                        fig_pca = px.scatter_3d(
                            resultado_pca, 
                            x='PC1', y='PC2', z='PC3', 
                            color=resultado_pca['Cluster'].astype(str),
                            hover_name=hover_names, 
                            title='Similaridade de Tickers (PCA/K-means - 3D)',
                            color_discrete_sequence=template_colors
                        )
                    else:
                        fig_pca = px.scatter(
                            resultado_pca, 
                            x='PC1', y='PC2', 
                            color=resultado_pca['Cluster'].astype(str),
                            hover_name=hover_names, 
                            title='Similaridade de Tickers (PCA/K-means - 2D)',
                            color_discrete_sequence=template_colors
                        )
                    
                    fig_pca.update_layout(**obter_template_grafico(), height=600)
                    st.plotly_chart(fig_pca, use_container_width=True)
                    
                    if ativo_selecionado in resultado_pca.index:
                        cluster_ativo = resultado_pca.loc[ativo_selecionado, 'Cluster']
                        ativos_similares_df = resultado_pca[resultado_pca['Cluster'] == cluster_ativo]
                        ativos_similares = [a for a in ativos_similares_df.index.tolist() if a != ativo_selecionado]
                        
                        st.success(f"**{ativo_selecionado.replace('.SA', '')}** pertence ao **Cluster {cluster_ativo}**")
                        
                        if ativos_similares:
                            st.markdown(f"#### Tickers Similares no Cluster {cluster_ativo}:")
                            st.write(", ".join([a.replace('.SA', '') for a in ativos_similares]))
                        else:
                            st.info("Nenhum outro ticker similar encontrado neste cluster.")

                else:
                    st.warning("Não foi possível realizar a clusterização.")
        
        except Exception as e:
            st.error(f"Erro ao analisar o ticker {ativo_selecionado}: {str(e)}")
            st.code(traceback.format_exc())

def aba_referencias():
    """Aba 5: Referências Bibliográficas Completas (V8.7 Original)"""
    
    st.markdown("## 📚 Referências e Bibliografia")
    st.markdown("Esta seção consolida as referências bibliográficas indicadas nas ementas das disciplinas relacionadas (GRDECO222 e GRDECO203).")

    st.markdown("---")
    
    st.markdown("### GRDECO222: Machine Learning (Prof. Rafael Martins de Souza)")
    
    st.markdown("#### Bibliografia Obrigatória")
    st.markdown(
        """
        <div class="reference-block">
            <p><strong>1. Jupter Notebooks apresentados em sala de aula.</strong></p>
            <p class="explanation">
            Explicação: O material principal do curso é prático, baseado nos códigos e exemplos desenvolvidos
            pelo professor durante as aulas.
            </p>
        </div>
        <div class="reference-block">
            <p><strong>2. Géron, A. Mãos à Obra: Aprendizado de Máquina com Scikit-Learn, Keras e TensorFlow.</strong></p>
            <p class="explanation">
            Explicação: Considerado um dos principais livros-texto práticos sobre Machine Learning.
            Cobre desde os fundamentos (Regressão, SVMs, Árvores de Decisão) até tópicos avançados
            de Deep Learning, com foco na implementação usando bibliotecas Python populares.
            </p>
        </div>
        """, unsafe_allow_html=True
    )
    
    st.markdown("#### Bibliografia Complementar")
    st.markdown(
        """
        <div class="reference-block">
            <p><strong>1. Coleman, C., Spencer Lyon, S., Jesse Perla, J. QuantEcon Data Science, Introduction to Economic Modeling and Data Science. (https://datascience.quantecon.org/)</strong></p>
            <p class="explanation">
            Explicação: Um recurso online focado na aplicação de Ciência de Dados especificamente
            para modelagem econômica, alinhado com os objetivos da disciplina.
            </p>
        </div>
        <div class="reference-block">
            <p><strong>2. Sargent, T. J., Stachurski, J., Quantitative Economics with Python. (https://python.quantecon.org/)</strong></p>
            <p class="explanation">
            Explicação: Outro projeto da QuantEcon, focado em métodos quantitativos e economia computacional
            usando Python. É uma referência padrão para economistas que programam.
            </p>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown("---")
    
    st.markdown("### GRDECO203: Laboratório de Ciência de Dados Aplicados à Finanças (Prof. Diogo Tavares Robaina)")

    st.markdown("#### Bibliografia Básica")
    st.markdown(
        """
        <div class="reference-block">
            <p><strong>1. HILPISCH, Y. J. Python for finance: analyze big financial dat. O'Reilly Media, 2015.</strong></p>
            <p class="explanation">
            Explicação: Uma referência clássica para finanças quantitativas em Python. Cobre manipulação
            de dados financeiros (séries temporais), análise de risco, e implementação de estratégias
            de trading e precificação de derivativos.
            </p>
        </div>
        <div class="reference-block">
            <p><strong>2. ARRATIA, A. Computational finance an introductory course with R. Atlantis, 2014.</strong></p>
            <p class="explanation">
            Explicação: Focado em finanças computacionais usando a linguagem R, abordando conceitos
            introdutórios e modelagem.
            </p>
        </div>
        <div class="reference-block">
            <p><strong>3. RASCHKA, S. Python machine learning: unlock deeper insights... Packt Publishing, 2015.</strong></p>
            <p class="explanation">
            Explicação: Um guia popular focado na aplicação prática de algoritmos de Machine Learning
            com Scikit-Learn em Python, similar ao livro de Géron.
            </p>
        </div>
        <div class="reference-block">
            <p><strong>4. MAINDONALD, J., and Braun, J. Data analysis and graphics using R: an example-based approach. Cambridge University Press, 2006.</strong></p>
            <p class="explanation">
            Explicação: Livro focado em análise de dados e visualização gráfica utilizando a linguagem R.
            </p>
        </div>
        <div class="reference-block">
            <p><strong>5. REYES, J. M. M. Introduction to Data Science for Social and Policy Research. Cambridge University Press, 2017.</strong></p>
            <p class="explanation">
            Explicação: Aborda a aplicação de Ciência de Dados no contexto de ciências sociais e pesquisa
            de políticas públicas, relevante para a análise econômica.
            </p>
        </div>
        """, unsafe_allow_html=True
    )
    
    st.markdown("#### Bibliografia Complementar")
    st.markdown(
        """
        <div class="reference-block">
            <p><strong>1. TEAM, R. Core. "R language definition." R foundation for statistical computing (2000).</strong></p>
            <p class="explanation">Explicação: A documentação oficial da linguagem R.</p>
        </div>
        <div class="reference-block">
            <p><strong>2. MISHRA, R.; RAM, B. Portfolio Selection Using R. Yugoslav Journal of Operations Research, 2020.</strong></p>
            <p class="explanation">Explicação: Um artigo de pesquisa focado especificamente na aplicação da
            linguagem R para otimização e seleção de portfólios, muito relevante para a disciplina.
            </p>
        </div>
        <div class="reference-block">
            <p><strong>3. WICKHAM, H., et al. (dplyr, Tidy data, Advanced R, ggplot2, R for data science).</strong></p>
            <p class="explanation">
            Explicação: Múltiplas referências de Hadley Wickham, o criador do "Tidyverse" em R.
            São os pacotes e livros fundamentais para a manipulação de dados moderna (dplyr),
            organização (Tidy data) e visualização (ggplot2) na linguagem R.
            </p>
        </div>
        """, unsafe_allow_html=True
    )

def main():
    if 'builder' not in st.session_state:
        st.session_state.builder = None
        st.session_state.builder_complete = False
        st.session_state.profile = {}
        st.session_state.ativos_para_analise = []
        st.session_state.analisar_ativo_triggered = False
        
    configurar_pagina()
    st.markdown('<h1 class="main-header">Sistema de Portfólios Adaptativos</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Metodologia", "🎯 Seleção de Ativos", "🏗️ Construtor de Portfólio", "🔍 Análise Individual", "📖 Referências"])
    
    with tab1: aba_introducao()
    with tab2: aba_selecao_ativos()
    with tab3: aba_construtor_portfolio()
    with tab4: aba_analise_individual()
    with tab5: aba_referencias()

if __name__ == "__main__":
    main()
