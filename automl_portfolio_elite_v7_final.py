# -*- coding: utf-8 -*-
"""
=============================================================================
SISTEMA DE PORTFÓLIOS ADAPTATIVOS - OTIMIZAÇÃO QUANTITATIVA (LIVE VERSION)
=============================================================================

Adaptação do Sistema AutoML para processamento "In-Live" (Tempo Real).
- Dados de Cotação: yfinance
- Dados Fundamentalistas: Scraper Fundamentus (Classe Dedicada)
- Machine Learning: Treinamento instantâneo (RandomForest)
- Design e Lógica de Otimização: Mantidos da v8.7.0

Versão: 9.0.0 (Live Data & Training)
=============================================================================
"""

# --- 1. CORE LIBRARIES & UTILITIES ---
import warnings
import numpy as np
import pandas as pd
import sys
import os
import time
from datetime import datetime, timedelta, timezone
import json
import traceback
import logging

# --- 2. WEB SCRAPING & DATA ---
import requests
from bs4 import BeautifulSoup
import yfinance as yf

# --- 3. SCIENTIFIC / STATISTICAL TOOLS ---
from scipy.optimize import minimize
from scipy.stats import zscore, norm

# --- 4. STREAMLIT & PLOTTING ---
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- 5. MACHINE LEARNING (SCIKIT-LEARN) ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# --- 6. TIME SERIES ---
from arch import arch_model

# --- CONFIGURATION ---
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURAÇÕES E CONSTANTES GLOBAIS
# =============================================================================

# Período estendido para garantir dados suficientes para TA e ML
PERIODO_DADOS = '5y' 
MIN_DIAS_HISTORICO = 252
NUM_ATIVOS_PORTFOLIO = 5
TAXA_LIVRE_RISCO = 0.1075
LOOKBACK_ML = 30
SCORE_PERCENTILE_THRESHOLD = 0.85 

# Mapeamento de Prazos 
LOOKBACK_ML_DAYS_MAP = {
    'curto_prazo': 84,
    'medio_prazo': 168,
    'longo_prazo': 252
}

# =============================================================================
# 2. CLASSE FUNDAMENTUS (FORNECIDA)
# =============================================================================

# Configuração básica de logging para a classe
logging.basicConfig(level=logging.INFO)

# URLs e Constantes do Fundamentus
URL_TICKERS_ACOES = "https://www.fundamentus.com.br/resultado.php"
URL_TICKERS_FIIS = "https://www.fundamentus.com.br/fii_resultado.php"
URL_KPIS_TICKER = "https://www.fundamentus.com.br/detalhes.php?papel="
REQUEST_HEADER = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}
VARIATION_HEADINGS = ["Dia", "Mês", "30 dias", "12 meses", "2023", "2022", "2021", "2020", "2019", "2018"]

# Metadados para mapeamento (Simplificado para uso interno, mas mantendo estrutura)
METADATA_COLS_ACOES = {
    "P/L": "pe_ratio", "P/VP": "pb_ratio", "ROE": "roe", "Div. Yield": "div_yield",
    "Cresc. Rec (5a)": "revenue_growth", "Dív.Brut/ Patrim.": "debt_to_equity",
    "Marg. Líquida": "net_margin", "Marg. EBIT": "operating_margin", "Vol $ med (2m)": "volume_medio_2m",
    "Setor": "sector", "Subsetor": "industry", "Liquidez Corr": "current_ratio"
}
# Adicionando mapeamento reverso para parsing
METADATA_COLS_ACOES_INV = {k: v for k, v in METADATA_COLS_ACOES.items()}

METADATA_COLS_FIIS = {
    "Dividend Yield": "div_yield", "P/VP": "pb_ratio", "Valor de Mercado": "market_cap",
    "Liquidez": "liquidity", "Qtd de imóveis": "num_properties", 
    "Vacância Média": "vacancy_rate"
}

def log_config(logger_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(logger_level)
    return logger

class Fundamentus:
    """Classe responsável por extrair informações do site Fundamentus."""

    def __init__(
        self,
        logger_level: int = logging.INFO,
        url_tickers_acoes: str = URL_TICKERS_ACOES,
        url_tickers_fiis: str = URL_TICKERS_FIIS,
        url_kpis_ticker: str = URL_KPIS_TICKER,
        request_header: dict = REQUEST_HEADER,
        variation_headings: list = VARIATION_HEADINGS,
        metadata_cols_acoes: dict = METADATA_COLS_ACOES,
        metadata_cols_fiis: dict = METADATA_COLS_FIIS
    ) -> None:
        self.logger_level = logger_level
        self.logger = log_config(logger_level=self.logger_level)
        self.url_tickers_acoes = url_tickers_acoes
        self.url_tickers_fiis = url_tickers_fiis
        self.url_kpis_ticker = url_kpis_ticker
        self.request_header = request_header
        self.variation_headings = variation_headings
        self.metadata_cols_acoes = metadata_cols_acoes
        self.metadata_cols_fiis = metadata_cols_fiis

    @staticmethod
    def _parse_float_cols(df: pd.DataFrame, cols_list: list) -> pd.DataFrame:
        for col in cols_list:
            if col in df.columns:
                df[col] = df[col].astype(str).replace('[^0-9,]', '', regex=True)
                df[col] = df[col].replace("", np.nan)
                df[col] = df[col].replace(",", ".", regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    @staticmethod
    def _parse_pct_cols(df: pd.DataFrame, cols_list: list) -> pd.DataFrame:
        for col in cols_list:
            if col in df.columns:
                df[col] = df[col].astype(str).replace("%", "", regex=False)
                df[col] = df[col].replace(",", ".", regex=True) # Garante ponto decimal
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col] / 100
        return df

    def extracao_tickers_de_ativos(self, tipo: str = "ações") -> list[str]:
        tipo_prep = tipo.strip().lower()
        if tipo_prep == "ações":
            url = self.url_tickers_acoes
        else:
            url = self.url_tickers_fiis

        try:
            html_content = requests.get(url, headers=self.request_header).text
            soup = BeautifulSoup(html_content, "lxml")
            tickers = [
                row.find_all("a")[0].text.strip()
                for row in soup.find_all("tr")[1:]
            ]
            return sorted(list(set(tickers)))
        except Exception as e:
            self.logger.error(f"Erro ao extrair tickers: {e}")
            return []

    def coleta_indicadores_de_ativo(self, ticker: str, parse_dtypes: bool = True) -> pd.DataFrame:
        url = self.url_kpis_ticker + ticker.strip().upper()
        try:
            html_content = requests.get(url=url, headers=self.request_header).text
            soup = BeautifulSoup(html_content, "lxml")
            tables = soup.find_all("table", attrs={'class': 'w728'})
            
            financial_data_raw = []
            for table in tables:
                rows = table.find_all("tr")
                for row in rows:
                    cells = row.find_all("td")
                    # A estrutura do fundamentus geralmente é Label | Valor | Label | Valor
                    # Vamos iterar de 2 em 2
                    for i in range(0, len(cells), 2):
                        if i+1 < len(cells):
                            label = cells[i].text.strip().replace("?", "")
                            value = cells[i+1].text.strip()
                            if label and value:
                                financial_data_raw.append({label: value})

            financial_data = {k: v for d in financial_data_raw for k, v in d.items()}
            
            # Determina se é ação ou FII baseado no retorno
            metadata_cols = self.metadata_cols_acoes # Default para Ações
            
            # Cria DataFrame
            df = pd.DataFrame(financial_data, index=[0])
            
            # Renomeia colunas
            df = df.rename(columns=metadata_cols)
            
            # Parse de tipos
            if parse_dtypes:
                # Colunas float (monetárias ou numéricas simples)
                float_cols = ['pe_ratio', 'pb_ratio', 'current_ratio', 'debt_to_equity', 'volume_medio_2m']
                df = self._parse_float_cols(df, float_cols)
                
                # Colunas percentuais
                pct_cols = ['roe', 'div_yield', 'revenue_growth', 'net_margin', 'operating_margin']
                df = self._parse_pct_cols(df, pct_cols)
            
            df['Ticker'] = ticker.upper()
            return df

        except Exception as e:
            # self.logger.error(f"Erro ao coletar dados de {ticker}: {e}")
            return pd.DataFrame()

# =============================================================================
# 3. PONDERAÇÕES E REGRAS (MANTIDAS)
# =============================================================================

WEIGHT_PERFORMANCE = 0.40
WEIGHT_FUNDAMENTAL = 0.30
WEIGHT_TECHNICAL = 0.30
WEIGHT_ML = 0.30
PESO_MIN = 0.10
PESO_MAX = 0.30

# =============================================================================
# 4. LISTAS DE ATIVOS (IBOVESPA)
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
ATIVOS_POR_SETOR = {k: v for k, v in ATIVOS_POR_SETOR_IBOV.items() if any(x in ATIVOS_IBOVESPA for x in v)}

# =============================================================================
# 5. MAPEAMENTOS DE PONTUAÇÃO (MANTIDOS)
# =============================================================================
# ... (Códigos de Mapeamento de Perfil - Mantidos idênticos ao original por brevidade, pois são estáticos) ...
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
MAP_CONCORDA = {OPTIONS_CONCORDA[i]: k for i, k in enumerate(SCORE_MAP_ORIGINAL.keys())}
OPTIONS_DISCORDA = [
    "CT: (Concordo Totalmente) - A preservação do capital é minha prioridade máxima, acima de qualquer ganho potencial.",
    "C: (Concordo) - É muito importante para mim evitar perdas, mesmo que isso limite o crescimento do meu portfólio.",
    "N: (Neutro) - Busco um equilíbrio; não quero perdas excessivas, mas sei que algum risco é necessário para crescer.",
    "D: (Discordo) - Estou focado no crescimento de longo prazo e entendo que perdas de curto prazo fazem parte do processo.",
    "DT: (Discordo Totalmente) - Meu foco é maximizar o retorno; perdas de curto prazo são irrelevantes se a tese de longo prazo for válida."
]
MAP_DISCORDA = {OPTIONS_DISCORDA[i]: k for i, k in enumerate(SCORE_MAP_ORIGINAL.keys())}
OPTIONS_REACTION_DETALHADA = ["A: (Vender Imediatamente) - Venderia a posição para evitar perdas maiores; prefiro realizar o prejuízo e reavaliar.", "B: (Manter e Reavaliar) - Manteria a calma, reavaliaria os fundamentos do ativo e o cenário macro para tomar uma decisão.", "C: (Comprar Mais) - Encararia como uma oportunidade de compra, aumentando a posição a um preço menor, se os fundamentos estiverem intactos."]
MAP_REACTION = {OPTIONS_REACTION_DETALHADA[i]: k for i, k in enumerate(SCORE_MAP_REACTION_ORIGINAL.keys())}
OPTIONS_CONHECIMENTO_DETALHADA = ["A: (Avançado) - Sinto-me confortável analisando balanços (fundamentalista), gráficos (técnica) e cenários macroeconômicos.", "B: (Intermediário) - Entendo os conceitos básicos (Renda Fixa vs. Variável, risco vs. retorno) e acompanho o mercado.", "C: (Iniciante) - Tenho pouca ou nenhuma experiência prática em investimentos além da poupança ou produtos bancários simples."]
MAP_CONHECIMENTO = {OPTIONS_CONHECIMENTO_DETALHADA[i]: k for i, k in enumerate(SCORE_MAP_CONHECIMENTO_ORIGINAL.keys())}
OPTIONS_TIME_HORIZON_DETALHADA = ['A: Curto (até 1 ano) - Meu objetivo é preservar capital ou realizar um ganho rápido, com alta liquidez.', 'B: Médio (1-5 anos) - Busco um crescimento balanceado e posso tolerar alguma flutuação neste período.', 'C: Longo (5+ anos) - Meu foco é a acumulação de patrimônio; flutuações de curto/médio prazo não me afetam.']
OPTIONS_LIQUIDEZ_DETALHADA = ['A: Menos de 6 meses - Posso precisar resgatar o valor a qualquer momento (ex: reserva de emergência).', 'B: Entre 6 meses e 2 anos - Não preciso do dinheiro imediatamente, mas tenho um objetivo de curto/médio prazo.', 'C: Mais de 2 anos - Este é um investimento de longo prazo; não tenho planos de resgatar nos próximos anos.']

# =============================================================================
# 6. CLASSES AUXILIARES (AnalisadorPerfilInvestidor e EngenheiroFeatures)
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
        if final_lookback >= 30: self.horizonte_tempo = "LONGO PRAZO"; self.dias_lookback_ml = 30
        elif final_lookback >= 20: self.horizonte_tempo = "MÉDIO PRAZO"; self.dias_lookback_ml = 20
        else: self.horizonte_tempo = "CURTO PRAZO"; self.dias_lookback_ml = 5
        return self.horizonte_tempo, self.dias_lookback_ml
    
    def calcular_perfil(self, respostas: dict) -> tuple[str, str, int, int]:
        score_risk = SCORE_MAP_ORIGINAL.get(respostas['risk_accept'], 3)
        score_gain = SCORE_MAP_ORIGINAL.get(respostas['max_gain'], 3)
        score_stable = SCORE_MAP_INV_ORIGINAL.get(respostas['stable_growth'], 3)
        score_loss = SCORE_MAP_INV_ORIGINAL.get(respostas['avoid_loss'], 3)
        score_know = SCORE_MAP_CONHECIMENTO_ORIGINAL.get(respostas['level'], 3)
        score_react = SCORE_MAP_REACTION_ORIGINAL.get(respostas['reaction'], 3)

        pontuacao = (score_risk * 5 + score_gain * 5 + score_stable * 5 + score_loss * 5 + score_know * 3 + score_react * 3)
        nivel_risco = self.determinar_nivel_risco(pontuacao)
        liq = respostas['liquidity'][0] if respostas['liquidity'] else 'C'
        obj = respostas['time_purpose'][0] if respostas['time_purpose'] else 'C'
        horizonte, ml_look = self.determinar_horizonte_ml(liq, obj)
        return nivel_risco, horizonte, ml_look, pontuacao

class EngenheiroFeatures:
    """Engenharia de Features e Cálculos Técnicos In-Live."""

    @staticmethod
    def calcular_rsi(series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calcular_macd(series, slow=26, fast=12, signal=9):
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        diff = macd - signal_line
        return macd, signal_line, diff

    @staticmethod
    def calcular_bollinger(series, window=20, num_std=2):
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        width = (upper - lower) / sma
        return upper, lower, width

    @staticmethod
    def calcular_momentum(series, window=60):
        return series.pct_change(window)

    @staticmethod
    def _normalizar(serie: pd.Series, maior_melhor: bool = True) -> pd.Series:
        if serie.isnull().all(): return pd.Series(0, index=serie.index)
        q_low = serie.quantile(0.02)
        q_high = serie.quantile(0.98)
        serie_clipped = serie.clip(q_low, q_high)
        min_val, max_val = serie_clipped.min(), serie_clipped.max()
        if max_val == min_val: return pd.Series(0.5, index=serie.index)
        
        if maior_melhor: return (serie_clipped - min_val) / (max_val - min_val)
        else: return (max_val - serie_clipped) / (max_val - min_val)

# =============================================================================
# 7. COLETOR DE DADOS LIVE (YFINANCE + FUNDAMENTUS)
# =============================================================================

class ColetorDadosLive:
    """Coleta dados em tempo real do YFinance e Fundamentus."""
    
    def __init__(self, periodo=PERIODO_DADOS):
        self.periodo = periodo
        self.dados_por_ativo = {}
        self.dados_fundamentalistas = pd.DataFrame()
        self.metricas_performance = pd.DataFrame()
        self.volatilidades_garch_raw = {}
        self.scraper_fundamentus = Fundamentus()

    def coletar_yfinance(self, simbolos):
        """Baixa histórico do Yahoo Finance."""
        dados = {}
        try:
            # Baixa em batch para ser mais rápido
            df_batch = yf.download(simbolos, period=self.periodo, progress=False, group_by='ticker', auto_adjust=True)
            
            for ticker in simbolos:
                try:
                    if len(simbolos) > 1:
                        df_ativo = df_batch[ticker].copy()
                    else:
                        df_ativo = df_batch.copy() # Se for só 1, estrutura é diferente
                    
                    if df_ativo.empty or 'Close' not in df_ativo.columns:
                        continue
                    
                    df_ativo = df_ativo.dropna(subset=['Close'])
                    if len(df_ativo) < MIN_DIAS_HISTORICO:
                        continue
                        
                    # Engenharia de Features Técnica (Live)
                    df_ativo['returns'] = df_ativo['Close'].pct_change()
                    df_ativo['rsi_14'] = EngenheiroFeatures.calcular_rsi(df_ativo['Close'])
                    _, _, df_ativo['macd_diff'] = EngenheiroFeatures.calcular_macd(df_ativo['Close'])
                    df_ativo['macd'], df_ativo['macd_signal'], _ = EngenheiroFeatures.calcular_macd(df_ativo['Close']) # Completo para gráfico
                    _, _, df_ativo['bb_width'] = EngenheiroFeatures.calcular_bollinger(df_ativo['Close'])
                    df_ativo['momentum_60'] = EngenheiroFeatures.calcular_momentum(df_ativo['Close'])
                    
                    # Limpeza pós cálculos
                    df_ativo.fillna(method='ffill', inplace=True)
                    df_ativo.dropna(inplace=True)
                    
                    dados[ticker] = df_ativo
                except Exception as e:
                    print(f"Erro ao processar {ticker} no YFinance: {e}")
                    continue
        except Exception as e:
            st.error(f"Erro geral no download YFinance: {e}")
        
        return dados

    def process(self, simbolos):
        """Pipeline principal de coleta."""
        self.dados_por_ativo = self.coletar_yfinance(simbolos)
        self.ativos_sucesso = list(self.dados_por_ativo.keys())
        
        lista_fund = []
        metricas_perf = []
        
        # Coleta dados fundamentalistas (Um por um - gargalo possível)
        # Usaremos um st.progress no main para acompanhar isso
        
        total_ativos = len(self.ativos_sucesso)
        
        # Barra de progresso interna se chamada fora do loop principal
        # mas aqui vamos apenas iterar
        
        for i, ticker in enumerate(self.ativos_sucesso):
            # 1. Performance (Dados Históricos)
            df = self.dados_por_ativo[ticker]
            retorno_anual = df['returns'].mean() * 252
            vol_anual = df['returns'].std() * np.sqrt(252)
            sharpe = (retorno_anual - TAXA_LIVRE_RISCO) / vol_anual if vol_anual > 0 else 0
            cum_ret = (1 + df['returns']).cumprod()
            max_dd = ((cum_ret - cum_ret.expanding().max()) / cum_ret.expanding().max()).min()
            
            metricas_perf.append({
                'Ticker': ticker,
                'sharpe': sharpe,
                'retorno_anual': retorno_anual,
                'volatilidade_anual': vol_anual,
                'max_drawdown': max_dd
            })
            
            # 2. GARCH Volatility (Cálculo Rápido)
            try:
                # Usa os últimos 1000 dias para rapidez
                returns_garch = df['returns'].iloc[-1000:] * 100 
                model = arch_model(returns_garch, vol='Garch', p=1, q=1, rescale=False)
                res = model.fit(disp='off', show_warning=False)
                garch_forecast = np.sqrt(res.forecast(horizon=1).variance.values[-1, :])[0]
                self.volatilidades_garch_raw[ticker] = (garch_forecast / 100) * np.sqrt(252)
            except:
                self.volatilidades_garch_raw[ticker] = vol_anual # Fallback

            # 3. Fundamentos (Live Scraping)
            ticker_clean = ticker.replace('.SA', '')
            df_fund = self.scraper_fundamentus.coleta_indicadores_de_ativo(ticker_clean)
            
            if not df_fund.empty:
                fund_data = df_fund.iloc[0].to_dict()
                fund_data['Ticker'] = ticker
                # Adiciona métricas calculadas
                fund_data['annual_return'] = retorno_anual
                fund_data['annual_volatility'] = vol_anual
                fund_data['sharpe_ratio'] = sharpe
                fund_data['garch_volatility'] = self.volatilidades_garch_raw[ticker]
                lista_fund.append(fund_data)
            else:
                # Cria dummy se falhar o scrape
                lista_fund.append({'Ticker': ticker, 'sector': 'Unknown', 'pe_ratio': np.nan})

        if lista_fund:
            self.dados_fundamentalistas = pd.DataFrame(lista_fund).set_index('Ticker')
        
        if metricas_perf:
            self.metricas_performance = pd.DataFrame(metricas_perf).set_index('Ticker')
            
        return len(self.ativos_sucesso) > 0

    def coletar_ativo_unico(self, ticker):
        """Para aba de análise individual."""
        if ticker in self.dados_por_ativo:
            df_tecnicos = self.dados_por_ativo[ticker]
        else:
            dados = self.coletar_yfinance([ticker])
            df_tecnicos = dados.get(ticker, pd.DataFrame())
            
        if ticker in self.dados_fundamentalistas.index:
            fund_series = self.dados_fundamentalistas.loc[ticker]
            fund_dict = fund_series.to_dict()
        else:
             # Scrape on demand
             ticker_clean = ticker.replace('.SA', '')
             df_fund = self.scraper_fundamentus.coleta_indicadores_de_ativo(ticker_clean)
             fund_dict = df_fund.iloc[0].to_dict() if not df_fund.empty else {}

        return df_tecnicos, fund_dict

# =============================================================================
# 8. OTIMIZADOR DE PORTFÓLIO (MANTIDO)
# =============================================================================

class OtimizadorPortfolioAvancado:
    """Otimização de portfólio com volatilidade GARCH e CVaR"""
    def __init__(self, returns_df, garch_vols=None):
        self.returns = returns_df
        self.mean_returns = returns_df.mean() * 252
        if garch_vols:
            self.cov_matrix = self._construir_matriz_cov_garch(returns_df, garch_vols)
        else:
            self.cov_matrix = returns_df.cov() * 252
        self.num_ativos = len(returns_df.columns)

    def _construir_matriz_cov_garch(self, returns_df, garch_vols):
        corr_matrix = returns_df.corr()
        vol_array = np.array([garch_vols.get(ativo, returns_df[ativo].std() * np.sqrt(252)) for ativo in returns_df.columns])
        cov_matrix = corr_matrix.values * np.outer(vol_array, vol_array)
        return pd.DataFrame(cov_matrix, index=returns_df.columns, columns=returns_df.columns)
    
    def estatisticas_portfolio(self, pesos):
        p_retorno = np.dot(pesos, self.mean_returns)
        p_vol = np.sqrt(np.dot(pesos.T, np.dot(self.cov_matrix, pesos)))
        return p_retorno, p_vol
    
    def sharpe_negativo(self, pesos):
        r, v = self.estatisticas_portfolio(pesos)
        return -(r - TAXA_LIVRE_RISCO) / v if v > 0 else 0
    
    def minimizar_volatilidade(self, pesos):
        return self.estatisticas_portfolio(pesos)[1]

    def cvar_negativo(self, pesos, confidence=0.95):
        portfolio_returns = self.returns @ pesos
        var_index = int(np.floor((1 - confidence) * len(portfolio_returns)))
        sorted_returns = np.sort(portfolio_returns)
        cvar = sorted_returns[:var_index].mean()
        return -cvar

    def otimizar(self, estrategia='MaxSharpe'):
        if self.num_ativos == 0: return {}
        restricoes = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        limites = tuple((PESO_MIN, PESO_MAX) for _ in range(self.num_ativos))
        chute = np.array([1.0/self.num_ativos]*self.num_ativos)
        
        if estrategia == 'MinVolatility': objetivo = self.minimizar_volatilidade
        elif estrategia == 'CVaR': objetivo = self.cvar_negativo
        else: objetivo = self.sharpe_negativo
        
        try:
            res = minimize(objetivo, chute, method='SLSQP', bounds=limites, constraints=restricoes)
            return {k: v for k, v in zip(self.returns.columns, res.x)} if res.success else {k: 1.0/self.num_ativos for k in self.returns.columns}
        except:
            return {k: 1.0/self.num_ativos for k in self.returns.columns}

# =============================================================================
# 9. CONSTRUTOR AUTOML (COM TREINAMENTO LIVE)
# =============================================================================

class ConstrutorPortfolioAutoML:
    """Orquestrador principal com treinamento 'in-live'."""
    
    def __init__(self, valor_investimento, periodo=PERIODO_DADOS):
        self.valor_investimento = valor_investimento
        self.periodo = periodo
        self.dados_por_ativo = {}
        self.dados_fundamentalistas = pd.DataFrame()
        self.metricas_performance = pd.DataFrame()
        self.volatilidades_garch = {}
        self.predicoes_ml = {}
        self.ml_metadata = {} # Armazena metadados live para a aba 4
        self.ativos_sucesso = []
        self.ativos_selecionados = []
        self.alocacao_portfolio = {}
        self.metricas_portfolio = {}
        self.metodo_alocacao_atual = "Não Aplicado"
        self.justificativas_selecao = {}
        self.scores_combinados = pd.DataFrame()
        self.coletor = ColetorDadosLive(periodo)

    def coletar_e_processar_dados(self, simbolos, progress_bar=None):
        """Usa o coletor Live."""
        simbolos_filtrados = [s for s in simbolos if s in TODOS_ATIVOS]
        if not simbolos_filtrados: return False
        
        # Simulando o progresso do scrape fundamentalista
        step = 100 / (len(simbolos_filtrados) + 1)
        
        # Executa o processamento completo
        sucesso = self.coletor.process(simbolos_filtrados)
        
        self.dados_por_ativo = self.coletor.dados_por_ativo
        self.dados_fundamentalistas = self.coletor.dados_fundamentalistas
        self.ativos_sucesso = self.coletor.ativos_sucesso
        self.metricas_performance = self.coletor.metricas_performance
        self.volatilidades_garch = self.coletor.volatilidades_garch_raw
        
        return sucesso

    def treinar_modelo_rapido(self, ativo):
        """Treina um RandomForest rápido para um ativo específico."""
        try:
            df = self.dados_por_ativo[ativo].copy()
            # Cria Target: Retorno > 0 nos próximos ML_LOOKBACK dias (simplificado para velocidade)
            horizon = 20 # ~1 mês
            df['Target'] = (df['Close'].shift(-horizon) > df['Close']).astype(int)
            
            features = ['rsi_14', 'macd_diff', 'bb_width', 'momentum_60', 'returns']
            df_model = df.dropna().copy()
            
            if len(df_model) < 100: return 0.5, 0.5, {} # Dados insuficientes
            
            X = df_model[features]
            y = df_model['Target']
            
            # TimeSeries Split simples
            split = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]
            
            model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            
            # Avaliação
            preds = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, preds) if len(np.unique(y_test)) > 1 else 0.5
            
            # Predição Atual (usando a última linha disponível de features)
            last_features = df[features].iloc[[-1]]
            prob_alta = model.predict_proba(last_features)[0, 1]
            
            # Feature Importance
            importances = dict(zip(features, model.feature_importances_))
            
            return prob_alta, auc, importances
            
        except Exception as e:
            # print(f"Erro ML {ativo}: {e}")
            return 0.5, 0.5, {}

    def treinar_modelos_ensemble(self, dias_lookback_ml=30, progress_bar=None):
        """Executa o treinamento live para todos os ativos."""
        total = len(self.ativos_sucesso)
        for i, ativo in enumerate(self.ativos_sucesso):
            prob, auc, imps = self.treinar_modelo_rapido(ativo)
            
            # Armazena resultados simples para o otimizador
            self.predicoes_ml[ativo] = {
                'predicted_proba_up': prob,
                'auc_roc_score': auc,
                'model_name': 'RandomForest Live'
            }
            
            # Armazena metadados ricos para a Aba 4 (compatibilidade)
            self.ml_metadata[ativo] = {
                'final_ensemble_proba': prob,
                'features_selected_count': len(imps),
                'features_original_count': 5,
                'target_days': 20,
                'models_data': json.dumps({'RandomForest': {'hpo_auc': auc, 'wfc_weight': 1.0, 'final_proba': prob}}),
                'rf_importances': json.dumps(imps),
                'features_selected_list': json.dumps(list(imps.keys()))
            }
            
            # Adiciona ao dataframe histórico para normalização no score
            self.dados_por_ativo[ativo].loc[self.dados_por_ativo[ativo].index[-1], 'ML_Proba'] = prob
            self.dados_por_ativo[ativo].loc[self.dados_por_ativo[ativo].index[-1], 'ML_Confidence'] = auc

    def pontuar_e_selecionar_ativos(self, horizonte_tempo):
        # (Lógica idêntica a v8.7.0)
        if horizonte_tempo == "CURTO PRAZO": WEIGHT_PERF, WEIGHT_FUND, WEIGHT_TECH = 0.40, 0.10, 0.20
        elif horizonte_tempo == "LONGO PRAZO": WEIGHT_PERF, WEIGHT_FUND, WEIGHT_TECH = 0.40, 0.50, 0.10
        else: WEIGHT_PERF, WEIGHT_FUND, WEIGHT_TECH = 0.40, 0.30, 0.30

        final_ml_weight = WEIGHT_ML
        total_non_ml_weight = WEIGHT_PERF + WEIGHT_FUND + WEIGHT_TECH
        scale_factor = (1.0 - final_ml_weight) / total_non_ml_weight if total_non_ml_weight > 0 else 0
        WEIGHT_PERF *= scale_factor; WEIGHT_FUND *= scale_factor; WEIGHT_TECH *= scale_factor
        
        self.pesos_atuais = {'Performance': WEIGHT_PERF, 'Fundamentos': WEIGHT_FUND, 'Técnicos': WEIGHT_TECH, 'ML': final_ml_weight}
        
        scores = pd.DataFrame(index=self.ativos_sucesso)
        last_metrics = {}
        for asset in self.ativos_sucesso:
            df = self.dados_por_ativo[asset]
            last_idx = df.last_valid_index()
            if last_idx: last_metrics[asset] = df.loc[last_idx].to_dict()
            
            # Merge com fundamentos
            if asset in self.dados_fundamentalistas.index:
                fund_row = self.dados_fundamentalistas.loc[asset].to_dict()
                last_metrics[asset].update(fund_row)

        combinado = pd.DataFrame(last_metrics).T
        
        # Normalização e Cálculo
        required_cols = ['sharpe_ratio', 'pe_ratio', 'roe', 'rsi_14', 'macd_diff', 'ML_Proba', 'ML_Confidence']
        for col in required_cols:
            if col not in combinado.columns: combinado[col] = np.nan

        scores['perf_score'] = EngenheiroFeatures._normalizar(combinado['sharpe_ratio'], True) * WEIGHT_PERF
        
        # Tratamento para P/L e ROE que podem vir como string ou float do scraper
        combinado['pe_ratio'] = pd.to_numeric(combinado['pe_ratio'], errors='coerce')
        combinado['roe'] = pd.to_numeric(combinado['roe'], errors='coerce')

        scores['fund_score'] = (EngenheiroFeatures._normalizar(combinado['pe_ratio'], False) * 0.5 + 
                                EngenheiroFeatures._normalizar(combinado['roe'], True) * 0.5) * WEIGHT_FUND
        
        scores['tech_score'] = (EngenheiroFeatures._normalizar(combinado['rsi_14'], False) * 0.5 + 
                                EngenheiroFeatures._normalizar(combinado['macd_diff'], True) * 0.5) * WEIGHT_TECH
        
        scores['ml_score'] = (EngenheiroFeatures._normalizar(combinado['ML_Proba'], True) * 0.6 + 
                              EngenheiroFeatures._normalizar(combinado['ML_Confidence'], True) * 0.4) * final_ml_weight
        
        scores['total_score'] = scores.sum(axis=1)
        self.scores_combinados = scores.join(combinado, rsuffix='_raw').sort_values('total_score', ascending=False)
        
        # Seleção via Cluster
        features_cluster = ['sharpe_ratio', 'annual_return', 'annual_volatility', 'pe_ratio', 'roe', 'div_yield']
        df_cluster = self.scores_combinados[features_cluster].fillna(0)
        
        if len(df_cluster) > 5:
            try:
                kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto').fit(StandardScaler().fit_transform(df_cluster))
                df_cluster['Cluster'] = kmeans.labels_
                self.scores_combinados['Cluster'] = kmeans.labels_
                
                candidates = self.scores_combinados[self.scores_combinados['total_score'] >= self.scores_combinados['total_score'].quantile(SCORE_PERCENTILE_THRESHOLD)].copy()
                candidates['rank'] = candidates.groupby('Cluster')['total_score'].rank(ascending=False)
                self.ativos_selecionados = candidates.sort_values(['rank', 'total_score']).head(NUM_ATIVOS_PORTFOLIO).index.tolist()
            except:
                self.ativos_selecionados = self.scores_combinados.head(NUM_ATIVOS_PORTFOLIO).index.tolist()
        else:
            self.ativos_selecionados = self.scores_combinados.head(NUM_ATIVOS_PORTFOLIO).index.tolist()
            
        if len(self.ativos_selecionados) < NUM_ATIVOS_PORTFOLIO:
            extras = [a for a in self.scores_combinados.index if a not in self.ativos_selecionados][:NUM_ATIVOS_PORTFOLIO - len(self.ativos_selecionados)]
            self.ativos_selecionados.extend(extras)

    def otimizar_alocacao(self, nivel_risco):
        returns_sel = pd.DataFrame({a: self.dados_por_ativo[a]['returns'] for a in self.ativos_selecionados}).dropna()
        vols = {a: self.volatilidades_garch.get(a, returns_sel[a].std()*np.sqrt(252)) for a in returns_sel.columns}
        opt = OtimizadorPortfolioAvancado(returns_sel, vols)
        
        strat = 'MaxSharpe'
        if 'CONSERVADOR' in nivel_risco: strat = 'MinVolatility'
        elif 'AVANÇADO' in nivel_risco: strat = 'CVaR'
        
        w = opt.otimizar(strat)
        self.metodo_alocacao_atual = f"Otimização {strat}"
        self.alocacao_portfolio = {a: {'weight': p, 'amount': self.valor_investimento * p} for a, p in w.items()}

    def executar_pipeline(self, simbolos_customizados, perfil_inputs, progress_bar=None):
        try:
            if progress_bar: progress_bar.progress(10, "Coletando dados Ao Vivo (YFinance + Fundamentus)...")
            if not self.coletar_e_processar_dados(simbolos_customizados): return False
            
            if progress_bar: progress_bar.progress(50, "Treinando Modelos ML em Tempo Real...")
            self.treinar_modelos_ensemble()
            
            if progress_bar: progress_bar.progress(70, "Ranqueando Ativos...")
            self.pontuar_e_selecionar_ativos(perfil_inputs.get('time_horizon'))
            
            if progress_bar: progress_bar.progress(90, "Otimizando Portfólio...")
            self.otimizar_alocacao(perfil_inputs.get('risk_level'))
            
            # Métricas Finais
            w = {k: v['weight'] for k, v in self.alocacao_portfolio.items()}
            df_ret = pd.DataFrame({k: self.dados_por_ativo[k]['returns'] for k in w.keys()}).dropna()
            if not df_ret.empty:
                port_ret = (df_ret * pd.Series(w)).sum(axis=1)
                self.metricas_portfolio = {
                    'annual_return': port_ret.mean()*252,
                    'annual_volatility': port_ret.std()*np.sqrt(252),
                    'sharpe_ratio': (port_ret.mean()*252 - TAXA_LIVRE_RISCO)/(port_ret.std()*np.sqrt(252)),
                    'max_drawdown': ((1+port_ret).cumprod() / (1+port_ret).cumprod().cummax() - 1).min()
                }
            
            # Justificativas
            self.justificativas_selecao = {}
            for a in self.ativos_selecionados:
                s = self.scores_combinados.loc[a]
                self.justificativas_selecao[a] = f"Score: {s['total_score']:.2f} | Sharpe: {s['sharpe_ratio']:.2f} | ML Proba: {s['ML_Proba']*100:.1f}% | P/L: {s['pe_ratio']}"

            if progress_bar: progress_bar.progress(100, "Concluído!")
            return True
        except Exception as e:
            st.error(f"Erro Pipeline: {e}")
            st.code(traceback.format_exc())
            return False

# =============================================================================
# 10. CLASSE ANALISADOR INDIVIDUAL (MANTIDA)
# =============================================================================
class AnalisadorIndividualAtivos:
    @staticmethod
    def realizar_clusterizacao_pca(dados_ativos: pd.DataFrame, max_clusters: int = 10):
        # Versão simplificada para visualização
        features = dados_ativos.select_dtypes(include=np.number).dropna(axis=1)
        if features.shape[1] < 2 or len(features) < 3: return None, None, None, None
        
        scaler = StandardScaler()
        X = scaler.fit_transform(features.fillna(0))
        pca = PCA(n_components=min(3, X.shape[1])).fit(X)
        X_pca = pca.transform(X)
        kmeans = KMeans(n_clusters=min(5, len(X)), random_state=42).fit(X_pca)
        
        res = pd.DataFrame(X_pca, index=features.index, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
        res['Cluster'] = kmeans.labels_
        return res, pca, kmeans, min(5, len(X))

# =============================================================================
# 11. FUNÇÕES DE ESTILO (MANTIDAS)
# =============================================================================
def obter_template_grafico() -> dict:
    return {
        'plot_bgcolor': '#f8f9fa', 'paper_bgcolor': 'white',
        'font': {'family': 'Arial', 'size': 12, 'color': '#343a40'},
        'title': {'font': {'size': 16, 'color': '#212529', 'weight': 'bold'}, 'x': 0.5},
        'colorway': ['#212529', '#495057', '#6c757d', '#adb5bd', '#ced4da']
    }

def configurar_pagina():
    st.set_page_config(page_title="Sistema de Portfólios Adaptativos (Live)", page_icon="📈", layout="wide")
    st.markdown("""
        <style>
        :root { --primary-color: #000000; --background-dark: #f8f9fa; --text-color: #212529; }
        .main-header { font-family: 'Arial'; color: var(--primary-color); text-align: center; border-bottom: 2px solid #dee2e6; font-size: 2.2rem; margin-bottom: 20px; }
        .stButton button { border: 1px solid black !important; color: black !important; }
        .stButton button:hover { background-color: black !important; color: white !important; }
        .info-box { background-color: var(--background-dark); border-left: 4px solid black; padding: 15px; margin: 10px 0; }
        </style>
    """, unsafe_allow_html=True)

# =============================================================================
# 12. ABAS DA INTERFACE (MANTIDAS E ADAPTADAS)
# =============================================================================

def aba_introducao():
    st.markdown("## 📚 Metodologia (Live Version)")
    st.info("Esta versão do sistema coleta dados **em tempo real** do **Yahoo Finance** e **Fundamentus**, e treina modelos de Machine Learning instantaneamente.")

def aba_selecao_ativos():
    st.markdown("## 🎯 Definição do Universo")
    modo = st.radio("Modo:", ["📊 Índice (Todos Ibovespa)", "🏢 Setorial", "✍️ Individual"])
    sel = []
    if "Índice" in modo: sel = ATIVOS_IBOVESPA
    elif "Setorial" in modo:
        sets = st.multiselect("Setores:", sorted(list(ATIVOS_POR_SETOR.keys())))
        for s in sets: sel.extend(ATIVOS_POR_SETOR[s])
    else:
        sel = st.multiselect("Tickers:", TODOS_ATIVOS)
    
    if sel:
        st.session_state.ativos_para_analise = list(set(sel))
        st.success(f"{len(sel)} ativos selecionados.")
    else: st.warning("Selecione ativos.")

def aba_construtor_portfolio():
    if not st.session_state.get('ativos_para_analise'): return st.warning("Selecione ativos na aba 2.")
    
    if not st.session_state.get('builder_complete'):
        with st.form("perfil"):
            st.markdown("### Perfil do Investidor")
            r_risk = st.radio("Tolerância:", OPTIONS_CONCORDA, index=2)
            r_gain = st.radio("Foco Ganho:", OPTIONS_CONCORDA, index=2)
            r_loss = st.radio("Aversão Perda:", OPTIONS_DISCORDA, index=2)
            r_time = st.radio("Horizonte:", OPTIONS_TIME_HORIZON_DETALHADA, index=1)
            inv = st.number_input("Investimento (R$):", value=10000)
            if st.form_submit_button("Otimizar Ao Vivo"):
                # Mock de respostas para o analisador
                respostas = {'risk_accept': MAP_CONCORDA[r_risk], 'max_gain': MAP_CONCORDA[r_gain], 
                             'stable_growth': 'N: Neutro', 'avoid_loss': MAP_DISCORDA[r_loss], 
                             'level': 'B: Intermediário', 'reaction': 'B: Manter',
                             'time_purpose': r_time, 'liquidity': 'C'}
                
                an = AnalisadorPerfilInvestidor()
                risk, hor, look, score = an.calcular_perfil(respostas)
                st.session_state.profile = {'risk_level': risk, 'time_horizon': hor}
                
                st.session_state.builder = ConstrutorPortfolioAutoML(inv)
                prog = st.progress(0, "Iniciando...")
                
                if st.session_state.builder.executar_pipeline(st.session_state.ativos_para_analise, st.session_state.profile, prog):
                    st.session_state.builder_complete = True
                    st.rerun()
    else:
        b = st.session_state.builder
        st.success("Otimização Concluída!")
        col1, col2 = st.columns(2)
        col1.metric("Retorno Esp. (a.a.)", f"{b.metricas_portfolio['annual_return']*100:.1f}%")
        col2.metric("Sharpe", f"{b.metricas_portfolio['sharpe_ratio']:.2f}")
        
        st.markdown("#### Alocação Final")
        df_alloc = pd.DataFrame([{'Ativo': k, 'Peso %': v['weight']*100, 'Valor R$': v['amount']} for k, v in b.alocacao_portfolio.items()])
        st.dataframe(df_alloc)
        
        if st.button("Reiniciar"):
            st.session_state.builder_complete = False
            st.rerun()

def aba_analise_individual():
    st.markdown("## 🔍 Análise Individual")
    if not st.session_state.get('ativos_para_analise'): return
    
    ativo = st.selectbox("Ativo:", st.session_state.ativos_para_analise)
    if st.button("Analisar Ativo", type="primary"):
        with st.spinner("Coletando e Analisando..."):
            # Usa o coletor live para pegar dados frescos ou do cache do builder
            coletor = st.session_state.builder.coletor if st.session_state.get('builder') else ColetorDadosLive()
            df_tec, df_fund = coletor.coletar_ativo_unico(ativo)
            
            if df_tec.empty: return st.error("Sem dados.")
            
            # Treina modelo rápido se não existir
            prob, auc, imps = 0.5, 0.5, {}
            if st.session_state.get('builder') and ativo in st.session_state.builder.ml_metadata:
                meta = st.session_state.builder.ml_metadata[ativo]
                prob = meta['final_ensemble_proba']
                imps = json.loads(meta['rf_importances'])
            else:
                # Treino on-the-fly para análise isolada
                construtor_temp = ConstrutorPortfolioAutoML(0)
                construtor_temp.dados_por_ativo[ativo] = df_tec
                prob, auc, imps = construtor_temp.treinar_modelo_rapido(ativo)

            tab1, tab2, tab3, tab4 = st.tabs(["Gráfico", "Fundamentos", "Técnico", "ML Live"])
            
            with tab1:
                fig = go.Figure(data=[go.Candlestick(x=df_tec.index, open=df_tec['Open'], high=df_tec['High'], low=df_tec['Low'], close=df_tec['Close'])])
                fig.update_layout(title=f"{ativo} - Preço", xaxis_rangeslider_visible=False, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.json(df_fund)
            
            with tab3:
                col1, col2 = st.columns(2)
                col1.metric("RSI", f"{df_tec['rsi_14'].iloc[-1]:.2f}")
                col2.metric("MACD Diff", f"{df_tec['macd_diff'].iloc[-1]:.4f}")
            
            with tab4:
                st.metric("Probabilidade Alta (RF Live)", f"{prob*100:.1f}%")
                if imps:
                    st.markdown("#### Importância das Features (Ao Vivo)")
                    df_imp = pd.Series(imps).sort_values(ascending=False).to_frame("Importância")
                    st.bar_chart(df_imp)

def aba_referencias():
    st.markdown("## Referências")
    st.markdown("- Fundamentus (Scraping Live)\n- Yahoo Finance (API Live)\n- Scikit-Learn (RandomForest)")

# =============================================================================
# 13. MAIN
# =============================================================================

def main():
    if 'builder' not in st.session_state:
        st.session_state.builder = None
        st.session_state.builder_complete = False
        st.session_state.profile = {}
        st.session_state.ativos_para_analise = []

    configurar_pagina()
    st.markdown('<h1 class="main-header">Sistema de Portfólios Adaptativos (Live v9.0)</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Metodologia", "🎯 Seleção", "🏗️ Construtor", "🔍 Individual", "📖 Referências"])
    
    with tab1: aba_introducao()
    with tab2: aba_selecao_ativos()
    with tab3: aba_construtor_portfolio()
    with tab4: aba_analise_individual()
    with tab5: aba_referencias()

if __name__ == "__main__":
    main()
