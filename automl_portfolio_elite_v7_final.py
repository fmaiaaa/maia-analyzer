# -*- coding: utf-8 -*-
"""
=============================================================================
SISTEMA DE PORTFÓLIOS ADAPTATIVOS - OTIMIZAÇÃO QUANTITATIVA (LIVE DATA)
=============================================================================

Versão: 9.0.0 (Live YFinance + Fundamentus Scraper + ML On-the-fly)

Alterações:
- Substituição do GCS por yfinance para dados de mercado.
- Integração da classe Fundamentus para dados fundamentalistas.
- Cálculo de indicadores técnicos (RSI, MACD, Volatilidade) em tempo real.
- Treinamento de modelos de Machine Learning em tempo real dentro do app.
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

# --- 2. EXTERNAL DATA LIBRARIES ---
import yfinance as yf
import requests
from bs4 import BeautifulSoup

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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit

# --- 6. CONFIGURATION ---
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURAÇÕES E CONSTANTES GLOBAIS
# =============================================================================

PERIODO_DADOS = '2y' # Reduzido para 2y para otimizar performance live
MIN_DIAS_HISTORICO = 120
NUM_ATIVOS_PORTFOLIO = 5
TAXA_LIVRE_RISCO = 0.1075
SCORE_PERCENTILE_THRESHOLD = 0.80 

# URLs Fundamentus (Mantidas da classe original)
URL_TICKERS_ACOES = "https://www.fundamentus.com.br/resultado.php"
URL_TICKERS_FIIS = "https://www.fundamentus.com.br/fii_resultado.php"
URL_KPIS_TICKER = "https://www.fundamentus.com.br/detalhes.php?papel="
REQUEST_HEADER = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Mapeamento de Prazos para Labels
LOOKBACK_ML_DAYS_MAP = {
    'curto_prazo': 20, # Dias úteis aprox 1 mês
    'medio_prazo': 60, # Dias úteis aprox 3 meses
    'longo_prazo': 120 # Dias úteis aprox 6 meses
}

# =============================================================================
# 2. PONDERAÇÕES E REGRAS DE OTIMIZAÇÃO
# =============================================================================

WEIGHT_PERFORMANCE = 0.40
WEIGHT_FUNDAMENTAL = 0.30
WEIGHT_TECHNICAL = 0.30
WEIGHT_ML = 0.30

PESO_MIN = 0.05
PESO_MAX = 0.35

# =============================================================================
# 3. DADOS ESTÁTICOS (TICKERS E SETORES)
# =============================================================================

# Lista Base IBOVESPA (Referência)
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

# Mapeamento setorial (Simplificado para referência)
ATIVOS_POR_SETOR_IBOV = {
    'Financeiro': ['BBDC4.SA', 'ITUB4.SA', 'BBAS3.SA', 'B3SA3.SA', 'ITSA4.SA', 'SANB11.SA', 'BPAC11.SA', 'BBSE3.SA'],
    'Materiais Básicos': ['VALE3.SA', 'GGBR4.SA', 'CSNA3.SA', 'USIM5.SA', 'BRAP4.SA', 'SUZB3.SA', 'KLBN11.SA'],
    'Petróleo e Gás': ['PETR4.SA', 'PETR3.SA', 'PRIO3.SA', 'UGPA3.SA', 'CSAN3.SA', 'VBBR3.SA', 'RRRP3.SA'],
    'Utilidade Pública': ['ELET3.SA', 'ELET6.SA', 'EQTL3.SA', 'CMIG4.SA', 'CPLE6.SA', 'SBSP3.SA', 'TAEE11.SA'],
    'Consumo e Varejo': ['MGLU3.SA', 'LREN3.SA', 'RENT3.SA', 'JBSS3.SA', 'BRFS3.SA', 'BEEF3.SA', 'ABEV3.SA', 'NTCO3.SA'],
    'Indústria e Construção': ['WEGE3.SA', 'EMBR3.SA', 'CYRE3.SA', 'MRVE3.SA', 'EZTC3.SA'],
    'Saúde': ['HAPV3.SA', 'RDOR3.SA', 'FLRY3.SA'],
    'Tecnologia': ['TOTS3.SA', 'LWSA3.SA'],
    'Educação': ['YDUQ3.SA', 'COGN3.SA']
}
TODOS_ATIVOS = sorted(list(set(ATIVOS_IBOVESPA)))

ATIVOS_POR_SETOR = {
    setor: [ativo for ativo in ativos if ativo in ATIVOS_IBOVESPA] 
    for setor, ativos in ATIVOS_POR_SETOR_IBOV.items()
    if any(ativo in ATIVOS_IBOVESPA for ativo in ativos)
}

# =============================================================================
# 4. MAPAS DE PONTUAÇÃO (INALTERADOS)
# =============================================================================
SCORE_MAP_ORIGINAL = {'CT: Concordo Totalmente': 5, 'C: Concordo': 4, 'N: Neutro': 3, 'D: Discordo': 2, 'DT: Discordo Totalmente': 1}
SCORE_MAP_INV_ORIGINAL = {'CT: Concordo Totalmente': 1, 'C: Concordo': 2, 'N: Neutro': 3, 'D: Discordo': 4, 'DT: Discordo Totalmente': 5}
SCORE_MAP_CONHECIMENTO_ORIGINAL = {'A: Avançado (Análise fundamentalista, macro e técnica)': 5, 'B: Intermediário (Conhecimento básico sobre mercados e ativos)': 3, 'C: Iniciante (Pouca ou nenhuma experiência em investimentos)': 1}
SCORE_MAP_REACTION_ORIGINAL = {'A: Venderia imediatamente': 1, 'B: Manteria e reavaliaria a tese': 3, 'C: Compraria mais para aproveitar preços baixos': 5}

OPTIONS_CONCORDA = ["CT: Concordo Totalmente", "C: Concordo", "N: Neutro", "D: Discordo", "DT: Discordo Totalmente"]
MAP_CONCORDA = {k: k for k in OPTIONS_CONCORDA} # Simplificação: Valor visual é a chave
OPTIONS_DISCORDA = ["CT: Concordo Totalmente", "C: Concordo", "N: Neutro", "D: Discordo", "DT: Discordo Totalmente"]
MAP_DISCORDA = {k: k for k in OPTIONS_DISCORDA}
OPTIONS_REACTION_DETALHADA = ["A: Venderia imediatamente", "B: Manteria e reavaliaria a tese", "C: Compraria mais para aproveitar preços baixos"]
MAP_REACTION = {k: k for k in OPTIONS_REACTION_DETALHADA}
OPTIONS_CONHECIMENTO_DETALHADA = ["A: Avançado (Análise fundamentalista, macro e técnica)", "B: Intermediário (Conhecimento básico sobre mercados e ativos)", "C: Iniciante (Pouca ou nenhuma experiência em investimentos)"]
MAP_CONHECIMENTO = {k: k for k in OPTIONS_CONHECIMENTO_DETALHADA}
OPTIONS_TIME_HORIZON_DETALHADA = ['A: Curto (até 1 ano)', 'B: Médio (1-5 anos)', 'C: Longo (5+ anos)']
OPTIONS_LIQUIDEZ_DETALHADA = ['A: Menos de 6 meses', 'B: Entre 6 meses e 2 anos', 'C: Mais de 2 anos']


# =============================================================================
# 5. CLASSE FUNDAMENTUS (DO CÓDIGO FORNECIDO)
# =============================================================================

class Fundamentus:
    """Classe responsável por extrair informações do site Fundamentus."""
    
    def __init__(
        self,
        logger_level: int = logging.INFO,
        url_tickers_acoes: str = URL_TICKERS_ACOES,
        url_tickers_fiis: str = URL_TICKERS_FIIS,
        url_kpis_ticker: str = URL_KPIS_TICKER,
        request_header: dict = REQUEST_HEADER,
        variation_headings: list = ["Dia", "Mês", "30 dias", "12 meses", "2024", "2023"], # Exemplo
        metadata_cols_acoes: dict = {'P/L': 'pe_ratio', 'ROE': 'roe', 'Div. Yield': 'div_yield', 'P/VP': 'pb_ratio', 'Dív.Brut/ Patrim.': 'debt_to_equity', 'Cres. Rec (5a)': 'revenue_growth', 'Marg. Oper.': 'operating_margin', 'Liquidez Corr.': 'current_ratio', 'Setor': 'sector', 'Subsetor': 'industry'},
        metadata_cols_fiis: dict = {}
    ) -> None:
        # Configurando logger básico se não configurado
        logging.basicConfig(level=logger_level)
        self.logger = logging.getLogger(__name__)

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
                df[col] = df[col].replace('[^0-9,]', '', regex=True)
                df[col] = df[col].replace("", np.nan)
                df[col] = df[col].replace(",", ".", regex=True)
                df[col] = df[col].astype(float)
        return df

    @staticmethod
    def _parse_pct_cols(df: pd.DataFrame, cols_list: list) -> pd.DataFrame:
        for col in cols_list:
            if col in df.columns:
                df[col] = df[col].replace("%", "")
                df[col] = df[col].replace(",", ".", regex=True)
                df[col] = df[col].replace("", np.nan)
                df[col] = df[col].astype(float)
                df[col] = df[col] / 100
        return df

    def extracao_tickers_de_ativos(self, tipo: str = "ações") -> list[str]:
        # Implementação simplificada para o contexto do app se necessário, 
        # mas aqui usaremos a lista fixa IBOVESPA para performance.
        return ATIVOS_IBOVESPA

    def coleta_indicadores_de_ativo(self, ticker: str, parse_dtypes: bool = True) -> pd.DataFrame:
        ticker_clean = ticker.replace('.SA', '').strip().upper()
        url = self.url_kpis_ticker + ticker_clean

        try:
            resp = requests.get(url=url, headers=self.request_header)
            resp.raise_for_status()
            html_content = resp.text
            soup = BeautifulSoup(html_content, "lxml")
            tables = soup.find_all("table", attrs={'class': 'w728'})
            
            financial_data_raw = []
            for table in tables:
                table_row = table.find_all("tr")
                for table_data in table_row:
                    cells_list = table_data.find_all("td")
                    # Lógica simplificada de extração par-valor
                    for i in range(0, len(cells_list), 2):
                        if i+1 < len(cells_list):
                            key = cells_list[i].text.replace("?", "").strip()
                            val = cells_list[i+1].text.strip()
                            if key and val:
                                financial_data_raw.append({key: val})

            financial_data = {name: value for dictionary in financial_data_raw for name, value in dictionary.items()}
            
            # Identificar Setor e Subsetor explicitamente (geralmente estão no topo)
            # Scraper específico para metadados de setor pode ser necessário se não vier na tabela
            # Fallback simples:
            links = soup.find_all('a')
            for link in links:
                if 'resultado.php?setor=' in str(link.get('href')):
                    financial_data['Setor'] = link.text.strip()
                    break
            
            if not financial_data:
                return pd.DataFrame()

            df_ativo = pd.DataFrame(financial_data, index=[0])
            
            # Renomear colunas
            df_indicadores = df_ativo.rename(columns=self.metadata_cols_acoes, errors="ignore")
            
            # Filtrar apenas colunas de interesse
            cols_interesse = list(self.metadata_cols_acoes.values())
            cols_finais = [c for c in cols_interesse if c in df_indicadores.columns]
            df_indicadores = df_indicadores[cols_finais]
            
            if parse_dtypes:
                # Heurística simples para detectar colunas float/pct
                pct_cols = ['roe', 'div_yield', 'revenue_growth', 'operating_margin']
                float_cols = ['pe_ratio', 'pb_ratio', 'debt_to_equity', 'current_ratio']
                
                df_indicadores = self._parse_float_cols(df_indicadores, [c for c in float_cols if c in df_indicadores.columns])
                df_indicadores = self._parse_pct_cols(df_indicadores, [c for c in pct_cols if c in df_indicadores.columns])

            return df_indicadores

        except Exception as e:
            # self.logger.error(f"Erro ao extrair {ticker}: {e}")
            return pd.DataFrame()

# =============================================================================
# 6. CLASSE: ANALISADOR DE PERFIL DO INVESTIDOR
# =============================================================================

class AnalisadorPerfilInvestidor:
    def __init__(self):
        self.nivel_risco = ""
        self.horizonte_tempo = ""
        self.dias_lookback_ml = 20
    
    def determinar_nivel_risco(self, pontuacao: int) -> str:
        if pontuacao <= 46: return "CONSERVADOR"
        elif pontuacao <= 67: return "INTERMEDIÁRIO"
        elif pontuacao <= 88: return "MODERADO"
        elif pontuacao <= 109: return "MODERADO-ARROJADO"
        else: return "AVANÇADO"
    
    def calcular_perfil(self, respostas_risco_originais: dict) -> tuple[str, str, int, int]:
        # Extrai apenas o texto da chave para pontuação (ex: 'CT: ...' -> 'CT: ...')
        # No novo UI, a value já é a chave.
        
        score_risk_accept = SCORE_MAP_ORIGINAL.get(respostas_risco_originais['risk_accept'], 3)
        score_max_gain = SCORE_MAP_ORIGINAL.get(respostas_risco_originais['max_gain'], 3)
        score_stable_growth = SCORE_MAP_INV_ORIGINAL.get(respostas_risco_originais['stable_growth'], 3)
        score_avoid_loss = SCORE_MAP_INV_ORIGINAL.get(respostas_risco_originais['avoid_loss'], 3)
        score_level = SCORE_MAP_CONHECIMENTO_ORIGINAL.get(respostas_risco_originais['level'], 3)
        score_reaction = SCORE_MAP_REACTION_ORIGINAL.get(respostas_risco_originais['reaction'], 3)

        pontuacao = (score_risk_accept * 5 + score_max_gain * 5 + score_stable_growth * 5 +
                     score_avoid_loss * 5 + score_level * 3 + score_reaction * 3)
        
        nivel_risco = self.determinar_nivel_risco(pontuacao)
        
        # Horizonte
        horizon_ans = respostas_risco_originais.get('time_purpose', 'B')[0]
        liquidez_ans = respostas_risco_originais.get('liquidity', 'B')[0]
        
        if horizon_ans == 'C' or liquidez_ans == 'C':
            horizonte_tempo = "LONGO PRAZO"
            ml_lookback = 120
        elif horizon_ans == 'B' or liquidez_ans == 'B':
            horizonte_tempo = "MÉDIO PRAZO"
            ml_lookback = 60
        else:
            horizonte_tempo = "CURTO PRAZO"
            ml_lookback = 20
            
        return nivel_risco, horizonte_tempo, ml_lookback, pontuacao

# =============================================================================
# 7. ENGENHARIA DE FEATURES (CÁLCULOS IN-LIVE)
# =============================================================================

class EngenheiroFeatures:
    """Calcula indicadores técnicos em tempo real."""

    @staticmethod
    def calcular_indicadores_tecnicos(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < 30: return df
        
        # RSI (14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD (12, 26, 9)
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # Bandas de Bollinger (20, 2)
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['std_20'] = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['sma_20'] + (df['std_20'] * 2)
        df['bb_lower'] = df['sma_20'] - (df['std_20'] * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma_20']
        
        # Volatilidade Histórica (Anualizada)
        df['returns'] = df['Close'].pct_change()
        df['volatility_21d'] = df['returns'].rolling(window=21).std() * np.sqrt(252)
        
        # Momentum (ROC)
        df['momentum_60'] = df['Close'].pct_change(periods=60)
        
        return df.fillna(0)

    @staticmethod
    def normalizar(serie: pd.Series, maior_melhor: bool = True) -> pd.Series:
        """Normaliza uma série para o range [0, 1]."""
        if serie.empty or serie.nunique() <= 1: return pd.Series(0.5, index=serie.index)
        
        q_low = serie.quantile(0.02)
        q_high = serie.quantile(0.98)
        serie_clipped = serie.clip(q_low, q_high)
        
        min_val = serie_clipped.min()
        max_val = serie_clipped.max()
        
        if max_val == min_val: return pd.Series(0.5, index=serie.index)
        
        norm = (serie_clipped - min_val) / (max_val - min_val)
        return norm if maior_melhor else (1 - norm)

# =============================================================================
# 8. COLETA DE DADOS E PROCESSAMENTO (LIVE)
# =============================================================================

class ColetorDadosLive:
    """Coleta dados do YFinance e Fundamentus em tempo real."""
    
    def __init__(self):
        self.fundamentus_scraper = Fundamentus()
        self.dados_por_ativo = {}
        self.dados_fundamentalistas = pd.DataFrame()
        self.metricas_performance = pd.DataFrame()

    def coletar(self, simbolos: list, periodo: str = PERIODO_DADOS) -> bool:
        dados_fund_list = []
        metricas_perf_list = []
        
        # 1. Coleta Técnica (YFinance - Bulk Download é mais rápido)
        try:
            st.toast(f"Baixando dados técnicos de {len(simbolos)} ativos...", icon="⬇️")
            tickers_yf = [s for s in simbolos]
            # Download em lote
            data_yf = yf.download(tickers_yf, period=periodo, group_by='ticker', auto_adjust=True, threads=True)
        except Exception as e:
            st.error(f"Erro no download YFinance: {e}")
            return False

        # 2. Processamento Individual
        ativos_validos = []
        
        # Barra de progresso para processamento (cálculos + fundamentus)
        progresso_texto = st.empty()
        bar_progresso = st.progress(0)
        
        total = len(simbolos)
        
        for i, simbolo in enumerate(simbolos):
            progresso_texto.text(f"Processando {simbolo} (Técnico + Fundamentalista)...")
            bar_progresso.progress((i + 1) / total)
            
            # Extrair DF do MultiIndex do YFinance
            if len(simbolos) > 1:
                try:
                    df_ativo = data_yf[simbolo].copy()
                except KeyError:
                    continue
            else:
                df_ativo = data_yf.copy() # Se for só 1 ativo, não tem nível superior
            
            # Validação básica
            if df_ativo.empty or len(df_ativo) < MIN_DIAS_HISTORICO:
                continue
            
            # Limpeza de colunas (YFinance as vezes traz colunas vazias)
            df_ativo = df_ativo.dropna(how='all')
            if 'Close' not in df_ativo.columns: continue
            
            # Engenharia de Features (Técnica)
            df_ativo = EngenheiroFeatures.calcular_indicadores_tecnicos(df_ativo)
            self.dados_por_ativo[simbolo] = df_ativo
            
            # Coleta Fundamentalista (Scraping)
            # Nota: Scraping sequencial pode ser lento. 
            try:
                df_fund = self.fundamentus_scraper.coleta_indicadores_de_ativo(simbolo)
                if not df_fund.empty:
                    fund_dict = df_fund.iloc[0].to_dict()
                    fund_dict['Ticker'] = simbolo
                    dados_fund_list.append(fund_dict)
            except Exception:
                pass # Segue sem fundamentos se falhar
            
            # Métricas de Performance Simples
            retorno_anual = df_ativo['returns'].mean() * 252
            vol_anual = df_ativo['returns'].std() * np.sqrt(252)
            sharpe = (retorno_anual - TAXA_LIVRE_RISCO) / vol_anual if vol_anual > 0 else 0
            max_dd = (df_ativo['Close'] / df_ativo['Close'].cummax() - 1).min()
            
            metricas_perf_list.append({
                'Ticker': simbolo,
                'sharpe': sharpe,
                'retorno_anual': retorno_anual,
                'volatilidade_anual': vol_anual,
                'max_drawdown': max_dd
            })
            
            ativos_validos.append(simbolo)
            
        progresso_texto.empty()
        bar_progresso.empty()
        
        if not ativos_validos:
            return False
            
        self.metricas_performance = pd.DataFrame(metricas_perf_list).set_index('Ticker')
        
        if dados_fund_list:
            self.dados_fundamentalistas = pd.DataFrame(dados_fund_list).set_index('Ticker')
            # Preencher setores faltantes com "Desconhecido"
            if 'sector' not in self.dados_fundamentalistas.columns:
                self.dados_fundamentalistas['sector'] = 'Desconhecido'
        else:
            # Cria DF vazio com colunas esperadas para não quebrar
            self.dados_fundamentalistas = pd.DataFrame(index=ativos_validos, columns=['pe_ratio', 'roe', 'sector'])
            
        return True

# =============================================================================
# 9. OTIMIZADOR E CONSTRUTOR
# =============================================================================

class OtimizadorPortfolio:
    """Otimização MPT (Markowitz)"""
    def __init__(self, returns_df):
        self.returns = returns_df
        self.mean_returns = returns_df.mean() * 252
        self.cov_matrix = returns_df.cov() * 252
        self.num_ativos = len(returns_df.columns)

    def otimizar(self, estrategia='MaxSharpe'):
        if self.num_ativos == 0: return {}
        
        def get_ret_vol_sr(weights):
            weights = np.array(weights)
            ret = np.sum(self.mean_returns * weights)
            vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            sr = (ret - TAXA_LIVRE_RISCO) / vol if vol > 0 else -10
            return np.array([ret, vol, sr])
        
        def neg_sharpe(weights): return -get_ret_vol_sr(weights)[2]
        def minimize_vol(weights): return get_ret_vol_sr(weights)[1]
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((PESO_MIN, PESO_MAX) for _ in range(self.num_ativos))
        init_guess = [1./self.num_ativos] * self.num_ativos
        
        try:
            if estrategia == 'MinVolatility':
                opt = minimize(minimize_vol, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
            else:
                opt = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
                
            return {ticker: weight for ticker, weight in zip(self.returns.columns, opt.x)}
        except:
            return {ticker: 1./self.num_ativos for ticker in self.returns.columns}

class ConstrutorPortfolioAutoML:
    """Orquestrador Live"""
    def __init__(self, valor_investimento):
        self.valor_investimento = valor_investimento
        self.dados_por_ativo = {}
        self.dados_fundamentalistas = pd.DataFrame()
        self.metricas_performance = pd.DataFrame()
        self.predicoes_ml = {}
        self.ativos_selecionados = []
        self.alocacao_portfolio = {}
        self.metricas_portfolio = {}
        self.scores_combinados = pd.DataFrame()
        self.metodo_alocacao_atual = ""
        self.justificativas_selecao = {}
        self.pesos_atuais = {}

    def executar_pipeline(self, simbolos, perfil_inputs, progress_bar):
        coletor = ColetorDadosLive()
        
        progress_bar.progress(10, text="Baixando e processando dados em tempo real...")
        if not coletor.coletar(simbolos):
            st.error("Falha ao coletar dados.")
            return False
            
        self.dados_por_ativo = coletor.dados_por_ativo
        self.dados_fundamentalistas = coletor.dados_fundamentalistas
        self.metricas_performance = coletor.metricas_performance
        
        progress_bar.progress(50, text="Treinando modelos de Machine Learning (In-Live)...")
        self.treinar_ml_live(perfil_inputs.get('ml_lookback_days', 20))
        
        progress_bar.progress(70, text="Calculando scores e selecionando ativos...")
        self.pontuar_e_selecionar(perfil_inputs.get('time_horizon', 'MÉDIO PRAZO'))
        
        progress_bar.progress(90, text="Otimizando pesos do portfólio...")
        self.otimizar_pesos(perfil_inputs.get('risk_level', 'MODERADO'))
        
        self.calcular_metricas_finais()
        self.gerar_justificativas()
        
        progress_bar.progress(100, text="Concluído!")
        time.sleep(1)
        return True

    def treinar_ml_live(self, lookback_days):
        """Treina um RandomForest para cada ativo para prever retorno > 0"""
        for ticker, df in self.dados_por_ativo.items():
            try:
                if len(df) < 60: 
                    self.predicoes_ml[ticker] = {'proba': 0.5, 'auc': 0.5}
                    continue
                
                # Criar Target: Retorno futuro > 0
                df['target'] = (df['Close'].shift(-lookback_days) > df['Close']).astype(int)
                
                features = ['rsi_14', 'macd_diff', 'volatility_21d', 'returns', 'momentum_60']
                df_model = df.dropna()
                
                X = df_model[features]
                y = df_model['target']
                
                # Train/Test Split temporal (treina no passado, prevê o último estado)
                split = int(len(X) * 0.8)
                X_train, y_train = X.iloc[:split], y.iloc[:split]
                X_test, y_test = X.iloc[split:], y.iloc[split:]
                
                model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
                model.fit(X_train, y_train)
                
                # Avaliação simples
                preds_test = model.predict_proba(X_test)[:, 1]
                auc = 0.5
                try:
                    if len(np.unique(y_test)) > 1:
                        auc = roc_auc_score(y_test, preds_test)
                except: pass
                
                # Predição Atual (Última linha do DF original)
                last_row = df.iloc[[-1]][features].fillna(0)
                prob_alta = model.predict_proba(last_row)[0, 1]
                
                self.predicoes_ml[ticker] = {'proba': prob_alta, 'auc': auc}
                
                # Salva no DF para uso posterior
                self.dados_por_ativo[ticker].loc[df.index[-1], 'ML_Proba'] = prob_alta
                
            except Exception as e:
                self.predicoes_ml[ticker] = {'proba': 0.5, 'auc': 0.5}

    def pontuar_e_selecionar(self, horizonte):
        # Pesos dinâmicos
        if horizonte == "CURTO PRAZO": w_p, w_f, w_t = 0.4, 0.1, 0.2
        elif horizonte == "LONGO PRAZO": w_p, w_f, w_t = 0.4, 0.5, 0.1
        else: w_p, w_f, w_t = 0.4, 0.3, 0.3
        
        # Rebalacear com ML
        total = w_p + w_f + w_t
        factor = (1 - WEIGHT_ML) / total
        w_p *= factor; w_f *= factor; w_t *= factor
        self.pesos_atuais = {'Performance': w_p, 'Fundamentos': w_f, 'Técnicos': w_t, 'ML': WEIGHT_ML}
        
        scores = {}
        for t in self.dados_por_ativo.keys():
            # Coleta dados
            perf = self.metricas_performance.loc[t]
            fund = self.dados_fundamentalistas.loc[t] if t in self.dados_fundamentalistas.index else pd.Series()
            ml = self.predicoes_ml.get(t, {'proba': 0.5})
            tech = self.dados_por_ativo[t].iloc[-1]
            
            # Scores Individuais
            # Performance: Sharpe
            s_perf = 0.5
            if not pd.isna(perf.get('sharpe')):
                s_perf = min(max(perf['sharpe'] / 3, 0), 1) # Normalização grosseira [-0, 3] -> [0, 1]
            
            # Fundamental: P/L (menor melhor) e ROE (maior melhor)
            s_fund = 0.5
            pl = fund.get('pe_ratio', np.nan)
            roe = fund.get('roe', np.nan)
            if not pd.isna(pl) and pl > 0: 
                s_pl = 1 / (1 + np.log1p(pl)) # Decaimento suave
            else: s_pl = 0.5
            if not pd.isna(roe): s_roe = min(max(roe, 0), 1)
            else: s_roe = 0.5
            s_fund = (s_pl + s_roe) / 2
            
            # Técnico: RSI (menor melhor p/ compra) e MACD (maior melhor)
            rsi = tech.get('rsi_14', 50)
            macd = tech.get('macd_diff', 0)
            s_rsi = 1 - (rsi / 100)
            s_macd = 0.5 + (np.tanh(macd) / 2) # Tanh para normalizar [-1, 1] -> [0, 1]
            s_tech = (s_rsi + s_macd) / 2
            
            # ML
            s_ml = ml['proba']
            
            total_score = (s_perf * w_p) + (s_fund * w_f) + (s_tech * w_t) + (s_ml * WEIGHT_ML)
            
            scores[t] = {
                'total_score': total_score,
                'sharpe': perf.get('sharpe', 0),
                'pe_ratio': pl,
                'roe': roe,
                'rsi_14': rsi,
                'ML_Proba': s_ml
            }
            
        self.scores_combinados = pd.DataFrame.from_dict(scores, orient='index').sort_values('total_score', ascending=False)
        
        # Clusterização simplificada para seleção
        # Se houver poucos ativos, pega Top N simples
        if len(self.scores_combinados) < 10:
            self.ativos_selecionados = self.scores_combinados.head(NUM_ATIVOS_PORTFOLIO).index.tolist()
        else:
            # Tenta selecionar diversificado (Clusterizacao K-Means com dados disponiveis)
            try:
                feats = self.scores_combinados[['sharpe', 'pe_ratio', 'rsi_14']].fillna(0)
                kmeans = KMeans(n_clusters=NUM_ATIVOS_PORTFOLIO, random_state=42).fit(feats)
                self.scores_combinados['cluster'] = kmeans.labels_
                
                # Seleciona o melhor de cada cluster
                selecionados = []
                for c in range(NUM_ATIVOS_PORTFOLIO):
                    cluster_members = self.scores_combinados[self.scores_combinados['cluster'] == c]
                    if not cluster_members.empty:
                        selecionados.append(cluster_members['total_score'].idxmax())
                self.ativos_selecionados = selecionados
            except:
                self.ativos_selecionados = self.scores_combinados.head(NUM_ATIVOS_PORTFOLIO).index.tolist()

    def otimizar_pesos(self, nivel_risco):
        if not self.ativos_selecionados: return
        
        tickers = self.ativos_selecionados
        # Monta DF de retornos apenas dos selecionados
        returns_list = [self.dados_por_ativo[t]['returns'] for t in tickers]
        df_returns = pd.concat(returns_list, axis=1, keys=tickers).dropna()
        
        otimizador = OtimizadorPortfolio(df_returns)
        
        if 'CONSERVADOR' in nivel_risco: strategy = 'MinVolatility'
        else: strategy = 'MaxSharpe'
        
        pesos = otimizador.otimizar(estrategia=strategy)
        self.metodo_alocacao_atual = strategy
        
        self.alocacao_portfolio = {
            t: {'weight': w, 'amount': w * self.valor_investimento}
            for t, w in pesos.items()
        }

    def calcular_metricas_finais(self):
        # Cálculo simples do portfólio consolidado
        if not self.alocacao_portfolio: return
        
        tickers = list(self.alocacao_portfolio.keys())
        weights = np.array([self.alocacao_portfolio[t]['weight'] for t in tickers])
        
        returns_list = [self.dados_por_ativo[t]['returns'] for t in tickers]
        df_returns = pd.concat(returns_list, axis=1, keys=tickers).dropna()
        
        port_rets = df_returns.dot(weights)
        
        ann_ret = port_rets.mean() * 252
        ann_vol = port_rets.std() * np.sqrt(252)
        sharpe = (ann_ret - TAXA_LIVRE_RISCO) / ann_vol if ann_vol > 0 else 0
        
        cum_ret = (1 + port_rets).cumprod()
        dd = (cum_ret / cum_ret.cummax() - 1).min()
        
        self.metricas_portfolio = {
            'annual_return': ann_ret,
            'annual_volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': dd
        }

    def gerar_justificativas(self):
        for t in self.ativos_selecionados:
            s = self.scores_combinados.loc[t]
            txt = f"Score Total: {s['total_score']:.2f} | Sharpe: {s['sharpe']:.2f} | ML Prob: {s['ML_Proba']*100:.1f}% | RSI: {s['rsi_14']:.1f}"
            self.justificativas_selecao[t] = txt

# =============================================================================
# 10. INTERFACE STREAMLIT
# =============================================================================

def obter_template_grafico():
    return {'template': 'plotly_white', 'margin': dict(l=20, r=20, t=40, b=20)}

def configurar_pagina():
    st.set_page_config(page_title="AutoML Portfolio Live", page_icon="📈", layout="wide")
    st.markdown("""<style>.main-header {font-size: 2rem; font-weight: bold; text-align: center; margin-bottom: 20px;}</style>""", unsafe_allow_html=True)

def aba_analise_individual_live():
    st.markdown("## 🔍 Análise Live de Ativos")
    
    ativo = st.selectbox("Selecione o Ativo:", TODOS_ATIVOS)
    if st.button("Analisar Ativo (Live Fetch)"):
        with st.spinner(f"Baixando dados de {ativo}..."):
            coletor = ColetorDadosLive()
            # Coleta apenas 1 ativo
            if coletor.coletar([ativo]):
                df = coletor.dados_por_ativo[ativo]
                fund = coletor.dados_fundamentalistas.loc[ativo]
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Preço Atual", f"R$ {df['Close'].iloc[-1]:.2f}")
                col2.metric("RSI (14)", f"{df['rsi_14'].iloc[-1]:.2f}")
                col3.metric("P/L", f"{fund.get('pe_ratio', 'N/A')}")
                
                fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
                fig.update_layout(title=f"{ativo} - Preço Histórico", height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(fund.to_frame().T, use_container_width=True)
            else:
                st.error("Erro ao baixar dados.")

def main():
    configurar_pagina()
    st.markdown('<div class="main-header">Sistema de Portfólios Adaptativos (Live Data v9.0)</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🎯 Setup & Perfil", "🏗️ Construção do Portfólio", "🔍 Análise Rápida"])
    
    # Variáveis de Sessão
    if 'builder' not in st.session_state: st.session_state.builder = None
    
    with tab1:
        st.info("Este sistema agora coleta dados em tempo real do Yahoo Finance e Fundamentus.")
        st.subheader("Definição de Universo")
        modo = st.radio("Modo de Seleção:", ["Índice Completo (IBOV)", "Seleção Setorial", "Manual"])
        
        if modo == "Manual":
            ativos = st.multiselect("Selecione Ativos:", TODOS_ATIVOS, default=['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'WEGE3.SA'])
        elif modo == "Seleção Setorial":
            setor = st.selectbox("Setor:", list(ATIVOS_POR_SETOR.keys()))
            ativos = ATIVOS_POR_SETOR[setor]
        else:
            ativos = ATIVOS_IBOVESPA
            st.warning("⚠️ Selecionar todos os ativos pode demorar alguns minutos devido ao Web Scraping sequencial dos fundamentos.")
            
        st.markdown("---")
        st.subheader("Perfil de Risco")
        # Formulário Simplificado
        c1, c2 = st.columns(2)
        with c1:
            risk = st.select_slider("Apetite a Risco:", options=["Conservador", "Moderado", "Arrojado"], value="Moderado")
            horizon = st.select_slider("Horizonte:", options=["Curto Prazo", "Médio Prazo", "Longo Prazo"], value="Médio Prazo")
        with c2:
            capital = st.number_input("Capital (R$):", value=10000.0)
            
        if st.button("Salvar Configuração", type="primary"):
            st.session_state.config = {
                'ativos': ativos,
                'risk_level': risk.upper(),
                'time_horizon': horizon.upper(),
                'capital': capital,
                # Mapeamento para o pipeline antigo
                'ml_lookback_days': LOOKBACK_ML_DAYS_MAP.get(horizon.lower().replace(' ', '_'), 60)
            }
            st.success(f"Configuração salva! {len(ativos)} ativos selecionados.")

    with tab2:
        if 'config' in st.session_state:
            st.write(f"Pronto para processar **{len(st.session_state.config['ativos'])}** ativos.")
            if st.button("🚀 Gerar Portfólio (Live)", type="primary"):
                builder = ConstrutorPortfolioAutoML(st.session_state.config['capital'])
                bar = st.progress(0, text="Iniciando...")
                
                success = builder.executar_pipeline(
                    st.session_state.config['ativos'],
                    st.session_state.config,
                    bar
                )
                
                if success:
                    st.session_state.builder = builder
                    st.rerun()
        else:
            st.warning("Configure o perfil na aba anterior.")
            
        # Exibição de Resultados
        if st.session_state.builder:
            b = st.session_state.builder
            st.markdown("### 🏆 Carteira Recomendada")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Retorno Esp. (a.a.)", f"{b.metricas_portfolio['annual_return']*100:.2f}%")
            c2.metric("Volatilidade (a.a.)", f"{b.metricas_portfolio['annual_volatility']*100:.2f}%")
            c3.metric("Sharpe", f"{b.metricas_portfolio['sharpe_ratio']:.2f}")
            
            # Gráfico de Pizza
            alloc = pd.DataFrame.from_dict(b.alocacao_portfolio, orient='index')
            if not alloc.empty:
                fig = px.pie(alloc, values='weight', names=alloc.index, title="Alocação Sugerida")
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("#### Detalhes e Justificativas")
                for t in b.ativos_selecionados:
                    st.info(f"**{t}**: {b.justificativas_selecao[t]}")

    with tab3:
        aba_analise_individual_live()

if __name__ == "__main__":
    main()
