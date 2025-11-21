# -*- coding: utf-8 -*-
"""
=============================================================================
SISTEMA DE PORTFÓLIOS ADAPTATIVOS - OTIMIZAÇÃO QUANTITATIVA (INTEGRADO)
=============================================================================

Adaptação do Sistema AutoML para usar lógica de portfolio_analyzer.py no backend,
mantendo o design e funcionalidades visuais do automl_portfolio_elite.

Fonte de Dados:
- Histórico de Preços: yfinance
- Fundamentos: Pynvest (Fundamentus)

Versão: 9.0.0 (Integração Backend Analyzer + Pynvest)
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
from tqdm import tqdm
import json
import traceback

# --- 2. SCIENTIFIC / STATISTICAL TOOLS ---
from scipy.optimize import minimize
from scipy.stats import zscore

# --- 3. STREAMLIT, DATA ACQUISITION, & PLOTTING ---
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf

# --- 4. MACHINE LEARNING (SCIKIT-LEARN) ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# --- 5. PYNVEST INSTALLATION & IMPORT ---
try:
    from pynvest.scrappers.fundamentus import Fundamentus
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pynvest", "lxml"])
    from pynvest.scrappers.fundamentus import Fundamentus

# --- 6. CONFIGURATION ---
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURAÇÕES E CONSTANTES GLOBAIS
# =============================================================================

# Constantes do Portfolio Analyzer
RISK_FREE_RATE = 0.1075 # Mantido do original (Selic aprox)
MIN_WEIGHT = 0.10
MAX_WEIGHT = 0.30
NUM_ATIVOS_PORTFOLIO = 5
FIXED_ANALYSIS_PERIOD = '5y' # Usando 5 anos para ter histórico suficiente para ML
MIN_HISTORY_DAYS = 252 # 1 ano de dados mínimos

# Mapeamento de Prazos para ML (4, 8, 12 meses em dias úteis aprox)
LOOKBACK_ML_DAYS_MAP = {
    'curto_prazo': 84,   # ~4 meses
    'medio_prazo': 168,  # ~8 meses
    'longo_prazo': 252   # ~12 meses
}

# =============================================================================
# 2. LISTAS DE ATIVOS E SETORES
# =============================================================================

# Lista Consolidada de Ativos (Ibovespa)
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

TODOS_ATIVOS = sorted(list(set(ATIVOS_IBOVESPA)))

# Mapeamento Setorial (Usado apenas para visualização/agrupamento inicial)
ATIVOS_POR_SETOR = {
    'Geral': TODOS_ATIVOS # Simplificado para garantir funcionamento, a lógica de setor virá do Pynvest
}


# =============================================================================
# 3. MAPEAMENTOS DE PONTUAÇÃO DO QUESTIONÁRIO
# =============================================================================
# (Mantido idêntico ao automl para compatibilidade com a interface)

OPTIONS_CONCORDA = [
    "CT: (Concordo Totalmente) - Estou confortável com altas flutuações...",
    "C: (Concordo) - Aceito alguma volatilidade...",
    "N: (Neutro) - Tenho dificuldade em opinar...",
    "D: (Discordo) - Prefiro estratégias mais cautelosas...",
    "DT: (Discordo Totalmente) - Não estou disposto a ver meu patrimônio flutuar..."
]
MAP_CONCORDA = {OPTIONS_CONCORDA[0]: 'CT: Concordo Totalmente', OPTIONS_CONCORDA[1]: 'C: Concordo', OPTIONS_CONCORDA[2]: 'N: Neutro', OPTIONS_CONCORDA[3]: 'D: Discordo', OPTIONS_CONCORDA[4]: 'DT: Discordo Totalmente'}

OPTIONS_DISCORDA = [
    "CT: (Concordo Totalmente) - A preservação do capital é minha prioridade...",
    "C: (Concordo) - É muito importante para mim evitar perdas...",
    "N: (Neutro) - Busco um equilíbrio...",
    "D: (Discordo) - Estou focado no crescimento...",
    "DT: (Discordo Totalmente) - Meu foco é maximizar o retorno..."
]
MAP_DISCORDA = {OPTIONS_DISCORDA[0]: 'CT: Concordo Totalmente', OPTIONS_DISCORDA[1]: 'C: Concordo', OPTIONS_DISCORDA[2]: 'N: Neutro', OPTIONS_DISCORDA[3]: 'D: Discordo', OPTIONS_DISCORDA[4]: 'DT: Discordo Totalmente'}

OPTIONS_REACTION_DETALHADA = ["A: (Vender Imediatamente)", "B: (Manter e Reavaliar)", "C: (Comprar Mais)"]
MAP_REACTION = {OPTIONS_REACTION_DETALHADA[0]: 'A: Venderia imediatamente', OPTIONS_REACTION_DETALHADA[1]: 'B: Manteria e reavaliaria a tese', OPTIONS_REACTION_DETALHADA[2]: 'C: Compraria mais para aproveitar preços baixos'}

OPTIONS_CONHECIMENTO_DETALHADA = ["A: (Avançado)", "B: (Intermediário)", "C: (Iniciante)"]
MAP_CONHECIMENTO = {OPTIONS_CONHECIMENTO_DETALHADA[0]: 'A: Avançado', OPTIONS_CONHECIMENTO_DETALHADA[1]: 'B: Intermediário', OPTIONS_CONHECIMENTO_DETALHADA[2]: 'C: Iniciante'}

OPTIONS_TIME_HORIZON_DETALHADA = ['A: Curto (até 1 ano)', 'B: Médio (1-5 anos)', 'C: Longo (5+ anos)']
OPTIONS_LIQUIDEZ_DETALHADA = ['A: Menos de 6 meses', 'B: Entre 6 meses e 2 anos', 'C: Mais de 2 anos']

SCORE_MAP_ORIGINAL = {'CT: Concordo Totalmente': 5, 'C: Concordo': 4, 'N: Neutro': 3, 'D: Discordo': 2, 'DT: Discordo Totalmente': 1}
SCORE_MAP_INV_ORIGINAL = {'CT: Concordo Totalmente': 1, 'C: Concordo': 2, 'N: Neutro': 3, 'D: Discordo': 4, 'DT: Discordo Totalmente': 5}
SCORE_MAP_CONHECIMENTO_ORIGINAL = {'A: Avançado': 5, 'B: Intermediário': 3, 'C: Iniciante': 1}
SCORE_MAP_REACTION_ORIGINAL = {'A: Venderia imediatamente': 1, 'B: Manteria e reavaliaria a tese': 3, 'C: Compraria mais para aproveitar preços baixos': 5}


# =============================================================================
# 4. CLASSE: ANALISADOR DE PERFIL DO INVESTIDOR
# =============================================================================

class AnalisadorPerfilInvestidor:
    """Analisa perfil de risco e horizonte temporal do investidor."""
    
    def __init__(self):
        self.risk_level = ""
        self.time_horizon = ""
        self.ml_lookback_days = 84 # Default curto prazo (4 meses)
    
    def determinar_nivel_risco(self, pontuacao: int) -> str:
        if pontuacao <= 46: return "CONSERVADOR"
        elif pontuacao <= 67: return "INTERMEDIÁRIO"
        elif pontuacao <= 88: return "MODERADO"
        elif pontuacao <= 109: return "MODERADO-ARROJADO"
        else: return "AVANÇADO"
    
    def determinar_horizonte_ml(self, liquidez_key: str, objetivo_key: str) -> tuple[str, int]:
        # Lógica adaptada para os prazos 4, 8 e 12 meses
        time_map = { 'A': 84, 'B': 168, 'C': 252 }
        final_lookback = max( time_map.get(liquidez_key, 84), time_map.get(objetivo_key, 84) )
        
        if final_lookback >= 252:
            self.time_horizon = "LONGO PRAZO"
        elif final_lookback >= 168:
            self.time_horizon = "MÉDIO PRAZO"
        else:
            self.time_horizon = "CURTO PRAZO"
        
        self.ml_lookback_days = final_lookback
        return self.time_horizon, self.ml_lookback_days
    
    def calcular_perfil(self, respostas_risco_originais: dict) -> tuple[str, str, int, int]:
        score_risk_accept = SCORE_MAP_ORIGINAL.get(respostas_risco_originais['risk_accept'], 3)
        score_max_gain = SCORE_MAP_ORIGINAL.get(respostas_risco_originais['max_gain'], 3)
        score_stable_growth = SCORE_MAP_INV_ORIGINAL.get(respostas_risco_originais['stable_growth'], 3)
        score_avoid_loss = SCORE_MAP_INV_ORIGINAL.get(respostas_risco_originais['avoid_loss'], 3)
        score_level = SCORE_MAP_CONHECIMENTO_ORIGINAL.get(respostas_risco_originais['level'], 3)
        score_reaction = SCORE_MAP_REACTION_ORIGINAL.get(respostas_risco_originais['reaction'], 3)

        pontuacao = (score_risk_accept * 5 + score_max_gain * 5 + score_stable_growth * 5 + 
                     score_avoid_loss * 5 + score_level * 3 + score_reaction * 3)
        
        nivel_risco = self.determinar_nivel_risco(pontuacao)
        
        liquidez_key = respostas_risco_originais['liquidity'][0] if isinstance(respostas_risco_originais['liquidity'], str) else 'C'
        objetivo_key = respostas_risco_originais['time_purpose'][0] if isinstance(respostas_risco_originais['time_purpose'], str) else 'C'
        
        horizonte_tempo, ml_lookback = self.determinar_horizonte_ml(liquidez_key, objetivo_key)
        return nivel_risco, horizonte_tempo, ml_lookback, pontuacao

# =============================================================================
# 5. FUNÇÕES DE ESTILO E VISUALIZAÇÃO
# =============================================================================

def obter_template_grafico() -> dict:
    return {
        'plot_bgcolor': '#f8f9fa', 'paper_bgcolor': 'white',
        'font': {'family': 'Arial, sans-serif', 'size': 12, 'color': '#343a40'},
        'title': {'font': {'family': 'Arial, sans-serif', 'size': 16, 'color': '#212529', 'weight': 'bold'}, 'x': 0.5, 'xanchor': 'center'},
        'xaxis': {'showgrid': True, 'gridcolor': '#e9ecef', 'showline': True, 'linecolor': '#ced4da', 'linewidth': 1},
        'yaxis': {'showgrid': True, 'gridcolor': '#e9ecef', 'showline': True, 'linecolor': '#ced4da', 'linewidth': 1},
        'legend': {'bgcolor': 'rgba(255, 255, 255, 0.8)', 'bordercolor': '#e9ecef', 'borderwidth': 1},
        'colorway': ['#212529', '#495057', '#6c757d', '#adb5bd', '#ced4da']
    }

# =============================================================================
# 6. ENGENHARIA DE FEATURES (BASEADO EM PORTFOLIO_ANALYZER.PY)
# =============================================================================

def calculate_technical_indicators(df):
    """Calcula indicadores técnicos (RSI, MACD, Volatilidade, Momentum, SMA)."""
    # Se não houver dados suficientes
    if df.empty or len(df) < 50: return pd.DataFrame()

    df['Returns'] = df['Close'].pct_change()
    
    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # MACD (12, 26, 9)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal'] # Adicionado Histograma para compatibilidade
    
    # Volatilidade Anualizada (20 dias)
    df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
    
    # Momentum (10 dias)
    df['Momentum'] = df['Close'] / df['Close'].shift(10) - 1
    
    # SMAs
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    return df.dropna()

def calculate_risk_metrics(returns):
    """Calcula métricas de risco e retorno."""
    if returns.empty: return {}
    
    annual_return = returns.mean() * 252
    annual_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = (annual_return - RISK_FREE_RATE) / annual_volatility if annual_volatility > 0 else 0
    
    cumulative = (1 + returns).cumprod()
    if cumulative.empty: return {}
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        'annual_return': annual_return, 'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio, 'max_drawdown': max_drawdown
    }

# =============================================================================
# 7. COLETA DE DADOS (YFINANCE + PYNVEST)
# =============================================================================

class ColetorDadosIntegrado:
    """Coleta dados via yfinance (preço) e Pynvest/Fundamentus (fundamentos)."""
    
    def __init__(self, periodo=FIXED_ANALYSIS_PERIOD):
        self.periodo = periodo
        self.pynvest_scrapper = Fundamentus()
        
        self.dados_por_ativo = {}
        self.fundamental_data = pd.DataFrame()
        self.performance_data = pd.DataFrame()
        self.successful_assets = []
        
    def _normalizar_ticker_fundamentus(self, ticker: str) -> str:
        """Remove .SA para usar no Fundamentus."""
        return ticker.replace('.SA', '').lower()

    def coletar_dados(self, simbolos: list, progress_bar=None):
        self.successful_assets = []
        lista_fundamentos = []
        
        total = len(simbolos)
        
        # 1. Coleta Dados Técnicos (Yfinance) - Lote ou Loop otimizado
        # Por simplicidade e controle de erro, faremos loop, mas yf.download em batch seria mais rápido
        # Mantendo a estrutura do analyzer original que faz um por um para robustez.
        
        for i, simbolo in enumerate(simbolos):
            if progress_bar: progress_bar.progress((i / total) * 0.4, text=f"Coletando histórico: {simbolo}")
            
            # --- YFINANCE ---
            try:
                ticker_yf = yf.Ticker(simbolo)
                hist = ticker_yf.history(period=self.periodo)
                
                if hist.empty or len(hist) < MIN_HISTORY_DAYS:
                    continue
                
                # Engenharia de Features Técnica
                df_tecnico = calculate_technical_indicators(hist.copy())
                if df_tecnico.empty: continue
                
                # --- PYNVEST (Fundamentos) ---
                ticker_clean = self._normalizar_ticker_fundamentus(simbolo)
                try:
                    # Tenta pegar dados do Pynvest
                    df_indicadores = self.pynvest_scrapper.coleta_indicadores_de_ativo(ticker_clean)
                    
                    # Mapeamento Pynvest -> Colunas do Analyzer
                    # P/L -> vlr_ind_p_sobre_l
                    # P/VP -> vlr_ind_p_sobre_vp
                    # DY -> vlr_ind_div_yield
                    # ROE -> vlr_ind_roe
                    # Sector -> nome_setor
                    
                    if not df_indicadores.empty:
                        # Extrai valores escalares (assumindo que vem 1 linha por ativo)
                        fund_row = {
                            'Ticker': simbolo,
                            'Sector': df_indicadores['nome_setor'].iloc[0] if 'nome_setor' in df_indicadores.columns else 'Unknown',
                            'PE_Ratio': df_indicadores['vlr_ind_p_sobre_l'].iloc[0] if 'vlr_ind_p_sobre_l' in df_indicadores.columns else np.nan,
                            'PB_Ratio': df_indicadores['vlr_ind_p_sobre_vp'].iloc[0] if 'vlr_ind_p_sobre_vp' in df_indicadores.columns else np.nan,
                            'Div_Yield': df_indicadores['vlr_ind_div_yield'].iloc[0] * 100 if 'vlr_ind_div_yield' in df_indicadores.columns else 0.0, # Pynvest vem decimal? Assumindo decimal -> %
                            'ROE': df_indicadores['vlr_ind_roe'].iloc[0] * 100 if 'vlr_ind_roe' in df_indicadores.columns else np.nan
                        }
                    else:
                         # Fallback simples se falhar
                        fund_row = {'Ticker': simbolo, 'Sector': 'Unknown', 'PE_Ratio': np.nan, 'PB_Ratio': np.nan, 'Div_Yield': 0, 'ROE': np.nan}
                        
                except Exception as e:
                     # Fallback se Pynvest falhar
                    fund_row = {'Ticker': simbolo, 'Sector': 'Unknown', 'PE_Ratio': np.nan, 'PB_Ratio': np.nan, 'Div_Yield': 0, 'ROE': np.nan}
                
                # Sucesso
                self.dados_por_ativo[simbolo] = df_tecnico
                lista_fundamentos.append(fund_row)
                self.successful_assets.append(simbolo)
                
            except Exception:
                continue

        if not self.successful_assets: return False
        
        # DataFrame de Fundamentos Consolidado
        self.fundamental_data = pd.DataFrame(lista_fundamentos).set_index('Ticker')
        
        # Limpeza de Fundamentos (Lógica do Analyzer)
        self.fundamental_data = self.fundamental_data.replace([np.inf, -np.inf], np.nan)
        self.fundamental_data.fillna(self.fundamental_data.median(numeric_only=True), inplace=True)
        self.fundamental_data.fillna(0, inplace=True)
        
        # Métricas de Performance
        metrics = {s: calculate_risk_metrics(self.dados_por_ativo[s]['Returns']) for s in self.successful_assets}
        self.performance_data = pd.DataFrame(metrics).T
        
        return True

    def calculate_cross_sectional_features(self):
        """Calcula features relativas ao setor (Analyzer)."""
        if self.fundamental_data.empty: return
        
        df_fund = self.fundamental_data.copy()
        # Se só tiver 'Unknown', a média por setor não ajuda muito, mas mantemos a lógica
        sector_means = df_fund.groupby('Sector')[['PE_Ratio', 'PB_Ratio']].transform('mean')
        
        df_fund['pe_rel_sector'] = df_fund['PE_Ratio'] / sector_means['PE_Ratio']
        df_fund['pb_rel_sector'] = df_fund['PB_Ratio'] / sector_means['PB_Ratio']
        
        df_fund = df_fund.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        self.fundamental_data = df_fund


# =============================================================================
# 8. OTIMIZADOR (MARKOWITZ) - LÓGICA ANALYZER
# =============================================================================

class PortfolioOptimizer:
    def __init__(self, returns_df):
        self.returns = returns_df
        self.mean_returns = returns_df.mean() * 252
        self.cov_matrix = returns_df.cov() * 252
        self.num_assets = len(returns_df.columns)

    def portfolio_stats(self, weights):
        p_return = np.dot(weights, self.mean_returns)
        p_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return p_return, p_std

    def negative_sharpe(self, weights):
        p_return, p_std = self.portfolio_stats(weights)
        return -(p_return - RISK_FREE_RATE) / p_std if p_std != 0 else -100

    def minimize_volatility(self, weights):
        return self.portfolio_stats(weights)[1]

    def optimize(self, strategy):
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((MIN_WEIGHT, MAX_WEIGHT) for _ in range(self.num_assets))
        initial_guess = np.array([1/self.num_assets] * self.num_assets)

        if strategy == 'MinVolatility':
            objective = self.minimize_volatility
        elif strategy == 'MaxSharpe':
            objective = self.negative_sharpe
        else:
            return None

        try:
            result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
            if result.success:
                return {asset: weight for asset, weight in zip(self.returns.columns, result.x)}
            else:
                return None
        except Exception:
            return None

# =============================================================================
# 9. CONSTRUTOR DE PORTFÓLIO INTEGRADO
# =============================================================================

class ConstrutorPortfolioAutoML:
    """Orquestra o pipeline usando a lógica do Analyzer."""
    
    def __init__(self, valor_investimento: float, periodo: str = FIXED_ANALYSIS_PERIOD):
        self.valor_investimento = valor_investimento
        self.periodo = periodo
        
        self.coletor = ColetorDadosIntegrado(periodo)
        
        # Dados
        self.ml_predictions = {} 
        self.combined_scores = pd.DataFrame()
        
        # Resultados
        self.ativos_selecionados = []
        self.alocacao_portfolio = {}
        self.metricas_portfolio = {}
        self.metodo_alocacao_atual = ""
        self.justificativas_selecao = {}
        self.pesos_atuais = {}

    def apply_clustering_and_ml(self, ml_lookback_days, progress_bar=None):
        """
        Lógica do Analyzer para Clustering e ML.
        Treina modelos para o horizonte especificado (4, 8 ou 12 meses).
        """
        fund_data = self.coletor.fundamental_data
        perf_data = self.coletor.performance_data
        
        # 1. Clustering
        clustering_df = fund_data[['PE_Ratio', 'PB_Ratio', 'Div_Yield', 'ROE']].join(
            perf_data[['sharpe_ratio', 'annual_volatility']], how='inner'
        ).fillna(0)
        
        if len(clustering_df) >= 5:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(clustering_df)
            n_comp = min(data_scaled.shape[1], 3)
            pca = PCA(n_components=n_comp)
            data_pca = pca.fit_transform(data_scaled)
            n_clusters = min(len(data_pca), 5)
            if n_clusters < 2: n_clusters = 1
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(data_pca)
            fund_data['Cluster'] = pd.Series(clusters, index=clustering_df.index).fillna(-1).astype(int)
        else:
            fund_data['Cluster'] = 0
            
        self.coletor.fundamental_data = fund_data # Atualiza
        
        # 2. ML (Random Forest)
        features_ml = [
            'RSI', 'MACD', 'Volatility', 'Momentum', 'SMA_50', 'SMA_200',
            'PE_Ratio', 'PB_Ratio', 'Div_Yield', 'ROE',
            'pe_rel_sector', 'pb_rel_sector', 'Cluster'
        ]
        
        assets = self.coletor.successful_assets
        total = len(assets)
        
        for i, symbol in enumerate(assets):
            if progress_bar: progress_bar.progress(0.5 + (i/total)*0.2, text=f"Treinando ML: {symbol}")
            
            df = self.coletor.dados_por_ativo[symbol].copy()
            
            # Target: Retorno > 0 daqui a N dias
            df['Future_Direction'] = np.where(
                df['Close'].pct_change(ml_lookback_days).shift(-ml_lookback_days) > 0, 1, 0
            )
            
            # Adiciona fundamentos ao DF temporal
            if symbol in fund_data.index:
                f_row = fund_data.loc[symbol].to_dict()
                for col in [f for f in features_ml if f not in df.columns]:
                    if col in f_row: df[col] = f_row[col]
            
            current_cols = [f for f in features_ml if f in df.columns]
            df.dropna(subset=current_cols + ['Future_Direction'], inplace=True)
            
            if len(df) < MIN_HISTORY_DAYS: continue
            
            X = df[current_cols].iloc[:-ml_lookback_days].copy()
            y = df['Future_Direction'].iloc[:-ml_lookback_days]
            
            if 'Cluster' in X.columns: X['Cluster'] = X['Cluster'].astype(str)
            
            numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = ['Cluster'] if 'Cluster' in X.columns else []
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), [f for f in numeric_cols if 'rel_sector' not in f]),
                    ('rel', 'passthrough', [f for f in numeric_cols if 'rel_sector' in f]),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
                ], remainder='passthrough'
            )
            
            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced'))
            ])
            
            try:
                if len(np.unique(y)) < 2:
                    self.ml_predictions[symbol] = {'predicted_proba_up': 0.5, 'auc_roc_score': 0.5, 'model_name': 'Classe Única'}
                    continue
                    
                model.fit(X, y)
                scores = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=5), scoring='roc_auc')
                auc = scores.mean()
                
                last_feat = df[current_cols].iloc[[-ml_lookback_days]].copy()
                if 'Cluster' in last_feat.columns: last_feat['Cluster'] = last_feat['Cluster'].astype(str)
                
                proba = model.predict_proba(last_feat)[0][1]
                
                self.ml_predictions[symbol] = {'predicted_proba_up': proba, 'auc_roc_score': auc, 'model_name': 'RandomForest'}
            except:
                self.ml_predictions[symbol] = {'predicted_proba_up': 0.5, 'auc_roc_score': 0.5, 'model_name': 'Erro'}

    def _normalize_score(self, series, higher_better=True):
        """Z-score robusto (0-100)."""
        clean = series.replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty or clean.std() == 0: return pd.Series(50, index=series.index)
        z = zscore(clean, nan_policy='omit')
        norm = pd.Series(50 + (z.clip(-3, 3)/3)*50, index=clean.index)
        if not higher_better: norm = 100 - norm
        return norm.reindex(series.index, fill_value=50)

    def score_and_rank_assets(self, time_horizon):
        """Pontuação e Ranking do Analyzer."""
        
        # Pesos conforme horizonte
        if "CURTO" in time_horizon:
            W_PERF, W_FUND, W_TECH = 0.40, 0.10, 0.50
        elif "LONGO" in time_horizon:
            W_PERF, W_FUND, W_TECH = 0.40, 0.50, 0.10
        else:
            W_PERF, W_FUND, W_TECH = 0.40, 0.30, 0.30
            
        self.pesos_atuais = {'Performance': W_PERF, 'Fundamentos': W_FUND, 'Técnicos': W_TECH, 'ML': 0.3} # ML fixo no cálculo final do analyzer
        
        combined = self.coletor.performance_data.join(self.coletor.fundamental_data, how='inner').copy()
        
        # Adiciona dados técnicos atuais
        for s in combined.index:
            if s in self.coletor.dados_por_ativo:
                df = self.coletor.dados_por_ativo[s]
                combined.loc[s, 'RSI_current'] = df['RSI'].iloc[-1]
                combined.loc[s, 'MACD_current'] = df['MACD'].iloc[-1]
        
        scores = pd.DataFrame(index=combined.index)
        
        # Scores Parciais
        scores['performance_score'] = self._normalize_score(combined['sharpe_ratio'], True) * W_PERF
        
        fund_score = self._normalize_score(combined.get('PE_Ratio', 50), False) * 0.5
        fund_score += self._normalize_score(combined.get('ROE', 50), True) * 0.5
        scores['fundamental_score'] = fund_score * W_FUND
        
        tech_score = 0
        if W_TECH > 0:
            rsi_prox = 100 - abs(combined.get('RSI_current', 50) - 50)
            tech_score += self._normalize_score(rsi_prox, True) * 0.5
            tech_score += self._normalize_score(combined.get('MACD_current', 50), True) * 0.5
        scores['technical_score'] = tech_score * W_TECH
        
        # ML Score
        ml_probs = pd.Series({s: self.ml_predictions.get(s, {'predicted_proba_up': 0.5})['predicted_proba_up'] for s in combined.index})
        ml_confs = pd.Series({s: self.ml_predictions.get(s, {'auc_roc_score': 0.5})['auc_roc_score'] for s in combined.index})
        
        combined['ML_Proba'] = ml_probs
        combined['ML_Confidence'] = ml_confs.fillna(0.5)
        
        ml_score_base = (self._normalize_score(combined['ML_Proba'], True) * 0.5 + 
                         self._normalize_score(combined['ML_Confidence'], True) * 0.5) * 0.3
        
        scores['ml_score_weighted'] = ml_score_base * combined['ML_Confidence']
        
        score_sum = W_PERF + W_FUND + W_TECH
        scores['base_score'] = scores[['performance_score', 'fundamental_score', 'technical_score']].sum(axis=1) / score_sum if score_sum > 0 else 50
        
        # TOTAL
        scores['total_score'] = scores['base_score'] * 0.7 + scores['ml_score_weighted']
        
        self.combined_scores = scores.join(combined).sort_values('total_score', ascending=False)

    def select_final_portfolio(self):
        """Seleção com diversificação setorial (Analyzer)."""
        ranked = self.combined_scores.index.tolist()
        final = []
        selected_sectors = set()
        
        fund_data = self.coletor.fundamental_data
        
        for asset in ranked:
            sector = fund_data.loc[asset, 'Sector'] if asset in fund_data.index else 'Unknown'
            
            if sector not in selected_sectors or sector == 'Unknown': # Relaxando Unknown
                final.append(asset)
                selected_sectors.add(sector)
            
            if len(final) >= NUM_ATIVOS_PORTFOLIO: break
            
        # Fallback se não preencher
        if len(final) < NUM_ATIVOS_PORTFOLIO:
            for asset in ranked:
                if asset not in final:
                    final.append(asset)
                if len(final) >= NUM_ATIVOS_PORTFOLIO: break
                
        self.ativos_selecionados = final

    def optimize_allocation(self, risk_level):
        if len(self.ativos_selecionados) < NUM_ATIVOS_PORTFOLIO:
            self.metodo_alocacao_atual = "Pesos Iguais (Insuficiente)"
            return {a: {'weight': 1/len(self.ativos_selecionados), 'amount': self.valor_investimento/len(self.ativos_selecionados)} for a in self.ativos_selecionados}
            
        # Retornos dos selecionados
        df_sel = pd.DataFrame({s: self.coletor.dados_por_ativo[s]['Returns'] for s in self.ativos_selecionados}).dropna()
        
        if df_sel.shape[0] < 50:
            weights = {a: 1/len(self.ativos_selecionados) for a in self.ativos_selecionados}
            self.metodo_alocacao_atual = "Pesos Iguais (Dados Curtos)"
        else:
            opt = PortfolioOptimizer(df_sel)
            if 'CONSERVADOR' in risk_level or 'INTERMEDIÁRIO' in risk_level:
                weights = opt.optimize('MinVolatility')
                self.metodo_alocacao_atual = "Min Volatility"
            else:
                weights = opt.optimize('MaxSharpe')
                self.metodo_alocacao_atual = "Max Sharpe"
                
            if weights is None:
                weights = {a: 1/len(self.ativos_selecionados) for a in self.ativos_selecionados}
                self.metodo_alocacao_atual = "Fallback (Pesos Iguais)"
        
        # Formata
        self.alocacao_portfolio = {
            a: {'weight': weights[a], 'amount': self.valor_investimento * weights[a]} 
            for a in weights
        }
        return self.alocacao_portfolio

    def calculate_portfolio_metrics(self):
        if not self.alocacao_portfolio: return
        
        weights = {a: self.alocacao_portfolio[a]['weight'] for a in self.ativos_selecionados}
        df_sel = pd.DataFrame({s: self.coletor.dados_por_ativo[s]['Returns'] for s in self.ativos_selecionados}).dropna()
        
        if df_sel.empty: return
        
        w_array = np.array([weights[c] for c in df_sel.columns])
        w_array = w_array / w_array.sum()
        
        port_ret = (df_sel * w_array).sum(axis=1)
        metrics = calculate_risk_metrics(port_ret)
        metrics['total_investment'] = self.valor_investimento
        self.metricas_portfolio = metrics

    def generate_justifications(self):
        for s in self.ativos_selecionados:
            just = []
            # Perf
            if s in self.coletor.performance_data.index:
                p = self.coletor.performance_data.loc[s]
                just.append(f"Sharpe {p['sharpe_ratio']:.2f}")
            # Fund
            if s in self.coletor.fundamental_data.index:
                f = self.coletor.fundamental_data.loc[s]
                just.append(f"P/L {f['PE_Ratio']:.1f}, ROE {f['ROE']:.1f}%")
            # ML
            if s in self.ml_predictions:
                ml = self.ml_predictions[s]
                just.append(f"ML Prob {ml['predicted_proba_up']*100:.0f}% (AUC {ml['auc_roc_score']:.2f})")
                
            self.justificativas_selecao[s] = " | ".join(just)

    def executar_pipeline(self, simbolos_customizados: list, perfil_inputs: dict, progress_bar=None) -> bool:
        """Pipeline principal."""
        
        # 1. Coleta (YFinance + Pynvest)
        if progress_bar: progress_bar.progress(10, text="Coletando Preços e Fundamentos...")
        if not self.coletor.coletar_dados(simbolos_customizados, progress_bar): return False
        
        # 2. Cross-sectional features
        self.coletor.calculate_cross_sectional_features()
        
        # 3. ML (com horizonte ajustado)
        if progress_bar: progress_bar.progress(50, text="Treinando Modelos ML...")
        self.apply_clustering_and_ml(perfil_inputs['ml_lookback_days'], progress_bar)
        
        # 4. Score & Rank
        if progress_bar: progress_bar.progress(70, text="Ranqueando Ativos...")
        self.score_and_rank_assets(perfil_inputs['time_horizon'])
        
        # 5. Seleção
        self.select_final_portfolio()
        
        # 6. Otimização
        if progress_bar: progress_bar.progress(90, text="Otimizando Alocação...")
        self.optimize_allocation(perfil_inputs['risk_level'])
        
        # 7. Métricas Finais
        self.calculate_portfolio_metrics()
        self.generate_justifications()
        
        if progress_bar: progress_bar.progress(100, text="Concluído!")
        time.sleep(1)
        return True

# =============================================================================
# 10. INTERFACE STREAMLIT (VISUAL ORIGINAL MANTIDO)
# =============================================================================

def configurar_pagina():
    st.set_page_config(page_title="Sistema de Portfólio Integrado", page_icon="📈", layout="wide")
    # CSS Mantido do original (automl)
    st.markdown("""
        <style>
        :root { --primary-color: #000000; --secondary-color: #6c757d; --background-light: #ffffff; --background-dark: #f8f9fa; --text-color: #212529; --text-color-light: #ffffff; --border-color: #dee2e6; }
        body { background-color: var(--background-light); color: var(--text-color); font-family: 'Arial', sans-serif; }
        .main-header { color: var(--primary-color); text-align: center; border-bottom: 2px solid var(--border-color); padding-bottom: 10px; font-size: 2.2rem !important; margin-bottom: 20px; font-weight: 600; }
        .stButton button, .stDownloadButton button { border: 1px solid var(--primary-color) !important; color: var(--primary-color) !important; border-radius: 6px; }
        .stButton button:hover, .stDownloadButton button:hover { background-color: var(--primary-color) !important; color: var(--text-color-light) !important; }
        .stButton button[kind="primary"], .stFormSubmitButton button { background-color: var(--primary-color) !important; color: var(--text-color-light) !important; border: none !important; }
        .stTabs [data-baseweb="tab-list"] { justify-content: center; width: 100%; }
        .stTabs [data-baseweb="tab"] { flex-grow: 0 !important; }
        .stTabs [aria-selected="true"] { border-bottom: 2px solid var(--primary-color); color: var(--primary-color); }
        .info-box { background-color: var(--background-dark); border-left: 4px solid var(--primary-color); padding: 15px; margin: 10px 0; border-radius: 6px; }
        .stMetric { background-color: var(--background-dark); border-radius: 6px; padding: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        </style>
    """, unsafe_allow_html=True)

def aba_introducao():
    st.markdown("## 📚 Metodologia Integrada")
    st.info("Este sistema combina a robustez da Teoria de Portfólio (Markowitz) com Machine Learning preditivo, utilizando dados em tempo real do YFinance e Fundamentus.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 1. Coleta & Engenharia")
        st.markdown("- **Preços:** YFinance (Histórico Completo).")
        st.markdown("- **Fundamentos:** Pynvest (P/L, ROE, DY, etc).")
        st.markdown("- **Indicadores:** RSI, MACD, Volatilidade, Momentum.")
    with col2:
        st.markdown("### 2. Machine Learning & Otimização")
        st.markdown("- **ML:** Random Forest para prever tendências de 4, 8 ou 12 meses.")
        st.markdown("- **Ranking:** Score multi-fatorial (Perf + Fund + Tec + ML).")
        st.markdown("- **Alocação:** Max Sharpe ou Min Volatility.")

def aba_selecao_ativos():
    st.markdown("## 🎯 Universo de Ativos")
    st.info("Selecione os ativos que farão parte da análise.")
    
    modo = st.radio("Modo:", ["IBOVESPA Completo", "Seleção Manual"], index=0)
    
    if modo == "IBOVESPA Completo":
        sel = TODOS_ATIVOS
        st.success(f"{len(sel)} ativos carregados.")
    else:
        sel = st.multiselect("Selecione:", TODOS_ATIVOS, default=TODOS_ATIVOS[:10])
        
    if sel:
        st.session_state.ativos_para_analise = sel
        st.metric("Ativos para Análise", len(sel))
    else:
        st.warning("Selecione pelo menos um ativo.")

def aba_construtor_portfolio():
    if not st.session_state.get('ativos_para_analise'):
        st.warning("Defina os ativos na aba anterior.")
        return

    # Placeholder barra progresso
    prog_bar = st.empty()

    if not st.session_state.builder_complete:
        st.markdown("## 📋 Perfil do Investidor")
        with st.form("form_perfil"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Risco")
                r_risk = st.radio("1. Volatilidade:", OPTIONS_CONCORDA, index=2)
                r_gain = st.radio("2. Retorno Máximo:", OPTIONS_CONCORDA, index=2)
                r_stab = st.radio("3. Estabilidade:", OPTIONS_DISCORDA, index=2)
            with col2:
                st.markdown("#### Horizonte")
                r_loss = st.radio("4. Aversão Perda:", OPTIONS_DISCORDA, index=2)
                r_time = st.radio("7. Horizonte:", OPTIONS_TIME_HORIZON_DETALHADA, index=1)
                r_liq = st.radio("8. Liquidez:", OPTIONS_LIQUIDEZ_DETALHADA, index=1)
                inv = st.number_input("Investimento (R$):", 1000, 1000000, 10000)
                
            # Campos ocultos/fixos para compatibilidade do analisador
            r_reac = OPTIONS_REACTION_DETALHADA[1]
            r_lev = OPTIONS_CONHECIMENTO_DETALHADA[1]

            if st.form_submit_button("🚀 Otimizar Portfólio"):
                answers = {
                    'risk_accept': MAP_CONCORDA[r_risk], 'max_gain': MAP_CONCORDA[r_gain],
                    'stable_growth': MAP_DISCORDA[r_stab], 'avoid_loss': MAP_DISCORDA[r_loss],
                    'reaction': MAP_REACTION[r_reac], 'level': MAP_CONHECIMENTO[r_lev],
                    'time_purpose': r_time[0], 'liquidity': r_liq[0]
                }
                
                analyser = AnalisadorPerfilInvestidor()
                risk, horiz, lookback, score = analyser.calcular_perfil(answers)
                
                st.session_state.profile = {'risk_level': risk, 'time_horizon': horiz, 'ml_lookback_days': lookback, 'score': score}
                
                builder = ConstrutorPortfolioAutoML(inv)
                st.session_state.builder = builder
                
                p_wid = prog_bar.progress(0, text="Iniciando...")
                
                success = builder.executar_pipeline(st.session_state.ativos_para_analise, st.session_state.profile, p_wid)
                prog_bar.empty()
                
                if success:
                    st.session_state.builder_complete = True
                    st.rerun()
                else:
                    st.error("Falha no pipeline.")
    else:
        b = st.session_state.builder
        p = st.session_state.profile
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Perfil", p['risk_level'])
        col2.metric("Horizonte ML", f"{p['ml_lookback_days']} dias (~{int(p['ml_lookback_days']/21)} meses)")
        col3.metric("Sharpe Portfólio", f"{b.metricas_portfolio.get('sharpe_ratio',0):.2f}")
        
        if st.button("🔄 Recomeçar"):
            st.session_state.builder_complete = False
            st.rerun()
            
        tab_res1, tab_res2, tab_res3 = st.tabs(["Alocação", "Performance", "Detalhes"])
        
        with tab_res1:
            df_alloc = pd.DataFrame([
                {'Ativo': k, 'Peso (%)': v['weight']*100, 'Valor (R$)': v['amount']}
                for k, v in b.alocacao_portfolio.items()
            ])
            fig = px.pie(df_alloc, values='Peso (%)', names='Ativo', title="Alocação Otimizada")
            fig.update_layout(obter_template_grafico())
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_alloc)
            
        with tab_res2:
            # Gráfico acumulado dos selecionados
            fig_cum = go.Figure()
            for s in b.ativos_selecionados:
                df = b.coletor.dados_por_ativo[s]
                cum = (1 + df['Returns']).cumprod()
                fig_cum.add_trace(go.Scatter(x=cum.index, y=cum, name=s))
            fig_cum.update_layout(obter_template_grafico(), title="Performance Histórica Acumulada")
            st.plotly_chart(fig_cum, use_container_width=True)
            
        with tab_res3:
            st.markdown("### Justificativas de Seleção")
            for s, txt in b.justificativas_selecao.items():
                st.markdown(f"**{s}**: {txt}")
                
            st.markdown("### Ranking Completo")
            st.dataframe(b.combined_scores[['total_score', 'performance_score', 'fundamental_score', 'ml_score_weighted']].head(20))

def aba_analise_individual():
    st.markdown("## 🔍 Análise Individual")
    if not st.session_state.get('builder'):
        st.warning("Execute o construtor primeiro para carregar dados.")
        return
        
    b = st.session_state.builder
    assets = sorted(b.coletor.successful_assets)
    sel = st.selectbox("Ativo:", assets)
    
    if sel:
        df = b.coletor.dados_por_ativo[sel]
        fund = b.coletor.fundamental_data.loc[sel]
        ml = b.ml_predictions.get(sel, {})
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Preço Atual", f"R$ {df['Close'].iloc[-1]:.2f}")
        col2.metric("P/L", f"{fund['PE_Ratio']:.1f}")
        col3.metric("ROE", f"{fund['ROE']:.1f}%")
        col4.metric("ML Prob", f"{ml.get('predicted_proba_up',0)*100:.0f}%")
        
        # Gráfico Técnico
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Preço'), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD Hist'), row=2, col=1)
        fig.update_layout(obter_template_grafico(), height=600, title=f"Análise Técnica: {sel}")
        st.plotly_chart(fig, use_container_width=True)

def aba_referencias():
    st.markdown("## 📖 Referências")
    st.markdown("1. Markowitz, H. (1952). Portfolio Selection.")
    st.markdown("2. Scikit-Learn Documentation (Random Forest).")
    st.markdown("3. Fundamentus (Dados Fundamentalistas).")

def main():
    if 'builder' not in st.session_state:
        st.session_state.builder = None
        st.session_state.builder_complete = False
        st.session_state.profile = {}
        st.session_state.ativos_para_analise = []

    configurar_pagina()
    st.markdown('<h1 class="main-header">Sistema de Portfólio Integrado (v9.0)</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Metodologia", "🎯 Seleção", "🏗️ Construtor", "🔍 Análise", "📖 Referências"])
    
    with tab1: aba_introducao()
    with tab2: aba_selecao_ativos()
    with tab3: aba_construtor_portfolio()
    with tab4: aba_analise_individual()
    with tab5: aba_referencias()

if __name__ == "__main__":
    main()
