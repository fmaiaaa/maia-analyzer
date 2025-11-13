# -*- coding: utf-8 -*-
"""
=============================================================================
SISTEMA DE PORTFÓLIOS ADAPTATIVOS - OTIMIZAÇÃO QUANTITATIVA
=============================================================================

Adaptação do Sistema AutoML para usar dados pré-processados (CSV/GCS)
gerados pelo gerador_financeiro.py, eliminando a dependência do yfinance
na interface Streamlit e adotando uma linguagem profissional.

Versão: 8.0.0 - Foco em Profissionalismo e Dados Offline
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

# --- 2. SCIENTIFIC / STATISTICAL TOOLS ---
from scipy.optimize import minimize
from scipy.stats import zscore, norm

# --- 3. STREAMLIT, DATA ACQUISITION, & PLOTTING ---
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- 4. FEATURE ENGINEERING / TECHNICAL ANALYSIS (TA) ---
# Apenas a importação das classes, o cálculo em si é assumido nos dados do GCS
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, CCIIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel, DonchianChannel
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, MFIIndicator, VolumeWeightedAveragePrice

# --- 5. MACHINE LEARNING (SCIKIT-LEARN) ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# --- 6. BOOSTED MODELS & OPTIMIZATION ---
# Mantidas apenas para o contexto, mas a lógica de treinamento complexo foi removida.
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
import shap

# --- 7. SPECIALIZED TIME SERIES & ECONOMETRICS ---
from arch import arch_model

# --- 8. CONFIGURATION (NOVO NOME PROFISSIONAL) ---
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURAÇÕES E CONSTANTES GLOBAIS
# =============================================================================

PERIODO_DADOS = '5y'
MIN_DIAS_HISTORICO = 252
NUM_ATIVOS_PORTFOLIO = 5
TAXA_LIVRE_RISCO = 0.1075
LOOKBACK_ML = 30

# =============================================================================
# 2. PONDERAÇÕES E REGRAS DE OTIMIZAÇÃO
# =============================================================================

WEIGHT_PERFORMANCE = 0.40
WEIGHT_FUNDAMENTAL = 0.30
WEIGHT_TECHNICAL = 0.30
WEIGHT_ML = 0.30

PESO_MIN = 0.10
PESO_MAX = 0.30

# =============================================================================
# 3. CAMINHOS DE DADOS E GCS (Configuração do gerador_financeiro.py)
# =============================================================================

GCS_BUCKET_NAME = 'meu-portfolio-dados-gratuitos'
GCS_FOLDER_PATH = 'dados_financeiros_etl/'
GCS_BASE_URL = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{GCS_FOLDER_PATH}"

# =============================================================================
# 4. LISTAS DE ATIVOS E SETORES (Inalteradas, pois são a base do sistema)
# =============================================================================

ATIVOS_IBOVESPA = [
    'ALOS3.SA', 'ABEV3.SA', 'ASAI3.SA', 'AURE3.SA', 'AZZA3.SA', 'B3SA3.SA',
    'BBSE3.SA', 'BBDC3.SA', 'BBDC4.SA', 'BRAP4.SA', 'BBAS3.SA', 'BRKM5.SA',
    'BRAV3.SA', 'BPAC11.SA', 'CXSE3.SA', 'CEAB3.SA', 'CMIG4.SA', 'COGN3.SA',
    'CPLE6.SA', 'CSAN3.SA', 'CPFE3.SA', 'CMIN3.SA', 'CURY3.SA', 'CVCB3.SA',
    'CYRE3.SA', 'DIRR3.SA', 'ELET3.SA', 'ELET6.SA', 'EMBR3.SA', 'ENGI11.SA',
    'ENEV3.SA', 'EGIE3.SA', 'EQTL3.SA', 'FLRY3.SA', 'GGBR4.SA', 'GOAU4.SA',
    'HAPV3.SA', 'HYPE3.SA', 'IGTI11.SA', 'IRBR3.SA', 'ISAE4.SA', 'ITSA4.SA',
    'ITUB4.SA', 'KLBN11.SA', 'RENT3.SA', 'LREN3.SA', 'MGLU3.SA', 'POMO4.SA',
    'MBRF3.SA', 'BEEF3.SA', 'MOTV3.SA', 'MRVE3.SA', 'MULT3.SA', 'NATU3.SA',
    'PCAR3.SA', 'PETR3.SA', 'PETR4.SA', 'RECV3.SA', 'PRIO3.SA', 'PSSA3.SA',
    'RADL3.SA', 'RAIZ4.SA', 'RDOR3.SA', 'RAIL3.SA', 'SBSP3.SA', 'SANB11.SA',
    'CSNA3.SA', 'SLCE3.SA', 'SMFT3.SA', 'SUZB3.SA', 'TAEE11.SA', 'VIVT3.SA',
    'TIMS3.SA', 'TOTS3.SA', 'UGPA3.SA', 'USIM5.SA', 'VALE3.SA', 'VAMO3.SA',
    'VBBR3.SA', 'VIVA3.SA', 'WEGE3.SA', 'YDUQ3.SA'
]

ATIVOS_POR_SETOR = {
    'Bens Industriais': ['NATU3.SA', 'AMOB3.SA', 'ISAE4.SA', 'BHIA3.SA', 'ZAMP3.SA', 'AERI3.SA',
                         'ICBR3.SA', 'DOTZ3.SA', 'GOLL3.SA', 'VIIA3.SA', 'ARML3.SA', 'MLAS3.SA',
                         'CBAV3.SA', 'TTEN3.SA', 'BRBI11.SA', 'REAG3.SA', 'ATEA3.SA', 'MODL4.SA',
                         'VITT3.SA', 'KRSA3.SA', 'CXSE3.SA', 'RIOS3.SA', 'HCAR3.SA', 'GGPS3.SA',
                         'MATD3.SA', 'ALLD3.SA', 'BLAU3.SA', 'ATMP3.SA', 'ASAI3.SA', 'JSLG3.SA',
                         'CMIN3.SA', 'ELMD3.SA', 'ORVR3.SA', 'OPCT3.SA', 'WEST3.SA', 'CSED3.SA',
                         'BMOB3.SA', 'JALL3.SA', 'TOKY3.SA', 'ESPA3.SA', 'VAMO3.SA', 'INTB3.SA',
                         'NGRD3.SA', 'AVLL3.SA', 'RRRP3.SA', 'ENJU3.SA', 'CASH3.SA', 'TFCO4.SA',
                         'CONX3.SA', 'GMAT3.SA', 'SEQL3.SA', 'PASS3.SA', 'BOAS3.SA', 'MELK3.SA',
                         'HBSA3.SA', 'SIMH3.SA', 'CURY3.SA', 'PLPL3.SA', 'PETZ3.SA', 'PGMN3.SA',
                         'LAVV3.SA', 'LJQQ3.SA', 'DMVF3.SA', 'SOMA3.SA', 'RIVA3.SA', 'AMBP3.SA', 'ALPK3.SA'],
    'Consumo Cíclico': ['AZZA3.SA', 'ALOS3.SA', 'VIIA3.SA', 'RDNI3.SA', 'SLED4.SA', 'RSID3.SA',
                        'MNDL3.SA', 'LEVE3.SA', 'CTKA4.SA', 'MYPK3.SA', 'GRND3.SA', 'LCAM3.SA',
                        'CEAB3.SA', 'VSTE3.SA', 'CGRA3.SA', 'ESTR4.SA', 'DIRR3.SA', 'CTNM3.SA',
                        'ANIM3.SA', 'EVEN3.SA', 'AMAR3.SA', 'MOVI3.SA', 'JHSF3.SA', 'HBOR3.SA',
                        'PDGR3.SA', 'ARZZ3.SA', 'EZTC3.SA', 'ALPA3.SA', 'RENT3.SA', 'MRVE3.SA',
                        'MGLU3.SA', 'LREN3.SA', 'COGN3.SA', 'WHRL4.SA', 'TCSA3.SA', 'SMLS3.SA',
                        'SEER3.SA', 'HOOT4.SA', 'GFSA3.SA', 'YDUQ3.SA', 'CYRE3.SA', 'CVCB3.SA', 'SBFG3.SA'],
    'Consumo não Cíclico': ['PRVA3.SA', 'SMTO3.SA', 'MDIA3.SA', 'CAML3.SA', 'AGRO3.SA', 'BEEF3.SA',
                             'VIVA3.SA', 'CRFB3.SA', 'PCAR3.SA', 'NTCO3.SA', 'NATU3.SA', 'MRFG3.SA',
                             'JBSS3.SA', 'BRFS3.SA'],
    'Financeiro': ['CSUD3.SA', 'INBR31.SA', 'BIDI3.SA', 'BIDI4.SA', 'IGTI11.SA', 'IGTI3.SA',
                   'XPBR31.SA', 'TRAD3.SA', 'BSLI4.SA', 'BTTL3.SA', 'BPAR3.SA', 'SCAR3.SA',
                   'LPSB3.SA', 'BMGB4.SA', 'IGBR3.SA', 'GSHP3.SA', 'PSSA3.SA', 'CARD3.SA',
                   'BBRK3.SA', 'BRPR3.SA', 'BRSR6.SA', 'SANB4.SA', 'SANB3.SA', 'MULT3.SA',
                   'ITUB3.SA', 'ITUB4.SA', 'ALSO3.SA', 'BMIN3.SA', 'MERC4.SA', 'LOGG3.SA',
                   'ITSA4.SA', 'IRBR3.SA', 'PDTC3.SA', 'SYNE3.SA', 'BBDC4.SA', 'BBDC3.SA',
                   'BRML3.SA', 'APER3.SA', 'BBSE3.SA', 'BPAN4.SA', 'BBAS3.SA'],
    'Materiais Básicos': ['LAND3.SA', 'DEXP4.SA', 'RANI3.SA', 'PMAM3.SA', 'FESA4.SA', 'EUCA3.SA',
                          'SUZB3.SA', 'KLBN4.SA', 'KLBN3.SA', 'VALE3.SA', 'VALE5.SA', 'UNIP6.SA',
                          'UNIP5.SA', 'GOAU4.SA', 'DXCO3.SA', 'CSNA3.SA', 'BRKM6.SA', 'BRKM5.SA',
                          'BRAP4.SA', 'BRAP3.SA'],
    'Petróleo, Gás e Biocombustíveis': ['SRNA3.SA', 'VBBR3.SA', 'RAIZ4.SA', 'RECV3.SA', 'PRIO3.SA',
                                        'OSXB3.SA', 'DMMO3.SA', 'RPMG3.SA', 'UGPA3.SA', 'PETR4.SA',
                                        'PETR3.SA', 'ENAT3.SA'],
    'Saúde': ['ONCO3.SA', 'VVEO3.SA', 'PARD3.SA', 'BIOM3.SA', 'BALM3.SA', 'PNVL3.SA', 'AALR3.SA',
              'ODPV3.SA', 'RADL3.SA', 'QUAL3.SA', 'OFSA3.SA', 'HYPE3.SA', 'FLRY3.SA'],
    'Tecnologia da Informação': ['CLSA3.SA', 'LVTC3.SA', 'G2DI33.SA', 'IFCM3.SA', 'GOGL35.SA',
                                 'LWSA3.SA', 'TOTS3.SA', 'LINX3.SA', 'POSI3.SA'],
    'Telecomunicações': ['BRIT3.SA', 'FIQE3.SA', 'DESK3.SA', 'TIMS3.SA', 'VIVT3.SA', 'TELB4.SA', 'TELB3.SA'],
    'Utilidade Pública': ['BRAV3.SA', 'AURE3.SA', 'MEGA3.SA', 'CEPE6.SA', 'CEED3.SA', 'EEEL4.SA',
                          'CASN4.SA', 'CEGR3.SA', 'CEBR3.SA', 'RNEW4.SA', 'COCE6.SA', 'CLSC4.SA',
                          'ALUP4.SA', 'ALUP3.SA', 'SAPR4.SA', 'SAPR3.SA', 'CPRE3.SA', 'CPLE5.SA',
                          'CPLE6.SA', 'CPLE3.SA', 'CPFE3.SA', 'CGAS3.SA', 'AESB3.SA', 'NEOE3.SA',
                          'TRPL4.SA', 'TRPL3.SA', 'EGIE3.SA', 'TAEE4.SA', 'TAEE3.SA', 'SBSP3.SA',
                          'GEPA4.SA', 'CESP6.SA', 'CMIG4.SA', 'CMIG3.SA', 'AFLT3.SA']
}

TODOS_ATIVOS = sorted(list(set([ativo for ativos in ATIVOS_POR_SETOR.values() for ativo in ativos])))


# =============================================================================
# 5. MAPEAMENTOS DE PONTUAÇÃO DO QUESTIONÁRIO (Inalterados)
# =============================================================================

SCORE_MAP = {
    'CT: Concordo Totalmente': 5, 'C: Concordo': 4, 'N: Neutro': 3, 'D: Discordo': 2, 'DT: Discordo Totalmente': 1
}
SCORE_MAP_INV = {
    'CT: Concordo Totalmente': 1, 'C: Concordo': 2, 'N: Neutro': 3, 'D: Discordo': 4, 'DT: Discordo Totalmente': 5
}
SCORE_MAP_CONHECIMENTO = {
    'A: Avançado': 5, 'B: Intermediário': 3, 'C: Iniciante': 1
}
SCORE_MAP_REACTION = {
    'A: Venderia': 1, 'B: Manteria': 3, 'C: Compraria mais': 5
}

# =============================================================================
# 6. CLASSE: ANALISADOR DE PERFIL DO INVESTIDOR (Inalterada)
# =============================================================================

class AnalisadorPerfilInvestidor:
    """Analisa perfil de risco e horizonte temporal do investidor."""
    
    def __init__(self):
        self.nivel_risco = ""
        self.horizonte_tempo = ""
        self.dias_lookback_ml = 5
    
    def determinar_nivel_risco(self, pontuacao: int) -> str:
        if pontuacao <= 18: return "CONSERVADOR"
        elif pontuacao <= 30: return "INTERMEDIÁRIO"
        elif pontuacao <= 45: return "MODERADO"
        elif pontuacao <= 60: return "MODERADO-ARROJADO"
        else: return "AVANÇADO"
    
    def determinar_horizonte_ml(self, liquidez_key: str, objetivo_key: str) -> tuple[str, int]:
        time_map = { 'A': 5, 'B': 20, 'C': 30 }
        final_lookback = max( time_map.get(liquidez_key, 5), time_map.get(objetivo_key, 5) )
        
        if final_lookback >= 30:
            self.horizonte_tempo = "LONGO PRAZO"; self.dias_lookback_ml = 30
        elif final_lookback >= 20:
            self.horizonte_tempo = "MÉDIO PRAZO"; self.dias_lookback_ml = 20
        else:
            self.horizonte_tempo = "CURTO PRAZO"; self.dias_lookback_ml = 5
        
        return self.horizonte_tempo, self.dias_lookback_ml
    
    def calcular_perfil(self, respostas_risco: dict) -> tuple[str, str, int, int]:
        pontuacao = (
            SCORE_MAP[respostas_risco['risk_accept']] * 5 +
            SCORE_MAP[respostas_risco['max_gain']] * 5 +
            SCORE_MAP_INV[respostas_risco['stable_growth']] * 5 +
            SCORE_MAP_INV[respostas_risco['avoid_loss']] * 5 +
            SCORE_MAP_CONHECIMENTO[respostas_risco['level']] * 3 +
            SCORE_MAP_REACTION[respostas_risco['reaction']] * 3
        )
        nivel_risco = self.determinar_nivel_risco(pontuacao)
        horizonte_tempo, ml_lookback = self.determinar_horizonte_ml(
            respostas_risco['liquidity'], respostas_risco['time_purpose']
        )
        return nivel_risco, horizonte_tempo, ml_lookback, pontuacao

# =============================================================================
# 7. FUNÇÕES DE ESTILO E VISUALIZAÇÃO (Inalteradas)
# =============================================================================

def obter_template_grafico() -> dict:
    """Retorna um template de layout otimizado para gráficos Plotly com estilo Times New Roman."""
    return {
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'font': {
            'family': 'Times New Roman, serif',
            'size': 12,
            'color': 'black'
        },
        'title': {
            'font': {
                'family': 'Times New Roman, serif',
                'size': 16,
                'color': '#2c3e50',
                'weight': 'bold'
            },
            'x': 0.5,
            'xanchor': 'center'
        },
        'xaxis': {
            'showgrid': True, 'gridcolor': 'lightgray', 'showline': True, 'linecolor': 'black', 'linewidth': 1,
            'tickfont': {'family': 'Times New Roman, serif', 'color': 'black'}, 'title': {'font': {'family': 'Times New Roman, serif', 'color': 'black'}}, 'zeroline': False
        },
        'yaxis': {
            'showgrid': True, 'gridcolor': 'lightgray', 'showline': True, 'linecolor': 'black', 'linewidth': 1,
            'tickfont': {'family': 'Times New Roman, serif', 'color': 'black'}, 'title': {'font': {'family': 'Times New Roman, serif', 'color': 'black'}}, 'zeroline': False
        },
        'legend': {
            'font': {'family': 'Times New Roman, serif', 'color': 'black'},
            'bgcolor': 'rgba(255, 255, 255, 0.8)',
            'bordercolor': 'lightgray',
            'borderwidth': 1
        },
        'colorway': ['#2c3e50', '#7f8c8d', '#3498db', '#e74c3c', '#27ae60']
    }

# =============================================================================
# 8. CLASSE: ENGENHEIRO DE FEATURES (Mantido, mas usado apenas para normalização)
# =============================================================================

class EngenheiroFeatures:
    """Funções utilitárias de features e normalização."""

    @staticmethod
    def _normalizar(serie: pd.Series, maior_melhor: bool = True) -> pd.Series:
        """Normaliza uma série de indicadores para o range [0, 1] (Min-Max Scaling)."""
        if serie.isnull().all():
            return pd.Series(0, index=serie.index)
        
        min_val = serie.min()
        max_val = serie.max()
        
        if max_val == min_val:
            return pd.Series(0.5, index=serie.index)
        
        if maior_melhor:
            return (serie - min_val) / (max_val - min_val)
        else:
            return (max_val - serie) / (max_val - min_val)

# =============================================================================
# 9. FUNÇÕES DE COLETA DE DADOS GCS (Corrigidas e Otimizadas)
# =============================================================================

def carregar_dados_ativo_gcs_csv(base_url: str, ticker: str, file_suffix: str) -> pd.DataFrame:
    """Carrega o DataFrame de um único ativo via URL pública do GCS (formato CSV)."""
    file_name = f"{ticker}{file_suffix}" 
    full_url = f"{base_url}{file_name}"
    
    try:
        df_ativo = pd.read_csv(full_url)
        
        if 'Date' in df_ativo.columns:
            df_ativo = df_ativo.set_index('Date')
        elif 'index' in df_ativo.columns:
            df_ativo = df_ativo.set_index('index')
            df_ativo.index.name = 'Date'
            
        if df_ativo.index.tz is not None:
             df_ativo.index = pd.to_datetime(df_ativo.index, utc=True).tz_convert(None) 
        else:
             df_ativo.index = pd.to_datetime(df_ativo.index)

        for col in df_ativo.columns:
            if col not in ['ticker', 'sector', 'industry', 'recommendation', 'Date', 'index']:
                df_ativo[col] = pd.to_numeric(df_ativo[col], errors='coerce')
        
        return df_ativo

    except Exception as e:
        # print(f"❌ Erro ao carregar {ticker} com sufixo {file_suffix} da URL: {full_url}. Erro: {e}")
        return pd.DataFrame()

class ColetorDadosGCS(object):
    """
    Coleta dados de mercado de arquivos CSV individuais no GCS,
    usando os 4 arquivos gerados pelo gerador_financeiro.py e consolidando-os.
    """
    
    def __init__(self, periodo=PERIODO_DADOS):
        self.periodo = periodo
        self.dados_por_ativo = {} 
        self.dados_fundamentalistas = pd.DataFrame() 
        self.metricas_performance = pd.DataFrame() 
        self.volatilidades_garch_raw = {}
        self.metricas_simples = {}

    def _get_fundamental_metrics_from_df(self, fund_row: pd.Series) -> dict:
        """Método auxiliar para extrair fundamentos e métricas de performance de uma linha de df_fundamentos."""
        fund_data = fund_row.filter(regex='^fund_').to_dict()
        fund_data = {k.replace('fund_', ''): v for k, v in fund_data.items()}
        
        fund_data['sharpe_ratio'] = fund_row.get('sharpe_ratio', np.nan)
        fund_data['annual_return'] = fund_row.get('annual_return', np.nan)
        fund_data['annual_volatility'] = fund_row.get('annual_volatility', np.nan)
        fund_data['max_drawdown'] = fund_row.get('max_drawdown', np.nan)
        fund_data['garch_volatility'] = fund_row.get('garch_volatility', np.nan)
        fund_data['sector'] = fund_row.get('sector', 'Unknown')
        fund_data['industry'] = fund_row.get('industry', 'Unknown')
        
        return fund_data

    def coletar_e_processar_dados(self, simbolos: list) -> bool:
        """Carrega os DataFrames para todos os ativos no pipeline."""
        
        self.ativos_sucesso = []
        lista_fundamentalistas = []
        garch_vols = {}
        metricas_simples_list = []

        MIN_DIAS_HISTORICO_FLEXIVEL = max(180, int(MIN_DIAS_HISTORICO * 0.7))

        for simbolo in tqdm(simbolos, desc="📥 Carregando ativos do GCS"):
            
            df_tecnicos = carregar_dados_ativo_gcs_csv(GCS_BASE_URL, simbolo, file_suffix='_tecnicos.csv')
            df_fundamentos = carregar_dados_ativo_gcs_csv(GCS_BASE_URL, simbolo, file_suffix='_fundamentos.csv')
            df_ml_results = carregar_dados_ativo_gcs_csv(GCS_BASE_URL, simbolo, file_suffix='_ml_results.csv')
            
            if df_tecnicos.empty or 'Close' not in df_tecnicos.columns or len(df_tecnicos) < MIN_DIAS_HISTORICO_FLEXIVEL or df_fundamentos.empty: continue

            # --- 3. Extração e Preparação ---
            
            self.dados_por_ativo[simbolo] = df_tecnicos.dropna(how='all')
            self.ativos_sucesso.append(simbolo)
            
            fund_row = df_fundamentos.iloc[0] 
            fund_data = self._get_fundamental_metrics_from_df(fund_row)
            
            # 4. Adiciona dados ML à última linha do DataFrame Temporal
            if not df_ml_results.empty:
                ml_row = df_ml_results.iloc[0] 
                ml_proba_col = next((c for c in ml_row.index if c.startswith('ml_proba_') and not pd.isna(ml_row[c])), 'ml_proba_252d')
                
                last_index = self.dados_por_ativo[simbolo].index[-1]
                
                self.dados_por_ativo[simbolo].loc[last_index, 'ML_Proba'] = ml_row[ml_proba_col] if ml_proba_col in ml_row else 0.5
                self.dados_por_ativo[simbolo].loc[last_index, 'ML_Confidence'] = 0.7 
            
            # 5. Cria o DataFrame Estático e Volatilidade GARCH
            fund_data['Ticker'] = simbolo
            lista_fundamentalistas.append(fund_data)
            garch_vols[simbolo] = fund_data.get('garch_volatility', np.nan)

            # 6. Preenche as Métricas Simples (usadas para a aba de Performance)
            metricas_simples_list.append({
                'Ticker': simbolo,
                'sharpe': fund_data.get('sharpe_ratio', np.nan),
                'retorno_anual': fund_data.get('annual_return', np.nan),
                'volatilidade_anual': fund_data.get('annual_volatility', np.nan),
                'max_drawdown': fund_data.get('max_drawdown', np.nan),
            })
        
        # --- 6. Finalização e Anexo dos Dados Estáticos ---
        
        if len(self.ativos_sucesso) < NUM_ATIVOS_PORTFOLIO: return False
            
        self.dados_fundamentalistas = pd.DataFrame(lista_fundamentalistas).set_index('Ticker')
        self.dados_fundamentalistas['garch_volatility'] = self.dados_fundamentalistas.index.map(garch_vols)
        self.volatilidades_garch_raw = garch_vols 
        
        self.metricas_performance = pd.DataFrame(metricas_simples_list).set_index('Ticker')
        
        for simbolo in self.ativos_sucesso:
            if simbolo in self.dados_fundamentalistas.index:
                last_index = self.dados_por_ativo[simbolo].index[-1]
                for col, value in self.dados_fundamentalistas.loc[simbolo].items():
                    self.dados_por_ativo[simbolo].loc[last_index, col] = value
                
        return True

    def coletar_ativo_unico_gcs(self, ativo_selecionado: str) -> tuple[pd.DataFrame | None, dict | None]:
        """Coleta e retorna dados de um único ativo sob demanda (Aba 4)."""
        
        df_tecnicos = carregar_dados_ativo_gcs_csv(GCS_BASE_URL, ativo_selecionado, file_suffix='_tecnicos.csv')
        df_fundamentos = carregar_dados_ativo_gcs_csv(GCS_BASE_URL, ativo_selecionado, file_suffix='_fundamentos.csv')
        
        if df_tecnicos.empty or df_fundamentos.empty or 'Close' not in df_tecnicos.columns:
            return None, None
        
        fund_row = df_fundamentos.iloc[0]
        features_fund = self._get_fundamental_metrics_from_df(fund_row)
        
        df_ml_results = carregar_dados_ativo_gcs_csv(GCS_BASE_URL, ativo_selecionado, file_suffix='_ml_results.csv')
        
        if not df_tecnicos.empty:
            last_index = df_tecnicos.index[-1]
            
            for key, value in features_fund.items():
                df_tecnicos.loc[last_index, key] = value
                
            if not df_ml_results.empty:
                ml_row = df_ml_results.iloc[0] 
                ml_proba_col = next((c for c in ml_row.index if c.startswith('ml_proba_') and not pd.isna(ml_row[c])), 'ml_proba_252d')
                
                df_tecnicos.loc[last_index, 'ML_Proba'] = ml_row[ml_proba_col] if ml_proba_col in ml_row else 0.5
                df_tecnicos.loc[last_index, 'ML_Confidence'] = 0.7 
        
        return df_tecnicos.dropna(how='all'), features_fund

# =============================================================================
# 10. CLASSE: OTIMIZADOR DE PORTFÓLIO (Inalterada)
# =============================================================================

class OtimizadorPortfolioAvancado:
    """Otimização de portfólio com volatilidade GARCH e CVaR"""
    
    def __init__(self, returns_df: pd.DataFrame, garch_vols: dict = None, fundamental_data: pd.DataFrame = None, ml_predictions: pd.Series = None):
        self.returns = returns_df
        self.mean_returns = returns_df.mean() * 252
        
        if garch_vols is not None and garch_vols:
            self.cov_matrix = self._construir_matriz_cov_garch(returns_df, garch_vols)
        else:
            self.cov_matrix = returns_df.cov() * 252
            
        self.num_ativos = len(returns_df.columns)
        self.fundamental_data = fundamental_data
        self.ml_predictions = ml_predictions

    def _construir_matriz_cov_garch(self, returns_df: pd.DataFrame, garch_vols: dict) -> pd.DataFrame:
        corr_matrix = returns_df.corr()
        
        vol_array = np.array([
            garch_vols.get(ativo, returns_df[ativo].std() * np.sqrt(252))
            for ativo in returns_df.columns
        ])
        
        if np.isnan(vol_array).all() or np.all(vol_array <= 1e-9):
            return returns_df.cov() * 252
            
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
    
    def calcular_cvar(self, pesos: np.ndarray, confidence: float = 0.95) -> float:
        portfolio_returns = self.returns @ pesos
        sorted_returns = np.sort(portfolio_returns)
        var_index = int(np.floor((1 - confidence) * len(sorted_returns)))
        var = sorted_returns[var_index]
        cvar = sorted_returns[sorted_returns <= var].mean()
        return cvar

    def cvar_negativo(self, pesos: np.ndarray, confidence: float = 0.95) -> float:
        return -self.calcular_cvar(pesos, confidence)

    def otimizar(self, estrategia: str = 'MaxSharpe', confidence_level: float = 0.95) -> dict:
        if self.num_ativos == 0: return {}

        restricoes = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        limites = tuple((PESO_MIN, PESO_MAX) for _ in range(self.num_ativos))
        chute_inicial = np.array([1.0 / self.num_ativos] * self.num_ativos)
        
        min_w, max_w = limites[0]
        chute_inicial = np.clip(chute_inicial, min_w, max_w)
        chute_inicial /= np.sum(chute_inicial)
        
        if estrategia == 'MinVolatility':
            objetivo = self.minimizar_volatilidade
        elif estrategia == 'CVaR':
            objetivo = lambda pesos: self.cvar_negativo(pesos, confidence=confidence_level)
        else:
            objetivo = self.sharpe_negativo
        
        try:
            resultado = minimize(
                objetivo, chute_inicial, method='SLSQP', bounds=limites, constraints=restricoes, options={'maxiter': 500, 'ftol': 1e-6} 
            )
            
            if resultado.success:
                final_weights = resultado.x / np.sum(resultado.x)
                return {ativo: peso for ativo, peso in zip(self.returns.columns, final_weights)}
            else:
                return {ativo: 1.0 / self.num_ativos for ativo in self.returns.columns}
        
        except Exception:
            return {ativo: 1.0 / self.num_ativos for ativo in self.returns.columns}

# =============================================================================
# 11. CLASSE PRINCIPAL: CONSTRUTOR DE PORTFÓLIO AUTOML (ADAPTADA)
# =============================================================================

class ConstrutorPortfolioAutoML:
    """Orquestrador principal para construção de portfólio adaptativo."""
    
    def __init__(self, valor_investimento: float, periodo: str = PERIODO_DADOS):
        self.valor_investimento = valor_investimento
        self.periodo = periodo
        
        self.dados_por_ativo = {}
        self.dados_fundamentalistas = pd.DataFrame()
        self.dados_performance = pd.DataFrame()
        self.volatilidades_garch = {}
        self.predicoes_ml = {}
        self.predicoes_estatisticas = {}
        self.ativos_sucesso = []
        self.metricas_performance = pd.DataFrame()
        
        self.ativos_selecionados = []
        self.alocacao_portfolio = {}
        self.metricas_portfolio = {}
        self.metodo_alocacao_atual = "Não Aplicado"
        self.justificativas_selecao = {}
        self.perfil_dashboard = {} 
        self.pesos_atuais = {}
        self.scores_combinados = pd.DataFrame()
        
    def coletar_e_processar_dados(self, simbolos: list) -> bool:
        """Coleta e processa dados de mercado (VIA GCS/CSV)."""
        coletor = ColetorDadosGCS(periodo=self.periodo)
        
        if not coletor.coletar_e_processar_dados(simbolos):
            return False
        
        self.dados_por_ativo = coletor.dados_por_ativo
        self.dados_fundamentalistas = coletor.dados_fundamentalistas
        self.ativos_sucesso = coletor.ativos_sucesso
        self.metricas_performance = coletor.metricas_performance
        self.volatilidades_garch = coletor.volatilidades_garch_raw 
        
        return True

    def calcular_volatilidades_garch(self):
        """Valida se as volatilidades foram carregadas, com fallback se necessário."""
        valid_vols = len([k for k, v in self.volatilidades_garch.items() if not np.isnan(v)])
        
        if valid_vols == 0:
             for ativo in self.ativos_sucesso:
                 if ativo in self.metricas_performance.index and 'volatilidade_anual' in self.metricas_performance.columns:
                      self.volatilidades_garch[ativo] = self.metricas_performance.loc[ativo, 'volatilidade_anual']
        
    def treinar_modelos_ensemble(self, dias_lookback_ml: int = LOOKBACK_ML, otimizar: bool = False):
        """
        Placeholder para o treinamento ML. Assume que os dados ML (ML_Proba e ML_Confidence)
        foram carregados na última linha do self.dados_por_ativo pelo ColetorDadosGCS.
        """
        
        ativos_com_ml = [
            ativo for ativo, df in self.dados_por_ativo.items() 
            if 'ML_Proba' in df.columns and not pd.isna(df['ML_Proba'].iloc[-1])
        ]
        
        # Popula self.predicoes_ml (necessário para a ABA 3/4 exibirem os resultados)
        for ativo in ativos_com_ml:
            last_row = self.dados_por_ativo[ativo].iloc[-1]
            self.predicoes_ml[ativo] = {
                'predicted_proba_up': last_row.get('ML_Proba', 0.5),
                'auc_roc_score': last_row.get('ML_Confidence', np.nan),
                'model_name': 'Ensemble GCS (Pré-calculado)'
            }

    def pontuar_e_selecionar_ativos(self, horizonte_tempo: str):
        """Pontua e ranqueia ativos usando sistema multi-fator (Perf, Fund, Tech, ML) e diversificação."""
        
        # 1. Pesos Adaptativos
        if horizonte_tempo == "CURTO PRAZO": WEIGHT_PERF, WEIGHT_FUND, WEIGHT_TECH = 0.40, 0.10, 0.20
        elif horizonte_tempo == "LONGO PRAZO": WEIGHT_PERF, WEIGHT_FUND, WEIGHT_TECH = 0.40, 0.50, 0.10
        else: WEIGHT_PERF, WEIGHT_FUND, WEIGHT_TECH = 0.40, 0.30, 0.30

        final_ml_weight = WEIGHT_ML
        total_non_ml_weight = WEIGHT_PERF + WEIGHT_FUND + WEIGHT_TECH
        scale_factor = (1.0 - final_ml_weight) / total_non_ml_weight if total_non_ml_weight > 0 else 0
        WEIGHT_PERF *= scale_factor; WEIGHT_FUND *= scale_factor; WEIGHT_TECH *= scale_factor

        self.pesos_atuais = {'Performance': WEIGHT_PERF, 'Fundamentos': WEIGHT_FUND, 'Técnicos': WEIGHT_TECH, 'ML': final_ml_weight}
        
        # 2. Score - Agrega as últimas métricas
        scores = pd.DataFrame(index=self.ativos_sucesso)
        last_metrics = {}
        for asset in self.ativos_sucesso:
            df = self.dados_por_ativo[asset]
            last_row = df.iloc[-1].to_dict()
            last_metrics[asset] = last_row
            
        combinado = pd.DataFrame(last_metrics).T
        
        required_cols = ['sharpe_ratio', 'pe_ratio', 'roe', 'rsi_14', 'macd', 'ML_Proba', 'ML_Confidence', 'sector']
        for col in required_cols:
             if col not in combinado.columns: combinado[col] = np.nan
        
        # 4. Cálculo dos Scores
        pe_col = 'pe_ratio' if 'pe_ratio' in combinado.columns else 'fund_pe_ratio'
        roe_col = 'roe' if 'roe' in combinado.columns else 'fund_roe'

        # 4.1. Score de Performance (Sharpe)
        scores['performance_score'] = EngenheiroFeatures._normalizar(combinado['sharpe_ratio'], maior_melhor=True) * WEIGHT_PERF
        
        # 4.2. Score Fundamentalista (P/L e ROE)
        pe_score = EngenheiroFeatures._normalizar(combinado[pe_col], maior_melhor=False)
        roe_score = EngenheiroFeatures._normalizar(combinado[roe_col], maior_melhor=True)
        scores['fundamental_score'] = (pe_score * 0.5 + roe_score * 0.5) * WEIGHT_FUND
        
        # 4.3. Score Técnico (RSI e MACD)
        rsi_norm = EngenheiroFeatures._normalizar(combinado['rsi_14'], maior_melhor=False)
        macd_norm = EngenheiroFeatures._normalizar(combinado['macd'], maior_melhor=True)
        scores['technical_score'] = (rsi_norm * 0.5 + macd_norm * 0.5) * WEIGHT_TECH

        # 4.4. Score de Machine Learning (Proba e Confiança)
        ml_proba_norm = EngenheiroFeatures._normalizar(combinado['ML_Proba'], maior_melhor=True)
        ml_confidence_norm = EngenheiroFeatures._normalizar(combinado['ML_Confidence'], maior_melhor=True)
        scores['ml_score_weighted'] = (ml_proba_norm * 0.6 + ml_confidence_norm * 0.4) * final_ml_weight
        
        scores['total_score'] = scores['performance_score'] + scores['fundamental_score'] + scores['technical_score'] + scores['ml_score_weighted']
        self.scores_combinados = scores.join(combinado, rsuffix='_combined').sort_values('total_score', ascending=False)
        
        # 5. Seleção Final (Diversificação Setorial)
        ranked_assets = self.scores_combinados.index.tolist()
        final_portfolio = []; selected_sectors = set(); num_assets_to_select = min(NUM_ATIVOS_PORTFOLIO, len(ranked_assets))
        
        for asset in ranked_assets:
            sector = combinado.loc[asset, 'sector'] if 'sector' in combinado.columns and asset in combinado.index else 'Unknown'
            if sector not in selected_sectors or len(final_portfolio) < num_assets_to_select:
                final_portfolio.append(asset); selected_sectors.add(sector)
            if len(final_portfolio) >= num_assets_to_select: break
        
        self.ativos_selecionados = final_portfolio
        return self.ativos_selecionados
    
    def otimizar_alocacao(self, nivel_risco: str):
        if not self.ativos_selecionados or len(self.ativos_selecionados) < 1:
            self.metodo_alocacao_atual = "ERRO: Ativos Insuficientes"; return {}
        
        available_assets_returns = {s: self.dados_por_ativo[s]['returns']
                                    for s in self.ativos_selecionados if s in self.dados_por_ativo and 'returns' in self.dados_por_ativo[s]}
        final_returns_df = pd.DataFrame(available_assets_returns).dropna()
        
        if final_returns_df.shape[0] < 50:
            weights = {asset: 1.0 / len(self.ativos_selecionados) for asset in self.ativos_selecionados}
            self.metodo_alocacao_atual = 'PESOS IGUAIS (Dados insuficientes)'; return self._formatar_alocacao(weights)

        garch_vols_filtered = {asset: self.volatilidades_garch.get(asset, final_returns_df[asset].std() * np.sqrt(252))
                               for asset in final_returns_df.columns}
                               
        optimizer = OtimizadorPortfolioAvancado(final_returns_df, garch_vols=garch_vols_filtered)
        strategy = 'MaxSharpe' 
        if 'CONSERVADOR' in nivel_risco or 'INTERMEDIÁRIO' in nivel_risco: strategy = 'MinVolatility'
        elif 'AVANÇADO' in nivel_risco: strategy = 'CVaR' 
        weights = optimizer.otimizar(estrategia=strategy)
        self.metodo_alocacao_atual = f'Otimização {strategy} (GARCH/Histórico)'; return self._formatar_alocacao(weights)
        
    def _formatar_alocacao(self, weights: dict) -> dict:
        if not weights or sum(weights.values()) == 0: return {}
        total_weight = sum(weights.values())
        return {s: {'weight': w / total_weight, 'amount': self.valor_investimento * (w / total_weight)}
                for s, w in weights.items() if s in self.ativos_selecionados}
    
    def calcular_metricas_portfolio(self):
        if not self.alocacao_portfolio: return {}
        weights_dict = {s: data['weight'] for s, data in self.alocacao_portfolio.items()}
        returns_df_raw = {s: self.dados_por_ativo[s]['returns'] for s in weights_dict.keys() if s in self.dados_por_ativo and 'returns' in self.dados_por_ativo[s]}
        returns_df = pd.DataFrame(returns_df_raw).dropna()
        if returns_df.empty: return {}
        weights = np.array([weights_dict[s] for s in returns_df.columns])
        weights = weights / np.sum(weights) 
        portfolio_returns = (returns_df * weights).sum(axis=1)
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - TAXA_LIVRE_RISCO) / annual_volatility if annual_volatility > 0 else 0
        cumulative_portfolio_returns = (1 + portfolio_returns).cumprod()
        running_max_portfolio = cumulative_portfolio_returns.expanding().max()
        max_drawdown = ((cumulative_portfolio_returns - running_max_portfolio) / running_max_portfolio).min()
        self.metricas_portfolio = {
            'annual_return': annual_return, 'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio, 'max_drawdown': max_drawdown, 'total_investment': self.valor_investimento
        }
        return self.metricas_portfolio

    def gerar_justificativas(self):
        self.justificativas_selecao = {}
        for simbolo in self.ativos_selecionados:
            justification = []
            
            perf = self.metricas_performance.loc[simbolo] if simbolo in self.metricas_performance.index else pd.Series({})
            fund = self.dados_fundamentalistas.loc[simbolo] if simbolo in self.dados_fundamentalistas.index else pd.Series({})
            
            sharpe = perf.get('sharpe', np.nan)
            ann_ret = perf.get('retorno_anual', np.nan)
            vol_garch = fund.get('garch_volatility', np.nan)
            justification.append(f"Performance: Sharpe {sharpe:.3f}, Retorno {ann_ret*100:.2f}%, Vol. {vol_garch*100:.2f}% (GARCH/Hist.)")
            
            pe_ratio = fund.get('pe_ratio', fund.get('fund_pe_ratio', np.nan))
            roe = fund.get('roe', fund.get('fund_roe', np.nan))
            justification.append(f"Fundamentos: P/L {pe_ratio:.2f}, ROE {roe:.2f}%")
            
            proba_up = self.predicoes_ml.get(simbolo, {}).get('predicted_proba_up', 0.5)
            auc_score = self.predicoes_ml.get(simbolo, {}).get('auc_roc_score', np.nan)
            auc_str = f"{auc_score:.3f}" if not pd.isna(auc_score) else "N/A"
            justification.append(f"ML: Prob. Alta {proba_up*100:.1f}% (Confiança: {auc_str})")
            
            self.justificativas_selecao[simbolo] = " | ".join(justification)
        
        return self.justificativas_selecao
        
    def executar_pipeline(self, simbolos_customizados: list, perfil_inputs: dict, otimizar_ml: bool = False) -> bool:
        
        self.perfil_dashboard = perfil_inputs
        ml_lookback_days = perfil_inputs.get('ml_lookback_days', LOOKBACK_ML)
        nivel_risco = perfil_inputs.get('risk_level', 'MODERADO')
        horizonte_tempo = perfil_inputs.get('time_horizon', 'MÉDIO PRAZO')
        
        if not self.coletar_e_processar_dados(simbolos_customizados): return False
        
        self.calcular_volatilidades_garch()
        self.treinar_modelos_ensemble(dias_lookback_ml=ml_lookback_days, otimizar=otimizar_ml)
        self.pontuar_e_selecionar_ativos(horizonte_tempo=horizonte_tempo)
        self.alocacao_portfolio = self.otimizar_alocacao(nivel_risco=nivel_risco)
        self.calcular_metricas_portfolio()
        self.gerar_justificativas()
        
        return True

# =============================================================================
# 12. CLASSE: ANALISADOR INDIVIDUAL DE ATIVOS (ADAPTADA)
# =============================================================================

class AnalisadorIndividualAtivos:
    """Análise completa de ativos individuais com máximo de features."""
    
    @staticmethod
    def realizar_clusterizacao_pca(dados_ativos: pd.DataFrame, n_clusters: int = 5) -> tuple[pd.DataFrame | None, PCA | None, KMeans | None]:
        """Realiza clusterização K-means após redução de dimensionalidade com PCA."""
        
        features_numericas = dados_ativos.select_dtypes(include=[np.number]).copy()
        features_numericas = features_numericas.replace([np.inf, -np.inf], np.nan)
        
        for col in features_numericas.columns:
            if features_numericas[col].isnull().any():
                median_val = features_numericas[col].median()
                features_numericas[col] = features_numericas[col].fillna(median_val)
        
        features_numericas = features_numericas.dropna(axis=1, how='all')
        features_numericas = features_numericas.loc[:, (features_numericas.std() > 1e-6)]

        if features_numericas.empty or len(features_numericas) < n_clusters:
            return None, None, None
            
        scaler = StandardScaler()
        dados_normalizados = scaler.fit_transform(features_numericas)
        
        n_pca_components = min(3, len(features_numericas.columns))
        pca = PCA(n_components=n_pca_components)
        componentes_pca = pca.fit_transform(dados_normalizados)
        
        actual_n_clusters = min(n_clusters, len(features_numericas))
        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(componentes_pca)
        
        resultado_pca = pd.DataFrame(
            componentes_pca,
            columns=[f'PC{i+1}' for i in range(componentes_pca.shape[1])],
            index=features_numericas.index
        )
        resultado_pca['Cluster'] = clusters
        
        return resultado_pca, pca, kmeans

# =============================================================================
# 13. INTERFACE STREAMLIT - REESTRUTURADA COM TEXTOS PROFISSIONAIS
# =============================================================================

def configurar_pagina():
    """Configura página Streamlit com novo título e estilo."""
    st.set_page_config(
        page_title="Sistema de Portfólios Adaptativos",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
        <style>
        .main-header {
            font-family: 'Times New Roman', serif;
            color: #2c3e50;
            text-align: center;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 10px;
            font-size: 2.2rem !important;
            margin-bottom: 20px;
        }
        html, body, [class*="st-"] {
            font-family: 'Times New Roman', serif;
        }
        .stButton button {
            border: 1px solid #2c3e50;
            color: #2c3e50;
            border-radius: 4px;
            padding: 8px 16px;
        }
        .stButton button:hover {
            background-color: #7f8c8d;
            color: white;
        }
        .stButton button[kind="primary"] {
            background-color: #2c3e50;
            color: white;
            border: none;
        }
        .info-box {
            background-color: #f8f9fa;
            border-left: 4px solid #2c3e50;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f8f9fa;
            border-radius: 4px 4px 0 0;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ffffff;
            border-top: 2px solid #2c3e50;
            border-left: 1px solid #e0e0e0;
            border-right: 1px solid #e0e0e0;
        }
        .stTabs [aria-selected="true"] span {
            font-weight: bold;
        }
        .stMetric {
            padding: 10px 15px;
            background-color: #ffffff;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
            margin-bottom: 10px;
        }
        .stMetric label { font-weight: bold; color: #555; }
        .stMetric delta { font-weight: bold; color: #28a745; }
        .stMetric delta[style*="color: red"] { color: #dc3545 !important; }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            color: #2c3e50;
            font-weight: bold;
        }
        .stMarkdown ul { padding-left: 20px; }
        .stMarkdown li { margin-bottom: 8px; }

        .alert-success { background-color: #d4edda; border-color: #c3e6cb; color: #155724; padding: .75rem 1.25rem; margin-bottom: 1rem; border: 1px solid transparent; border-radius: .25rem; }
        .alert-warning { background-color: #fff3cd; border-color: #ffeeba; color: #856404; padding: .75rem 1.25rem; margin-bottom: 1rem; border: 1px solid transparent; border-radius: .25rem; }
        .alert-error { background-color: #f8d7da; border-color: #f5c6cb; color: #721c24; padding: .75rem 1.25rem; margin-bottom: 1rem; border: 1px solid transparent; border-radius: .25rem; }
        .alert-info { background-color: #d1ecf1; border-color: #bee5eb; color: #0c5460; padding: .75rem 1.25rem; margin-bottom: 1rem; border: 1px solid transparent; border-radius: .25rem; }
        </style>
    """, unsafe_allow_html=True)

def aba_introducao():
    """Aba 1: Introdução e Metodologia (Textos revisados)"""
    
    st.markdown("## 📚 Metodologia de Otimização Quantitativa")
    
    st.markdown("""
    <div class="info-box">
    <h3>🎯 Proposta de Valor</h3>
    <p>O <strong>Sistema de Portfólios Adaptativos</strong> é uma plataforma avançada que utiliza 
    <strong>Machine Learning, Modelagem Estatística e Teoria Moderna de Portfólio</strong> 
    para construir e otimizar alocações de capital. Nossa metodologia adapta a seleção de ativos e 
    os critérios de otimização de acordo com o perfil de risco do investidor e o horizonte temporal.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔬 Pipeline Analítico Estruturado")
        st.markdown("""
        **1. Avaliação do Perfil de Risco**
        - Definição do horizonte temporal e tolerância à volatilidade, essenciais para a calibração dos fatores de pontuação.
        
        **2. Fatorização e Engenharia de Features**
        - Integração de dados de *Valuation* (P/L, ROE, etc.), *Momentum* (RSI, MACD) e *Volatilidade* (GARCH, ATR).
        - Utilização de séries temporais de alta granularidade e fatores de risco macroeconômicos.
        
        **3. Previsão Preditiva (Machine Learning)**
        - Modelos de Ensemble (LightGBM, XGBoost, etc.) pré-treinados e otimizados via Optuna.
        - Geração de probabilidades de alta futura, ponderadas pela confiança (AUC-ROC), incorporando uma visão preditiva na seleção.
        """)
    
    with col2:
        st.markdown("### ⚙️ Algoritmos de Otimização e Alocação")
        st.markdown("""
        **Otimização de Portfólio**
        - Estruturação da Matriz de Covariância usando volatilidade condicional **GARCH** para maior precisão de risco.
        - Objetivos de Otimização: Maximização do Sharpe Ratio, Minimização da Volatilidade ou Minimização do Risco Condicional (*CVaR*).
        
        **Seleção Multi-Fatorial**
        - Ranqueamento dos ativos com base em um *Score Composto* que balanceia Fatores (Performance Histórica, Fundamentos, Análise Técnica e Predição ML).
        
        **Governança e Diversificação**
        - Restrições de diversificação setorial para reduzir o risco não-sistemático.
        - Limites de concentração por ativo (Peso Mínimo/Máximo), garantindo a robustez da carteira.
        """)
    
    st.markdown("---")
    
    st.markdown("### 📊 O Score Composto para a Seleção Final")
    
    st.markdown("""
    <div class="info-box">
    <h4>Fatores de Pontuação Adaptativos</h4>
    <p>A seleção dos <strong>5 ativos</strong> é baseada em um modelo de ranqueamento que adapta a relevância de cada fator ao seu perfil:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        **📈 Fator Performance**
        - Relação Retorno/Risco (Sharpe Ratio)
        - Retorno e Drawdown
        """)
    
    with col2:
        st.markdown("""
        **💼 Fator Fundamentalista**
        - Métrica de Valor (P/L, P/VP)
        - Métrica de Qualidade (ROE, ROIC)
        """)
    
    with col3:
        st.markdown("""
        **🔧 Fator Técnico**
        - Indicadores de Curto Prazo (RSI, MACD)
        - Análise de Tendência e Volatilidade
        """)
    
    with col4:
        st.markdown("""
        **🤖 Fator Predição ML**
        - Probabilidade de Movimento Direcional (Alta)
        - Consistência do Modelo (AUC-ROC)
        """)
    
    st.markdown("---")
    
    st.markdown("### ⚖️ Otimização vs. Perfil")
    
    perfil_table = pd.DataFrame({
        'Perfil': ['Conservador', 'Intermediário', 'Moderado', 'Moderado-Arrojado', 'Avançado'],
        'Estratégia de Otimização': ['Minimização de Volatilidade', 'Minimização de Volatilidade', 'Maximização do Sharpe', 'Maximização do Sharpe', 'Otimização do CVaR'],
        'Foco dos Fatores': ['Qualidade (Longo Prazo)', 'Equilíbrio c/ Fundamentos', 'Equilíbrio Geral (Médio Prazo)', 'Momentum (Curto Prazo)', 'Visão de Curto Prazo/ML']
    })
    
    st.table(perfil_table)
    
    st.markdown("---")
    
    st.info("""
    **Próxima Etapa:**
    Navegue até a aba **'Construtor de Portfólio'** para responder ao questionário e gerar sua alocação otimizada.
    """)

def aba_selecao_ativos():
    """Aba 2: Seleção de Ativos (Inalterada, apenas pequenos ajustes de texto)"""
    
    st.markdown("## 🎯 Definição do Universo de Análise")
    
    st.markdown("""
    <div class="info-box">
    <p>Selecione o universo de ativos a ser considerado. O motor de otimização irá 
    analisar o histórico e as métricas de todos os ativos selecionados para escolher os 
    <strong>5 ativos ideais</strong> com base no seu perfil de risco.</p>
    </div>
    """, unsafe_allow_html=True)
    
    modo_selecao = st.radio(
        "**Modo de Seleção:**",
        [
            "📊 Índice de Referência (Ibovespa - 82 ativos)",
            "🌐 Universo Expandido (259 ativos)",
            "🏢 Setores Específicos",
            "✍️ Entrada Manual de Tickers"
        ],
        index=0
    )
    
    ativos_selecionados = []
    
    if "Índice de Referência" in modo_selecao:
        ativos_selecionados = ATIVOS_IBOVESPA.copy()
        st.success(f"✓ **{len(ativos_selecionados)} ativos de referência** selecionados")
        
        with st.expander("📋 Ver tickers do Índice"):
            ibov_display = pd.DataFrame({
                'Ticker': ATIVOS_IBOVESPA,
                'Código': [a.replace('.SA', '') for a in ATIVOS_IBOVESPA]
            })
            
            cols = st.columns(4)
            chunk_size = len(ibov_display) // 4 + 1
            
            for i, col in enumerate(cols):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size
                col.dataframe(
                    ibov_display.iloc[start_idx:end_idx],
                    hide_index=True,
                    use_container_width=True
                )
    
    elif "Universo Expandido" in modo_selecao:
        ativos_selecionados = TODOS_ATIVOS.copy()
        st.success(f"✓ **{len(ativos_selecionados)} ativos** selecionados de todo o universo B3")
        
        with st.expander("📊 Distribuição Setorial"):
            setor_counts = {setor: len(ativos) for setor, ativos in ATIVOS_POR_SETOR.items()}
            df_setores = pd.DataFrame({'Setor': list(setor_counts.keys()), 'Quantidade': list(setor_counts.values())}).sort_values('Quantidade', ascending=False)
            
            fig = px.bar(df_setores, x='Setor', y='Quantidade', title='Ativos por Setor')
            fig.update_layout(**obter_template_grafico(), xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    elif "Setores Específicos" in modo_selecao:
        st.markdown("### 🏢 Seleção por Setor")
        setores_disponiveis = list(ATIVOS_POR_SETOR.keys())
        col1, col2 = st.columns([2, 1])
        
        with col1:
            setores_selecionados = st.multiselect(
                "Escolha um ou mais setores para compor o universo de análise:",
                options=setores_disponiveis,
                default=setores_disponiveis[:3] if setores_disponiveis else [],
            )
        
        if setores_selecionados:
            for setor in setores_selecionados: ativos_selecionados.extend(ATIVOS_POR_SETOR[setor])
            
            with col2:
                st.metric("Setores Selecionados", len(setores_selecionados))
                st.metric("Total de Ativos", len(ativos_selecionados))
            
            with st.expander("📋 Ver ativos por setor"):
                for setor in setores_selecionados:
                    st.markdown(f"**{setor}** ({len(ATIVOS_POR_SETOR[setor])} ativos)")
                    st.write(", ".join([a.replace('.SA', '') for a in ATIVOS_POR_SETOR[setor]]))
        else:
            st.warning("⚠️ Selecione pelo menos um setor.")
    
    elif "Entrada Manual de Tickers" in modo_selecao:
        st.markdown("### ✍️ Inserção Manual de Tickers")
        
        ativos_com_setor = {}
        for setor, ativos in ATIVOS_POR_SETOR.items():
            for ativo in ativos: ativos_com_setor[ativo] = setor
        
        todos_tickers = sorted(list(ativos_com_setor.keys()))
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("#### 📝 Selecione Tickers da Lista")
            ativos_da_lista = st.multiselect(
                "Pesquise e selecione tickers disponíveis:",
                options=todos_tickers,
                format_func=lambda x: f"{x.replace('.SA', '')} - {ativos_com_setor.get(x, 'Desconhecido')}",
            )
        
        with col2:
            st.metric("Tickers Selecionados", len(ativos_da_lista))
        
        st.markdown("---")
        st.markdown("#### ✏️ Ou Adicione Tickers Novos/Personalizados")
        
        col3, col4 = st.columns(2)
        
        with col3:
            novos_ativos_input = st.text_area(
                "Digite os códigos dos ativos (um por linha):",
                height=150,
                placeholder="Ex: TOTS3\nBBDC4\n...",
                help="Digite os códigos sem o '.SA'. Um código por linha."
            )
        
        with col4:
            setores_disponiveis_manual = ["Selecione o setor..."] + list(ATIVOS_POR_SETOR.keys())
            setor_novos_ativos = st.selectbox("Setor principal dos tickers manuais:", options=setores_disponiveis_manual)
            st.markdown("**Ou digite um nome de setor:**")
            setor_customizado = st.text_input("Setor personalizado:")
        
        # Process manual inputs
        novos_ativos = []
        if novos_ativos_input.strip():
            linhas = novos_ativos_input.strip().split('\n')
            for linha in linhas:
                ticker = linha.strip().upper()
                if ticker:
                    if not ticker.endswith('.SA'): ticker = f"{ticker}.SA"
                    novos_ativos.append(ticker)
        
        ativos_selecionados = list(set(ativos_da_lista + novos_ativos))
        
        if ativos_selecionados:
            st.success(f"✓ **{len(ativos_selecionados)} ativos** definidos para análise")
            
            with st.expander("📋 Ver tickers definidos"):
                df_selecionados = pd.DataFrame({
                    'Ticker': [a.replace('.SA', '') for a in ativos_selecionados],
                    'Código Completo': ativos_selecionados,
                    'Setor Estimado': [ativos_com_setor.get(a, setor_customizado or setor_novos_ativos or 'Não especificado') 
                             for a in ativos_selecionados]
                })
                st.dataframe(df_selecionados, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ Nenhum ativo definido. Por favor, faça uma seleção.")
    
    if ativos_selecionados:
        st.session_state.ativos_para_analise = ativos_selecionados
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("✓ Tickers Definidos", len(ativos_selecionados))
        col2.metric("→ Serão Ranqueados", len(ativos_selecionados))
        col3.metric("→ Carteira Final", NUM_ATIVOS_PORTFOLIO)
        
        st.success("✓ Definição concluída! Vá para a aba **'Construtor de Portfólio'** para gerar a alocação.")
    else:
        st.warning("⚠️ O universo de análise está vazio.")

# =============================================================================
# Aba 3: Questionário e Construção de Portfólio (Textos revisados)
# =============================================================================

def aba_construtor_portfolio():
    """Aba 3: Questionário e Construção de Portfólio"""
    
    if 'ativos_para_analise' not in st.session_state or not st.session_state.ativos_para_analise:
        st.warning("⚠️ Por favor, defina o universo de análise na aba **'Seleção de Ativos'** primeiro.")
        return
    
    if 'builder' not in st.session_state: st.session_state.builder = None
    if 'profile' not in st.session_state: st.session_state.profile = {}
    if 'builder_complete' not in st.session_state: st.session_state.builder_complete = False
    
    # FASE 1: QUESTIONÁRIO
    if not st.session_state.builder_complete:
        st.markdown('## 📋 Questionário de Calibração do Perfil')
        
        st.info(f"✓ {len(st.session_state.ativos_para_analise)} ativos prontos para o ranqueamento.")
        
        col_question1, col_question2 = st.columns(2)
        
        with st.form("investor_profile_form"):
            options_score = [
                'CT: Concordo Totalmente', 'C: Concordo', 'N: Neutro', 'D: Discordo', 'DT: Discordo Totalmente'
            ]
            options_reaction = ['A: Venderia imediatamente', 'B: Manteria e reavaliaria a tese', 'C: Compraria mais para aproveitar preços baixos']
            options_level_abc = ['A: Avançado (Análise fundamentalista, macro e técnica)', 'B: Intermediário (Conhecimento básico sobre mercados e ativos)', 'C: Iniciante (Pouca ou nenhuma experiência em investimentos)']
            options_time_horizon = [
                'A: Curto (até 1 ano - Foco em liquidez)', 'B: Médio (1-5 anos - Foco em crescimento balanceado)', 'C: Longo (5+ anos - Foco em valor e qualidade)'
            ]
            options_liquidez = [
                'A: Menos de 6 meses (Alta liquidez requerida)', 'B: Entre 6 meses e 2 anos (Liquidez moderada)', 'C: Mais de 2 anos (Baixa necessidade de liquidez imediata)'
            ]
            
            with col_question1:
                st.markdown("#### Avaliação da Tolerância ao Risco")
                p2_risk = st.radio("**1. Estou disposto a aceitar volatilidade de curto prazo em busca de retornos superiores:**", options=options_score, index=2, key='risk_accept_radio')
                p3_gain = st.radio("**2. Meu objetivo principal é a maximização do retorno, mesmo com maior exposição ao risco:**", options=options_score, index=2, key='max_gain_radio')
                p4_stable = st.radio("**3. Priorizo a estabilidade e a preservação do capital em detrimento de altos ganhos:**", options=options_score, index=2, key='stable_growth_radio')
                p5_loss = st.radio("**4. A prevenção de perdas é mais crítica para mim do que a busca por crescimento acelerado:**", options=options_score, index=2, key='avoid_loss_radio')
                p511_reaction = st.radio("**5. Diante de uma queda de 10% no portfólio, minha reação seria:**", options=options_reaction, index=1, key='reaction_radio')
                p_level = st.radio("**6. Meu nível de conhecimento sobre o mercado financeiro:**", options=options_level_abc, index=1, key='level_radio')
            
            with col_question2:
                st.markdown("#### Definição de Horizonte e Capital")
                p211_time = st.radio("**7. O prazo máximo para reavaliação estratégica do portfólio é:**", options=options_time_horizon, index=2, key='time_purpose_radio')[0]
                p311_liquid = st.radio("**8. Meu prazo mínimo de resgate/necessidade de liquidez é:**", options=options_liquidez, index=2, key='liquidity_radio')[0]
                
                st.markdown("---")
                investment = st.number_input(
                    "Capital Total a ser Alocado (R$)",
                    min_value=1000, max_value=10000000, value=100000, step=10000, key='investment_amount'
                )
            
            with st.expander("Parâmetros de Otimização Avançada"):
                otimizar_ml = st.checkbox("Executar Otimização de Hiperparâmetros (HPO) nos modelos ML (Aumenta o tempo de processamento)", value=False, key='optimize_ml_checkbox')
            
            submitted = st.form_submit_button("🚀 Iniciar Otimização Adaptativa", type="primary")
            
            if submitted:
                # 1. Analisa perfil
                risk_answers = {
                    'risk_accept': p2_risk, 'max_gain': p3_gain, 'stable_growth': p4_stable, 'avoid_loss': p5_loss,
                    'reaction': p511_reaction, 'level': p_level, 'time_purpose': p211_time, 'liquidity': p311_liquid
                }
                analyzer = AnalisadorPerfilInvestidor()
                risk_level, horizon, lookback, score = analyzer.calcular_perfil(risk_answers)
                
                st.session_state.profile = {
                    'risk_level': risk_level, 'time_horizon': horizon, 'ml_lookback_days': lookback, 'risk_score': score
                }
                
                # 2. Cria construtor
                try:
                    builder_local = ConstrutorPortfolioAutoML(investment)
                    st.session_state.builder = builder_local
                except Exception as e:
                    st.error(f"Erro fatal ao inicializar o construtor do portfólio: {e}")
                    return

                # 3. Executa pipeline
                with st.spinner(f'Executando pipeline de fatores para **PERFIL {risk_level}** ({horizon})...'):
                    success = builder_local.executar_pipeline(
                        simbolos_customizados=st.session_state.ativos_para_analise,
                        perfil_inputs=st.session_state.profile,
                        otimizar_ml=otimizar_ml
                    )
                    
                    if not success:
                        st.error("Falha na aquisição ou processamento dos dados. Certifique-se de que os arquivos CSV no GCS estão disponíveis e válidos.")
                        st.session_state.builder = None; st.session_state.profile = {}; return
                    
                    st.session_state.builder_complete = True
                    st.rerun()
    
    # FASE 2: RESULTADOS
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
        
        if st.button("🔄 Recalibrar Perfil e Otimizar", key='recomecar_analysis'):
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
                    {'Ativo': a, 'Peso (%)': allocation[a]['weight'] * 100}
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
                        colorscale='RdYlGn', 
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
                st.warning("Não há dados de Predição ML para exibir. O gerador de dados pode ter falhado nesta etapa.")
        
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

                fig_garch.add_trace(go.Bar(name='Volatilidade Histórica', x=plot_df_garch['Ticker'], y=plot_df_garch['Vol. Histórica (%)'], marker=dict(color='#7f8c8d'), opacity=0.7))
                fig_garch.add_trace(go.Bar(name='Volatilidade Condicional', x=plot_df_garch['Ticker'], y=plot_df_garch['Vol. Condicional (%)'], marker=dict(color='#3498db')))
                
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
            
            df_scores_display = builder.scores_combinados[['total_score', 'performance_score', 'fundamental_score', 'technical_score', 'ml_score_weighted', 'sharpe_ratio', 'pe_ratio', 'roe', 'rsi_14', 'macd', 'ML_Proba']].copy()
            df_scores_display.columns = ['Score Total', 'Score Perf.', 'Score Fund.', 'Score Téc.', 'Score ML', 'Sharpe', 'P/L', 'ROE', 'RSI 14', 'MACD', 'Prob. Alta ML']
            df_scores_display = df_scores_display.iloc[:15]
            
            st.markdown("##### Ranqueamento Ponderado Multi-Fatorial (Top 15 Tickers)")
            st.dataframe(df_scores_display.style.background_gradient(cmap='YlGn', subset=['Score Total']), use_container_width=True)
            
            st.markdown("---")
            st.markdown('##### Resumo da Seleção de Ativos')
            
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


# =============================================================================
# Aba 4: Análise Individual (Textos revisados)
# =============================================================================

def aba_analise_individual():
    """Aba 4: Análise Individual de Ativos - Usando dados pré-carregados ou sob demanda do GCS."""
    
    st.markdown("## 🔍 Análise de Fatores por Ticker")
    
    if 'ativos_para_analise' in st.session_state and st.session_state.ativos_para_analise:
        ativos_disponiveis = st.session_state.ativos_para_analise
    else:
        ativos_disponiveis = ATIVOS_IBOVESPA 
            
    if not ativos_disponiveis:
        st.error("Nenhum ativo disponível. Verifique a seleção ou o universo padrão.")
        return

    if 'individual_asset_select' not in st.session_state and ativos_disponiveis:
        st.session_state.individual_asset_select = ativos_disponiveis[0]

    col1, col2 = st.columns([3, 1])
    
    with col1:
        ativo_selecionado = st.selectbox(
            "Selecione um ticker para análise detalhada:",
            options=ativos_disponiveis,
            format_func=lambda x: x.replace('.SA', '') if isinstance(x, str) else x,
            key='individual_asset_select'
        )
    
    with col2:
        if st.button("🔄 Executar Análise", key='analyze_asset_button', type="primary"):
            st.session_state.analisar_ativo_triggered = True 
    
    if 'analisar_ativo_triggered' not in st.session_state or not st.session_state.analisar_ativo_triggered:
        st.info("👆 Selecione um ticker e clique em 'Executar Análise' para obter o relatório completo.")
        return
    
    # --- 3. Execução da Análise (Com Coleta Sob Demanda do GCS) ---
    with st.spinner(f"Processando análise de fatores para {ativo_selecionado} (Leitura GCS)..."):
        try:
            df_completo = None
            features_fund = None
            
            # 1. Tenta usar o cache do construtor (se executado)
            builder_existe = 'builder' in st.session_state and st.session_state.builder is not None
            if builder_existe and ativo_selecionado in st.session_state.builder.dados_por_ativo:
                builder = st.session_state.builder
                df_completo = builder.dados_por_ativo[ativo_selecionado].copy().dropna(how='all')
                features_fund = builder.dados_fundamentalistas.loc[ativo_selecionado].to_dict()
            
            # 2. Se falhar ou não houver cache, coleta sob demanda
            if df_completo is None or df_completo.empty or features_fund is None:
                df_completo, features_fund = ColetorDadosGCS().coletar_ativo_unico_gcs(ativo_selecionado)
                if df_completo is not None: df_completo = df_completo.dropna(how='all')

            if df_completo is None or df_completo.empty or 'Close' not in df_completo.columns:
                st.error(f"❌ Não foi possível obter dados (Histórico/Features) válidos do GCS para **{ativo_selecionado.replace('.SA', '')}**. Verifique a configuração do GCS.")
                return

            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Histórico e Visão Geral",
                "💼 Fatores Fundamentalistas",
                "🔧 Fatores Técnicos e Momentum",
                "🔬 Similaridade e Clusterização"
            ])
            
            with tab1:
                st.markdown(f"### {ativo_selecionado.replace('.SA', '')} - Fatores Chave de Mercado")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                preco_atual = df_completo['Close'].iloc[-1]
                variacao_dia = df_completo['returns'].iloc[-1] * 100
                volume_medio = df_completo['Volume'].mean()
                
                col1.metric("Preço de Fechamento", f"R$ {preco_atual:.2f}", f"{variacao_dia:+.2f}%")
                col2.metric("Volume Médio Recente", f"{volume_medio:,.0f}")
                col3.metric("Setor", features_fund.get('sector', 'N/A'))
                col4.metric("Indústria", features_fund.get('industry', 'N/A'))
                col5.metric("Vol. Anualizada", f"{features_fund.get('annual_volatility', np.nan)*100:.2f}%" if not pd.isna(features_fund.get('annual_volatility')) else "N/A")
                
                if not df_completo.empty and 'Open' in df_completo.columns and 'Volume' in df_completo.columns:
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                    
                    fig.add_trace(go.Candlestick(x=df_completo.index, open=df_completo['Open'], high=df_completo['High'], low=df_completo['Low'], close=df_completo['Close'], name='Preço'), row=1, col=1)
                    fig.add_trace(go.Bar(x=df_completo.index, y=df_completo['Volume'], name='Volume', marker=dict(color='lightblue'), opacity=0.7), row=2, col=1)
                    
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
                col3.metric("ROE (Rentabilidade)", f"{features_fund.get('roe', np.nan):.2f}%" if not pd.isna(features_fund.get('roe')) else "N/A")
                col4.metric("Margem Operacional", f"{features_fund.get('operating_margin', np.nan):.2f}%" if not pd.isna(features_fund.get('operating_margin')) else "N/A")
                col5.metric("Cresc. Receita Anual", f"{features_fund.get('revenue_growth', np.nan):.2f}%" if not pd.isna(features_fund.get('revenue_growth')) else "N/A")
                
                st.markdown("#### Saúde Financeira e Dividendo")
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Dívida/Patrimônio", f"{features_fund.get('debt_to_equity', np.nan):.2f}" if not pd.isna(features_fund.get('debt_to_equity')) else "N/A")
                col2.metric("Current Ratio", f"{features_fund.get('current_ratio', np.nan):.2f}" if not pd.isna(features_fund.get('current_ratio')) else "N/A")
                col3.metric("Dividend Yield", f"{features_fund.get('div_yield', np.nan):.2f}%" if not pd.isna(features_fund.get('div_yield')) else "N/A")
                col4.metric("Beta (Risco Sistêmico)", f"{features_fund.get('beta', np.nan):.2f}" if not pd.isna(features_fund.get('beta')) else "N/A")
                
                st.markdown("---")
                st.markdown("#### Tabela de Fatores Fundamentais (GCS)")
                
                df_fund_display = pd.DataFrame({
                    'Métrica': list(features_fund.keys()),
                    'Valor': [f"{v:.4f}" if isinstance(v, (int, float)) and not pd.isna(v) else str(v) for v in features_fund.values()]
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
                
                if 'rsi_14' in df_completo.columns:
                    fig_osc.add_trace(go.Scatter(x=df_completo.index, y=df_completo['rsi_14'], name='RSI', line=dict(color='#3498db')), row=1, col=1)
                    fig_osc.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
                    fig_osc.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
                
                if 'macd' in df_completo.columns and 'macd_signal' in df_completo.columns:
                    fig_osc.add_trace(go.Scatter(x=df_completo.index, y=df_completo['macd'], name='MACD', line=dict(color='#e74c3c')), row=2, col=1)
                    fig_osc.add_trace(go.Scatter(x=df_completo.index, y=df_completo['macd_signal'], name='Signal', line=dict(color='#7f8c8d')), row=2, col=1)
                    fig_osc.add_trace(go.Bar(x=df_completo.index, y=df_completo['macd_diff'], name='Histograma', marker=dict(color='lightgray')), row=2, col=1)
                
                fig_layout = obter_template_grafico()
                fig_layout['height'] = 550
                fig_osc.update_layout(**fig_layout)
                
                st.plotly_chart(fig_osc, use_container_width=True)

            with tab4:
                st.markdown("### Análise de Similaridade e Clusterização")
                
                if not builder_existe or builder.metricas_performance.empty:
                    st.warning("A Clusterização está desabilitada. É necessário executar a **'Otimização Adaptativa'** (Aba 3) para carregar os dados de comparação de múltiplos ativos.")
                    return
                
                df_comparacao = builder.metricas_performance.join(builder.dados_fundamentalistas, how='inner', rsuffix='_fund')
                
                features_cluster = ['sharpe', 'retorno_anual', 'volatilidade_anual', 'max_drawdown', 
                                    'pe_ratio', 'pb_ratio', 'roe', 'debt_to_equity', 'revenue_growth']
                
                df_comparacao = df_comparacao.filter(items=features_cluster).dropna()
                
                if len(df_comparacao) < 10:
                    st.warning("Dados insuficientes para realizar a clusterização (menos de 10 ativos com métricas completas).")
                    return
                    
                resultado_pca, pca, kmeans = AnalisadorIndividualAtivos.realizar_clusterizacao_pca(df_comparacao, n_clusters=5)
                
                if resultado_pca is not None:
                    if 'PC3' in resultado_pca.columns:
                        fig_pca = px.scatter_3d(resultado_pca, x='PC1', y='PC2', z='PC3', color='Cluster', hover_name=resultado_pca.index.str.replace('.SA', ''), title='Similaridade de Tickers (PCA/K-means - 3D)')
                    else:
                        fig_pca = px.scatter(resultado_pca, x='PC1', y='PC2', color='Cluster', hover_name=resultado_pca.index.str.replace('.SA', ''), title='Similaridade de Tickers (PCA/K-means - 2D)')
                    
                    fig_pca.update_layout(**obter_template_grafico(), height=600)
                    st.plotly_chart(fig_pca, use_container_width=True)
                    
                    if ativo_selecionado in resultado_pca.index:
                        cluster_ativo = resultado_pca.loc[ativo_selecionado, 'Cluster']
                        ativos_similares_df = resultado_pca[resultado_pca['Cluster'] == cluster_ativo]
                        ativos_similares = [a for a in ativos_similares_df.index.tolist() if a != ativo_selecionado]
                        
                        st.success(f"**{ativo_selecionado.replace('.SA', '')}** pertence ao Cluster {cluster_ativo}")
                        
                        if ativos_similares:
                            st.markdown(f"#### Tickers Similares no Cluster {cluster_ativo}:")
                            st.write(", ".join([a.replace('.SA', '') for a in ativos_similares]))
                        else:
                            st.info("Nenhum outro ticker similar encontrado neste cluster.")

                else:
                    st.warning("Não foi possível realizar a clusterização (erro de dimensionalidade ou dados nulos).")
        
        except Exception as e:
            st.error(f"Erro ao analisar o ticker {ativo_selecionado}: {str(e)}")
            
def main():
    """Função principal que orquestra a interface Streamlit."""
    
    if 'builder' not in st.session_state:
        st.session_state.builder = None
        st.session_state.builder_complete = False
        st.session_state.profile = {}
        st.session_state.ativos_para_analise = []
        st.session_state.analisar_ativo_triggered = False
        
    configurar_pagina()
    
    # Novo Título Principal
    st.markdown('<h1 class="main-header">Sistema de Portfólios Adaptativos</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📚 Metodologia",
        "🎯 Seleção de Ativos",
        "🏗️ Construtor de Portfólio",
        "🔍 Análise Individual"
    ])
    
    with tab1:
        aba_introducao()
    
    with tab2:
        aba_selecao_ativos()
    
    with tab3:
        aba_construtor_portfolio()
    
    with tab4:
        aba_analise_individual()

if __name__ == "__main__":
    main()
