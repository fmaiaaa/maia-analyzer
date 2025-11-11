"""
=============================================================================
SISTEMA AUTOML AVANÇADO - OTIMIZAÇÃO DE PORTFÓLIO FINANCEIRO
=============================================================================

Sistema completo de otimização de portfólio com:
- Questionário de perfil de investidor
- Seleção de ativos por setor
- Ensemble de modelos ML (XGBoost, LightGBM, RandomForest)
- Modelagem de volatilidade GARCH
- Otimização de hiperparâmetros com Optuna
- Engenharia massiva de features
- Smart Beta Factors
- Dashboard interativo completo

Versão: 5.0.0 - Sistema AutoML Completo
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

# --- 2. SCIENTIFIC / STATISTICAL TOOLS ---
from scipy.optimize import minimize
from scipy.stats import zscore, norm

# --- 3. STREAMLIT, DATA ACQUISITION, & PLOTTING ---
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- 4. FEATURE ENGINEERING / TECHNICAL ANALYSIS (TA) ---
import ta
# Trend Indicators
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, CCIIndicator
# Momentum Indicators
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
# Volatility Indicators
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel, DonchianChannel
# Volume Indicators
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, MFIIndicator, VolumeWeightedAveragePrice

# --- 5. MACHINE LEARNING (SCIKIT-LEARN) ---

# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression, BayesianRidge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Preprocessing, Features & Clustering
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

# Metrics
from sklearn.metrics import silhouette_score, mean_squared_error, mean_absolute_error, r2_score, roc_auc_score

# --- 6. BOOSTED MODELS ---
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# --- 7. SPECIALIZED TIME SERIES & ECONOMETRICS ---
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.varmax import VARMAX
from prophet import Prophet
from arch import arch_model

# --- 8. OPTIMIZATION AND EXPLAINABILITY (XAI) ---
import optuna
import shap
import lime
import lime.lime_tabular

# --- 9. DEEP LEARNING (TENSORFLOW/KERAS) ---
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, MultiHeadAttention, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# --- 10. CONFIGURATION ---
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURAÇÕES GLOBAIS
# =============================================================================

# Período de coleta de dados. 'max' indica o máximo disponível no yfinance.
PERIODO_DADOS = 'max'
# Mínimo de dias úteis para considerar um histórico válido (aprox. 1 ano)
MIN_DIAS_HISTORICO = 252
# Número de ativos a serem selecionados para compor o portfólio final
NUM_ATIVOS_PORTFOLIO = 5
# Taxa Livre de Risco (e.g., CDI/SELIC anualizada) usada no cálculo do Sharpe Ratio
TAXA_LIVRE_RISCO = 0.1075
# Janela de lookback (dias) para a previsão dos modelos de Machine Learning
LOOKBACK_ML = 30

# =============================================================================
# 2. PONDERAÇÕES E REGRAS DE OTIMIZAÇÃO
# =============================================================================

# Ponderações padrão para o score agregado (a soma deve ser 1.0 ou próxima)
WEIGHT_PERFORMANCE = 0.40   # Desempenho histórico (Retorno, Volatilidade, Sharpe)
WEIGHT_FUNDAMENTAL = 0.30   # Indicadores fundamentalistas
WEIGHT_TECHNICAL = 0.30     # Indicadores técnicos (RSI, MACD, etc.)
WEIGHT_ML = 0.30            # Ponderação do score de Machine Learning (e.g., previsão de alta)

# Limites de peso por ativo no portfólio final (para diversificação)
PESO_MIN = 0.10
PESO_MAX = 0.30

# =============================================================================
# 3. CAMINHOS DE DADOS
# =============================================================================

DATA_PATH = './dados_financeiros/'
ARQUIVO_HISTORICO = DATA_PATH + 'dados_historicos.parquet'
ARQUIVO_FUNDAMENTALISTA = DATA_PATH + 'dados_fundamentalistas.parquet'
ARQUIVO_METRICAS = DATA_PATH + 'metricas_performance.parquet'
ARQUIVO_MACRO = DATA_PATH + 'dados_macro.parquet'
ARQUIVO_METADATA = DATA_PATH + 'metadata.parquet'

# =============================================================================
# 4. LISTAS DE ATIVOS E SETORES
# =============================================================================

# Lista oficial de ativos do IBOVESPA (aproximada, usada como base para a tela principal)
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

# Mapeamento estendido de ativos por setor (usado para diversificação)
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

# Lista completa de todos os ativos únicos mapeados por setor
TODOS_ATIVOS = sorted(list(set([ativo for ativos in ATIVOS_POR_SETOR.values() for ativo in ativos])))


# =============================================================================
# 5. CONSTANTES DE GOVERNANÇA E MLOPS (Monitoramento de Modelos)
# =============================================================================

# Mínimo aceitável para a métrica AUC (Área sob a Curva ROC)
AUC_THRESHOLD_MIN = 0.65
# Queda percentual no AUC que dispara um alerta de degradação do modelo
AUC_DROP_THRESHOLD = 0.05
# Janela de observação para monitoramento de desvio de dados (data drift)
DRIFT_WINDOW = 20
# Número de desvios-padrão para simular choques em testes de estresse
STRESS_TEST_SIGMA = 2.0


# =============================================================================
# 6. MAPEAMENTOS DE PONTUAÇÃO DO QUESTIONÁRIO (Perfil do Investidor)
# =============================================================================

# Mapeamento de score padrão (quanto mais extremo, maior o score)
SCORE_MAP = {
    'CT: Concordo Totalmente': 5,
    'C: Concordo': 4,
    'N: Neutro': 3,
    'D: Discordo': 2,
    'DT: Discordo Totalmente': 1
}

# Mapeamento de score invertido (para perguntas onde "Discordo" indica maior risco)
SCORE_MAP_INV = {
    'CT: Concordo Totalmente': 1,
    'C: Concordo': 2,
    'N: Neutro': 3,
    'D: Discordo': 4,
    'DT: Discordo Totalmente': 5
}

# Mapeamento de score para nível de conhecimento
SCORE_MAP_CONHECIMENTO = {
    'A: Avançado': 5,
    'B: Intermediário': 3,
    'C: Iniciante': 1
}

# Mapeamento de score para reação a perdas (tolerância a risco)
SCORE_MAP_REACTION = {
    'A: Venderia': 1,
    'B: Manteria': 3,
    'C: Compraria mais': 5
}

# =============================================================================
# 1. CONSTANTES DE GOVERNANÇA (Extraídas da Configuração Global)
# =============================================================================

# Mínimo aceitável para a métrica AUC (Área sob a Curva ROC)
AUC_THRESHOLD_MIN = 0.65
# Queda percentual no AUC que dispara um alerta de degradação do modelo
AUC_DROP_THRESHOLD = 0.05
# Janela de observação para monitoramento de drift (usado como max_historico na classe)
DRIFT_WINDOW = 20
# Número de desvios-padrão para simular choques em testes de estresse (não usado na classe abaixo, mas mantido para contexto)
STRESS_TEST_SIGMA = 2.0

# =============================================================================
# 1. MAPEAMENTOS DE PONTUAÇÃO DO QUESTIONÁRIO (Replicados para auto-contenção)
# =============================================================================

SCORE_MAP = {
    'CT: Concordo Totalmente': 5,
    'C: Concordo': 4,
    'N: Neutro': 3,
    'D: Discordo': 2,
    'DT: Discordo Totalmente': 1
}

SCORE_MAP_INV = {
    'CT: Concordo Totalmente': 1,
    'C: Concordo': 2,
    'N: Neutro': 3,
    'D: Discordo': 4,
    'DT: Discordo Totalmente': 5
}

SCORE_MAP_CONHECIMENTO = {
    'A: Avançado': 5,
    'B: Intermediário': 3,
    'C: Iniciante': 1
}

SCORE_MAP_REACTION = {
    'A: Venderia': 1,
    'B: Manteria': 3,
    'C: Compraria mais': 5
}

# =============================================================================
# 2. CLASSE: ANALISADOR DE PERFIL DO INVESTIDOR
# =============================================================================

class AnalisadorPerfilInvestidor:
    """
    Analisa perfil de risco e horizonte temporal do investidor com base 
    em um questionário padronizado.
    """
    
    def __init__(self):
        self.nivel_risco = ""
        self.horizonte_tempo = ""
        self.dias_lookback_ml = 5 # Janela de previsão padrão
    
    def determinar_nivel_risco(self, pontuacao: int) -> str:
        """Traduz a pontuação total do questionário em perfil de risco."""
        # Os thresholds de pontuação definem as categorias de risco
        if pontuacao <= 18:
            return "CONSERVADOR"
        elif pontuacao <= 30:
            return "INTERMEDIÁRIO"
        elif pontuacao <= 45:
            return "MODERADO"
        elif pontuacao <= 60:
            return "MODERADO-ARROJADO"
        else:
            return "AVANÇADO"
    
    def determinar_horizonte_ml(self, liquidez_key: str, objetivo_key: str) -> tuple[str, int]:
        """
        Define o horizonte temporal e a janela de lookback para modelos de ML 
        com base nas respostas de liquidez e objetivo.
        """
        # Mapeia as chaves de resposta para o número de dias úteis (lookback/horizonte)
        time_map = {
            'A': 5,  # Curto prazo (1 semana)
            'B': 20, # Médio prazo (1 mês)
            'C': 30  # Longo prazo (mais de 1 mês)
        }
        
        # Escolhe o lookback mais longo entre as duas perguntas
        final_lookback = max(
            time_map.get(liquidez_key, 5),
            time_map.get(objetivo_key, 5)
        )
        
        if final_lookback >= 30:
            self.horizonte_tempo = "LONGO PRAZO"
            self.dias_lookback_ml = 30
        elif final_lookback >= 20:
            self.horizonte_tempo = "MÉDIO PRAZO"
            self.dias_lookback_ml = 20
        else:
            self.horizonte_tempo = "CURTO PRAZO"
            self.dias_lookback_ml = 5
        
        return self.horizonte_tempo, self.dias_lookback_ml
    
    def calcular_perfil(self, respostas_risco: dict) -> tuple[str, str, int, int]:
        """
        Calcula o perfil completo do investidor.
        
        Args:
            respostas_risco (dict): Dicionário contendo as chaves de resposta.
                                    Ex: {'risk_accept': 'CT: Concordo Totalmente', ...}
                                    
        Returns:
            tuple: (nivel_risco, horizonte_tempo, ml_lookback, pontuacao_total)
        """
        
        # Cálculo da pontuação total (a ponderação reflete a importância da pergunta)
        pontuacao = (
            SCORE_MAP[respostas_risco['risk_accept']] * 5 +          # Alto Risco
            SCORE_MAP[respostas_risco['max_gain']] * 5 +             # Alto Retorno
            SCORE_MAP_INV[respostas_risco['stable_growth']] * 5 +    # Evita crescimento estável
            SCORE_MAP_INV[respostas_risco['avoid_loss']] * 5 +       # Evita perdas (Invertido)
            SCORE_MAP_CONHECIMENTO[respostas_risco['level']] * 3 +   # Conhecimento (Peso Médio)
            SCORE_MAP_REACTION[respostas_risco['reaction']] * 3      # Tolerância a perdas (Peso Médio)
        )
        
        nivel_risco = self.determinar_nivel_risco(pontuacao)
        
        horizonte_tempo, ml_lookback = self.determinar_horizonte_ml(
            respostas_risco['liquidity'],
            respostas_risco['time_purpose']
        )
        
        return nivel_risco, horizonte_tempo, ml_lookback, pontuacao

# =============================================================================
# 3. FUNÇÕES DE ESTILO E VISUALIZAÇÃO
# =============================================================================

def obter_template_grafico() -> dict:
    """Retorna um template de layout otimizado para gráficos Plotly com estilo Times New Roman."""
    return {
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'font': {
            'family': 'Times New Roman, serif', # Fonte primária com fallback
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
            'showgrid': True,
            'gridcolor': 'lightgray',
            'showline': True,
            'linecolor': 'black',
            'linewidth': 1,
            'tickfont': {'family': 'Times New Roman, serif', 'color': 'black'},
            'title': {'font': {'family': 'Times New Roman, serif', 'color': 'black'}},
            'zeroline': False
        },
        'yaxis': {
            'showgrid': True,
            'gridcolor': 'lightgray',
            'showline': True,
            'linecolor': 'black',
            'linewidth': 1,
            'tickfont': {'family': 'Times New Roman, serif', 'color': 'black'},
            'title': {'font': {'family': 'Times New Roman, serif', 'color': 'black'}},
            'zeroline': False
        },
        'legend': {
            'font': {'family': 'Times New Roman, serif', 'color': 'black'},
            'bgcolor': 'rgba(255, 255, 255, 0.8)',
            'bordercolor': 'lightgray',
            'borderwidth': 1
        },
        # Esquema de cores corporativo ou neutro
        'colorway': ['#2c3e50', '#7f8c8d', '#3498db', '#e74c3c', '#27ae60']
    }

# =============================================================================
# CLASSE: ENGENHEIRO DE FEATURES
# =============================================================================

class EngenheiroFeatures:
    """Calcula indicadores técnicos e fundamentalistas com máxima profundidade"""
    
# DENTRO DA CLASSE ENGENHEIRO DE FEATURES

    @staticmethod
    def calcular_indicadores_tecnicos(hist: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula indicadores técnicos simplificados (similar ao p_a) 
        para otimizar o tempo, mantendo a compatibilidade do scoring.
        """
        df = hist.copy()
        
        # --- 1. Retornos e Volatilidade (BASE) ---
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        # Volatilidade anualizada (janela de 20 dias - MANTIDO como 'volatility_20')
        df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # --- 2. Médias Móveis (SMA 50 e 200 - Essenciais) ---
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        df['sma_200'] = df['Close'].rolling(window=200).mean()
        
        # --- 3. Momentum (RSI e MACD - Essenciais para Scoring) ---
        
        # RSI (14 períodos - Renomeado para 'rsi_14' para compatibilidade)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['rsi_14'] = 100 - (100 / (1 + (gain / loss)))
        
        # MACD (12, 26, 9 - Renomeado para 'macd' e componentes para compatibilidade)
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26 
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal'] # Usado em algumas features avançadas originais
        
        # Momentum (10 períodos - p_a usa 10)
        df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        
        # --- 4. Indicadores Necessários para o CORE PIPELINE (Drawdown e BBands) ---
        
        # Drawdown (Necessário para cálculo de Max Drawdown)
        cumulative_returns = (1 + df['returns']).cumprod()
        running_max = cumulative_returns.expanding().max()
        df['drawdown'] = (cumulative_returns - running_max) / running_max
        df['max_drawdown_252'] = df['drawdown'].rolling(252).min()
        
        # Bandas de Bollinger (Apenas os componentes necessários para 'bb_position')
        bb_window = 20
        df['bb_middle'] = df['Close'].rolling(window=bb_window).mean()
        bb_std = df['Close'].rolling(window=bb_window).std()
        df['bb_upper'] = df['bb_middle'] + 2 * bb_std
        df['bb_lower'] = df['bb_middle'] - 2 * bb_std
        # Coluna crítica para o Scoring: bb_position
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # --- 5. Features Temporais (Necessárias para ML) ---
        if not df.index.empty:
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            df['day_of_month'] = df.index.day
            df['week_of_year'] = df.index.isocalendar().week.astype(int)
            
        # Garante que a coluna 'Volume' está presente
        if 'Volume' not in df.columns and 'Volume' in hist.columns:
            df['Volume'] = hist['Volume'].reindex(df.index)
            
        return df.dropna()
    
    @staticmethod
    def calcular_features_fundamentalistas(info: dict) -> dict:
        """Extrai features fundamentalistas expandidas de um dicionário de info do ativo."""
        return {
            'pe_ratio': info.get('trailingPE', np.nan),
            'forward_pe': info.get('forwardPE', np.nan),
            'pb_ratio': info.get('priceToBook', np.nan),
            'ps_ratio': info.get('priceToSalesTrailing12Months', np.nan),
            'peg_ratio': info.get('pegRatio', np.nan),
            'ev_ebitda': info.get('enterpriseToEbitda', np.nan),
            'div_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else np.nan,
            'payout_ratio': info.get('payoutRatio', np.nan) * 100 if info.get('payoutRatio') else np.nan,
            'roe': info.get('returnOnEquity', np.nan) * 100 if info.get('returnOnEquity') else np.nan,
            'roa': info.get('returnOnAssets', np.nan) * 100 if info.get('returnOnAssets') else np.nan,
            'roic': info.get('returnOnCapital', np.nan) * 100 if info.get('returnOnCapital') else np.nan,
            'profit_margin': info.get('profitMargins', np.nan) * 100 if info.get('profitMargins') else np.nan,
            'operating_margin': info.get('operatingMargins', np.nan) * 100 if info.get('operatingMargins') else np.nan,
            'gross_margin': info.get('grossMargins', np.nan) * 100 if info.get('grossMargins') else np.nan,
            'debt_to_equity': info.get('debtToEquity', np.nan),
            'current_ratio': info.get('currentRatio', np.nan),
            'quick_ratio': info.get('quickRatio', np.nan),
            'revenue_growth': info.get('revenueGrowth', np.nan) * 100 if info.get('revenueGrowth') else np.nan,
            'earnings_growth': info.get('earningsGrowth', np.nan) * 100 if info.get('earningsGrowth') else np.nan,
            'market_cap': info.get('marketCap', np.nan),
            'enterprise_value': info.get('enterpriseValue', np.nan),
            'beta': info.get('beta', np.nan),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown')
        }
    
    @staticmethod
    def _normalizar(serie: pd.Series, maior_melhor: bool = True) -> pd.Series:
        """Normaliza uma série de indicadores para o range [0, 1] (Min-Max Scaling)."""
        if serie.isnull().all():
            return pd.Series(0, index=serie.index)
        
        min_val = serie.min()
        max_val = serie.max()
        
        if max_val == min_val:
            # Retorna 0.5 se todos os valores forem iguais (neutro)
            return pd.Series(0.5, index=serie.index)
        
        # Aplica a normalização
        if maior_melhor:
            return (serie - min_val) / (max_val - min_val)
        else:
            # Normalização invertida (quanto menor, melhor)
            return (max_val - serie) / (max_val - min_val)

# =============================================================================
# FUNÇÃO AUXILIAR: COLETA ROBusta
# =============================================================================

# FUNÇÃO AUXILIAR: COLETA ROBusta E RESILIENTE (VERSÃO p_a ADAPTADA)
def coletar_historico_ativo_robusto(ticker, periodo, min_dias_historico, max_retries=3, initial_delay=3): # Delay aumentado para Cloud
    """
    Coleta dados históricos do ativo usando yfinance com retentativas, 
    variações de ticker e validação de tamanho mínimo.
    
    Retorna: (DataFrame com histórico, mensagem_de_erro)
    """
    import yfinance as yf
    import time
    
    def get_ticker_variations_b3(symbol):
        """ Gera variações do ticker para aumentar chance de sucesso. """
        variations = [symbol]
        if symbol.endswith('.SA'):
            base = symbol[:-3]
            variations.append(base)
            variations.append(base + '.SAO')
        else:
            if len(symbol) <= 6 and not symbol.endswith(('.L', '.PA', '.NY')):
                variations.append(symbol + '.SA')
                variations.append(symbol + '.SAO')
        
        # Remove duplicatas e garante que o original está no início
        return list(dict.fromkeys(variations))

    simbolo_completo = ticker if ticker.endswith('.SA') else f"{ticker}.SA"
    min_dias_flexivel = max(180, int(min_dias_historico * 0.7))
    
    for attempt in range(max_retries):
        found_valid_data = False
        last_error = "Coleta não tentada."
        
        for attempt_ticker in get_ticker_variations_b3(simbolo_completo):
            try:
                # Prioriza yf.Ticker().history() (mais robusto)
                ticker_obj = yf.Ticker(attempt_ticker)
                hist = ticker_obj.history(
                    period=periodo,
                    interval="1d",
                    auto_adjust=True,
                    timeout=15 
                )
                
                # Fallback para yf.download se o Ticker().history falhar ou retornar pouco
                if hist.empty or len(hist) < 5: 
                    hist = yf.download(
                        attempt_ticker,
                        period=periodo,
                        progress=False,
                        timeout=10
                    )

                if not hist.empty and len(hist) >= min_dias_flexivel:
                    found_valid_data = True
                    break
                    
            except Exception as e:
                last_error = str(e)
                continue

        if found_valid_data:
            return hist, None
        
        if attempt < max_retries - 1:
            delay = initial_delay * (2 ** attempt)
            print(f"  [Tentativa {attempt+1}] Falha em {simbolo_completo}. Esperando {delay}s...")
            time.sleep(delay)
            continue
        else:
            return None, f"Erro na coleta após {max_retries} tentativas: {last_error[:50]}"
            
    return None, "Falha desconhecida na coleta."


# =============================================================================
# CLASSE: COLETOR DE DADOS (APENAS YFINANCE)
# =============================================================================

class ColetorDados:
    """Coleta e processa dados de mercado com profundidade máxima, usando yfinance."""
    
    def __init__(self, periodo=PERIODO_DADOS):
        self.periodo = periodo
        self.dados_por_ativo = {}
        self.dados_fundamentalistas = pd.DataFrame()
        self.ativos_sucesso = []
        self.dados_macro = {}
        self.metricas_performance = pd.DataFrame() # Initialize metric dataframe
    
    def coletar_dados_macroeconomicos(self):
        """Coleta dados macroeconômicos (índices) para features externas, usando yfinance."""
        print("\n📊 Coletando dados macroeconômicos...")
        
        try:
            # Índices de referência
            indices = {
                'IBOV': '^BVSP',  # Ibovespa
                'SP500': '^GSPC',  # S&P 500
                'VIX': '^VIX',  # Volatility Index
                'USD_BRL': 'BRL=X',  # Dólar
                'GOLD': 'GC=F',  # Ouro
                'OIL': 'CL=F'  # Petróleo WTI
            }
            
            for nome, simbolo in indices.items():
                try:
                    ticker = yf.Ticker(simbolo)
                    hist = ticker.history(period=self.periodo)
                    
                    if not hist.empty:
                        self.dados_macro[nome] = hist['Close'].pct_change()
                        # print(f"  ✓ {nome}: {len(hist)} dias")
                    else:
                        # print(f"  ⚠️ {nome}: Sem dados históricos")
                        self.dados_macro[nome] = pd.Series()
                except Exception as e:
                    # print(f"  ⚠️ {nome}: Erro - {str(e)[:50]}")
                    self.dados_macro[nome] = pd.Series()
            
            print(f"✓ Dados macroeconômicos coletados: {len(self.dados_macro)} indicadores")
            
        except Exception as e:
            print(f"❌ Erro ao coletar dados macro: {str(e)}")
    
    def adicionar_correlacoes_macro(self, df, simbolo):
        """Adiciona correlações com indicadores macroeconômicos (lógica inalterada)"""
        if not self.dados_macro or 'returns' not in df.columns:
            return df
        
        try:
            if df['returns'].isnull().all():
                # print(f"  ⚠️ {simbolo}: Coluna 'returns' está vazia, pulando correlações macro.")
                return df

            for nome, serie_macro in self.dados_macro.items():
                if serie_macro.empty or serie_macro.isnull().all():
                    continue
                
                df_returns_aligned = df['returns'].reindex(df.index)
                
                if df_returns_aligned.isnull().all() or serie_macro.isnull().all():
                    continue

                combined_df = pd.DataFrame({
                    'asset_returns': df_returns_aligned,
                    'macro_returns': serie_macro.reindex(df.index)
                }).dropna()

                if len(combined_df) > 60:
                    corr_rolling = combined_df['asset_returns'].rolling(60).corr(combined_df['macro_returns'])
                    df[f'corr_{nome.lower()}'] = corr_rolling.reindex(df.index)
                else:
                    df[f'corr_{nome.lower()}'] = np.nan
        except Exception as e:
            print(f"  ⚠️ {simbolo}: Erro ao calcular correlações macro - {str(e)[:80]}")
        
        return df
    
    def coletar_e_processar_dados(self, simbolos):
        """Coleta e processa dados de mercado para todos os ativos solicitados."""
        self.ativos_sucesso = []
        lista_fundamentalistas = []
        
        # 1. Coleta dados macro
        self.coletar_dados_macroeconomicos()
        
        print(f"\n{'='*60}")
        print(f"INICIANDO COLETA E PROCESSAMENTO - {len(simbolos)} ativos")
        print(f"Período: {self.periodo} (MÁXIMO DISPONÍVEL)")
        print(f"Mínimo de dias: {MIN_DIAS_HISTORICO}")
        print(f"{'='*60}\n")
        
        erros_detalhados = []
        
        for simbolo in tqdm(simbolos, desc="📥 Coletando dados"):
            simbolo_completo = simbolo if simbolo.endswith('.SA') else f"{simbolo}.SA"
            
            # A. Coleta histórica robusta
            hist, erro_coleta = coletar_historico_ativo_robusto(
                simbolo, 
                self.periodo, 
                MIN_DIAS_HISTORICO
            )
            
            if hist is None:
                erros_detalhados.append(f"{simbolo}: {erro_coleta}")
                continue
            
            try:
                # B. Engenharia de Features Técnicas
                df = EngenheiroFeatures.calcular_indicadores_tecnicos(hist)
                
                # C. Adiciona Correlações Macro
                df = self.adicionar_correlacoes_macro(df, simbolo_completo)
                
                # Revalidação de tamanho após a remoção de NaNs
                min_dias_flexivel = max(180, int(MIN_DIAS_HISTORICO * 0.7))
                if len(df) < min_dias_flexivel:
                    erros_detalhados.append(f"{simbolo}: Dados insuficientes após features: {len(df)} dias")
                    continue
                
                # D. Coleta de Dados Fundamentalistas (yf.Ticker().info)
                ticker = yf.Ticker(simbolo_completo)
                info = ticker.info
                features_fund = EngenheiroFeatures.calcular_features_fundamentalistas(info)
                features_fund['Ticker'] = simbolo_completo
                lista_fundamentalistas.append(features_fund)
                
                # E. Sucesso - armazena dados
                self.dados_por_ativo[simbolo_completo] = df
                self.ativos_sucesso.append(simbolo_completo)
                
            except Exception as e:
                erros_detalhados.append(f"{simbolo}: Erro no processamento - {str(e)[:50]}")
                continue
        
        # Log detalhado de erros
        if erros_detalhados:
            print(f"\n⚠️ Ativos com problemas ({len(erros_detalhados)}):")
            for erro in erros_detalhados[:10]:
                print(f"  • {erro}")
            if len(erros_detalhados) > 10:
                print(f"  ... e mais {len(erros_detalhados) - 10} ativos")
        
        print(f"\n✓ Total de ativos válidos: {len(self.ativos_sucesso)} ativos")
        
        # 2. Validação final
        if len(self.ativos_sucesso) < NUM_ATIVOS_PORTFOLIO:
            print(f"\n❌ ERRO: Apenas {len(self.ativos_sucesso)} ativos coletados.")
            print(f"    Necessário: {NUM_ATIVOS_PORTFOLIO} ativos mínimos")
            return False
        
        # 3. Processamento e Escalonamento Fundamentalista
        self.dados_fundamentalistas = pd.DataFrame(lista_fundamentalistas).set_index('Ticker')
        self.dados_fundamentalistas = self.dados_fundamentalistas.replace([np.inf, -np.inf], np.nan)
        
        scaler = RobustScaler()
        numeric_cols = self.dados_fundamentalistas.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.dados_fundamentalistas[col].isnull().any():
                median_val = self.dados_fundamentalistas[col].median()
                self.dados_fundamentalistas[col] = self.dados_fundamentalistas[col].fillna(median_val)
        
        self.dados_fundamentalistas[numeric_cols] = scaler.fit_transform(self.dados_fundamentalistas[numeric_cols])

        # 4. Cálculo de Métricas de Performance
        metricas = {}
        for simbolo in self.ativos_sucesso:
            if 'returns' in self.dados_por_ativo[simbolo] and 'drawdown' in self.dados_por_ativo[simbolo]:
                returns = self.dados_por_ativo[simbolo]['returns']
                volatilidade_anual = returns.std() * np.sqrt(252)
                retorno_anual = returns.mean() * 252
                metricas[simbolo] = {
                    'retorno_anual': retorno_anual,
                    'volatilidade_anual': volatilidade_anual,
                    'sharpe': (retorno_anual - TAXA_LIVRE_RISCO) / volatilidade_anual if volatilidade_anual > 0 else 0,
                    'max_drawdown': self.dados_por_ativo[simbolo]['drawdown'].min()
                }

        self.metricas_performance = pd.DataFrame(metricas).T
        
        return True

# =============================================================================
# CLASSE: MODELAGEM DE VOLATILIDADE GARCH
# =============================================================================

# Certifique-se de que as importações abaixo estão no topo do seu script principal:
# import numpy as np
# import pandas as pd
# from arch import arch_model

class VolatilidadeGARCH:
    """Modelagem de volatilidade GARCH/EGARCH"""
    
    @staticmethod
    def ajustar_garch(returns: pd.Series, tipo_modelo: str = 'GARCH') -> float:
        """
        Ajusta modelo GARCH ou EGARCH e prevê a volatilidade anualizada para o próximo dia.
        
        Args:
            returns (pd.Series): Série temporal dos retornos diários do ativo.
            tipo_modelo (str): 'GARCH' (padrão) ou 'EGARCH'.
            
        Returns:
            float: Volatilidade anualizada prevista pelo modelo, ou np.nan em caso de falha.
        """
        try:
            # 1. Preparação dos dados
            # Multiplica por 100 para evitar problemas de otimização/underflow com arch
            returns_limpo = returns.dropna() * 100 
            
            # 2. Validações iniciais
            if len(returns_limpo) < 100: # Mínimo de pontos para ajuste estatístico
                # print("GARCH ERRO: Dados insuficientes (< 100)")
                return np.nan
            
            if returns_limpo.std() == 0: # Evita erro se a variância for zero
                # print("GARCH ERRO: Variância zero")
                return np.nan
            
            # 3. Definição e ajuste do modelo (ARMA(0,0) na média)
            if tipo_modelo == 'EGARCH':
                modelo = arch_model(returns_limpo, vol='EGARCH', p=1, q=1, rescale=False)
            else: # Default to GARCH
                modelo = arch_model(returns_limpo, vol='Garch', p=1, q=1, rescale=False)
            
            # Ajuste do modelo, suprimindo o output e warnings
            resultado = modelo.fit(disp='off', show_warning=False, options={'maxiter': 1000})
            
            if resultado is None or not resultado.params.any(): # Verifica se o ajuste foi bem-sucedido
                # print("GARCH ERRO: Falha no fit do modelo")
                return np.nan
            
            # 4. Previsão da Volatilidade
            # Previsão da variância (h.squared) para o próximo período (horizon=1)
            previsao = resultado.forecast(horizon=1)
            
            # Calcula a volatilidade diária (raiz quadrada da variância), desescalando (/100)
            volatilidade_diaria = np.sqrt(previsao.variance.values[-1, 0]) / 100 
            
            # 5. Validação da Saída e Anualização
            if np.isnan(volatilidade_diaria) or np.isinf(volatilidade_diaria) or volatilidade_diaria < 0:
                # print("GARCH ERRO: Volatilidade inválida (NaN/Inf)")
                return np.nan
            
            # Anualiza a volatilidade (multiplica por raiz de 252 dias úteis)
            return volatilidade_diaria * np.sqrt(252)
            
        except Exception as e:
            # print(f"Erro GARCH: {str(e)}")
            return np.nan

# =============================================================================
# CLASSE: MODELOS ESTATÍSTICOS DE SÉRIES TEMPORAIS
# =============================================================================

# Certifique-se de que as seguintes bibliotecas estão importadas no topo do seu arquivo:
# import numpy as np
# import pandas as pd
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.tsa.statespace.varmax import VARMAX
# from prophet import Prophet
# import logging 

class ModelosEstatisticos:
    """Modelos estatísticos para previsão de séries temporais financeiras"""
    
    @staticmethod
    def ajustar_arima(series: pd.Series, order: tuple = (1, 1, 1), horizon: int = 1) -> dict:
        """
        Ajusta modelo ARIMA e faz previsão.
        """
        try:
            series_limpa = series.dropna()
            
            if len(series_limpa) < 50:
                return {'forecast': np.nan, 'conf_int_lower': np.nan, 'conf_int_upper': np.nan}
            
            # Fit ARIMA model
            modelo = ARIMA(series_limpa, order=order)
            resultado = modelo.fit()
            
            # Forecast
            previsao = resultado.forecast(steps=horizon)
            conf_int = resultado.get_forecast(steps=horizon).conf_int()
            
            return {
                'forecast': previsao.iloc[-1] if hasattr(previsao, 'iloc') else previsao[-1],
                'conf_int_lower': conf_int.iloc[-1, 0] if hasattr(conf_int, 'iloc') else conf_int[-1, 0],
                'conf_int_upper': conf_int.iloc[-1, 1] if hasattr(conf_int, 'iloc') else conf_int[-1, 1],
                'model': 'ARIMA',
                'aic': resultado.aic,
                'bic': resultado.bic
            }
            
        except Exception as e:
            # print(f"Erro ARIMA: {str(e)}")
            return {'forecast': np.nan, 'conf_int_lower': np.nan, 'conf_int_upper': np.nan}
    
    @staticmethod
    def ajustar_sarima(series: pd.Series, order: tuple = (1, 1, 1), seasonal_order: tuple = (1, 1, 1, 12), horizon: int = 1) -> dict:
        """
        Ajusta modelo SARIMA (ARIMA com sazonalidade) e faz previsão.
        """
        try:
            series_limpa = series.dropna()
            
            if len(series_limpa) < 100:  # SARIMA needs more data
                return {'forecast': np.nan, 'conf_int_lower': np.nan, 'conf_int_upper': np.nan}
            
            # Fit SARIMA model
            modelo = SARIMAX(series_limpa, order=order, seasonal_order=seasonal_order)
            resultado = modelo.fit(disp=False, maxiter=100)
            
            # Forecast
            previsao = resultado.forecast(steps=horizon)
            conf_int = resultado.get_forecast(steps=horizon).conf_int()
            
            return {
                'forecast': previsao.iloc[-1] if hasattr(previsao, 'iloc') else previsao[-1],
                'conf_int_lower': conf_int.iloc[-1, 0] if hasattr(conf_int, 'iloc') else conf_int[-1, 0],
                'conf_int_upper': conf_int.iloc[-1, 1] if hasattr(conf_int, 'iloc') else conf_int[-1, 1],
                'model': 'SARIMA',
                'aic': resultado.aic,
                'bic': resultado.bic
            }
            
        except Exception as e:
            # print(f"Erro SARIMA: {str(e)}")
            return {'forecast': np.nan, 'conf_int_lower': np.nan, 'conf_int_upper': np.nan}
    
    @staticmethod
    def ajustar_var(dataframe_series: pd.DataFrame, maxlags: int = 5, horizon: int = 1) -> dict:
        """
        Ajusta modelo VAR (Vector Autoregression) para múltiplas séries.
        """
        try:
            df_limpo = dataframe_series.dropna()
            
            if len(df_limpo) < 100 or df_limpo.shape[1] < 2:
                return {'forecasts': {col: np.nan for col in dataframe_series.columns}}
            
            # Fit VAR model (usando VARMAX com ordem MA=0)
            modelo = VARMAX(df_limpo, order=(maxlags, 0))
            resultado = modelo.fit(disp=False, maxiter=100)
            
            # Forecast
            previsao = resultado.forecast(steps=horizon)
            
            forecasts = {}
            for i, col in enumerate(dataframe_series.columns):
                forecasts[col] = previsao.iloc[-1, i] if hasattr(previsao, 'iloc') else previsao[-1, i]
            
            return {
                'forecasts': forecasts,
                'model': 'VAR',
                'aic': resultado.aic,
                'bic': resultado.bic
            }
            
        except Exception as e:
            # print(f"Erro VAR: {str(e)}")
            return {'forecasts': {col: np.nan for col in dataframe_series.columns}}
    
    @staticmethod
    def ajustar_prophet(series: pd.Series, horizon: int = 30) -> dict:
        """
        Ajusta modelo Prophet (Facebook) para previsão de séries temporais.
        """
        try:
            # Prepare data for Prophet (requires 'ds' and 'y' columns)
            df_prophet = pd.DataFrame({
                'ds': series.index,
                'y': series.values
            })
            
            df_prophet = df_prophet.dropna()
            
            if len(df_prophet) < 50:
                return {'forecast': np.nan, 'trend': np.nan, 'yhat_lower': np.nan, 'yhat_upper': np.nan}
            
            # Fit Prophet model
            from prophet import Prophet
            import logging
            logging.getLogger('prophet').setLevel(logging.WARNING)
            
            modelo = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            
            modelo.fit(df_prophet)
            
            # Create future dataframe
            future = modelo.make_future_dataframe(periods=horizon)
            previsao = modelo.predict(future)
            
            # Get the last forecast value
            ultimo_forecast = previsao.iloc[-1]
            
            return {
                'forecast': ultimo_forecast['yhat'],
                'trend': ultimo_forecast['trend'],
                'yhat_lower': ultimo_forecast['yhat_lower'],
                'yhat_upper': ultimo_forecast['yhat_upper'],
                'model': 'Prophet',
                'components': previsao[['ds', 'yhat', 'trend', 'yhat_lower', 'yhat_upper']].tail(horizon)
            }
            
        except Exception as e:
            # print(f"Erro Prophet: {str(e)}")
            return {'forecast': np.nan, 'trend': np.nan, 'yhat_lower': np.nan, 'yhat_upper': np.nan}
    
    @staticmethod
    def ensemble_estatistico(series: pd.Series, horizon: int = 1) -> dict:
        """
        Cria ensemble de modelos estatísticos (ARIMA, SARIMA, Prophet) com ponderação
        baseada em critérios de informação (AIC) ou largura do intervalo de confiança.
        """
        try:
            previsoes = {}
            pesos = {}
            
            # ARIMA
            resultado_arima = ModelosEstatisticos.ajustar_arima(series, order=(1, 1, 1), horizon=horizon)
            if not np.isnan(resultado_arima['forecast']):
                previsoes['ARIMA'] = resultado_arima['forecast']
                # Ponderação pelo inverso do AIC (menor AIC é melhor)
                pesos['ARIMA'] = 1.0 / (resultado_arima.get('aic', 1000) + 1)
            
            # SARIMA (apenas se houver dados suficientes)
            if len(series.dropna()) >= 100:
                resultado_sarima = ModelosEstatisticos.ajustar_sarima(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), horizon=horizon)
                if not np.isnan(resultado_sarima['forecast']):
                    previsoes['SARIMA'] = resultado_sarima['forecast']
                    pesos['SARIMA'] = 1.0 / (resultado_sarima.get('aic', 1000) + 1)
            
            # Prophet
            resultado_prophet = ModelosEstatisticos.ajustar_prophet(series, horizon=horizon)
            if not np.isnan(resultado_prophet['forecast']):
                previsoes['Prophet'] = resultado_prophet['forecast']
                # Pondera por uma métrica proxy (inverso da largura do IC)
                conf_width = resultado_prophet.get('yhat_upper', resultado_prophet['forecast']) - resultado_prophet.get('yhat_lower', resultado_prophet['forecast'])
                pesos['Prophet'] = 1.0 / (conf_width + 1) if not np.isnan(conf_width) and conf_width > 0 else 1.0
            
            if not previsoes:
                return {'ensemble_forecast': np.nan, 'individual_forecasts': {}}
            
            # Normalizar pesos
            total_peso = sum(pesos.values())
            pesos_norm = {k: v / total_peso for k, v in pesos.items()}
            
            # Média ponderada
            forecast_ensemble = sum(previsoes[k] * pesos_norm[k] for k in previsoes.keys())
            
            return {
                'ensemble_forecast': forecast_ensemble,
                'individual_forecasts': previsoes,
                'weights': pesos_norm,
                'model': 'Ensemble Estatístico'
            }
            
        except Exception as e:
            # print(f"Erro Ensemble Estatístico: {str(e)}")
            return {'ensemble_forecast': np.nan, 'individual_forecasts': {}}

# =============================================================================
# CLASSE: ENSEMBLE DE MODELOS ML (REDUZIDA A FUNÇÕES DE OTIMIZAÇÃO)
# =============================================================================

class EnsembleML:
    """
    Mantida APENAS para encapsular a lógica de otimização Optuna para LightGBM e XGBoost (fallback).
    Os métodos treinar_ensemble e prever_ensemble_ponderado foram removidos para garantir a 
    velocidade e o uso de um modelo único no pipeline principal.
    """
    
    # --- Métodos de Otimização Optuna (Otimizados para velocidade: n_splits=2, n_trials=5) ---
    
    @staticmethod
    def _otimizar_xgboost(X, y):
        """Otimiza hiperparâmetros do XGBoost com Optuna."""
        from sklearn.model_selection import TimeSeriesSplit
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
            modelo = xgb.XGBClassifier(**params, random_state=42)
            scores = []
            tscv = TimeSeriesSplit(n_splits=2) # Otimizado para 2 folds
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2: continue
                try:
                    modelo.fit(X_train, y_train)
                    proba = modelo.predict_proba(X_val)[:, 1]
                    score = roc_auc_score(y_val, proba)
                    scores.append(score)
                except ValueError: continue
            return np.mean(scores) if scores else 0.0
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=5, show_progress_bar=False) # Otimizado para 5 trials
        return study.best_params

    @staticmethod
    def _otimizar_lightgbm(X, y):
        """Otimiza hiperparâmetros do LightGBM com Optuna."""
        from sklearn.model_selection import TimeSeriesSplit
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 400),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'objective': 'binary',
                'metric': 'auc'
            }
            modelo = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
            scores = []
            tscv = TimeSeriesSplit(n_splits=2) # Otimizado para 2 folds
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2: continue
                try:
                    modelo.fit(X_train, y_train)
                    proba = modelo.predict_proba(X_val)[:, 1]
                    score = roc_auc_score(y_val, proba)
                    scores.append(score)
                except ValueError: continue
            return np.mean(scores) if scores else 0.0
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=5, show_progress_bar=False) # Otimizado para 5 trials
        return study.best_params
# =============================================================================
# CLASSE: OTIMIZADOR DE PORTFÓLIO
# =============================================================================

# Certifique-se de que as seguintes bibliotecas e constantes estão importadas/definidas no topo:
# import numpy as np
# import pandas as pd
# from scipy.optimize import minimize
# TAXA_LIVRE_RISCO (e.g., 0.1075)
# PESO_MIN (e.g., 0.10)
# PESO_MAX (e.g., 0.30)

class OtimizadorPortfolioAvancado:
    """Otimização de portfólio com volatilidade GARCH e CVaR"""
    
    def __init__(self, returns_df: pd.DataFrame, garch_vols: dict = None, fundamental_data: pd.DataFrame = None, ml_predictions: pd.Series = None):
        """
        Inicializa o otimizador.
        
        Args:
            returns_df (pd.DataFrame): Retornos diários históricos dos ativos.
            garch_vols (dict): Volatilidades anuais previstas pelo GARCH/EGARCH.
            fundamental_data (pd.DataFrame): Dados fundamentalistas (para otimizações futuras).
            ml_predictions (pd.Series): Previsões de probabilidade de alta do Ensemble ML.
        """
        self.returns = returns_df
        self.mean_returns = returns_df.mean() * 252 # Retornos médios anualizados
        
        # 1. Construção da Matriz de Covariância (GARCH-baseada)
        if garch_vols is not None and garch_vols: # Verifica se garch_vols não é None ou vazio
            self.cov_matrix = self._construir_matriz_cov_garch(returns_df, garch_vols)
        else:
            self.cov_matrix = returns_df.cov() * 252 # Fallback: Covariância histórica
            print("  ⚠️ Volatilidades GARCH não disponíveis, usando covariância histórica.")
        
        self.num_ativos = len(returns_df.columns)
        self.fundamental_data = fundamental_data
        self.ml_predictions = ml_predictions

    def _construir_matriz_cov_garch(self, returns_df: pd.DataFrame, garch_vols: dict) -> pd.DataFrame:
        """Constrói matriz de covariância usando correlações históricas e volatilidades GARCH."""
        
        # Correlação histórica (assumimos que a correlação é mais estável que a volatilidade)
        corr_matrix = returns_df.corr()
        
        # Vetor de volatilidades anuais, com fallback para desvio padrão histórico
        vol_array = np.array([
            garch_vols.get(ativo, returns_df[ativo].std() * np.sqrt(252))
            for ativo in returns_df.columns
        ])
        
        # Validação de segurança para o vetor de volatilidade
        if np.isnan(vol_array).all() or np.all(vol_array <= 1e-9):
            print("  ⚠️ Volatilidades GARCH inválidas. Voltando para covariância histórica.")
            return returns_df.cov() * 252
            
        # Recombina: Covariância = Correlação * Vol(i) * Vol(j)
        cov_matrix = corr_matrix.values * np.outer(vol_array, vol_array)
        
        return pd.DataFrame(cov_matrix, index=returns_df.columns, columns=returns_df.columns)
    
    # --- Funções de Otimização ---

    def estatisticas_portfolio(self, pesos: np.ndarray) -> tuple[float, float]:
        """Calcula Retorno (anualizado) e Volatilidade (anualizada) do portfólio."""
        p_retorno = np.dot(pesos, self.mean_returns)
        # p_vol = np.sqrt(pesos.T @ self.cov_matrix @ pesos)
        p_vol = np.sqrt(np.dot(pesos.T, np.dot(self.cov_matrix, pesos)))
        return p_retorno, p_vol
    
    def sharpe_negativo(self, pesos: np.ndarray) -> float:
        """Objetivo: Minimizar o Sharpe Ratio Negativo (Equivale a Maximizar o Sharpe Ratio)."""
        p_retorno, p_vol = self.estatisticas_portfolio(pesos)
        
        if p_vol <= 1e-9: # Proteção contra divisão por zero/volatilidade nula
            return -100.0 
            
        return -(p_retorno - TAXA_LIVRE_RISCO) / p_vol
    
    def minimizar_volatilidade(self, pesos: np.ndarray) -> float:
        """Objetivo: Minimizar a Volatilidade do Portfólio."""
        return self.estatisticas_portfolio(pesos)[1]
    
    def calcular_cvar(self, pesos: np.ndarray, confidence: float = 0.95) -> float:
        """Calcula Conditional Value at Risk (CVaR) diário, NÃO anualizado."""
        portfolio_returns = self.returns @ pesos
        sorted_returns = np.sort(portfolio_returns)
        
        # O VaR é o retorno no nível (1-confidence) dos dados
        var_index = int(np.floor((1 - confidence) * len(sorted_returns)))
        var = sorted_returns[var_index]
        
        # CVaR é a média dos retornos abaixo do VaR
        cvar = sorted_returns[sorted_returns <= var].mean()
        # Nota: O CVaR é naturalmente um número negativo (perda).
        return cvar

    def cvar_negativo(self, pesos: np.ndarray, confidence: float = 0.95) -> float:
        """Objetivo: Minimizar o CVaR Negativo (Equivale a Minimizar o CVaR Positivo)."""
        # Minimizar -CVaR é o mesmo que maximizar o retorno da cauda de perda, ou seja, minimizar a perda média.
        return -self.calcular_cvar(pesos, confidence)

    # --- Execução da Otimização ---

    def otimizar(self, estrategia: str = 'MaxSharpe', confidence_level: float = 0.95) -> dict:
        """
        Executa otimização do portfólio para a estratégia selecionada.
        """
        if self.num_ativos == 0:
            return {}

        # 1. Restrições e Limites
        restricoes = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # Soma dos pesos = 1
        
        # Definir limites (Bounds)
        # Assume-se que PESO_MIN e PESO_MAX são constantes globais
        limites = tuple((PESO_MIN, PESO_MAX) for _ in range(self.num_ativos))
        
        # Chute Inicial (Pesos Iguais)
        chute_inicial = np.array([1.0 / self.num_ativos] * self.num_ativos)
        
        # Garantir que o chute inicial respeite os limites, se necessário
        min_w, max_w = limites[0]
        chute_inicial = np.clip(chute_inicial, min_w, max_w)
        chute_inicial /= np.sum(chute_inicial) # Re-normaliza após o clip
        
        # 2. Definição do Objetivo
        if estrategia == 'MinVolatility':
            objetivo = self.minimizar_volatilidade
        elif estrategia == 'CVaR':
            objetivo = lambda pesos: self.cvar_negativo(pesos, confidence=confidence_level)
        else: # Default: MaxSharpe
            objetivo = self.sharpe_negativo
        
        # 3. Execução
        try:
            resultado = minimize(
                objetivo,
                chute_inicial,
                method='SLSQP',
                bounds=limites,
                constraints=restricoes,
                options={'maxiter': 500, 'ftol': 1e-6} 
            )
            
            # 4. Processamento dos Resultados
            if resultado.success:
                final_weights = resultado.x / np.sum(resultado.x) # Normaliza para garantir soma = 1.0
                return {ativo: peso for ativo, peso in zip(self.returns.columns, final_weights)}
            else:
                print(f"  ✗ Otimização falhou ({estrategia}): {resultado.message}")
                # Fallback: pesos iguais
                return {ativo: 1.0 / self.num_ativos for ativo in self.returns.columns}
        
        except Exception as e:
            print(f"  ✗ Erro fatal na otimização ({estrategia}): {str(e)}")
            # Fallback: pesos iguais
            return {ativo: 1.0 / self.num_ativos for ativo in self.returns.columns}
    
# =============================================================================
# CLASSE PRINCIPAL: CONSTRUTOR DE PORTFÓLIO AUTOML (VERSÃO FINAL E CORRIGIDA)
# =============================================================================

class ConstrutorPortfolioAutoML:
    """
    Orquestrador principal para construção de portfólio AutoML
    Coordena coleta, modelagem (GARCH, ML, Est.) e otimização.
    """
    
    def __init__(self, valor_investimento: float, periodo: str = PERIODO_DADOS):
        self.valor_investimento = valor_investimento
        self.periodo = periodo
        
        # Estruturas de Dados
        self.dados_por_ativo = {}
        self.dados_fundamentalistas = pd.DataFrame()
        self.dados_performance = pd.DataFrame()
        self.volatilidades_garch = {}
        self.predicoes_ml = {}
        self.predicoes_estatisticas = {}
        self.ativos_sucesso = []
        self.dados_macro = {}
        self.metricas_performance = pd.DataFrame()
        
        # Resultados do Pipeline
        self.modelos_ml = {}
        self.auc_scores = {}
        self.ativos_selecionados = []
        self.alocacao_portfolio = {}
        self.metricas_portfolio = {}
        self.metodo_alocacao_atual = "Não Aplicado"
        self.justificativas_selecao = {}
        self.perfil_dashboard = {} 
        self.pesos_atuais = {}
        self.scores_combinados = pd.DataFrame()
        
    def coletar_e_processar_dados(self, simbolos: list) -> bool:
        """Coleta e processa dados de mercado com engenharia de features (via ColetorDados)"""
        
        coletor = ColetorDados(periodo=self.periodo)
        if not coletor.coletar_e_processar_dados(simbolos):
            return False
        
        self.dados_por_ativo = coletor.dados_por_ativo
        self.dados_fundamentalistas = coletor.dados_fundamentalistas
        self.ativos_sucesso = coletor.ativos_sucesso
        self.dados_macro = coletor.dados_macro
        self.dados_performance = coletor.metricas_performance
        
        print(f"\n✓ Coleta concluída: {len(self.ativos_sucesso)} ativos válidos\n")
        return True
    
    def calcular_volatilidades_garch(self):
        """Calcula volatilidades GARCH/EGARCH para todos os ativos, com fallback."""
        print("\n📊 Calculando volatilidades GARCH...")
        
        for simbolo in tqdm(self.ativos_sucesso, desc="Modelagem GARCH"):
            if simbolo not in self.dados_por_ativo or 'returns' not in self.dados_por_ativo[simbolo]:
                continue
                
            returns = self.dados_por_ativo[simbolo]['returns']
            
            # Tenta GARCH
            garch_vol = VolatilidadeGARCH.ajustar_garch(returns, tipo_modelo='GARCH')
            
            # Tenta EGARCH se GARCH falhar
            if np.isnan(garch_vol):
                garch_vol = VolatilidadeGARCH.ajustar_garch(returns, tipo_modelo='EGARCH')
            
            if np.isnan(garch_vol):
                # Fallback para volatilidade histórica anualizada
                garch_vol = returns.std() * np.sqrt(252) if not returns.isnull().all() and returns.std() > 0 else np.nan
                
            self.volatilidades_garch[simbolo] = garch_vol
        
        print(f"✓ Volatilidades GARCH calculadas para {len([k for k, v in self.volatilidades_garch.items() if not np.isnan(v)])} ativos válidos\n")
    
    def treinar_modelos_ensemble(self, dias_lookback_ml: int = LOOKBACK_ML, otimizar: bool = False):
        """
        Treina modelos ML e estatísticos, aplicando ensemble e governança, usando APENAS LightGBM 
        para o modelo de Machine Learning.
        """
        
        print("\n🤖 Treinando Modelo de Machine Learning Único (LightGBM)...")
        
        # Colunas de features 
        colunas_features_base = [col for df in self.dados_por_ativo.values() for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'log_returns']]
        colunas_features_base = sorted(list(set(colunas_features_base)))
        fundamental_features = list(self.dados_fundamentalistas.columns) if not self.dados_fundamentalistas.empty else []
        macro_features = list(self.dados_macro.keys()) if self.dados_macro else []
        colunas_features_totais = colunas_features_base + [f'fund_{f}' for f in fundamental_features] + [f'macro_{f.lower()}' for f in macro_features]

        self.predicoes_estatisticas = {}
        self.predicoes_ml = {}
        self.modelos_ml = {} # Limpar modelos de ensemble
        self.auc_scores = {} # Limpar scores de ensemble
        
        for simbolo in tqdm(self.ativos_sucesso, desc="Treinamento ML + Estatístico (LightGBM)"):
            if simbolo not in self.dados_por_ativo: continue
            df = self.dados_por_ativo[simbolo].copy()
            
            # 1. Preparação dos dados (integração Fundamentalista e Macro)
            if simbolo in self.dados_fundamentalistas.index:
                for f_fund in fundamental_features:
                    df[f'fund_{f_fund}'] = self.dados_fundamentalistas.loc[simbolo, f_fund]
            
            if self.dados_macro:
                for f_macro in macro_features:
                    if f_macro in self.dados_macro and not self.dados_macro[f_macro].empty:
                        df[f'macro_{f_macro.lower()}'] = self.dados_macro[f_macro].reindex(df.index, method='ffill')
            
            # 2. Modelos Estatísticos 
            if 'Close' in df.columns and len(df) >= 100:
                try:
                    close_series = df['Close']
                    resultado_estatistico = ModelosEstatisticos.ensemble_estatistico(close_series, horizon=dias_lookback_ml)
                    
                    self.predicoes_estatisticas[simbolo] = {
                        'forecast': resultado_estatistico.get('ensemble_forecast', np.nan),
                        'current_price': close_series.iloc[-1] if not close_series.empty else np.nan,
                        'predicted_direction': 1 if resultado_estatistico.get('ensemble_forecast', 0) > close_series.iloc[-1] else 0,
                    }
                except:
                    self.predicoes_estatisticas[simbolo] = {'forecast': np.nan, 'predicted_direction': 0.5}

            # 3. Modelagem de Machine Learning (APENAS LightGBM)
            
            # Cria target (previsão da direção do preço futuro)
            df['Future_Direction'] = np.where(
                df['Close'].pct_change(dias_lookback_ml).shift(-dias_lookback_ml) > 0,
                1, 0
            )
            
            features_para_treino = [f for f in colunas_features_totais if f in df.columns]
            df_treino = df[features_para_treino + ['Future_Direction']].dropna()
            
            if len(df_treino) < MIN_DIAS_HISTORICO or len(np.unique(df_treino['Future_Direction'])) < 2:
                self.predicoes_ml[simbolo] = {'predicted_proba_up': 0.5, 'auc_roc_score': np.nan}
                continue

            X = df_treino[features_para_treino]
            y = df_treino['Future_Direction']
            
            try:
                # 3.1. Configuração do LightGBM (Modelo Único)
                lgbm_model = lgb.LGBMClassifier(
                    n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, 
                    verbose=-1, objective='binary', metric='auc'
                )
                
                # Otimização Optuna (Se ativada)
                if otimizar:
                    # Usa o método _otimizar_lightgbm da classe EnsembleML (assumida como redefinida)
                    best_params = EnsembleML._otimizar_lightgbm(X, y)
                    lgbm_model = lgb.LGBMClassifier(**best_params, random_state=42, verbose=-1, objective='binary', metric='auc')
                    print(f"    ✓ {simbolo.replace('.SA', '')}: LGBM Otimizado por Optuna")

                # 3.2. Treinamento Final
                lgbm_model.fit(X, y)
                self.modelos_ml[simbolo] = lgbm_model

                # 3.3. Validação Cruzada (2 Folds para velocidade)
                tscv = TimeSeriesSplit(n_splits=2)
                auc_scores_cv = cross_val_score(
                    lgbm_model, X, y, cv=tscv, scoring='roc_auc', error_score='raise'
                )
                auc_score_cv = np.mean(auc_scores_cv) if len(auc_scores_cv) > 0 else 0.5
                self.auc_scores[simbolo] = {'lightgbm': auc_score_cv} 

                # 3.4. Previsão final
                last_features = df[features_para_treino].iloc[[-dias_lookback_ml]]
                proba_final = lgbm_model.predict_proba(last_features)[:, 1][0]
                
                # Armazenamento da Previsão ML
                self.predicoes_ml[simbolo] = {
                    'predicted_proba_up': proba_final,
                    'auc_roc_score': auc_score_cv,
                    'model_name': 'LightGBM Único'
                }

            except Exception as e:
                self.predicoes_ml[simbolo] = {'predicted_proba_up': 0.5, 'auc_roc_score': np.nan}

        print(f"✓ Modelo LightGBM treinado para {len(self.predicoes_ml)} ativos")

    def pontuar_e_selecionar_ativos(self, horizonte_tempo: str):
        """Pontua e ranqueia ativos usando sistema multi-fator (Perf, Fund, Tech, ML) e diversificação."""
        
        if horizonte_tempo == "CURTO PRAZO":
            WEIGHT_PERF, WEIGHT_FUND, WEIGHT_TECH = 0.40, 0.10, 0.20
        elif horizonte_tempo == "LONGO PRAZO":
            WEIGHT_PERF, WEIGHT_FUND, WEIGHT_TECH = 0.40, 0.50, 0.10
        else:
            WEIGHT_PERF, WEIGHT_FUND, WEIGHT_TECH = 0.40, 0.30, 0.30

        final_ml_weight = WEIGHT_ML
        total_non_ml_weight = WEIGHT_PERF + WEIGHT_FUND + WEIGHT_TECH
        
        scale_factor = (1.0 - final_ml_weight) / total_non_ml_weight if total_non_ml_weight > 0 else 0
        WEIGHT_PERF *= scale_factor
        WEIGHT_FUND *= scale_factor
        WEIGHT_TECH *= scale_factor

        self.pesos_atuais = {
            'Performance': WEIGHT_PERF,
            'Fundamentos': WEIGHT_FUND,
            'Técnicos': WEIGHT_TECH,
            'ML': final_ml_weight
        }
        
        combinado = self.dados_performance.join(self.dados_fundamentalistas, how='inner').copy()
        
        for asset in combinado.index:
            if asset in self.dados_por_ativo and 'rsi_14' in self.dados_por_ativo[asset].columns:
                df = self.dados_por_ativo[asset]
                combinado.loc[asset, 'rsi_current'] = df['rsi_14'].iloc[-1]
                combinado.loc[asset, 'macd_current'] = df['macd'].iloc[-1]
                combinado.loc[asset, 'bb_position_current'] = df['bb_position'].iloc[-1]

            if asset in self.predicoes_ml:
                ml_info = self.predicoes_ml[asset]
                combinado.loc[asset, 'ML_Proba'] = ml_info.get('predicted_proba_up', 0.5)
                combinado.loc[asset, 'ML_Confidence'] = ml_info.get('auc_roc_score', 0.5)

        scores = pd.DataFrame(index=combinado.index)
        
        scores['performance_score'] = EngenheiroFeatures._normalizar(combinado.get('sharpe', pd.Series(0, index=combinado.index)), maior_melhor=True) * WEIGHT_PERF
        
        pe_score = EngenheiroFeatures._normalizar(combinado.get('pe_ratio', pd.Series(combinado['pe_ratio'].median(), index=combinado.index)), maior_melhor=False)
        roe_score = EngenheiroFeatures._normalizar(combinado.get('roe', pd.Series(combinado['roe'].median(), index=combinado.index)), maior_melhor=True)
        scores['fundamental_score'] = (pe_score * 0.5 + roe_score * 0.5) * WEIGHT_FUND
        
        rsi_norm = EngenheiroFeatures._normalizar(combinado.get('rsi_current', pd.Series(50, index=combinado.index)), maior_melhor=False)
        macd_norm = EngenheiroFeatures._normalizar(combinado.get('macd_current', pd.Series(0, index=combinado.index)), maior_melhor=True)
        scores['technical_score'] = (rsi_norm * 0.5 + macd_norm * 0.5) * WEIGHT_TECH

        ml_proba_norm = EngenheiroFeatures._normalizar(combinado.get('ML_Proba', pd.Series(0.5, index=combinado.index)), maior_melhor=True)
        ml_confidence_norm = EngenheiroFeatures._normalizar(combinado.get('ML_Confidence', pd.Series(0.5, index=combinado.index)), maior_melhor=True)
        scores['ml_score_weighted'] = (ml_proba_norm * 0.6 + ml_confidence_norm * 0.4) * final_ml_weight
        
        scores['total_score'] = scores['performance_score'] + scores['fundamental_score'] + scores['technical_score'] + scores['ml_score_weighted']
        
        self.scores_combinados = scores.join(combinado).sort_values('total_score', ascending=False)
        
        ranked_assets = self.scores_combinados.index.tolist()
        final_portfolio = []
        selected_sectors = set()
        num_assets_to_select = min(NUM_ATIVOS_PORTFOLIO, len(ranked_assets))

        for asset in ranked_assets:
            sector = self.dados_fundamentalistas.loc[asset, 'sector'] if asset in self.dados_fundamentalistas.index and 'sector' in self.dados_fundamentalistas.columns else 'Unknown'
            
            if sector not in selected_sectors or len(final_portfolio) < num_assets_to_select:
                final_portfolio.append(asset)
                selected_sectors.add(sector)
            
            if len(final_portfolio) >= num_assets_to_select:
                break
        
        self.ativos_selecionados = final_portfolio
        return self.ativos_selecionados
        
    def otimizar_alocacao(self, nivel_risco: str):
        """Otimiza alocação de capital usando Markowitz/CVaR com volatilidades GARCH."""
        
        if not self.ativos_selecionados or len(self.ativos_selecionados) < 1:
            self.metodo_alocacao_atual = "ERRO: Ativos Insuficientes"
            return {}
        
        available_assets_returns = {s: self.dados_por_ativo[s]['returns']
                                    for s in self.ativos_selecionados if s in self.dados_por_ativo and 'returns' in self.dados_por_ativo[s]}
        
        final_returns_df = pd.DataFrame(available_assets_returns).dropna()
        
        if final_returns_df.shape[0] < 50:
            weights = {asset: 1.0 / len(self.ativos_selecionados) for asset in self.ativos_selecionados}
            self.metodo_alocacao_atual = 'PESOS IGUAIS (Dados insuficientes)'
            return self._formatar_alocacao(weights)

        garch_vols_filtered = {asset: self.volatilidades_garch.get(asset, final_returns_df[asset].std() * np.sqrt(252))
                               for asset in final_returns_df.columns}

        optimizer = OtimizadorPortfolioAvancado(final_returns_df, garch_vols=garch_vols_filtered)
        
        strategy = 'MaxSharpe' 
        if 'CONSERVADOR' in nivel_risco or 'INTERMEDIÁRIO' in nivel_risco:
            strategy = 'MinVolatility'
        elif 'AVANÇADO' in nivel_risco:
            strategy = 'CVaR' 
            
        weights = optimizer.otimizar(estrategia=strategy)
        self.metodo_alocacao_atual = f'{strategy} (GARCH/Histórico)'
        
        return self._formatar_alocacao(weights)
        
    def _formatar_alocacao(self, weights: dict) -> dict:
        """Formata os pesos em valores monetários e garante a normalização."""
        if not weights or sum(weights.values()) == 0:
            return {}
            
        total_weight = sum(weights.values())
        return {
            s: {
                'weight': w / total_weight,
                'amount': self.valor_investimento * (w / total_weight)
            }
            for s, w in weights.items() if s in self.ativos_selecionados
        }
    
    def calcular_metricas_portfolio(self):
        """Calcula métricas consolidadas do portfólio (Retorno, Vol, Sharpe, Max Drawdown)."""
        
        if not self.alocacao_portfolio: return {}
        
        weights_dict = {s: data['weight'] for s, data in self.alocacao_portfolio.items()}
        returns_df_raw = {s: self.dados_por_ativo[s]['returns'] 
                          for s in weights_dict.keys() if s in self.dados_por_ativo and 'returns' in self.dados_por_ativo[s]}
        
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
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_investment': self.valor_investimento
        }
        
        return self.metricas_portfolio
    
    def gerar_justificativas(self):
        """Gera justificativas textuais para seleção e performance dos ativos."""
        
        self.justificativas_selecao = {}
        for simbolo in self.ativos_selecionados:
            justification = []
            
            perf = self.dados_performance.loc[simbolo] if simbolo in self.dados_performance.index else {}
            justification.append(f"Perf: Sharpe {perf.get('sharpe', np.nan):.3f}, Retorno {perf.get('retorno_anual', np.nan)*100:.2f}%, Vol. {self.volatilidades_garch.get(simbolo, perf.get('volatilidade_anual', np.nan))*100:.2f}% (GARCH/Hist.)")
            
            fund = self.dados_fundamentalistas.loc[simbolo] if simbolo in self.dados_fundamentalistas.index else {}
            justification.append(f"Fund: P/L {fund.get('pe_ratio', np.nan):.2f}, ROE {fund.get('roe', np.nan):.2f}%")
            
            if simbolo in self.predicoes_ml:
                ml = self.predicoes_ml[simbolo]
                proba_up = ml.get('predicted_proba_up', 0.5)
                auc_score = ml.get('auc_roc_score', np.nan)
                auc_str = f"{auc_score:.3f}" if not pd.isna(auc_score) else "N/A"
                justification.append(f"ML: Prob. Alta {proba_up*100:.1f}% (AUC {auc_str})")
            
            if simbolo in self.predicoes_estatisticas:
                stat_pred = self.predicoes_estatisticas[simbolo]
                forecast_price = stat_pred.get('forecast')
                current_price = stat_pred.get('current_price')
                
                if forecast_price is not None and current_price is not None and not np.isnan(forecast_price) and current_price != 0:
                    pred_change_pct = ((forecast_price - current_price) / current_price) * 100
                    justification.append(f"Estatístico: Prev. Preço R${forecast_price:.2f} ({pred_change_pct:.2f}%)")
                
            self.justificativas_selecao[simbolo] = " | ".join(justification)
        
        return self.justificativas_selecao
        
    def executar_pipeline(self, simbolos_customizados: list, perfil_inputs: dict, otimizar_ml: bool = False) -> bool:
        """Executa pipeline completo: Coleta -> Modelagem -> Pontuação -> Otimização."""
        
        self.perfil_dashboard = perfil_inputs
        ml_lookback_days = perfil_inputs.get('ml_lookback_days', LOOKBACK_ML)
        nivel_risco = perfil_inputs.get('risk_level', 'MODERADO')
        horizonte_tempo = perfil_inputs.get('time_horizon', 'MÉDIO PRAZO')
        
        # 1. Coleta de dados
        if not self.coletar_e_processar_dados(simbolos_customizados): return False
        
        # 2. Volatilidades GARCH
        self.calcular_volatilidades_garch()
        
        # 3. Treinamento ML e Estatístico
        self.treinar_modelos_ensemble(dias_lookback_ml=ml_lookback_days, otimizar=otimizar_ml)
        
        # 4. Pontuação e seleção
        self.pontuar_e_selecionar_ativos(horizonte_tempo=horizonte_tempo)
        
        # 5. Otimização de alocação
        self.otimizar_alocacao(nivel_risco=nivel_risco)
        
        # 6. Métricas e Justificativas
        self.calcular_metricas_portfolio()
        self.gerar_justificativas()
        
        print("\n✅ Pipeline de Otimização AutoML Concluído!")
        
        return True

# =============================================================================
# CLASSE: ANALISADOR INDIVIDUAL DE ATIVOS
# =============================================================================

# Certifique-se de que as seguintes importações estão no topo do seu arquivo:
# import numpy as np
# import pandas as pd
# import ta
# from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, CCIIndicator
# from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
# from ta.volatility import BollingerBands, AverageTrueRange
# from ta.volume import ChaikinMoneyFlowIndicator, MFIIndicator, OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans

class AnalisadorIndividualAtivos:
    """Análise completa de ativos individuais com máximo de features"""
    
    @staticmethod
    def calcular_todos_indicadores_tecnicos(hist: pd.DataFrame) -> pd.DataFrame:
        """
        [ATUALIZAR] Redireciona para a função de engenharia de features simplificada 
        para consistência e velocidade em todas as abas.
        """
        # Chama a função de engenharia de features simplificada (o que era o EngenheiroFeatures)
        df = EngenheiroFeatures.calcular_indicadores_tecnicos(hist)
        
        # Adiciona o volume de volta (se EngenheiroFeatures não o fez)
        if 'Volume' not in df.columns and 'Volume' in hist.columns:
             df['Volume'] = hist['Volume'].reindex(df.index)
        
        # Renomeia colunas para os nomes esperados pela aba de análise individual do p_a
        # O p_a original usava 'RSI' e 'MACD', mas o complexo usa 'rsi_14' e 'macd'.
        # Mantenha os nomes do complexo para evitar refatorar a Aba 4 inteira.
        
        return df.dropna(axis=1, how='all')
    
    @staticmethod
    def calcular_features_fundamentalistas_expandidas(ticker_obj: yf.Ticker) -> dict:
        """Extrai o máximo de features fundamentalistas e de consenso (yfinance .info)."""
        info = ticker_obj.info
        
        features = {
            # Valuation
            'pe_ratio': info.get('trailingPE', np.nan),
            'forward_pe': info.get('forwardPE', np.nan),
            'peg_ratio': info.get('pegRatio', np.nan),
            'pb_ratio': info.get('priceToBook', np.nan),
            'ps_ratio': info.get('priceToSalesTrailing12Months', np.nan),
            'enterprise_value': info.get('enterpriseValue', np.nan),
            'ev_to_revenue': info.get('enterpriseToRevenue', np.nan),
            'ev_to_ebitda': info.get('enterpriseToEbitda', np.nan),
            
            # Rentabilidade (%)
            'profit_margin': info.get('profitMargins', np.nan) * 100 if info.get('profitMargins') is not None else np.nan,
            'operating_margin': info.get('operatingMargins', np.nan) * 100 if info.get('operatingMargins') is not None else np.nan,
            'gross_margin': info.get('grossMargins', np.nan) * 100 if info.get('grossMargins') is not None else np.nan,
            'roe': info.get('returnOnEquity', np.nan) * 100 if info.get('returnOnEquity') is not None else np.nan,
            'roa': info.get('returnOnAssets', np.nan) * 100 if info.get('returnOnAssets') is not None else np.nan,
            'roic': info.get('returnOnCapital', np.nan) * 100 if info.get('returnOnCapital') is not None else np.nan,
            
            # Dividendos
            'div_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') is not None else np.nan,
            'payout_ratio': info.get('payoutRatio', np.nan) * 100 if info.get('payoutRatio') is not None else np.nan,
            'five_year_avg_div_yield': info.get('fiveYearAvgDividendYield', np.nan),
            
            # Crescimento (%)
            'revenue_growth': info.get('revenueGrowth', np.nan) * 100 if info.get('revenueGrowth') is not None else np.nan,
            'earnings_growth': info.get('earningsGrowth', np.nan) * 100 if info.get('earningsGrowth') is not None else np.nan,
            'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth', np.nan) * 100 if info.get('earningsQuarterlyGrowth') is not None else np.nan,
            
            # Saúde Financeira
            'current_ratio': info.get('currentRatio', np.nan),
            'quick_ratio': info.get('quickRatio', np.nan),
            'debt_to_equity': info.get('debtToEquity', np.nan),
            'total_debt': info.get('totalDebt', np.nan),
            'total_cash': info.get('totalCash', np.nan),
            'free_cashflow': info.get('freeCashflow', np.nan),
            'operating_cashflow': info.get('operatingCashflow', np.nan),
            
            # Informações Gerais
            'market_cap': info.get('marketCap', np.nan),
            'beta': info.get('beta', np.nan),
            'shares_outstanding': info.get('sharesOutstanding', np.nan),
            'float_shares': info.get('floatShares', np.nan),
            'shares_short': info.get('sharesShort', np.nan),
            'short_ratio': info.get('shortRatio', np.nan),
            
            # Setor e Indústria
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            
            # Preço e Consenso
            'current_price': info.get('currentPrice', np.nan),
            'target_high_price': info.get('targetHighPrice', np.nan),
            'target_low_price': info.get('targetLowPrice', np.nan),
            'target_mean_price': info.get('targetMeanPrice', np.nan),
            'recommendation': info.get('recommendationKey', 'none')
        }
        
        # Limpeza final: Garante que todos os valores None/null do yfinance se tornem np.nan
        for key, value in features.items():
            if value is None:
                features[key] = np.nan
        
        return features
    
    @staticmethod
    def realizar_clusterizacao_pca(dados_ativos: pd.DataFrame, n_clusters: int = 5) -> tuple[pd.DataFrame | None, PCA | None, KMeans | None]:
        """
        Realiza clusterização K-means após redução de dimensionalidade com PCA.
        
        Args:
            dados_ativos (pd.DataFrame): DataFrame de ativos (índice = tickers) com features numéricas
                                         e fundamentalistas (já normalizadas se vierem do ColetorDados).
            n_clusters (int): Número de clusters para o K-means.
            
        Returns:
            tuple: (DataFrame de resultados PCA+Cluster, Objeto PCA, Objeto KMeans)
        """
        
        # 1. Pré-processamento
        features_numericas = dados_ativos.select_dtypes(include=[np.number]).copy()
        features_numericas = features_numericas.replace([np.inf, -np.inf], np.nan)
        
        # Imputação com mediana (para garantir que a escalagem funcione)
        for col in features_numericas.columns:
            if features_numericas[col].isnull().any():
                median_val = features_numericas[col].median()
                features_numericas[col] = features_numericas[col].fillna(median_val)
        
        # Remove colunas sem variância ou completamente vazias
        features_numericas = features_numericas.dropna(axis=1, how='all')
        features_numericas = features_numericas.loc[:, (features_numericas.std() > 1e-6)]

        if features_numericas.empty or len(features_numericas) < n_clusters:
            print(f"  ⚠️ Dados insuficientes para clustering: {len(features_numericas)} pontos.")
            return None, None, None
            
        # 2. Normalização (Padronização Z-score)
        scaler = StandardScaler()
        dados_normalizados = scaler.fit_transform(features_numericas)
        
        # 3. PCA (Redução de Dimensionalidade)
        n_pca_components = min(3, len(features_numericas.columns))
        pca = PCA(n_components=n_pca_components)
        componentes_pca = pca.fit_transform(dados_normalizados)
        
        # 4. K-means
        actual_n_clusters = min(n_clusters, len(features_numericas))
        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(componentes_pca) # Clusteriza nos componentes PCA
        
        # 5. Criação do DataFrame de Resultados
        resultado_pca = pd.DataFrame(
            componentes_pca,
            columns=[f'PC{i+1}' for i in range(componentes_pca.shape[1])],
            index=features_numericas.index
        )
        resultado_pca['Cluster'] = clusters
        
        return resultado_pca, pca, kmeans
    
# =============================================================================
# INTERFACE STREAMLIT - REESTRUTURADA COM 5 ABAS
# =============================================================================

def configurar_pagina():
    """Configura página Streamlit"""
    st.set_page_config(
        page_title="Portfolio AutoML Elite",
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
            margin-bottom: 20px; /* Add some space below the header */
        }
        html, body, [class*="st-"] {
            font-family: 'Times New Roman', serif;
        }
        .stButton button {
            border: 1px solid #2c3e50;
            color: #2c3e50;
            border-radius: 4px;
            padding: 8px 16px; /* Slightly larger padding */
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
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); /* Subtle shadow */
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px; /* Space between tabs */
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px; /* Make tabs taller */
            white-space: pre-wrap;
            background-color: #f8f9fa; /* Light background for inactive tabs */
            border-radius: 4px 4px 0 0;
            gap: 1px;
            padding-top: 10px; /* Vertical padding */
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ffffff; /* White background for active tab */
            border-top: 2px solid #2c3e50; /* Highlight active tab */
            border-left: 1px solid #e0e0e0;
            border-right: 1px solid #e0e0e0;
        }
        .stTabs [aria-selected="true"] span {
            font-weight: bold; /* Bold text for active tab */
        }
        .stMetric { /* Style for metrics */
            padding: 10px 15px;
            background-color: #ffffff;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
            margin-bottom: 10px;
        }
        .stMetric label { font-weight: bold; color: #555; }
        .stMetric delta { font-weight: bold; color: #28a745; } /* Example color for positive delta */
        .stMetric delta[style*="color: red"] { color: #dc3545 !important; } /* Example for negative delta */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            color: #2c3e50; /* Dark blue for headers */
            font-weight: bold;
        }
        .stMarkdown ul { padding-left: 20px; }
        .stMarkdown li { margin-bottom: 8px; }

        /* Alerts for Governance Tab */
        .alert-success { background-color: #d4edda; border-color: #c3e6cb; color: #155724; padding: .75rem 1.25rem; margin-bottom: 1rem; border: 1px solid transparent; border-radius: .25rem; }
        .alert-warning { background-color: #fff3cd; border-color: #ffeeba; color: #856404; padding: .75rem 1.25rem; margin-bottom: 1rem; border: 1px solid transparent; border-radius: .25rem; }
        .alert-error { background-color: #f8d7da; border-color: #f5c6cb; color: #721c24; padding: .75rem 1.25rem; margin-bottom: 1rem; border: 1px solid transparent; border-radius: .25rem; }
        .alert-info { background-color: #d1ecf1; border-color: #bee5eb; color: #0c5460; padding: .75rem 1.25rem; margin-bottom: 1rem; border: 1px solid transparent; border-radius: .25rem; }
        </style>
    """, unsafe_allow_html=True)

def aba_introducao():
    """Aba 1: Introdução e Metodologia"""
    
    st.markdown("## 📚 Bem-vindo ao Sistema AutoML de Otimização de Portfólio")
    
    st.markdown("""
    <div class="info-box">
    <h3>🎯 O que este sistema faz?</h3>
    <p>Este é um sistema avançado de construção e otimização de portfólios de investimento que utiliza 
    <strong>Machine Learning</strong>, <strong>modelagem estatística</strong> e <strong>teoria moderna de portfólio</strong> 
    para criar carteiras personalizadas baseadas no seu perfil de risco e objetivos.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔬 Metodologia Científica")
        st.markdown("""
        **1. Análise de Perfil do Investidor**
        - Questionário baseado em normas CVM
        - Avaliação de tolerância ao risco
        - Definição de horizonte temporal
        - Análise de experiência e conhecimento
        
        **2. Coleta e Processamento de Dados**
        - Dados históricos de preços (máximo disponível)
        - Indicadores técnicos (30+ indicadores com a lib `ta`)
        - Fundamentos financeiros (20+ métricas expandidas)
        - Dados macroeconômicos (correlações)
        - Volume e liquidez
        
        **3. Engenharia de Features**
        - Indicadores técnicos avançados: RSI, MACD, Bollinger, Stochastic, ADX, ATR, CCI, Williams %R, OBV, MFI, Ichimoku, Keltner, Donchian
        - Indicadores fundamentalistas detalhados
        - Smart Beta Factors: Qualidade, Valor, Momentum
        - Modelagem de volatilidade (GARCH/EGARCH) e correlações
        - Lags e estatísticas rolling de preço e volume
        - Codificação temporal (dia da semana, mês, etc.)
        """)
    
    with col2:
        st.markdown("### 🤖 Tecnologias Utilizadas")
        st.markdown("""
        **Machine Learning Ensemble**
        - XGBoost, LightGBM, CatBoost, Random Forest, Extra Trees, KNN, SVC, Logistic Regression, Gaussian Naive Bayes
        - Otimização Optuna (Opcional): Hyperparameter tuning automático
        - Ponderação por AUC-ROC: Ensemble inteligente
        
        **Modelagem Estatística e Séries Temporais**
        - ARIMA, SARIMA, VAR, Prophet
        - Ensemble de modelos estatísticos para previsão de preços
        
        **Modelagem de Volatilidade**
        - GARCH(1,1) / EGARCH: Modelagem de volatilidade condicional
        - Previsão de volatilidade futura
        
        **Otimização de Portfólio**
        - Teoria de Markowitz: Fronteira eficiente com GARCH
        - Maximização de Sharpe Ratio
        - Minimização de volatilidade
        - Otimização de CVaR (Conditional Value at Risk)
        - Restrições de peso (10-30% por ativo)
        
        **Governança de Modelo**
        - Monitoramento de AUC-ROC, Precision, Recall, F1-Score
        - Alertas de degradação e drift
        """)
    
    st.markdown("---")
    
    st.markdown("### 📊 Como Funciona a Seleção dos 5 Ativos?")
    
    st.markdown("""
    <div class="info-box">
    <h4>Sistema de Pontuação Multi-Fator Adaptativo</h4>
    <p>O sistema utiliza um <strong>score composto</strong> que combina múltiplas dimensões de análise, com <strong>ponderações adaptativas</strong> ao seu perfil:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        **📈 Score de Performance (até 40%)**
        - Sharpe Ratio histórico
        - Retorno anualizado ajustado ao risco
        - Drawdown máximo
        """)
    
    with col2:
        st.markdown("""
        **💼 Score Fundamentalista (até 50%)**
        - Qualidade: ROE, margens, ROIC
        - Valor: P/L, P/VP baixos
        - Crescimento: Receita, Lucros
        - Saúde financeira: Dívida/Patrimônio, Liquidez
        """)
    
    with col3:
        st.markdown("""
        **🔧 Score Técnico (até 50%)**
        - Indicadores de Momentum (MACD, RSI)
        - Volatilidade (Bandas de Bollinger, ATR)
        - Tendência (ADX, Médias Móveis)
        - Padrões de preço
        """)
    
    with col4:
        st.markdown("""
        **🤖 Score de Machine Learning (até 30%)**
        - Probabilidade de alta prevista pelo ensemble ponderado por AUC
        - Confiança do modelo (AUC-ROC médio)
        - Validação cruzada temporal
        """)
    
    st.markdown("---")
    
    st.markdown("### ⚖️ Ponderação Adaptativa por Perfil")
    
    perfil_table = pd.DataFrame({
        'Perfil': ['Conservador', 'Intermediário', 'Moderado', 'Moderado-Arrojado', 'Avançado'],
        'Horizonte': ['Longo Prazo', 'Longo Prazo', 'Médio Prazo', 'Curto Prazo', 'Curto Prazo'],
        'Performance': ['40%', '40%', '40%', '40%', '40%'],
        'Fundamentos': ['50%', '40%', '30%', '20%', '10%'],
        'Técnicos': ['10%', '20%', '30%', '40%', '50%'],
        'ML': ['30%', '30%', '30%', '30%', '30%'], # ML weight is constant
        'Foco': ['Qualidade e Estabilidade', 'Equilíbrio com Foco em Fundamentos', 'Equilíbrio Geral', 'Momentum e Curto Prazo', 'Visão de Curto Prazo e Momentum']
    })
    
    st.table(perfil_table)
    
    st.markdown("---")
    
    st.markdown("### 🎯 Diversificação Setorial e de Risco")
    
    st.markdown("""
    <div class="info-box">
    <p>O sistema garante <strong>diversificação</strong> e <strong>gerenciamento de risco</strong>:</p>
    <ul>
        <li>Máximo de 2 ativos por setor (quando possível)</li>
        <li>Prioriza ativos de setores diferentes</li>
        <li>Reduz risco de concentração</li>
        <li>Modelagem de volatilidade GARCH e otimização de CVaR para robustez</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🚀 Próximos Passos")
    
    st.info("""
    **Navegue pelas abas acima para:**
    1. **Seleção de Ativos**: Escolha entre Ibovespa, setores específicos, número fixo ou todos os ativos disponíveis.
    2. **Construtor de Portfólio**: Responda o questionário para definir seu perfil e gerar seu portfólio otimizado.
    3. **Análise Individual**: Explore ativos isolados com análise técnica e fundamentalista completa, incluindo M.L. e Clusterização.
    4. **Governança de Modelo**: Monitore a performance dos modelos ML e receba alertas.
    """)

def aba_selecao_ativos():
    """Aba 2: Seleção de Ativos - Enhanced with 4 selection modes"""
    
    st.markdown("## 🎯 Seleção de Ativos para Análise")
    
    st.markdown("""
    <div class="info-box">
    <p>Escolha quais ativos você deseja incluir na análise. O sistema irá avaliar todos os ativos selecionados 
    e escolher os <strong>5 melhores</strong> baseado no seu perfil de risco e nos scores multi-fator.</p>
    </div>
    """, unsafe_allow_html=True)
    
    modo_selecao = st.radio(
        "**Modo de Seleção:**",
        [
            "📊 Ibovespa Completo (82 ativos)",
            "🌐 B3 Completa (259 ativos)",
            "🏢 Setores Específicos",
            "✍️ Digitar Ativos Manualmente"
        ],
        index=0
    )
    
    ativos_selecionados = []
    
    if "Ibovespa Completo" in modo_selecao:
        ativos_selecionados = ATIVOS_IBOVESPA.copy()
        
        st.success(f"✓ **{len(ativos_selecionados)} ativos do Ibovespa** selecionados")
        
        with st.expander("📋 Ver lista completa do Ibovespa"):
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
    
    elif "Lista Completa" in modo_selecao:
        ativos_selecionados = TODOS_ATIVOS.copy()
        
        st.success(f"✓ **{len(ativos_selecionados)} ativos** selecionados de todos os setores")
        
        with st.expander("📊 Distribuição por Setor"):
            setor_counts = {}
            for setor, ativos in ATIVOS_POR_SETOR.items():
                setor_counts[setor] = len(ativos)
            
            df_setores = pd.DataFrame({
                'Setor': list(setor_counts.keys()),
                'Quantidade': list(setor_counts.values())
            }).sort_values('Quantidade', ascending=False)
            
            fig = px.bar(
                df_setores,
                x='Setor',
                y='Quantidade',
                title='Ativos por Setor'
            )
            fig.update_layout(
                **obter_template_grafico(),
                xaxis_tickangle=-45,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif "Setores Específicos" in modo_selecao:
        st.markdown("### 🏢 Selecione os Setores")
        
        setores_disponiveis = list(ATIVOS_POR_SETOR.keys())
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            setores_selecionados = st.multiselect(
                "Escolha um ou mais setores:",
                options=setores_disponiveis,
                default=setores_disponiveis[:3] if setores_disponiveis else [],
                help="Selecione os setores que deseja incluir na análise"
            )
        
        if setores_selecionados:
            for setor in setores_selecionados:
                ativos_selecionados.extend(ATIVOS_POR_SETOR[setor])
            
            with col2:
                st.metric("Setores Selecionados", len(setores_selecionados))
                st.metric("Total de Ativos", len(ativos_selecionados))
            
            with st.expander("📋 Ver ativos por setor"):
                for setor in setores_selecionados:
                    st.markdown(f"**{setor}** ({len(ATIVOS_POR_SETOR[setor])} ativos)")
                    ativos_setor = [a.replace('.SA', '') for a in ATIVOS_POR_SETOR[setor]]
                    st.write(", ".join(ativos_setor))
        else:
            st.warning("⚠️ Selecione pelo menos um setor")
    
    elif "Digitar Ativos" in modo_selecao:
        st.markdown("### ✍️ Digite os Ativos Manualmente")
        
        st.info("💡 **Dica**: Você pode pesquisar ativos digitando parte do nome ou código na lista suspensa.")
        
        # Create a comprehensive list of all available assets with their sectors
        ativos_com_setor = {}
        for setor, ativos in ATIVOS_POR_SETOR.items():
            for ativo in ativos:
                ativos_com_setor[ativo] = setor
        
        # All available tickers for selection
        todos_tickers = sorted(list(ativos_com_setor.keys()))
        todos_tickers_display = [f"{t.replace('.SA', '')} ({ativos_com_setor[t]})" for t in todos_tickers]
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("#### 📝 Selecione Ativos da Lista")
            ativos_da_lista = st.multiselect(
                "Pesquise e selecione ativos:",
                options=todos_tickers,
                format_func=lambda x: f"{x.replace('.SA', '')} - {ativos_com_setor.get(x, 'Desconhecido')}",
                help="Digite para pesquisar. Ex: 'PETR', 'Vale', etc."
            )
        
        with col2:
            st.metric("Ativos Selecionados", len(ativos_da_lista))
        
        st.markdown("---")
        st.markdown("#### ✏️ Ou Digite Novos Ativos")
        
        col3, col4 = st.columns(2)
        
        with col3:
            novos_ativos_input = st.text_area(
                "Digite os códigos dos ativos (um por linha):",
                height=150,
                placeholder="PETR4\nVALE3\nITUB4\n...",
                help="Digite os códigos sem o '.SA'. Um código por linha."
            )
        
        with col4:
            setores_disponiveis_manual = ["Selecione o setor..."] + list(ATIVOS_POR_SETOR.keys())
            
            setor_novos_ativos = st.selectbox(
                "Setor dos novos ativos:",
                options=setores_disponiveis_manual,
                help="Selecione o setor ao qual os ativos digitados pertencem"
            )
            
            st.markdown("**Ou digite um novo setor:**")
            setor_customizado = st.text_input(
                "Setor personalizado:",
                placeholder="Ex: Tecnologia, Saúde, etc.",
                help="Digite um nome de setor se não estiver na lista"
            )
        
        # Process manual inputs
        novos_ativos = []
        if novos_ativos_input.strip():
            linhas = novos_ativos_input.strip().split('\n')
            for linha in linhas:
                ticker = linha.strip().upper()
                if ticker:
                    # Add .SA if not present
                    if not ticker.endswith('.SA'):
                        ticker = f"{ticker}.SA"
                    novos_ativos.append(ticker)
        
        # Combine selected and manually entered assets
        ativos_selecionados = list(set(ativos_da_lista + novos_ativos))
        
        if ativos_selecionados:
            st.success(f"✓ **{len(ativos_selecionados)} ativos** selecionados")
            
            # Display selected assets with their sectors
            with st.expander("📋 Ver ativos selecionados"):
                df_selecionados = pd.DataFrame({
                    'Ticker': [a.replace('.SA', '') for a in ativos_selecionados],
                    'Código Completo': ativos_selecionados,
                    'Setor': [ativos_com_setor.get(a, setor_customizado or setor_novos_ativos or 'Não especificado') 
                             for a in ativos_selecionados]
                })
                st.dataframe(df_selecionados, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ Nenhum ativo selecionado. Use a lista suspensa ou digite manualmente.")
    
    # Save selection to session state
    if ativos_selecionados:
        st.session_state.ativos_para_analise = ativos_selecionados
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("✓ Ativos Selecionados", len(ativos_selecionados))
        col2.metric("→ Serão Avaliados", len(ativos_selecionados))
        col3.metric("→ Portfólio Final", NUM_ATIVOS_PORTFOLIO)
        
        st.success("✓ Seleção confirmada! Vá para a aba **'Construtor de Portfólio'** para continuar.")
    else:
        st.warning("⚠️ Nenhum ativo selecionado. Por favor, faça uma seleção.")

# =============================================================================
# Aba 3: Questionário e Construção de Portfólio (Corrigida)
# =============================================================================

def aba_construtor_portfolio():
    """Aba 3: Questionário e Construção de Portfólio"""
    
    if 'ativos_para_analise' not in st.session_state or not st.session_state.ativos_para_analise:
        st.warning("⚠️ Por favor, selecione os ativos na aba **'Seleção de Ativos'** primeiro.")
        return
    
    # Initialize builder and profile in session state if they don't exist
    if 'builder' not in st.session_state:
        st.session_state.builder = None
    if 'profile' not in st.session_state:
        st.session_state.profile = {}
    if 'builder_complete' not in st.session_state:
        st.session_state.builder_complete = False
    
    # FASE 1: QUESTIONÁRIO
    if not st.session_state.builder_complete:
        st.markdown('## 📋 Questionário de Perfil do Investidor')
        
        st.info(f"✓ {len(st.session_state.ativos_para_analise)} ativos selecionados para análise")
        
        # Use columns for better layout of questions
        col_question1, col_question2 = st.columns(2)
        
        with st.form("investor_profile_form"):
            options_score = [
                'CT: Concordo Totalmente',
                'C: Concordo',
                'N: Neutro',
                'D: Discordo',
                'DT: Discordo Totalmente'
            ]
            options_reaction = ['A: Venderia', 'B: Manteria', 'C: Compraria mais']
            options_level_abc = ['A: Avançado', 'B: Intermediário', 'C: Iniciante']
            options_time_horizon = [
                'A: Curto (até 1 ano)',
                'B: Médio (1-5 anos)',
                'C: Longo (5+ anos)'
            ]
            options_liquidez = [
                'A: Menos de 6 meses',
                'B: Entre 6 meses e 2 anos',
                'C: Mais de 2 anos'
            ]
            
            with col_question1:
                st.markdown("#### Tolerância ao Risco")
                p2_risk = st.radio(
                    "**1. Aceito risco de curto prazo por retorno de longo prazo**",
                    options=options_score,
                    index=2, key='risk_accept_radio'
                )
                p3_gain = st.radio(
                    "**2. Ganhar o máximo é minha prioridade, mesmo com risco**",
                    options=options_score,
                    index=2, key='max_gain_radio'
                )
                p4_stable = st.radio(
                    "**3. Prefiro crescimento constante, sem volatilidade**",
                    options=options_score,
                    index=2, key='stable_growth_radio'
                )
                p5_loss = st.radio(
                    "**4. Evitar perdas é mais importante que crescimento**",
                    options=options_score,
                    index=2, key='avoid_loss_radio'
                )
                p511_reaction = st.radio(
                    "**5. Se meus investimentos caíssem 10%, eu:**",
                    options=options_reaction,
                    index=1, key='reaction_radio'
                )
                p_level = st.radio(
                    "**6. Meu nível de conhecimento em investimentos:**",
                    options=options_level_abc,
                    index=1, key='level_radio'
                )
            
            with col_question2:
                st.markdown("#### Horizonte Temporal e Capital")
                p211_time = st.radio(
                    "**7. Prazo máximo para reavaliação de estratégia:**",
                    options=options_time_horizon,
                    index=2, key='time_purpose_radio'
                )[0] # Get the key (A, B, C)
                
                p311_liquid = st.radio(
                    "**8. Necessidade de liquidez (prazo mínimo para resgate):**",
                    options=options_liquidez,
                    index=2, key='liquidity_radio'
                )[0] # Get the key (A, B, C)
                
                st.markdown("---")
                investment = st.number_input(
                    "Valor de Investimento (R$)",
                    min_value=1000,
                    max_value=10000000,
                    value=100000,
                    step=10000,
                    key='investment_amount'
                )
            
            # Opções avançadas
            with st.expander("Opções Avançadas"):
                # Otimização Optuna é para o LightGBM (único modelo agora)
                otimizar_ml = st.checkbox("Ativar otimização Optuna (mais lento, melhor precisão)", value=False, key='optimize_ml_checkbox')
            
            submitted = st.form_submit_button("🚀 Gerar Portfólio Otimizado", type="primary")
            
            if submitted:
                # 1. Analisa perfil
                risk_answers = {
                    'risk_accept': p2_risk,
                    'max_gain': p3_gain,
                    'stable_growth': p4_stable,
                    'avoid_loss': p5_loss,
                    'reaction': p511_reaction,
                    'level': p_level,
                    'time_purpose': p211_time,
                    'liquidity': p311_liquid
                }
                
                analyzer = AnalisadorPerfilInvestidor()
                risk_level, horizon, lookback, score = analyzer.calcular_perfil(risk_answers)
                
                st.session_state.profile = {
                    'risk_level': risk_level,
                    'time_horizon': horizon,
                    'ml_lookback_days': lookback,
                    'risk_score': score
                }
                
                # 2. Cria construtor
                try:
                    # Cria a instância localmente e armazena em session_state
                    builder_local = ConstrutorPortfolioAutoML(investment)
                    st.session_state.builder = builder_local
                except Exception as e:
                    st.error(f"Erro fatal ao inicializar o construtor do portfólio: {e}")
                    return

                # 3. Executa pipeline
                with st.spinner(f'Criando portfólio para **PERFIL {risk_level}** ({horizon})...'):
                    try:
                        # CHAMA O MÉTODO USANDO A VARIÁVEL LOCAL ONDE O OBJETO FOI CRIADO
                        success = builder_local.executar_pipeline(
                            simbolos_customizados=st.session_state.ativos_para_analise,
                            perfil_inputs=st.session_state.profile,
                            otimizar_ml=otimizar_ml
                        )
                    except AttributeError as e:
                        st.error(f"Erro de Atributo: O objeto ConstrutorPortfolioAutoML não tem o método 'executar_pipeline'. Erro: {e}")
                        st.error("Verifique se a classe ConstrutorPortfolioAutoML foi definida corretamente.")
                        return
                    
                    if not success:
                        st.error("Falha ao coletar dados suficientes ou processar os ativos. Tente novamente com uma seleção diferente de ativos ou verifique sua conexão.")
                        # Limpa o estado para permitir uma nova tentativa
                        st.session_state.builder = None
                        st.session_state.profile = {}
                        return
                    
                    st.session_state.builder_complete = True
                    st.rerun() # Rerun the app to show results
    
    # FASE 2: RESULTADOS
    else:
        builder = st.session_state.builder
        # Adiciona verificação de segurança, embora o erro tenha ocorrido na fase 1
        if builder is None:
            st.error("Objeto construtor não encontrado. Recomece a análise.")
            st.session_state.builder_complete = False
            return
            
        profile = st.session_state.profile
        assets = builder.ativos_selecionados
        allocation = builder.alocacao_portfolio
        
        st.markdown('## ✅ Portfólio Otimizado Gerado')
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Perfil de Risco", profile.get('risk_level', 'N/A'), f"Score: {profile.get('risk_score', 'N/A')}")
        col2.metric("Horizonte", profile.get('time_horizon', 'N/A'))
        col3.metric("Sharpe Ratio", f"{builder.metricas_portfolio.get('sharpe_ratio', 0):.3f}")
        col4.metric("Estratégia", builder.metodo_alocacao_atual.split('(')[0].strip())
        
        # Button to restart analysis
        if st.button("🔄 Recomeçar Análise", key='recomecar_analysis'):
            # Clear relevant session state variables to restart
            st.session_state.builder_complete = False
            st.session_state.builder = None
            st.session_state.profile = {}
            st.session_state.ativos_para_analise = [] 
            st.rerun()
        
        st.markdown("---")
        
        # Dashboard de resultados (código de exibição inalterado)
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Alocação", "📈 Performance", "🔬 Análise ML", "📉 Volatilidade GARCH", "❓ Justificativas"
        ])
        
        with tab1:
            col_alloc, col_table = st.columns([1, 2])
            
            with col_alloc:
                st.markdown('#### Alocação de Capital')
                alloc_data = pd.DataFrame([
                    {'Ativo': a, 'Peso (%)': allocation[a]['weight'] * 100}
                    for a in assets if a in allocation and allocation[a]['weight'] > 0.001
                ])
                
                if not alloc_data.empty:
                    fig_alloc = px.pie(
                        alloc_data,
                        values='Peso (%)',
                        names='Ativo',
                        hole=0.3
                    )
                    fig_layout = obter_template_grafico()
                    fig_layout['title']['text'] = "Distribuição do Portfólio"
                    fig_alloc.update_layout(**fig_layout)
                    st.plotly_chart(fig_alloc, use_container_width=True)
                else:
                    st.warning("Nenhuma alocação significativa para exibir no gráfico de pizza.")
            
            with col_table:
                st.markdown('#### Detalhamento dos Ativos')
                
                alloc_table = []
                for asset in assets:
                    if asset in allocation and allocation[asset]['weight'] > 0:
                        weight = allocation[asset]['weight']
                        amount = allocation[asset]['amount']
                        sector = builder.dados_fundamentalistas.loc[asset, 'sector'] if asset in builder.dados_fundamentalistas.index and 'sector' in builder.dados_fundamentalistas.columns else 'Unknown'
                        ml_info = builder.predicoes_ml.get(asset, {})
                        stat_info = builder.predicoes_estatisticas.get(asset, {})
                        
                        alloc_table.append({
                            'Ativo': asset.replace('.SA', ''), 
                            'Setor': sector,
                            'Peso (%)': f"{weight * 100:.2f}",
                            'Valor (R$)': f"R$ {amount:,.2f}",
                            'ML Prob. Alta (%)': f"{ml_info.get('predicted_proba_up', 0.5)*100:.1f}",
                            'ML AUC': f"{ml_info.get('auc_roc_score', 0):.3f}" if not pd.isna(ml_info.get('auc_roc_score')) else "N/A",
                            'Estatístico Dir.': f"{stat_info.get('predicted_direction', 0.5)*100:.0f}%" if stat_info.get('predicted_direction') is not None else "N/A",
                            'Estatístico Prev.': f"R$ {stat_info.get('forecast', np.nan):,.2f}" if not np.isnan(stat_info.get('forecast', np.nan)) else "N/A"
                        })
                
                df_alloc = pd.DataFrame(alloc_table)
                st.dataframe(df_alloc, use_container_width=True, hide_index=True)
        
        with tab2:
            st.markdown('#### Métricas de Performance do Portfólio')
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Retorno Anual", f"{builder.metricas_portfolio.get('annual_return', 0)*100:.2f}%")
            col2.metric("Volatilidade Anual", f"{builder.metricas_portfolio.get('annual_volatility', 0)*100:.2f}%")
            col3.metric("Sharpe Ratio", f"{builder.metricas_portfolio.get('sharpe_ratio', 0):.3f}")
            col4.metric("Max Drawdown", f"{builder.metricas_portfolio.get('max_drawdown', 0)*100:.2f}%")
            
            st.markdown("---")
            st.markdown('#### Evolução dos Retornos Cumulativos dos Ativos')
            
            fig_cum = go.Figure()
            
            for asset in assets:
                if asset in builder.dados_por_ativo and 'returns' in builder.dados_por_ativo[asset]:
                    returns = builder.dados_por_ativo[asset]['returns']
                    cum_returns = (1 + returns).cumprod()
                    
                    fig_cum.add_trace(go.Scatter(
                        x=cum_returns.index,
                        y=cum_returns.values,
                        name=asset.replace('.SA', ''),
                        mode='lines'
                    ))
            
            fig_layout = obter_template_grafico()
            fig_layout['title']['text'] = "Evolução dos Retornos Cumulativos dos Ativos Selecionados"
            fig_layout['yaxis']['title'] = "Retorno Acumulado (Base 1)"
            fig_layout['xaxis']['title'] = "Data"
            fig_cum.update_layout(**fig_layout, height=500)
            
            st.plotly_chart(fig_cum, use_container_width=True)
        
        with tab3:
            st.markdown('#### Análise de Machine Learning')
            
            ml_data = []
            for asset in assets:
                if asset in builder.predicoes_ml:
                    ml_info = builder.predicoes_ml[asset]
                    ml_data.append({
                        'Ativo': asset.replace('.SA', ''),
                        'Prob. Alta (%)': ml_info.get('predicted_proba_up', 0.5) * 100,
                        'AUC-ROC (CV)': ml_info.get('auc_roc_score', np.nan),
                        'Modelo': ml_info.get('model_name', 'N/A'),
                        'Nº Modelos': 1 # Agora é modelo único
                    })
            
            df_ml = pd.DataFrame(ml_data)
            
            if not df_ml.empty:
                fig_ml = go.Figure()
                
                fig_ml.add_trace(go.Bar(
                    x=df_ml['Ativo'],
                    y=df_ml['Prob. Alta (%)'],
                    marker=dict(
                        color=df_ml['Prob. Alta (%)'],
                        colorscale='RdYlGn', 
                        showscale=True,
                        colorbar=dict(title="Prob. (%)")
                    ),
                    text=df_ml['Prob. Alta (%)'].round(1),
                    textposition='outside'
                ))
                
                fig_layout = obter_template_grafico()
                fig_layout['title']['text'] = "Probabilidade de Alta Futura (LightGBM Único)"
                fig_layout['yaxis']['title'] = "Probabilidade (%)"
                fig_layout['xaxis']['title'] = "Ativo"
                fig_ml.update_layout(**fig_layout, height=400)
                
                st.plotly_chart(fig_ml, use_container_width=True)
                
                st.markdown("---")
                st.markdown('#### Métricas Detalhadas do ML')
                
                df_ml_display = df_ml.copy()
                df_ml_display['Prob. Alta (%)'] = df_ml_display['Prob. Alta (%)'].round(2)
                df_ml_display['AUC-ROC (CV)'] = df_ml_display['AUC-ROC (CV)'].apply(
                    lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A"
                )
                
                st.dataframe(df_ml_display, use_container_width=True, hide_index=True)
            else:
                st.warning("Não há dados de Machine Learning para exibir.")
        
        with tab4:
            st.markdown('#### Análise de Volatilidade GARCH')
            
            dados_garch = []
            for ativo in assets:
                if ativo in builder.dados_performance.index and ativo in builder.volatilidades_garch:
                    vol_hist = builder.dados_performance.loc[ativo, 'volatilidade_anual'] if 'volatilidade_anual' in builder.dados_performance.columns else np.nan
                    vol_garch = builder.volatilidades_garch.get(ativo)
                    
                    if vol_garch is not None and not np.isnan(vol_garch):
                        status = '✓ GARCH Ajustado'
                        vol_display = vol_garch
                    elif vol_hist is not None and not np.isnan(vol_hist): 
                        status = '⚠️ Histórica (GARCH Falhou)'
                        vol_display = vol_hist
                    else:
                        status = '❌ Dados Indisponíveis'
                        vol_display = np.nan
                    
                    dados_garch.append({
                        'Ativo': ativo.replace('.SA', ''),
                        'Vol. Histórica (%)': vol_hist * 100 if not np.isnan(vol_hist) else 'N/A',
                        'Vol. GARCH (%)': vol_display * 100 if vol_display is not None and not np.isnan(vol_display) else 'N/A',
                        'Status': status
                    })
            
            df_garch = pd.DataFrame(dados_garch)
            
            if not df_garch.empty:
                fig_garch = go.Figure()
                
                plot_df_garch = df_garch[df_garch['Vol. GARCH (%)'] != 'N/A'].copy() 
                plot_df_garch['Vol. GARCH (%)'] = plot_df_garch['Vol. GARCH (%)'].astype(float)
                plot_df_garch['Vol. Histórica (%)'] = plot_df_garch['Vol. Histórica (%)'].apply(lambda x: float(x) if x != 'N/A' else np.nan)

                fig_garch.add_trace(go.Bar(
                    name='Volatilidade Histórica',
                    x=plot_df_garch['Ativo'],
                    y=plot_df_garch['Vol. Histórica (%)'],
                    marker=dict(color='#7f8c8d'),
                    opacity=0.7
                ))
                
                fig_garch.add_trace(go.Bar(
                    name='Volatilidade GARCH Ajustada',
                    x=plot_df_garch['Ativo'],
                    y=plot_df_garch['Vol. GARCH (%)'],
                    marker=dict(color='#3498db')
                ))
                
                fig_layout = obter_template_grafico()
                fig_layout['title']['text'] = "Comparação: Volatilidade Histórica vs GARCH"
                fig_layout['yaxis']['title'] = "Volatilidade Anual (%)"
                fig_layout['xaxis']['title'] = "Ativo"
                fig_layout['barmode'] = 'group'
                fig_garch.update_layout(**fig_layout, height=400)
                
                st.plotly_chart(fig_garch, use_container_width=True)
                
                st.markdown("---")
                st.markdown('#### Detalhamento das Volatilidades')
                
                st.dataframe(df_garch, use_container_width=True, hide_index=True)
            else:
                st.warning("Não há dados de volatilidade para exibir.")
        
        with tab5:
            st.markdown('#### Justificativas de Seleção e Alocação')
            
            if not builder.justificativas_selecao:
                st.warning("Nenhuma justificativa gerada.")
            else:
                for asset, justification in builder.justificativas_selecao.items():
                    weight = builder.alocacao_portfolio.get(asset, {}).get('weight', 0)
                    st.markdown(f"""
                    <div class="info-box">
                    <h4>{asset.replace('.SA', '')} ({weight*100:.2f}%)</h4>
                    <p>{justification}</p>
                    </div>
                    """, unsafe_allow_html=True)

# =============================================================================
# FUNÇÃO: INTERFACE STREAMLIT - ABA ANÁLISE INDIVIDUAL (OTIMIZADA)
# =============================================================================

def aba_analise_individual():
    """Aba 4: Análise Individual de Ativos - Otimizada para velocidade."""
    
    st.markdown("## 🔍 Análise Individual Completa de Ativos")
    
    # Determine available assets for selection
    if 'ativos_para_analise' in st.session_state and st.session_state.ativos_para_analise:
        ativos_disponiveis = st.session_state.ativos_para_analise
    else:
        ativos_disponiveis = ATIVOS_IBOVESPA 
        if not ativos_disponiveis:
            ativos_disponiveis = TODOS_ATIVOS 
            
    if not ativos_disponiveis:
        st.error("Nenhum ativo disponível para análise. Verifique as configurações ou selecione ativos.")
        return

    col1, col2 = st.columns([3, 1])
    
    with col1:
        ativo_selecionado = st.selectbox(
            "Selecione um ativo para análise detalhada:",
            options=ativos_disponiveis,
            format_func=lambda x: x.replace('.SA', '') if isinstance(x, str) else x,
            key='individual_asset_select'
        )
    
    with col2:
        if st.button("🔄 Analisar Ativo", key='analyze_asset_button', type="primary"):
            st.session_state.analisar_ativo_triggered = True 
    
    if 'analisar_ativo_triggered' not in st.session_state or not st.session_state.analisar_ativo_triggered:
        st.info("👆 Selecione um ativo e clique em 'Analisar Ativo' para começar a análise completa.")
        return
    
    # Execute analysis
    with st.spinner(f"Analisando {ativo_selecionado}..."):
        try:
            ticker = yf.Ticker(ativo_selecionado)
            
            # --- OTIMIZAÇÃO: Reduzir período histórico para 5 anos ---
            hist = ticker.history(period='5y') 
            
            if hist.empty:
                st.error(f"Não foi possível obter dados históricos para {ativo_selecionado}.")
                return
            
            # Calculate all indicators
            df_completo = AnalisadorIndividualAtivos.calcular_todos_indicadores_tecnicos(hist)
            features_fund = AnalisadorIndividualAtivos.calcular_features_fundamentalistas_expandidas(ticker)
            
            # Tabs for analysis sections
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Visão Geral",
                "📈 Análise Técnica",
                "💼 Análise Fundamentalista",
                "🤖 Machine Learning", 
                "🔬 Clusterização e Similaridade"
            ])
            
            with tab1:
                st.markdown(f"### {ativo_selecionado.replace('.SA', '')} - Visão Geral")
                
                # Display key metrics (Code unchanged)
                col1, col2, col3, col4, col5 = st.columns(5)
                
                preco_atual = df_completo['Close'].iloc[-1] if not df_completo.empty and 'Close' in df_completo.columns else np.nan
                variacao_dia = df_completo['returns'].iloc[-1] * 100 if not df_completo.empty and 'returns' in df_completo.columns else np.nan
                volume_medio = df_completo['Volume'].mean() if not df_completo.empty and 'Volume' in df_completo.columns else np.nan
                
                col1.metric("Preço Atual", f"R$ {preco_atual:.2f}" if not np.isnan(preco_atual) else "N/A", f"{variacao_dia:+.2f}%" if not np.isnan(variacao_dia) else "N/A")
                col2.metric("Volume Médio", f"{volume_medio:,.0f}" if not np.isnan(volume_medio) else "N/A")
                col3.metric("Setor", features_fund.get('sector', 'N/A'))
                col4.metric("Indústria", features_fund.get('industry', 'N/A'))
                col5.metric("Beta", f"{features_fund.get('beta', np.nan):.2f}" if not np.isnan(features_fund.get('beta')) else "N/A")
                
                # Candlestick chart with Volume (Code unchanged)
                if not df_completo.empty:
                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.7, 0.3]
                    )
                    
                    fig.add_trace(
                        go.Candlestick(
                            x=df_completo.index,
                            open=df_completo['Open'],
                            high=df_completo['High'],
                            low=df_completo['Low'],
                            close=df_completo['Close'],
                            name='Preço'
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Bar(
                            x=df_completo.index,
                            y=df_completo['Volume'],
                            name='Volume',
                            marker=dict(color='lightblue'),
                            opacity=0.7
                        ),
                        row=2, col=1
                    )
                    
                    fig_layout = obter_template_grafico()
                    fig_layout['title']['text'] = f"Histórico de Preços e Volume - {ativo_selecionado.replace('.SA', '')}"
                    fig_layout['height'] = 600
                    fig.update_layout(**fig_layout)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Dados de histórico incompletos para gráfico.")

            with tab2:
                # ... (Análise Técnica inalterada) ...
                st.markdown("### Indicadores Técnicos")
                
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                
                col1.metric("RSI (14)", f"{df_completo['rsi_14'].iloc[-1]:.2f}" if 'rsi_14' in df_completo.columns else "N/A")
                col2.metric("MACD", f"{df_completo['macd'].iloc[-1]:.4f}" if 'macd' in df_completo.columns else "N/A")
                col3.metric("Stoch %K", f"{df_completo['stoch_k'].iloc[-1]:.2f}" if 'stoch_k' in df_completo.columns else "N/A")
                col4.metric("ADX", f"{df_completo['adx'].iloc[-1]:.2f}" if 'adx' in df_completo.columns else "N/A")
                col5.metric("CCI", f"{df_completo['cci'].iloc[-1]:.2f}" if 'cci' in df_completo.columns else "N/A")
                col6.metric("ATR (%)", f"{df_completo['atr_percent'].iloc[-1]:.2f}%" if 'atr_percent' in df_completo.columns else "N/A")

                st.markdown("#### RSI e Stochastic Oscillator")
                
                fig_osc = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                        subplot_titles=("RSI (14)", "Stochastic Oscillator (%K & %D)"))
                
                if 'rsi_14' in df_completo.columns:
                    fig_osc.add_trace(
                        go.Scatter(x=df_completo.index, y=df_completo['rsi_14'], name='RSI', line=dict(color='#3498db')),
                        row=1, col=1
                    )
                    fig_osc.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1, annotation_text="Overbought (70)")
                    fig_osc.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1, annotation_text="Oversold (30)")
                
                if 'stoch_k' in df_completo.columns and 'stoch_d' in df_completo.columns:
                    fig_osc.add_trace(
                        go.Scatter(x=df_completo.index, y=df_completo['stoch_k'], name='Stochastic %K', line=dict(color='#e74c3c')),
                        row=2, col=1
                    )
                    fig_osc.add_trace(
                        go.Scatter(x=df_completo.index, y=df_completo['stoch_d'], name='Stochastic %D', line=dict(color='#7f8c8d')),
                        row=2, col=1
                    )
                    fig_osc.add_hline(y=80, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Overbought (80)")
                    fig_osc.add_hline(y=20, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Oversold (20)")
                
                fig_layout = obter_template_grafico()
                fig_layout['height'] = 550
                fig_osc.update_layout(**fig_layout)
                
                st.plotly_chart(fig_osc, use_container_width=True)
                
                st.markdown("#### Valores Atuais dos Indicadores Técnicos")
                
                current_indicators = {}
                available_indicator_cols = [col for col in df_completo.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'log_returns', 'day_of_week', 'month', 'quarter', 'day_of_month', 'week_of_year']]
                
                for col in available_indicator_cols:
                    if col in df_completo.columns and not df_completo[col].empty:
                        current_indicators[col] = df_completo[col].iloc[-1]
                
                if current_indicators:
                    df_indicadores = pd.DataFrame({
                        'Indicador': list(current_indicators.keys()),
                        'Valor Atual': [f"{v:.4f}" if isinstance(v, (int, float)) else str(v) for v in current_indicators.values()]
                    })
                    
                    st.dataframe(df_indicadores, use_container_width=True, hide_index=True)
                else:
                    st.warning("Nenhum indicador técnico com dados atuais disponível.")

            with tab3:
                # ... (Análise Fundamentalista inalterada) ...
                st.markdown("### Análise Fundamentalista Expandida")
                
                st.markdown("#### Valuation")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                col1.metric("P/L (TTM)", f"{features_fund.get('pe_ratio', np.nan):.2f}" if not pd.isna(features_fund.get('pe_ratio')) else "N/A")
                col2.metric("P/VP", f"{features_fund.get('pb_ratio', np.nan):.2f}" if not pd.isna(features_fund.get('pb_ratio')) else "N/A")
                col3.metric("P/VPA (Vendas)", f"{features_fund.get('ps_ratio', np.nan):.2f}" if not pd.isna(features_fund.get('ps_ratio')) else "N/A")
                col4.metric("PEG", f"{features_fund.get('peg_ratio', np.nan):.2f}" if not pd.isna(features_fund.get('peg_ratio')) else "N/A")
                col5.metric("EV/EBITDA", f"{features_fund.get('ev_to_ebitda', np.nan):.2f}" if not pd.isna(features_fund.get('ev_to_ebitda')) else "N/A")
                
                st.markdown("#### Rentabilidade")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                col1.metric("ROE", f"{features_fund.get('roe', np.nan):.2f}%" if not pd.isna(features_fund.get('roe')) else "N/A")
                col2.metric("ROA", f"{features_fund.get('roa', np.nan):.2f}%" if not pd.isna(features_fund.get('roa')) else "N/A")
                col3.metric("ROIC", f"{features_fund.get('roic', np.nan):.2f}%" if not pd.isna(features_fund.get('roic')) else "N/A")
                col4.metric("Margem Operacional", f"{features_fund.get('operating_margin', np.nan):.2f}%" if not pd.isna(features_fund.get('operating_margin')) else "N/A")
                col5.metric("Margem Bruta", f"{features_fund.get('gross_margin', np.nan):.2f}%" if not pd.isna(features_fund.get('gross_margin')) else "N/A")
                
                st.markdown("#### Dividendos")
                col1, col2, col3 = st.columns(3)
                
                col1.metric("Dividend Yield", f"{features_fund.get('div_yield', np.nan):.2f}%" if not pd.isna(features_fund.get('div_yield')) else "N/A")
                col2.metric("Payout Ratio", f"{features_fund.get('payout_ratio', np.nan):.2f}%" if not pd.isna(features_fund.get('payout_ratio')) else "N/A")
                col3.metric("DY Médio 5A", f"{features_fund.get('five_year_avg_div_yield', np.nan):.2f}%" if not pd.isna(features_fund.get('five_year_avg_div_yield')) else "N/A")
                
                st.markdown("#### Crescimento")
                col1, col2, col3 = st.columns(3)
                col1.metric("Cresc. Receita", f"{features_fund.get('revenue_growth', np.nan):.2f}%" if not pd.isna(features_fund.get('revenue_growth')) else "N/A")
                col2.metric("Cresc. Lucros", f"{features_fund.get('earnings_growth', np.nan):.2f}%" if not pd.isna(features_fund.get('earnings_growth')) else "N/A")
                col3.metric("Cresc. Lucros (Q)", f"{features_fund.get('earnings_quarterly_growth', np.nan):.2f}%" if not pd.isna(features_fund.get('earnings_quarterly_growth')) else "N/A")

                st.markdown("#### Saúde Financeira")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Dívida/Patrimônio", f"{features_fund.get('debt_to_equity', np.nan):.2f}" if not pd.isna(features_fund.get('debt_to_equity')) else "N/A")
                col2.metric("Current Ratio", f"{features_fund.get('current_ratio', np.nan):.2f}" if not pd.isna(features_fund.get('current_ratio')) else "N/A")
                col3.metric("Quick Ratio", f"{features_fund.get('quick_ratio', np.nan):.2f}" if not pd.isna(features_fund.get('quick_ratio')) else "N/A")
                col4.metric("Fluxo de Caixa Livre", f"R$ {features_fund.get('free_cashflow', np.nan):,.0f}" if not pd.isna(features_fund.get('free_cashflow')) else "N/A")
                
                st.markdown("---")
                st.markdown("#### Todos os Fundamentos Disponíveis")
                
                df_fund_display = pd.DataFrame({
                    'Métrica': list(features_fund.keys()),
                    'Valor': [f"{v:.4f}" if isinstance(v, (int, float)) and not pd.isna(v) else str(v) for v in features_fund.values()]
                })
                
                st.dataframe(df_fund_display, use_container_width=True, hide_index=True)
            
            with tab4:
                st.markdown("### Análise de Machine Learning (Otimizada para Velocidade)")
                
                st.info("Utilizando um único modelo **LightGBM** com 2-Fold CV para previsão rápida da direção futura.")
                
                # Prepare data for ML 
                features_from_eng = [col for col in df_completo.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'log_returns']] 
                features_from_eng = [f for f in features_from_eng if pd.api.types.is_numeric_dtype(df_completo[f])] 
                
                all_potential_features = features_from_eng
                final_features_for_ml = [f for f in all_potential_features if f in df_completo.columns and pd.api.types.is_numeric_dtype(df_completo[f])]
                
                df_ml_data = df_completo[final_features_for_ml + ['Close']].copy() 

                prediction_horizon = LOOKBACK_ML 
                df_ml_data['Future_Direction'] = np.where(
                    df_ml_data['Close'].pct_change(prediction_horizon).shift(-prediction_horizon) > 0,
                    1, 0
                )
                
                df_ml_data = df_ml_data.drop(columns=['Close'])
                df_ml_data = df_ml_data.dropna()
                
                # Treinamento do Modelo Único
                if len(df_ml_data) > 100 and len(np.unique(df_ml_data['Future_Direction'])) >= 2:
                    X = df_ml_data.drop('Future_Direction', axis=1)
                    y = df_ml_data['Future_Direction']
                    
                    try:
                        # 1. Configurar e treinar o modelo único e rápido (LightGBM)
                        modelo_unico = lgb.LGBMClassifier(
                            n_estimators=50, 
                            max_depth=5,     
                            learning_rate=0.1, 
                            random_state=42, 
                            verbose=-1, 
                            objective='binary', 
                            metric='auc'
                        )
                        modelo_unico.fit(X, y)
                        
                        # 2. Previsão final
                        last_features_row = X.iloc[[-1]] 
                        proba_final = modelo_unico.predict_proba(last_features_row)[:, 1][0]

                        # 3. Simular AUC-ROC com validação cruzada rápida (2 folds)
                        tscv_quick = TimeSeriesSplit(n_splits=2)
                        auc_scores_quick = cross_val_score(
                            modelo_unico, 
                            X, 
                            y, 
                            cv=tscv_quick, 
                            scoring='roc_auc', 
                            error_score='raise'
                        )
                        auc_medio = np.mean(auc_scores_quick) if len(auc_scores_quick) > 0 else 0.5
                        
                        # Resultados
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Probabilidade de Alta Futura (LGBM)", f"{proba_final*100:.2f}%")
                        col2.metric("AUC-ROC Médio (2-Fold CV)", f"{auc_medio:.3f}" if not pd.isna(auc_medio) else "N/A")
                        col3.metric("Nº Features Usadas", len(X.columns))
                        
                        # Feature Importance
                        if hasattr(modelo_unico, 'feature_importances_'):
                            st.markdown("#### Feature Importance (LightGBM)")
                            
                            importances = modelo_unico.feature_importances_
                            feature_names = X.columns
                            
                            df_importance = pd.DataFrame({
                                'Feature': feature_names,
                                'Importance': importances
                            }).sort_values('Importance', ascending=False).head(20)
                            
                            fig_imp = px.bar(
                                df_importance,
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                title='Top 20 Features Mais Importantes (LightGBM)'
                            )
                            fig_imp.update_layout(**obter_template_grafico())
                            st.plotly_chart(fig_imp, use_container_width=True)
                            
                    except Exception as e:
                        st.warning(f"Falha na análise ML do ativo: {str(e)}")
                else:
                    st.warning("Dados insuficientes ou classe única encontrada para análise ML detalhada (mínimo de 100 dias com features válidas).")
            
            with tab5:
                # ... (Clusterização otimizada com limite de 20 ativos) ...
                st.markdown("### Clusterização e Análise de Similaridade (Otimizada)")
                
                st.info("Comparando este ativo com outros similares usando K-means + PCA...")
                
                max_assets_for_clustering = 82 # Limite otimizado
                
                assets_to_cluster = []
                if 'sector' in features_fund and features_fund['sector'] != 'Unknown':
                    sector_assets = ATIVOS_POR_SETOR.get(features_fund['sector'], [])
                    
                    if ativo_selecionado in sector_assets:
                        assets_to_cluster.append(ativo_selecionado)
                    
                    other_sector_assets = [a for a in sector_assets if a != ativo_selecionado]
                    assets_to_cluster.extend(other_sector_assets[:max_assets_for_clustering - len(assets_to_cluster)])
                
                if len(assets_to_cluster) < max_assets_for_clustering:
                    if ativo_selecionado not in assets_to_cluster:
                        assets_to_cluster.append(ativo_selecionado)
                    
                    other_assets = [a for a in ATIVOS_IBOVESPA if a not in assets_to_cluster]
                    assets_to_cluster.extend(other_assets[:max_assets_for_clustering - len(assets_to_cluster)])
                    
                comparison_data = {}
                if len(assets_to_cluster) > 5:
                    with st.spinner(f"Coletando dados para {len(assets_to_cluster)} ativos de comparação..."):
                        for asset_comp in assets_to_cluster:
                            try:
                                ticker_comp = yf.Ticker(asset_comp)
                                hist_comp = ticker_comp.history(period='2y') 
                                
                                if not hist_comp.empty:
                                    returns_comp = hist_comp['Close'].pct_change().dropna()
                                    
                                    if len(returns_comp) > 50: 
                                        comp_data = {
                                            'retorno_anual': returns_comp.mean() * 252,
                                            'volatilidade_anual': returns_comp.std() * np.sqrt(252),
                                            'sharpe': (returns_comp.mean() * 252 - TAXA_LIVRE_RISCO) / (returns_comp.std() * np.sqrt(252)) if returns_comp.std() > 0 else 0,
                                            'max_drawdown': ((1 + returns_comp).cumprod() / (1 + returns_comp).cumprod().expanding().max() - 1).min() if not returns_comp.empty else np.nan
                                        }
                                        
                                        info_comp = ticker_comp.info
                                        fund_metrics = AnalisadorIndividualAtivos.calcular_features_fundamentalistas_expandidas(ticker_comp)
                                        
                                        for metric in ['pe_ratio', 'pb_ratio', 'roe', 'debt_to_equity', 'revenue_growth']:
                                             comp_data[metric] = fund_metrics.get(metric, np.nan)

                                        comparison_data[asset_comp] = comp_data
                            except Exception:
                                continue
                
                if len(comparison_data) > 5:
                    df_comparacao = pd.DataFrame(comparison_data).T
                    
                    resultado_pca, pca, kmeans = AnalisadorIndividualAtivos.realizar_clusterizacao_pca(
                        df_comparacao,
                        n_clusters=5 
                    )
                    
                    if resultado_pca is not None:
                        if 'PC3' in resultado_pca.columns:
                            fig_pca = px.scatter_3d(
                                resultado_pca, x='PC1', y='PC2', z='PC3', color='Cluster',
                                hover_name=resultado_pca.index.str.replace('.SA', ''),
                                title='Clusterização K-means + PCA (3D) - Similaridade de Ativos'
                            )
                        else:
                            fig_pca = px.scatter(
                                resultado_pca, x='PC1', y='PC2', color='Cluster',
                                hover_name=resultado_pca.index.str.replace('.SA', ''),
                                title='Clusterização K-means + PCA (2D) - Similaridade de Ativos'
                            )
                        
                        fig_pca.update_layout(**obter_template_grafico(), height=600)
                        st.plotly_chart(fig_pca, use_container_width=True)
                        
                        if ativo_selecionado in resultado_pca.index:
                            cluster_ativo = resultado_pca.loc[ativo_selecionado, 'Cluster']
                            ativos_similares_df = resultado_pca[resultado_pca['Cluster'] == cluster_ativo]
                            ativos_similares = [a for a in ativos_similares_df.index.tolist() if a != ativo_selecionado]
                            
                            st.success(f"**{ativo_selecionado.replace('.SA', '')}** pertence ao Cluster {cluster_ativo}")
                            
                            if ativos_similares:
                                st.markdown(f"#### Outros Ativos no Cluster {cluster_ativo}:")
                                st.write(", ".join([a.replace('.SA', '') for a in ativos_similares[:15]]))
                        
                        st.markdown("#### Variância Explicada por Componente Principal")
                        var_exp = pca.explained_variance_ratio_ * 100
                        
                        df_var = pd.DataFrame({
                            'Componente': [f'PC{i+1}' for i in range(len(var_exp))],
                            'Variância (%)': var_exp
                        })
                        
                        fig_var = px.bar(df_var, x='Componente', y='Variância (%)', title='Variância Explicada por Componente Principal')
                        fig_var.update_layout(**obter_template_grafico())
                        st.plotly_chart(fig_var, use_container_width=True)
                else:
                    st.warning("Dados insuficientes para realizar a clusterização e análise de similaridade.")
        
        except Exception as e:
            st.error(f"Erro ao analisar o ativo {ativo_selecionado}: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
def main():
    """Função principal com estrutura de 4 abas (Sidebar removida)"""
    
    # Initialize session state variables if they don't exist
    if 'builder' not in st.session_state:
        st.session_state.builder = None
        st.session_state.builder_complete = False
        st.session_state.profile = {}
        st.session_state.ativos_para_analise = []
        st.session_state.analisar_ativo_triggered = False
        
    # Certifique-se de que 'configurar_pagina()' usa 'layout="wide"'
    # Exemplo: st.set_page_config(layout="wide", page_title="AutoML Elite")
    configurar_pagina()
    
    # O CÓDIGO DA SIDEBAR FOI REMOVIDO AQUI:
    # st.sidebar.markdown(...)
    # st.sidebar.markdown("---")
    # ... e assim por diante.
    
    # Main title
    st.markdown('<h1 class="main-header">Sistema AutoML Elite - Otimização Quantitativa de Portfólio</h1>', unsafe_allow_html=True)
    
    # Configuração das 4 abas
    tab1, tab2, tab3, tab4 = st.tabs([
        "📚 Introdução",
        "🎯 Seleção de Ativos",
        "🏗️ Construtor de Portfólio",
        "🔍 Análise Individual"
    ])
    
    # Render content for each tab
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
