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

Versão: 7.0.0 - Sistema AutoML Elite Final
=============================================================================
"""

import warnings
import numpy as np
import pandas as pd
import subprocess
import sys
import time # Added for retry logic
from datetime import datetime, timedelta
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.stats import zscore, norm
# Advanced feature engineering libraries
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, CCIIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel, DonchianChannel
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, MFIIndicator, VolumeWeightedAveragePrice
# Statistical models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.varmax import VARMAX
from prophet import Prophet
# Deep Learning (TensorFlow/Keras)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, MultiHeadAttention, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
# For file handling and paths
import os

warnings.filterwarnings('ignore')

# =============================================================================
# INSTALAÇÃO AUTOMÁTICA DE DEPENDÊNCIAS
# =============================================================================

REQUIRED_PACKAGES = {
    'yfinance': 'yfinance',
    'plotly': 'plotly',
    'streamlit': 'streamlit',
    'sklearn': 'scikit-learn',
    'scipy': 'scipy',
    'xgboost': 'xgboost',
    'lightgbm': 'lightgbm',
    'catboost': 'catboost',
    'arch': 'arch',
    'optuna': 'optuna',
    'ta': 'ta',
    'shap': 'shap',
    'lime': 'lime',
    'statsmodels': 'statsmodels',
    'prophet': 'prophet',
    'tensorflow': 'tensorflow',
    'keras': 'keras' # Explicitly add keras if not implicitly covered by tensorflow
}

def ensure_package(module_name, package_name):
    """Instala pacote se não estiver disponível"""
    try:
        __import__(module_name)
    except ImportError:
        print(f"Instalando {package_name}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', package_name])
        
        # Check if streamlit is loaded to prompt for restart
        if 'streamlit' in sys.modules:
            import streamlit as st
            st.warning(f"{package_name} foi instalado. Por favor, **reexecute** o servidor Streamlit para aplicar as mudanças.")
            st.stop() # Stop execution if in a streamlit app to ensure restart

# Instala todas as dependências
try:
    for module, package in REQUIRED_PACKAGES.items():
        ensure_package(module.split('.')[0], package)
    
    import streamlit as st
    import yfinance as yf
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier # Added Classifier variants
    from sklearn.linear_model import RidgeClassifier, LogisticRegression, BayesianRidge # Added Classifier variants
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import silhouette_score, mean_squared_error, mean_absolute_error, r2_score, roc_auc_score # Added roc_auc_score
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier # Added Classifier variant
    from arch import arch_model
    import optuna
    import shap
    import lime
    import lime.lime_tabular
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
except Exception as e:
    if 'streamlit' in sys.modules:
        st.error(f"Erro ao carregar bibliotecas: {e}")
    else:
        print(f"Erro ao carregar bibliotecas: {e}")
    sys.exit(1)

# =============================================================================
# CONSTANTES GLOBAIS E CONFIGURAÇÕES
# =============================================================================

# Updated data collection period to maximum available and adjusted other constants
# Configurações Globais
PERIODO_DADOS = 'max'  # Changed from '2y' to 'max' for maximum historical depth
MIN_DIAS_HISTORICO = 252  # Reduced minimum to accommodate max period
NUM_ATIVOS_PORTFOLIO = 5
TAXA_LIVRE_RISCO = 0.1075 # Updated risk-free rate
LOOKBACK_ML = 30  # Extended prediction horizon to 30 days

# Ponderações padrão para os scores
WEIGHT_PERFORMANCE = 0.40
WEIGHT_FUNDAMENTAL = 0.30
WEIGHT_TECHNICAL = 0.30
WEIGHT_ML = 0.30 # Adicionado peso para ML no score total

# Limites de peso por ativo na otimização
PESO_MIN = 0.10
PESO_MAX = 0.30

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

# Caminhos dos arquivos Parquet (gerados pelo coleta_diaria.py)
DATA_PATH = './dados_financeiros/'
ARQUIVO_HISTORICO = DATA_PATH + 'dados_historicos.parquet'
ARQUIVO_FUNDAMENTALISTA = DATA_PATH + 'dados_fundamentalistas.parquet'
ARQUIVO_METRICAS = DATA_PATH + 'metricas_performance.parquet'
ARQUIVO_MACRO = DATA_PATH + 'dados_macro.parquet'
ARQUIVO_METADATA = DATA_PATH + 'metadata.parquet'

# =============================================================================
# MAPEAMENTO DE ATIVOS POR SETOR
# =============================================================================

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

# Lista completa de todos os ativos
TODOS_ATIVOS = []
for setor, ativos in ATIVOS_POR_SETOR.items():
    TODOS_ATIVOS.extend(ativos)
TODOS_ATIVOS = sorted(list(set(TODOS_ATIVOS)))

# =============================================================================
# CONSTANTES DE GOVERNANÇA (NEW)
# =============================================================================

AUC_THRESHOLD_MIN = 0.65  # Alerta se AUC cair abaixo deste valor
AUC_DROP_THRESHOLD = 0.05   # Alerta se queda de 5% no AUC
DRIFT_WINDOW = 20           # Janela para monitoramento de drift
STRESS_TEST_SIGMA = 2.0     # Número de desvios-padrão para choque

# =============================================================================
# CLASSE: GOVERNANÇA DE MODELO (NEW)
# =============================================================================

class GovernancaModelo:
    """
    NEW: Classe para monitoramento e governança de modelos ML
    Rastreia AUC-ROC, Precision, Recall, F1-Score ao longo do tempo
    Emite alertas quando performance degrada
    """
    
    def __init__(self, ativo, max_historico=DRIFT_WINDOW):
        self.ativo = ativo
        self.max_historico = max_historico
        self.historico_auc = []
        self.historico_precision = []
        self.historico_recall = []
        self.historico_f1 = []
        self.auc_maximo = 0.0
        
    def adicionar_metricas(self, auc, precision, recall, f1):
        """Adiciona novas métricas ao histórico"""
        self.historico_auc.append(auc)
        self.historico_precision.append(precision)
        self.historico_recall.append(recall)
        self.historico_f1.append(f1)
        
        # Mantém apenas os últimos N registros
        if len(self.historico_auc) > self.max_historico:
            self.historico_auc.pop(0)
            self.historico_precision.pop(0)
            self.historico_recall.pop(0)
            self.historico_f1.pop(0)
        
        # Atualiza AUC máximo
        if auc > self.auc_maximo:
            self.auc_maximo = auc
    
    def verificar_alertas(self):
        """Verifica se há alertas de degradação de performance"""
        if not self.historico_auc:
            return []
        
        alertas = []
        auc_atual = self.historico_auc[-1]
        
        # Alerta 1: AUC abaixo do mínimo aceitável
        if auc_atual < AUC_THRESHOLD_MIN:
            alertas.append({
                'tipo': 'CRÍTICO',
                'mensagem': f'AUC ({auc_atual:.3f}) abaixo do mínimo aceitável ({AUC_THRESHOLD_MIN})'
            })
        
        # Alerta 2: Degradação significativa em relação ao máximo
        if self.auc_maximo > 0:
            degradacao = (self.auc_maximo - auc_atual) / self.auc_maximo
            if degradacao > AUC_DROP_THRESHOLD:
                alertas.append({
                    'tipo': 'ATENÇÃO',
                    'mensagem': f'Degradação de {degradacao*100:.1f}% em relação ao máximo ({self.auc_maximo:.3f})'
                })
        
        # Alerta 3: Tendência de queda consistente
        if len(self.historico_auc) >= 5:
            ultimos_5 = self.historico_auc[-5:]
            if all(ultimos_5[i] > ultimos_5[i+1] for i in range(len(ultimos_5)-1)):
                alertas.append({
                    'tipo': 'ATENÇÃO',
                    'mensagem': 'Tendência de queda consistente nos últimos 5 períodos'
                })
        
        return alertas
    
    def gerar_relatorio(self):
        """Gera relatório completo de governança"""
        if not self.historico_auc:
            return {
                'status': 'Sem dados suficientes',
                'severidade': 'info',
                'metricas': {},
                'alertas': [],
                'historico': {}
            }
        
        alertas = self.verificar_alertas()
        
        # Determina severidade geral
        if any(a['tipo'] == 'CRÍTICO' for a in alertas):
            severidade = 'error'
            status = 'Modelo requer atenção imediata'
        elif any(a['tipo'] == 'ATENÇÃO' for a in alertas):
            severidade = 'warning'
            status = 'Modelo em monitoramento'
        else:
            severidade = 'success'
            status = 'Modelo operando normalmente'
        
        return {
            'status': status,
            'severidade': severidade,
            'metricas': {
                'AUC Atual': self.historico_auc[-1],
                'AUC Médio': np.mean(self.historico_auc),
                'AUC Máximo': self.auc_maximo,
                'Precision Média': np.mean(self.historico_precision),
                'Recall Médio': np.mean(self.historico_recall),
                'F1-Score Médio': np.mean(self.historico_f1)
            },
            'alertas': alertas,
            'historico': {
                'AUC': self.historico_auc,
                'Precision': self.historico_precision,
                'Recall': self.historico_recall,
                'F1': self.historico_f1
            }
        }

# =============================================================================
# MAPEAMENTOS DE PONTUAÇÃO DO QUESTIONÁRIO
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
# CLASSE: ANALISADOR DE PERFIL DO INVESTIDOR
# =============================================================================

class AnalisadorPerfilInvestidor:
    """Analisa perfil de risco e horizonte temporal do investidor"""
    
    def __init__(self):
        self.nivel_risco = ""
        self.horizonte_tempo = ""
        self.dias_lookback_ml = 5
    
    def determinar_nivel_risco(self, pontuacao):
        """Traduz pontuação em perfil de risco"""
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
    
    def determinar_horizonte_ml(self, liquidez_key, objetivo_key):
        """Define horizonte temporal e janela ML"""
        time_map = {
            'A': 5, # Curto prazo
            'B': 20, # Médio prazo
            'C': 30 # Longo prazo
        }
        
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
    
    def calcular_perfil(self, respostas_risco):
        """Calcula perfil completo do investidor"""
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
            respostas_risco['liquidity'],
            respostas_risco['time_purpose']
        )
        
        return nivel_risco, horizonte_tempo, ml_lookback, pontuacao

# =============================================================================
# FUNÇÕES DE ESTILO E VISUALIZAÇÃO
# =============================================================================

def obter_template_grafico():
    """Template de layout para gráficos Plotly"""
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
                'color': '#2c3e50'
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
        'colorway': ['#2c3e50', '#7f8c8d', '#3498db', '#e74c3c', '#27ae60']
    }

# =============================================================================
# CLASSE: ENGENHEIRO DE FEATURES
# =============================================================================

class EngenheiroFeatures:
    """Calcula indicadores técnicos e fundamentalistas com máxima profundidade"""
    
    @staticmethod
    def calcular_indicadores_tecnicos(hist):
        """Calcula indicadores técnicos completos usando ta library"""
        df = hist.copy()
        
        # Retornos e Volatilidade
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        df['volatility_60'] = df['returns'].rolling(window=60).std() * np.sqrt(252)
        df['volatility_252'] = df['returns'].rolling(window=252).std() * np.sqrt(252) # Added long-term volatility
        
        # Médias Móveis (SMA, EMA, WMA, HMA)
        for periodo in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{periodo}'] = SMAIndicator(close=df['Close'], window=periodo).sma_indicator()
            df[f'ema_{periodo}'] = EMAIndicator(close=df['Close'], window=periodo).ema_indicator()
            # WMA (Weighted Moving Average)
            weights = np.arange(1, periodo + 1)
            df[f'wma_{periodo}'] = df['Close'].rolling(periodo).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        
        # Hull Moving Average (HMA)
        for periodo in [20, 50]:
            # Corrected WMA calculation for HMA
            wma_half_series = df['Close'].rolling(periodo // 2).apply(lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(), raw=True)
            wma_full_series = df['Close'].rolling(periodo).apply(lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(), raw=True)
            df[f'hma_{periodo}'] = (2 * wma_half_series - wma_full_series).rolling(int(np.sqrt(periodo))).mean()
        
        # Razões de preço e cruzamentos
        df['price_sma20_ratio'] = df['Close'] / df['sma_20']
        df['price_sma50_ratio'] = df['Close'] / df['sma_50']
        df['price_sma200_ratio'] = df['Close'] / df['sma_200']
        df['sma20_sma50_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
        df['sma50_sma200_cross'] = (df['sma_50'] > df['sma_200']).astype(int)
        df['death_cross'] = (df['Close'] < df['sma_200']).astype(int)
        
        # RSI (múltiplos períodos)
        for periodo in [7, 14, 21, 28]:
            df[f'rsi_{periodo}'] = RSIIndicator(close=df['Close'], window=periodo).rsi()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        df['williams_r'] = WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close'], lbp=14).williams_r()
        
        # MACD (múltiplas configurações)
        macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # MACD alternativo (5, 35, 5)
        macd_alt = MACD(close=df['Close'], window_slow=35, window_fast=5, window_sign=5)
        df['macd_alt'] = macd_alt.macd()
        
        # Bollinger Bands
        bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_pband'] = bb.bollinger_pband()
        
        # Keltner Channel
        kc = KeltnerChannel(high=df['High'], low=df['Low'], close=df['Close'], window=20, window_atr=10)
        df['kc_upper'] = kc.keltner_channel_hband()
        df['kc_lower'] = kc.keltner_channel_lband()
        df['kc_middle'] = kc.keltner_channel_mband()
        df['kc_width'] = (df['kc_upper'] - df['kc_lower']) / df['kc_middle']
        
        # Donchian Channel
        dc = DonchianChannel(high=df['High'], low=df['Low'], close=df['Close'], window=20)
        df['dc_upper'] = dc.donchian_channel_hband()
        df['dc_lower'] = dc.donchian_channel_lband()
        df['dc_middle'] = dc.donchian_channel_mband()
        
        # ATR (Average True Range)
        atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
        df['atr'] = atr.average_true_range()
        df['atr_percent'] = (df['atr'] / df['Close']) * 100
        
        # ADX (Average Directional Index)
        adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        
        # CCI (Commodity Channel Index)
        df['cci'] = CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=20).cci()
        
        # Momentum indicators
        df['momentum_10'] = ROCIndicator(close=df['Close'], window=10).roc()
        df['momentum_20'] = ROCIndicator(close=df['Close'], window=20).roc()
        df['momentum_60'] = ROCIndicator(close=df['Close'], window=60).roc()
        
        # Volume indicators
        df['obv'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
        df['cmf'] = ChaikinMoneyFlowIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=20).chaikin_money_flow()
        df['mfi'] = MFIIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=14).money_flow_index()
        df['vwap'] = VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).volume_weighted_average_price()
        
        # Drawdown
        cumulative_returns = (1 + df['returns']).cumprod()
        running_max = cumulative_returns.expanding().max()
        df['drawdown'] = (cumulative_returns - running_max) / running_max
        df['max_drawdown_252'] = df['drawdown'].rolling(252).min()
        
        # Lags (temporal features)
        for lag in [1, 5, 10, 20, 60]:
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
        
        # Rolling statistics
        for window in [5, 20, 60]:
            df[f'returns_mean_{window}'] = df['returns'].rolling(window).mean()
            df[f'returns_std_{window}'] = df['returns'].rolling(window).std()
            df[f'returns_skew_{window}'] = df['returns'].rolling(window).skew()
            df[f'returns_kurt_{window}'] = df['returns'].rolling(window).kurt()
            df[f'volume_mean_{window}'] = df['Volume'].rolling(window).mean()
            df[f'volume_std_{window}'] = df['Volume'].rolling(window).std()
        
        # Autocorrelation
        for lag in [1, 5, 10]:
            df[f'autocorr_{lag}'] = df['returns'].rolling(60).apply(lambda x: x.autocorr(lag=lag), raw=False)
        
        # Price patterns
        df['higher_high'] = ((df['High'] > df['High'].shift(1)) & (df['High'].shift(1) > df['High'].shift(2))).astype(int)
        df['lower_low'] = ((df['Low'] < df['Low'].shift(1)) & (df['Low'].shift(1) < df['Low'].shift(2))).astype(int)
        
        # Temporal encoding (day of week, month, quarter)
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['day_of_month'] = df.index.day
        df['week_of_year'] = df.index.isocalendar().week
        
        return df.dropna()
    
    @staticmethod
    def calcular_features_fundamentalistas(info):
        """Extrai features fundamentalistas expandidas"""
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
    def _normalizar(serie, maior_melhor=True):
        """Normaliza uma série para o range [0, 1]"""
        if serie.isnull().all():
            return pd.Series(0, index=serie.index) # Return zeros if all NaN
        
        min_val = serie.min()
        max_val = serie.max()
        
        if max_val == min_val: # Handle cases where all values are the same
            return pd.Series(0.5 if maior_melhor else 0.5, index=serie.index)
        
        if maior_melhor:
            return (serie - min_val) / (max_val - min_val)
        else:
            return (max_val - serie) / (max_val - min_val)

# =============================================================================
# CLASSE: COLETOR DE DADOS
# =============================================================================

class ColetorDados:
    """Coleta e processa dados de mercado com profundidade máxima"""
    
    def __init__(self, periodo=PERIODO_DADOS):
        self.periodo = periodo
        self.dados_por_ativo = {}
        self.dados_fundamentalistas = pd.DataFrame()
        self.ativos_sucesso = []
        self.dados_macro = {}
        self.metricas_performance = pd.DataFrame() # Initialize metric dataframe
    
    def coletar_dados_macroeconomicos(self):
        """Carrega dados macroeconômicos do arquivo Parquet"""
        print("\n📊 Carregando dados macroeconômicos do cache...")
        
        try:
            if os.path.exists(ARQUIVO_MACRO):
                df_macro = pd.read_parquet(ARQUIVO_MACRO)
                
                # Converter DataFrame para dicionário de Series
                for coluna in df_macro.columns:
                    self.dados_macro[coluna] = df_macro[coluna]
                    print(f"  ✓ {coluna}: {len(df_macro[coluna])} dias")
                
                print(f"✓ Dados macroeconômicos carregados: {len(self.dados_macro)} indicadores")
            else:
                print(f"⚠️ Arquivo {ARQUIVO_MACRO} não encontrado. Execute coleta_diaria.py primeiro.")
                self.dados_macro = {}
                
        except Exception as e:
            print(f"❌ Erro ao carregar dados macro: {str(e)}")
            self.dados_macro = {}
    
    def adicionar_correlacoes_macro(self, df, simbolo):
        """Adiciona correlações com indicadores macroeconômicos"""
        if not self.dados_macro or 'returns' not in df.columns:
            return df
        
        try:
            # Ensure that 'returns' column is available and not all NaN
            if df['returns'].isnull().all():
                print(f"  ⚠️ {simbolo}: Coluna 'returns' está vazia, pulando correlações macro.")
                return df

            for nome, serie_macro in self.dados_macro.items():
                if serie_macro.empty or serie_macro.isnull().all():
                    continue
                
                # Alinha as datas
                # Use reindex to align based on df's index, then align serie_macro to it
                df_returns_aligned = df['returns'].reindex(df.index)
                
                # Ensure both series have data after alignment and before calculating rolling corr
                if df_returns_aligned.isnull().all() or serie_macro.isnull().all():
                    continue

                # Align serie_macro to df_returns_aligned's index and fillna for rolling corr
                # Using inner join implicitly via reindex and subsequent operations
                combined_df = pd.DataFrame({
                    'asset_returns': df_returns_aligned,
                    'macro_returns': serie_macro.reindex(df.index) # Align macro to asset index
                }).dropna() # Drop rows where either asset or macro returns are missing for this period

                if len(combined_df) > 60:
                    # Correlação rolling
                    corr_rolling = combined_df['asset_returns'].rolling(60).corr(combined_df['macro_returns'])
                    df[f'corr_{nome.lower()}'] = corr_rolling.reindex(df.index) # Reindex to original df index
                else:
                    df[f'corr_{nome.lower()}'] = np.nan # Not enough data
        except Exception as e:
            print(f"  ⚠️ {simbolo}: Erro ao calcular correlações macro - {str(e)[:80]}") # Increased length for more detail
        
        return df
    
    def coletar_e_processar_dados(self, simbolos):
        """
        Carrega dados processados dos arquivos Parquet e filtra pelos símbolos selecionados
        
        IMPORTANTE: Este método agora apenas LÊ dados pré-processados.
        Execute coleta_diaria.py para atualizar os dados.
        """
        self.ativos_sucesso = []
        
        print(f"\n{'='*60}")
        print(f"CARREGANDO DADOS DO CACHE PARQUET")
        print(f"Ativos solicitados: {len(simbolos)}")
        print(f"{'='*60}\n")
        
        # Verificar se arquivos existem
        arquivos_necessarios = [
            ARQUIVO_HISTORICO,
            ARQUIVO_FUNDAMENTALISTA,
            ARQUIVO_METRICAS,
            ARQUIVO_MACRO
        ]
        
        arquivos_faltando = [arq for arq in arquivos_necessarios if not os.path.exists(arq)]
        
        if arquivos_faltando:
            print("❌ ERRO: Arquivos de dados não encontrados:")
            for arq in arquivos_faltando:
                print(f"  • {arq}")
            print("\n💡 Solução: Execute o script de coleta primeiro:")
            print("   python coleta_diaria.py")
            return False
        
        try:
            # 1. Carregar dados macroeconômicos
            self.coletar_dados_macroeconomicos()
            
            # 2. Carregar dados históricos
            print("📥 Carregando dados históricos...")
            df_historico_completo = pd.read_parquet(ARQUIVO_HISTORICO)
            
            # Filtrar apenas os símbolos solicitados
            simbolos_disponiveis = df_historico_completo['ticker'].unique()
            simbolos_validos = [s for s in simbolos if s in simbolos_disponiveis]
            
            if not simbolos_validos:
                print(f"❌ Nenhum dos símbolos solicitados está disponível no cache.")
                print(f"   Símbolos solicitados: {simbolos[:10]}...")
                print(f"   Símbolos disponíveis: {list(simbolos_disponiveis)[:10]}...")
                return False
            
            print(f"  ✓ {len(simbolos_validos)}/{len(simbolos)} símbolos encontrados no cache")
            
            # Reestruturar dados históricos para formato de dicionário
            for simbolo in tqdm(simbolos_validos, desc="⚙️ Processando dados"):
                df_ativo = df_historico_completo[df_historico_completo['ticker'] == simbolo].copy()
                df_ativo = df_ativo.drop(columns=['ticker'])
                
                if len(df_ativo) >= MIN_DIAS_HISTORICO * 0.7:
                    self.dados_por_ativo[simbolo] = df_ativo
                    self.ativos_sucesso.append(simbolo)
                else:
                    print(f"  ⚠️ {simbolo}: Dados insuficientes ({len(df_ativo)} dias)")
            
            # 3. Carregar dados fundamentalistas
            print("📥 Carregando dados fundamentalistas...")
            df_fundamentalista_completo = pd.read_parquet(ARQUIVO_FUNDAMENTALISTA)
            
            # Filtrar apenas símbolos válidos
            self.dados_fundamentalistas = df_fundamentalista_completo.loc[
                df_fundamentalista_completo.index.isin(self.ativos_sucesso)
            ]
            print(f"  ✓ {len(self.dados_fundamentalistas)} ativos com dados fundamentalistas")
            
            # 4. Carregar métricas de performance
            print("📥 Carregando métricas de performance...")
            df_metricas_completo = pd.read_parquet(ARQUIVO_METRICAS)
            
            # Filtrar apenas símbolos válidos
            self.metricas_performance = df_metricas_completo.loc[
                df_metricas_completo.index.isin(self.ativos_sucesso)
            ]
            print(f"  ✓ {len(self.metricas_performance)} ativos com métricas")
            
            # 5. Verificar se temos ativos suficientes
            if len(self.ativos_sucesso) < NUM_ATIVOS_PORTFOLIO:
                print(f"\n❌ ERRO: Apenas {len(self.ativos_sucesso)} ativos válidos carregados.")
                print(f"    Necessário: {NUM_ATIVOS_PORTFOLIO} ativos mínimos")
                print(f"\n💡 Dicas:")
                print(f"  • Execute coleta_diaria.py para atualizar os dados")
                print(f"  • Selecione mais ativos ou setores diferentes")
                print(f"  • Verifique os logs do ETL para ver quais ativos falharam\n")
                return False
            
            # 6. Exibir informações sobre a última coleta
            if os.path.exists(ARQUIVO_METADATA):
                df_metadata = pd.read_parquet(ARQUIVO_METADATA)
                ultima_coleta = df_metadata.iloc[-1]
                print(f"\n📅 Última coleta: {ultima_coleta['data_coleta']}")
                print(f"   Ativos coletados: {ultima_coleta['ativos_sucesso']}/{ultima_coleta['total_ativos_tentados']}")
            
            print(f"\n✓ Dados carregados com sucesso!")
            print(f"  • {len(self.ativos_sucesso)} ativos prontos para análise")
            print(f"  • {len(self.dados_fundamentalistas)} com dados fundamentalistas")
            print(f"  • {len(self.metricas_performance)} com métricas de performance")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados do cache: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

# A coleta agora é feita pelo coleta_diaria.py
# def coletar_dados_ativos(lista_ativos, periodo='max'):
#     """
#     Coleta dados históricos de múltiplos ativos com tratamento robusto de erros
#     """
#     dados_coletados = {}
#     ativos_com_erro = []
    
#     progress_bar = st.progress(0)
#     status_text = st.empty()
    
#     total = len(lista_ativos)
    
#     for idx, ticker in enumerate(tqdm(lista_ativos, desc="📥 Coletando dados")):
#         try:
#             status_text.text(f"Coletando {ticker} ({idx+1}/{total})...")
#             progress_bar.progress((idx + 1) / total)
            
#             # Add retry logic with exponential backoff
#             max_retries = 3
#             retry_delay = 1
            
#             for attempt in range(max_retries):
#                 try:
#                     # Download with extended timeout
#                     df = yf.download(
#                         ticker, 
#                         period=periodo, 
#                         progress=False,
#                         timeout=10,
#                         # Add headers to avoid rate limiting
#                         headers={'User-Agent': 'Mozilla/5.0'}
#                     )
                    
#                     if df.empty or len(df) < MIN_DIAS_HISTORICO:
#                         if attempt < max_retries - 1:
#                             time.sleep(retry_delay)
#                             retry_delay *= 2
#                             continue
#                         else:
#                             ativos_com_erro.append((ticker, "Sem dados históricos suficientes"))
#                             break
                    
#                     # Successful download
#                     dados_coletados[ticker] = df
#                     break
                    
#                 except Exception as e:
#                     if attempt < max_retries - 1:
#                         time.sleep(retry_delay)
#                         retry_delay *= 2
#                     else:
#                         ativos_com_erro.append((ticker, str(e)))
                        
#         except Exception as e:
#             ativos_com_erro.append((ticker, str(e)))
#             continue
    
#     progress_bar.empty()
#     status_text.empty()
    
#     # Display collection summary
#     if ativos_com_erro:
#         with st.expander(f"⚠️ Ativos com problemas ({len(ativos_com_erro)}):"):
#             for ticker, erro in ativos_com_erro[:10]:  # Show first 10
#                 st.text(f"  • {ticker}: {erro}")
#             if len(ativos_com_erro) > 10:
#                 st.text(f"  ... e mais {len(ativos_com_erro) - 10} ativos")
    
#     st.success(f"✓ Total coletado: {len(dados_coletados)}")
    
#     # Validate minimum assets collected
#     if len(dados_coletados) < NUM_ATIVOS_PORTFOLIO:
#         st.error("""
#         ❌ ERRO: Apenas {len(dados_coletados)} ativos coletados.
#             Necessário: {NUM_ATIVOS_PORTFOLIO} ativos mínimos
        
#         💡 Dicas:
#           • Verifique sua conexão com a internet
#           • Tente selecionar mais ativos ou setores diferentes
#           • Alguns ativos podem estar sem dados recentes
#           • Aguarde alguns minutos e tente novamente (possível rate limiting)
#         """.format(len=len, NUM_ATIVOS_PORTFOLIO=NUM_ATIVOS_PORTFOLIO))
#         return None
    
#     return dados_coletados

# =============================================================================
# CLASSE: MODELAGEM DE VOLATILIDADE GARCH
# =============================================================================

class VolatilidadeGARCH:
    """Modelagem de volatilidade GARCH/EGARCH"""
    
    @staticmethod
    def ajustar_garch(returns, tipo_modelo='GARCH'):
        """
        Ajusta modelo GARCH e prevê volatilidade
        Melhorado tratamento de erros e validação
        """
        try:
            returns_limpo = returns.dropna() * 100 # Scale returns for GARCH
            
            if len(returns_limpo) < 100: # Minimum data points for GARCH
                return np.nan
            
            if returns_limpo.std() == 0: # Avoid division by zero if no variance
                return np.nan
            
            if tipo_modelo == 'EGARCH':
                modelo = arch_model(returns_limpo, vol='EGARCH', p=1, q=1, rescale=False)
            else: # Default to GARCH
                modelo = arch_model(returns_limpo, vol='Garch', p=1, q=1, rescale=False)
            
            resultado = modelo.fit(disp='off', show_warning=False, options={'maxiter': 1000})
            
            if resultado is None or not resultado.params.any(): # Check if fitting was successful
                return np.nan
            
            previsao = resultado.forecast(horizon=1)
            volatilidade = np.sqrt(previsao.variance.values[-1, 0]) / 100 # Unscale prediction
            
            if np.isnan(volatilidade) or np.isinf(volatilidade):
                return np.nan
            
            return volatilidade * np.sqrt(252) # Annualize volatility
            
        except Exception as e:
            # print(f"Erro GARCH: {str(e)}")
            return np.nan

# =============================================================================
# CLASSE: MODELOS ESTATÍSTICOS DE SÉRIES TEMPORAIS
# =============================================================================

class ModelosEstatisticos:
    """Modelos estatísticos para previsão de séries temporais financeiras"""
    
    @staticmethod
    def ajustar_arima(series, order=(1, 1, 1), horizon=1):
        """
        Ajusta modelo ARIMA e faz previsão
        
        Args:
            series: Série temporal de preços ou retornos
            order: Tupla (p, d, q) para ARIMA
            horizon: Número de períodos à frente para prever
            
        Returns:
            Previsão e intervalo de confiança
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
    def ajustar_sarima(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), horizon=1):
        """
        Ajusta modelo SARIMA (ARIMA com sazonalidade) e faz previsão
        
        Args:
            series: Série temporal de preços ou retornos
            order: Tupla (p, d, q) para parte não-sazonal
            seasonal_order: Tupla (P, D, Q, s) para parte sazonal
            horizon: Número de períodos à frente para prever
            
        Returns:
            Previsão e intervalo de confiança
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
    def ajustar_var(dataframe_series, maxlags=5, horizon=1):
        """
        Ajusta modelo VAR (Vector Autoregression) para múltiplas séries
        
        Args:
            dataframe_series: DataFrame com múltiplas séries temporais
            maxlags: Número máximo de lags a considerar
            horizon: Número de períodos à frente para prever
            
        Returns:
            Previsões para todas as séries
        """
        try:
            df_limpo = dataframe_series.dropna()
            
            if len(df_limpo) < 100 or df_limpo.shape[1] < 2:
                return {'forecasts': {col: np.nan for col in dataframe_series.columns}}
            
            # Fit VAR model
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
    def ajustar_prophet(series, horizon=30):
        """
        Ajusta modelo Prophet (Facebook) para previsão de séries temporais
        
        Args:
            series: Série temporal com índice datetime
            horizon: Número de períodos à frente para prever
            
        Returns:
            Previsão e componentes
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
            modelo = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            
            # Suppress Prophet's verbose output
            import logging
            logging.getLogger('prophet').setLevel(logging.WARNING)
            
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
    def ensemble_estatistico(series, horizon=1):
        """
        Cria ensemble de modelos estatísticos (ARIMA, SARIMA, Prophet)
        
        Args:
            series: Série temporal
            horizon: Horizonte de previsão
            
        Returns:
            Previsão combinada e previsões individuais
        """
        try:
            previsoes = {}
            pesos = {}
            
            # ARIMA
            resultado_arima = ModelosEstatisticos.ajustar_arima(series, order=(1, 1, 1), horizon=horizon)
            if not np.isnan(resultado_arima['forecast']):
                previsoes['ARIMA'] = resultado_arima['forecast']
                # Weight by inverse AIC (lower AIC is better)
                pesos['ARIMA'] = 1.0 / (resultado_arima.get('aic', 1000) + 1)
            
            # SARIMA (only if enough data)
            if len(series.dropna()) >= 100:
                resultado_sarima = ModelosEstatisticos.ajustar_sarima(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), horizon=horizon)
                if not np.isnan(resultado_sarima['forecast']):
                    previsoes['SARIMA'] = resultado_sarima['forecast']
                    pesos['SARIMA'] = 1.0 / (resultado_sarima.get('aic', 1000) + 1)
            
            # Prophet
            resultado_prophet = ModelosEstatisticos.ajustar_prophet(series, horizon=horizon)
            if not np.isnan(resultado_prophet['forecast']):
                previsoes['Prophet'] = resultado_prophet['forecast']
                # Prophet doesn't have AIC, use equal weight or based on confidence interval width
                conf_width = resultado_prophet.get('yhat_upper', resultado_prophet['forecast']) - resultado_prophet.get('yhat_lower', resultado_prophet['forecast'])
                pesos['Prophet'] = 1.0 / (conf_width + 1) if not np.isnan(conf_width) and conf_width > 0 else 1.0
            
            if not previsoes:
                return {'ensemble_forecast': np.nan, 'individual_forecasts': {}}
            
            # Normalize weights
            total_peso = sum(pesos.values())
            pesos_norm = {k: v / total_peso for k, v in pesos.items()}
            
            # Weighted average
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
# CLASSE: ENSEMBLE DE MODELOS ML
# =============================================================================

class EnsembleML:
    """
    UPDATED: Ensemble de modelos ML com ponderação por AUC-ROC
    Agora inclui 9 modelos: XGBoost, LightGBM, CatBoost, RF, ET, KNN, SVC, LR, GNB
    """
    
    @staticmethod
    def treinar_ensemble(X, y, otimizar_optuna=False):
        """
        UPDATED: Treina ensemble expandido e retorna (modelos, auc_scores)
        """
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import GaussianNB
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
        
        modelos = {}
        auc_scores = {}
        
        # Configurações dos modelos
        configs = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            ),
            'catboost': CatBoostClassifier(
                iterations=100,
                depth=5,
                learning_rate=0.1,
                random_state=42,
                verbose=False
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'svc': SVC(probability=True, kernel='rbf', random_state=42),
            'logistic': LogisticRegression(max_iter=1000, random_state=42),
            'gaussian_nb': GaussianNB()
        }
        
        # Treina cada modelo com validação cruzada temporal
        tscv = TimeSeriesSplit(n_splits=3)
        
        for nome, modelo in configs.items():
            try:
                auc_fold_scores = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Treina modelo
                    modelo_fold = configs[nome]
                    modelo_fold.fit(X_train, y_train)
                    
                    # Calcula AUC
                    if hasattr(modelo_fold, 'predict_proba'):
                        y_proba = modelo_fold.predict_proba(X_val)[:, 1]
                    else:
                        y_proba = modelo_fold.decision_function(X_val)
                    
                    if len(np.unique(y_val)) >= 2:
                        auc = roc_auc_score(y_val, y_proba)
                        auc_fold_scores.append(auc)
                
                # AUC médio do modelo
                auc_medio = np.mean(auc_fold_scores) if auc_fold_scores else 0.5
                
                # Treina modelo final com todos os dados
                modelo.fit(X, y)
                modelos[nome] = modelo
                auc_scores[nome] = auc_medio
                
                print(f"  ✓ {nome}: AUC = {auc_medio:.3f}")
                
            except Exception as e:
                print(f"  ⚠️ {nome}: Erro - {str(e)[:50]}")
                continue
        
        return modelos, auc_scores
    
    @staticmethod
    def prever_ensemble_ponderado(modelos, auc_scores, X):
        """
        NEW: Previsão ponderada por AUC-ROC
        Apenas modelos com AUC > 0.50 são considerados
        """
        previsoes_ponderadas = []
        pesos_normalizados = []
        
        # Filtra modelos com AUC > 0.50
        modelos_validos = {nome: modelo for nome, modelo in modelos.items() 
                          if auc_scores.get(nome, 0) > 0.50}
        
        if not modelos_validos:
            # Fallback: usa todos os modelos com peso igual
            return EnsembleML.prever_ensemble(modelos, X)
        
        # Calcula pesos normalizados
        auc_validos = {nome: auc_scores[nome] for nome in modelos_validos.keys()}
        soma_auc = sum(auc_validos.values())
        
        for nome, modelo in modelos_validos.items():
            try:
                if hasattr(modelo, 'predict_proba'):
                    proba = modelo.predict_proba(X)[:, 1]
                else:
                    proba = modelo.decision_function(X)
                    proba = (proba - proba.min()) / (proba.max() - proba.min())
                
                peso = auc_validos[nome] / soma_auc
                previsoes_ponderadas.append(proba * peso)
                pesos_normalizados.append(peso)
                
            except Exception as e:
                print(f"  ⚠️ Erro ao prever com {nome}: {str(e)[:50]}")
                continue
        
        if not previsoes_ponderadas:
            return np.full(len(X), 0.5)
        
        return np.sum(previsoes_ponderadas, axis=0)
    
    @staticmethod
    def prever_ensemble(modelos, X):
        """Previsão simples (média) - mantido para compatibilidade"""
        previsoes = []
        
        for nome, modelo in modelos.items():
            try:
                if hasattr(modelo, 'predict_proba'):
                    proba = modelo.predict_proba(X)[:, 1]
                else:
                    proba = modelo.decision_function(X)
                    proba = (proba - proba.min()) / (proba.max() - proba.min())
                
                previsoes.append(proba)
            except:
                continue
        
        if not previsoes:
            return np.full(len(X), 0.5)
        
        return np.mean(previsoes, axis=0)
    
    @staticmethod
    def _otimizar_xgboost(X, y):
        """Otimiza hiperparâmetros do XGBoost com Optuna"""
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
            
            tscv = TimeSeriesSplit(n_splits=3)
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2: continue # Skip if classes are not mixed
                
                try:
                    modelo.fit(X_train, y_train)
                    proba = modelo.predict_proba(X_val)[:, 1]
                    score = roc_auc_score(y_val, proba)
                    scores.append(score)
                except ValueError: # Handle cases with only one class in split
                    continue
            
            return np.mean(scores) if scores else 0.0
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30, show_progress_bar=False) # Increased trials
        
        return study.best_params
    
    @staticmethod
    def _otimizar_lightgbm(X, y):
        """Otimiza hiperparâmetros do LightGBM com Optuna"""
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
            
            tscv = TimeSeriesSplit(n_splits=3)
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2: continue
                
                try:
                    modelo.fit(X_train, y_train)
                    proba = modelo.predict_proba(X_val)[:, 1]
                    score = roc_auc_score(y_val, proba)
                    scores.append(score)
                except ValueError:
                    continue
            
            return np.mean(scores) if scores else 0.0
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30, show_progress_bar=False) # Increased trials
        
        return study.best_params

# =============================================================================
# CLASSE: OTIMIZADOR DE PORTFÓLIO
# =============================================================================

class OtimizadorPortfolioAvancado:
    """Otimização de portfólio com volatilidade GARCH e CVaR"""
    
    def __init__(self, returns_df, garch_vols=None, fundamental_data=None, ml_predictions=None):
        self.returns = returns_df
        self.mean_returns = returns_df.mean() * 252
        
        if garch_vols is not None and not garch_vols.empty:
            self.cov_matrix = self._construir_matriz_cov_garch(returns_df, garch_vols)
        else:
            self.cov_matrix = returns_df.cov() * 252 # Fallback to historical covariance
            print("  ⚠️ GARCH volatilities not fully available, using historical covariance.")
        
        self.num_ativos = len(returns_df.columns)
        self.fundamental_data = fundamental_data # Store fundamental data
        self.ml_predictions = ml_predictions # Store ML predictions
    
    def _construir_matriz_cov_garch(self, returns_df, garch_vols):
        """Constrói matriz de covariância usando volatilidades GARCH"""
        corr_matrix = returns_df.corr()
        
        vol_array = np.array([
            garch_vols.get(ativo, returns_df[ativo].std() * np.sqrt(252)) # Fallback to std dev
            for ativo in returns_df.columns
        ])
        
        # Ensure vol_array is not all NaNs or zeros before outer product
        if np.isnan(vol_array).all() or np.all(vol_array == 0):
            print("  ⚠️ GARCH volatilities resulted in NaNs or zeros. Falling back to historical covariance.")
            return returns_df.cov() * 252
            
        cov_matrix = corr_matrix.values * np.outer(vol_array, vol_array)
        
        return pd.DataFrame(cov_matrix, index=returns_df.columns, columns=returns_df.columns)
    
    def estatisticas_portfolio(self, pesos):
        """Calcula retorno e volatilidade do portfólio"""
        p_retorno = np.dot(pesos, self.mean_returns)
        p_vol = np.sqrt(np.dot(pesos.T, np.dot(self.cov_matrix, pesos)))
        return p_retorno, p_vol
    
    def sharpe_negativo(self, pesos):
        """Sharpe ratio negativo para minimização"""
        p_retorno, p_vol = self.estatisticas_portfolio(pesos)
        # Handle cases where p_vol might be zero or very small
        if p_vol <= 1e-9: # Using a small epsilon to check for near zero
            return -100.0 # Return a very large negative value to avoid division by zero
        return -(p_retorno - TAXA_LIVRE_RISCO) / p_vol
    
    def minimizar_volatilidade(self, pesos):
        """Volatilidade do portfólio para minimização"""
        return self.estatisticas_portfolio(pesos)[1]
    
    # Added CVaR optimization as an option
    def calcular_cvar(self, pesos, confidence=0.95):
        """Calcula Conditional Value at Risk (CVaR)"""
        portfolio_returns = self.returns @ pesos
        sorted_returns = np.sort(portfolio_returns)
        
        # Calculate VaR (Value at Risk)
        var_index = int(np.floor((1 - confidence) * len(sorted_returns)))
        var = sorted_returns[var_index]
        
        # Calculate CVaR
        cvar = sorted_returns[sorted_returns <= var].mean()
        return cvar

    def cvar_negativo(self, pesos, confidence=0.95):
        """CVaR negativo para minimização"""
        return -self.calcular_cvar(pesos, confidence)

    def otimizar(self, estrategia='MaxSharpe', confidence_level=0.95):
        """Executa otimização do portfólio"""
        restricoes = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        # Adjusted limits to be more flexible for fewer assets
        num_assets = self.num_ativos
        limites = tuple((PESO_MIN, PESO_MAX) for _ in range(num_assets)) if num_assets > 1 else ((0,1),)
        chute_inicial = np.array([1.0 / num_assets] * num_assets)
        
        # Ensure initial guess respects bounds if necessary
        if num_assets > 0:
            min_w, max_w = limites[0]
            chute_inicial = np.clip(chute_inicial, min_w, max_w)
            chute_inicial /= np.sum(chute_inicial) # Re-normalize

        if estrategia == 'MinVolatility':
            objetivo = self.minimizar_volatilidade
        elif estrategia == 'CVaR':
            objetivo = lambda pesos: self.cvar_negativo(pesos, confidence=confidence_level)
            # Adjust constraints for CVaR if needed (e.g., allowing slightly more flexibility)
        else: # Default to MaxSharpe
            objetivo = self.sharpe_negativo
        
        try:
            # Increased max iterations and tolerance for SLSQP
            resultado = minimize(
                objetivo,
                chute_inicial,
                method='SLSQP',
                bounds=limites,
                constraints=restricoes,
                options={'maxiter': 500, 'ftol': 1e-6} # Increased iterations and tolerance
            )
            
            if resultado.success:
                # Normalize weights to sum to 1, especially important if min/max weights were tight
                final_weights = resultado.x / np.sum(resultado.x)
                return {ativo: peso for ativo, peso in zip(self.returns.columns, final_weights)}
            else:
                print(f"  ✗ Otimização falhou: {resultado.message}")
                # Fallback: equal weights if optimization fails
                return {ativo: 1.0 / self.num_ativos for ativo in self.returns.columns} if self.num_ativos > 0 else {}
        except Exception as e:
            print(f"  ✗ Erro na otimização: {str(e)}")
            # Fallback: equal weights
            return {ativo: 1.0 / self.num_ativos for ativo in self.returns.columns} if self.num_ativos > 0 else {}

# =============================================================================
# CLASSE PRINCIPAL: CONSTRUTOR DE PORTFÓLIO AUTOML
# =============================================================================

class ConstrutorPortfolioAutoML:
    """
    UPDATED: Construtor principal com governança e ensemble ponderado por AUC
    """
    
    def __init__(self, valor_investimento, periodo=PERIODO_DADOS):
        self.valor_investimento = valor_investimento
        self.periodo = periodo
        
        self.dados_por_ativo = {}
        self.dados_fundamentalistas = pd.DataFrame()
        self.dados_performance = pd.DataFrame()
        self.volatilidades_garch = {}
        self.predicoes_ml = {}
        self.predicoes_estatisticas = {} # Initialize for statistical predictions
        self.ativos_sucesso = []
        self.dados_macro = {} # Initialize for macro data
        self.metricas_performance = pd.DataFrame() # Initialize for performance metrics
        
        # NEW Attributes for Governance and Weighted Ensemble
        self.modelos_ml = {}
        self.auc_scores = {}  # NEW: Armazena AUC scores por ativo
        self.governanca_por_ativo = {}  # NEW: Governança por ativo
        
        self.ativos_selecionados = []
        self.alocacao_portfolio = {}
        self.metricas_portfolio = {}
        self.metodo_alocacao_atual = "Não Aplicado"
        self.justificativas_selecao = {}
        self.perfil_dashboard = {}
        self.pesos_atuais = {}
        self.scores_combinados = pd.DataFrame()
        self.matriz_covariancia = None # For optimizer
        self.retornos_esperados = {} # For optimizer
    
    def coletar_e_processar_dados(self, simbolos):
        """Coleta e processa dados de mercado com engenharia de features"""
        
        coletor = ColetorDados(periodo=self.periodo)
        if not coletor.coletar_e_processar_dados(simbolos):
            return False
        
        self.dados_por_ativo = coletor.dados_por_ativo
        self.dados_fundamentalistas = coletor.dados_fundamentalistas
        self.ativos_sucesso = coletor.ativos_sucesso
        self.dados_macro = coletor.dados_macro
        self.dados_performance = coletor.metricas_performance # Use the dataframe from ColetorDados
        
        print(f"\n✓ Coleta concluída: {len(self.ativos_sucesso)} ativos válidos\n")
        return True
    
    def calcular_volatilidades_garch(self):
        """
        Calcula volatilidades GARCH para todos os ativos
        Melhorado tratamento de erros e fallback
        """
        print("\n📊 Calculando volatilidades GARCH...")
        
        for simbolo in tqdm(self.ativos_sucesso, desc="Modelagem GARCH"):
            if simbolo not in self.dados_por_ativo or 'returns' not in self.dados_por_ativo[simbolo]:
                continue
                
            returns = self.dados_por_ativo[simbolo]['returns']
            
            garch_vol = VolatilidadeGARCH.ajustar_garch(returns, tipo_modelo='GARCH')
            
            if np.isnan(garch_vol):
                garch_vol = VolatilidadeGARCH.ajustar_garch(returns, tipo_modelo='EGARCH')
            
            if np.isnan(garch_vol):
                # Fallback to historical volatility if GARCH/EGARCH fails
                garch_vol = returns.std() * np.sqrt(252) if not returns.isnull().all() and returns.std() > 0 else np.nan
                if not np.isnan(garch_vol):
                    print(f"  ⚠️ {simbolo}: Usando volatilidade histórica (GARCH falhou)")
            
            self.volatilidades_garch[simbolo] = garch_vol
        
        print(f"✓ Volatilidades GARCH calculadas para {len([k for k, v in self.volatilidades_garch.items() if not np.isnan(v)])} ativos válidos\n")
    
    def treinar_modelos_ensemble(self, dias_lookback_ml=LOOKBACK_ML, otimizar=False):
        """
        UPDATED: Treina modelos ML com governança e AUC weighting
        """
        print("\n🤖 Treinando Modelos de Machine Learning...")
        
        # Expanded list of features to include more technical and macro indicators
        colunas_features_base = [
            'returns', 'log_returns', 'volatility_20', 'volatility_60', 'volatility_252',
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200',
            'ema_9', 'ema_12', 'ema_26', 'ema_50', 'ema_200',
            'wma_20', 'hma_20',
            'price_sma20_ratio', 'price_sma50_ratio', 'price_sma200_ratio',
            'sma20_sma50_cross', 'sma50_sma200_cross', 'death_cross',
            'rsi_7', 'rsi_14', 'rsi_21', 'rsi_28',
            'stoch_k', 'stoch_d', 'williams_r',
            'macd', 'macd_signal', 'macd_diff', 'macd_alt',
            'bb_width', 'bb_position', 'bb_pband',
            'kc_upper', 'kc_lower', 'kc_middle', 'kc_width',
            'dc_upper', 'dc_lower', 'dc_middle',
            'atr', 'atr_percent',
            'adx', 'adx_pos', 'adx_neg', 'cci',
            'momentum_10', 'momentum_20', 'momentum_60',
            'obv', 'cmf', 'mfi', 'vwap',
            'drawdown', 'max_drawdown_252',
            'close_lag_1', 'close_lag_5', 'close_lag_10', 'close_lag_20', 'close_lag_60',
            'returns_lag_1', 'returns_lag_5', 'returns_lag_10', 'returns_lag_20', 'returns_lag_60',
            'volume_lag_1', 'volume_lag_5', 'volume_lag_10', 'volume_lag_20', 'volume_lag_60',
            'returns_mean_5', 'returns_std_5', 'returns_skew_5', 'returns_kurt_5',
            'returns_mean_20', 'returns_std_20', 'returns_skew_20', 'returns_kurt_20',
            'returns_mean_60', 'returns_std_60', 'returns_skew_60', 'returns_kurt_60',
            'volume_mean_5', 'volume_std_5',
            'volume_mean_20', 'volume_std_20',
            'volume_mean_60', 'volume_std_60',
            'autocorr_1', 'autocorr_5', 'autocorr_10',
            'higher_high', 'lower_low',
            'day_of_week', 'month', 'quarter', 'day_of_month', 'week_of_year'
        ]
        
        # Adiciona features fundamentalistas e macro
        fundamental_features = list(self.dados_fundamentalistas.columns) if not self.dados_fundamentalistas.empty else []
        macro_features = list(self.dados_macro.keys()) if self.dados_macro else []
        
        colunas_features_totais = colunas_features_base + [f'fund_{f}' for f in fundamental_features] + [f'macro_{f.lower()}' for f in macro_features]
        
        self.predicoes_estatisticas = {}
        
        for ativo in tqdm(self.ativos_sucesso, desc="Treinamento ML"):
            if ativo not in self.dados_por_ativo:
                continue

            df = self.dados_por_ativo[ativo].copy()
            
            # Adiciona features fundamentalistas
            if not self.dados_fundamentalistas.empty and ativo in self.dados_fundamentalistas.index:
                for f_fund in fundamental_features:
                    col_name = f'fund_{f_fund}'
                    if col_name not in df.columns: # Avoid duplicating if already present
                        df[col_name] = self.dados_fundamentalistas.loc[ativo, f_fund]
            
            # Adiciona features macroeconômicas
            if self.dados_macro:
                for f_macro in macro_features:
                    col_name = f'macro_{f_macro.lower()}'
                    if col_name not in df.columns: # Avoid duplicating if already present
                        if f_macro in self.dados_macro and not self.dados_macro[f_macro].empty:
                           df[col_name] = self.dados_macro[f_macro].reindex(df.index, method='ffill') # Forward fill macro data
                        else:
                           df[col_name] = np.nan # Ensure column exists even if no data

            if 'Close' in df.columns and len(df) >= 100:
                try:
                    # Use closing prices for statistical models
                    close_series = df['Close']
                    
                    # Ensemble of statistical models
                    resultado_estatistico = ModelosEstatisticos.ensemble_estatistico(
                        close_series, 
                        horizon=dias_lookback_ml
                    )
                    
                    # Store statistical predictions
                    self.predicoes_estatisticas[ativo] = {
                        'forecast': resultado_estatistico.get('ensemble_forecast', np.nan),
                        'individual_forecasts': resultado_estatistico.get('individual_forecasts', {}),
                        'weights': resultado_estatistico.get('weights', {}),
                        'current_price': close_series.iloc[-1],
                        'predicted_direction': 1 if resultado_estatistico.get('ensemble_forecast', 0) > close_series.iloc[-1] else 0
                    }
                except Exception as e:
                    print(f"  ⚠️ Erro ao treinar modelos estatísticos para {ativo}: {str(e)[:50]}")
                    self.predicoes_estatisticas[ativo] = {
                        'forecast': np.nan,
                        'individual_forecasts': {},
                        'predicted_direction': 0.5
                    }
            
            # Cria target (previsão da direção do preço futuro) for ML models
            # Using shift(-dias_lookback_ml) to predict the direction 'dias_lookback_ml' days ahead
            df['Future_Direction'] = np.where(
                df['Close'].pct_change(dias_lookback_ml).shift(-dias_lookback_ml) > 0,
                1,
                0
            )
            
            # Seleciona features e remove NaNs
            features_para_treino = [f for f in colunas_features_totais if f in df.columns]
            df_treino = df[features_para_treino + ['Future_Direction']].dropna()
            
            if len(df_treino) < MIN_DIAS_HISTORICO: # Ensure enough data points after feature engineering
                self.previsoes_ml[ativo] = {
                    'predicted_proba_up': 0.5,
                    'auc_roc_score': np.nan,
                    'model_name': 'Dados Insuficientes'
                }
                continue
            
            X = df_treino[features_para_treino]
            y = df_treino['Future_Direction']
            
            if len(np.unique(y)) < 2: # Ensure there are at least two classes for classification
                self.previsoes_ml[ativo] = {
                    'predicted_proba_up': 0.5,
                    'auc_roc_score': np.nan,
                    'model_name': 'Classe Única'
                }
                continue
            
            try:
                # Treina ensemble
                modelos, auc_scores = EnsembleML.treinar_ensemble(X, y, otimizar_optuna=otimizar)
                
                if not modelos: # If no models were trained successfully
                    self.previsoes_ml[ativo] = {
                        'predicted_proba_up': 0.5,
                        'auc_roc_score': np.nan,
                        'model_name': 'Erro no Treino'
                    }
                    continue
                
                # Guarda modelos e scores para governança e previsão ponderada
                self.modelos_ml[ativo] = modelos
                self.auc_scores[ativo] = auc_scores
                
                # NEW: Inicializa governança
                self.governanca_por_ativo[ativo] = GovernancaModelo(ativo)
                
                # Adiciona métricas iniciais (treinando no dataset completo para ter as métricas finais)
                auc_medio = np.mean(list(auc_scores.values())) if auc_scores else 0.5
                
                # Calcula precision, recall, f1 (simplificado, pode ser melhorado com CV)
                # Usar previsão ponderada para calcular métricas de forma mais fiel
                y_pred_final = (EnsembleML.prever_ensemble_ponderado(modelos, auc_scores, X) > 0.5).astype(int)
                
                precision = precision_score(y, y_pred_final, zero_division=0)
                recall = recall_score(y, y_pred_final, zero_division=0)
                f1 = f1_score(y, y_pred_final, zero_division=0)
                
                self.governanca_por_ativo[ativo].adicionar_metricas(
                    auc_medio, precision, recall, f1
                )
                
                # Validação cruzada com TimeSeriesSplit
                scores = []
                # Adjust n_splits based on data length for meaningful CV
                n_splits_cv = min(5, len(X) // 100) if len(X) > 100 else 1
                tscv = TimeSeriesSplit(n_splits=max(2, n_splits_cv)) # Ensure at least 2 splits
                
                for train_idx, val_idx in tscv.split(X):
                    if len(train_idx) == 0 or len(val_idx) == 0: continue
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2: continue # Skip if classes are not mixed
                    
                    # Retrain models for CV fold (simplificado, pode ser mais complexo)
                    modelos_cv, auc_scores_cv = EnsembleML.treinar_ensemble(X_train, y_train, otimizar_optuna=False)
                    if not modelos_cv: continue

                    # NEW: Usa previsão ponderada para a validação
                    proba = EnsembleML.prever_ensemble_ponderado(modelos_cv, auc_scores_cv, X_val)
                    
                    if len(np.unique(y_val)) >= 2:
                        score = roc_auc_score(y_val, proba)
                        scores.append(score)
                
                auc_score_cv = np.mean(scores) if scores else np.nan
                
                # NEW: Usa previsão ponderada por AUC final
                # Use the last 'dias_lookback_ml' rows for prediction (features for the future point)
                # The actual prediction point is 'dias_lookback_ml' days ahead of the last data point
                last_features = df[features_para_treino].iloc[[-dias_lookback_ml]]
                
                if last_features.empty:
                     proba_final = 0.5
                else:
                    # Use the trained models and their AUC scores for weighted prediction
                    proba_final = EnsembleML.prever_ensemble_ponderado(modelos, auc_scores, last_features)[0]
                
                self.previsoes_ml[ativo] = {
                    'predicted_proba_up': proba_final,
                    'auc_roc_score': auc_score_cv, # Store CV score
                    'model_name': 'Ensemble Ponderado',
                    'num_models': len(modelos)
                }
                
                print(f"  ✓ {ativo}: Proba={proba_final:.3f}, AUC Médio CV={auc_score_cv:.3f}")
                
            except Exception as e:
                print(f"  ✗ Erro ML em {ativo}: {str(e)}")
                self.previsoes_ml[ativo] = {
                    'predicted_proba_up': 0.5,
                    'auc_roc_score': np.nan,
                    'model_name': 'Erro no Treino'
                }
        
        print(f"✓ Modelos ML treinados para {len(self.previsoes_ml)} ativos")
        print(f"✓ Modelos estatísticos treinados para {len(self.predicoes_estatisticas)} ativos\n")
    
    def pontuar_e_selecionar_ativos(self, horizonte_tempo):
        """Pontua e ranqueia ativos usando sistema multi-fator"""
        
        # Adjusted weights based on horizon and added ML weight dynamically
        if horizonte_tempo == "CURTO PRAZO":
            WEIGHT_PERF, WEIGHT_FUND, WEIGHT_TECH = 0.40, 0.10, 0.20 # Reduced tech for short-term
        elif horizonte_tempo == "LONGO PRAZO":
            WEIGHT_PERF, WEIGHT_FUND, WEIGHT_TECH = 0.40, 0.50, 0.10 # Increased fundamental for long-term
        else: # Médio Prazo
            WEIGHT_PERF, WEIGHT_FUND, WEIGHT_TECH = 0.40, 0.30, 0.30
        
        # Dynamically adjust total weights and incorporate ML
        total_non_ml_weight = WEIGHT_PERF + WEIGHT_FUND + WEIGHT_TECH
        
        # Ensure ML weight is factored in, adjusting others proportionally if needed
        final_ml_weight = WEIGHT_ML
        
        # If weights sum to more than 1, normalize or distribute
        if total_non_ml_weight + final_ml_weight > 1.0:
             # Scale down non-ML weights proportionally to make space for ML weight
             scale_factor = (1.0 - final_ml_weight) / total_non_ml_weight if total_non_ml_weight > 0 else 0
             WEIGHT_PERF *= scale_factor
             WEIGHT_FUND *= scale_factor
             WEIGHT_TECH *= scale_factor

        self.pesos_atuais = {
            'Performance': WEIGHT_PERF,
            'Fundamentos': WEIGHT_FUND,
            'Técnicos': WEIGHT_TECH,
            'ML': final_ml_weight # Store ML weight
        }
        
        # Combine dataframes
        combinado = self.dados_performance.join(self.dados_fundamentalistas, how='inner').copy()
        
        # Add current technical indicators and ML scores if available
        for asset in combinado.index:
            if asset in self.dados_por_ativo and 'rsi_14' in self.dados_por_ativo[asset].columns:
                df = self.dados_por_ativo[asset]
                combinado.loc[asset, 'rsi_current'] = df['rsi_14'].iloc[-1]
                combinado.loc[asset, 'macd_current'] = df['macd'].iloc[-1]
                combinado.loc[asset, 'bb_position_current'] = df['bb_position'].iloc[-1]

            if asset in self.predicoes_ml:
                ml_info = self.predicoes_ml[asset]
                combinado.loc[asset, 'ML_Proba'] = ml_info.get('predicted_proba_up', 0.5)
                combinado.loc[asset, 'ML_Confidence'] = ml_info.get('auc_roc_score', np.nan) if not pd.isna(ml_info.get('auc_roc_score')) else 0.5
        
        # Add statistical model predictions
        for asset in combinado.index:
            if asset in self.predicoes_estatisticas:
                stat_info = self.predicoes_estatisticas[asset]
                combinado.loc[asset, 'Stat_Forecast_Dir'] = stat_info.get('predicted_direction', 0.5) # 0 or 1 for direction
                combinado.loc[asset, 'Stat_Forecast_Price'] = stat_info.get('forecast', np.nan)

        # Calculate scores
        scores = pd.DataFrame(index=combinado.index)
        
        # Score de Performance (using Sharpe Ratio)
        # Normalize Sharpe Ratio (assuming typical range, adjust if needed)
        performance_norm = EngenheiroFeatures._normalizar(combinado.get('sharpe', pd.Series(0, index=combinado.index)), maior_melhor=True)
        scores['performance_score'] = performance_norm * WEIGHT_PERF
        
        # Score Fundamentalista
        # Example: Combine P/E (lower is better) and ROE (higher is better)
        pe_score = EngenheiroFeatures._normalizar(combinado.get('pe_ratio', pd.Series(combinado['pe_ratio'].median(), index=combinado.index)), maior_melhor=False)
        roe_score = EngenheiroFeatures._normalizar(combinado.get('roe', pd.Series(combinado['roe'].median(), index=combinado.index)), maior_melhor=True)
        fund_score = (pe_score * 0.5 + roe_score * 0.5) * WEIGHT_FUND
        scores['fundamental_score'] = fund_score
        
        # Score Técnico
        tech_score = 0
        if WEIGHT_TECH > 0:
            # RSI: Score higher when closer to 50 (less overbought/oversold)
            rsi_proximity_score = 100 - abs(combinado.get('rsi_current', pd.Series(50, index=combinado.index)) - 50)
            rsi_norm = EngenheiroFeatures._normalizar(rsi_proximity_score.clip(0, 100), maior_melhor=True)
            
            # MACD: Score higher for positive MACD values (uptrend momentum)
            macd_norm = EngenheiroFeatures._normalizar(combinado.get('macd_current', pd.Series(0, index=combinado.index)), maior_melhor=True)
            
            # Bollinger Band Position: Score higher when price is not at extremes (e.g., mid-band)
            bb_pos_norm = EngenheiroFeatures._normalizar(combinado.get('bb_position_current', pd.Series(0.5, index=combinado.index)), maior_melhor=False) # Higher score for mid-band (0.5)
            
            tech_score += rsi_norm * 0.4 # Weight RSI
            tech_score += macd_norm * 0.3 # Weight MACD
            tech_score += bb_pos_norm * 0.3 # Weight Bollinger Position
            
        scores['technical_score'] = tech_score * WEIGHT_TECH
        
        # Score de ML
        ml_proba_norm = EngenheiroFeatures._normalizar(combinado.get('ML_Proba', pd.Series(0.5, index=combinado.index)), maior_melhor=True)
        ml_confidence_norm = EngenheiroFeatures._normalizar(combinado.get('ML_Confidence', pd.Series(0.5, index=combinado.index)), maior_melhor=True)
        scores['ml_score_weighted'] = (ml_proba_norm * 0.6 + ml_confidence_norm * 0.4) * WEIGHT_ML # Weighted ML score
        
        # Score de Modelo Estatístico (direção prevista)
        stat_dir_score = EngenheiroFeatures._normalizar(combinado.get('Stat_Forecast_Dir', pd.Series(0.5, index=combinado.index)), maior_melhor=True)
        # Add weight for statistical models if desired, e.g., 0.1 * stat_dir_score
        
        # Score total combinado
        total_score_base = (WEIGHT_PERF + WEIGHT_FUND + WEIGHT_TECH)
        scores['base_score'] = scores[[
            'performance_score',
            'fundamental_score',
            'technical_score'
        ]].sum(axis=1) / total_score_base if total_score_base > 0 else pd.Series(0, index=scores.index)
        
        scores['total_score'] = scores['base_score'] + scores['ml_score_weighted'] #+ (0.1 * stat_dir_score if WEIGHT_STAT > 0 else 0)
        
        self.scores_combinados = scores.join(combinado).sort_values('total_score', ascending=False)
        
        # Seleção final com diversificação setorial
        ranked_assets = self.scores_combinados.index.tolist()
        final_portfolio = []
        selected_sectors = set()
        
        # Ensure NUM_ATIVOS_PORTFOLIO is not greater than available assets
        num_assets_to_select = min(NUM_ATIVOS_PORTFOLIO, len(ranked_assets))

        for asset in ranked_assets:
            sector = self.dados_fundamentalistas.loc[asset, 'sector'] if asset in self.dados_fundamentalistas.index and 'sector' in self.dados_fundamentalistas.columns else 'Unknown'
            
            # Diversification logic: try to add assets from different sectors, up to 2 per sector if possible
            if sector not in selected_sectors or len(final_portfolio) < NUM_ATIVOS_PORTFOLIO:
                final_portfolio.append(asset)
                selected_sectors.add(sector)
            
            if len(final_portfolio) >= num_assets_to_select:
                break
        
        # If diversification logic didn't yield enough assets (e.g., all top assets in same sector)
        # Fill up to NUM_ATIVOS_PORTFOLIO from the next best ranked assets regardless of sector
        if len(final_portfolio) < num_assets_to_select:
            for asset in ranked_assets:
                if asset not in final_portfolio:
                    final_portfolio.append(asset)
                    if len(final_portfolio) >= num_assets_to_select:
                        break

        self.ativos_selecionados = final_portfolio
        return self.ativos_selecionados
    
    def otimizar_alocacao(self, nivel_risco):
        """Otimiza alocação de capital usando teoria de Markowitz com GARCH volatilities"""
        
        if not self.ativos_selecionados or len(self.ativos_selecionados) < 1: # Require at least one asset
            self.metodo_alocacao_atual = "ERRO: Ativos Insuficientes"
            return {}
        
        # Select only the returns data for the chosen assets
        available_assets_returns = {s: self.dados_por_ativo[s]['returns']
                                    for s in self.ativos_selecionados if s in self.dados_por_ativo and 'returns' in self.dados_por_ativo[s]}
        
        if not available_assets_returns:
            print("  ✗ Não foi possível obter dados de retorno para os ativos selecionados.")
            self.metodo_alocacao_atual = "ERRO: Dados de Retorno Insuficientes"
            return {}
            
        final_returns_df = pd.DataFrame(available_assets_returns).dropna()
        
        # Ensure there are enough data points for optimization
        if final_returns_df.shape[0] < 50: # Minimum number of data points for meaningful optimization
            print(f"  ⚠️ Dados de retorno insuficientes ({final_returns_df.shape[0]} dias) para otimização complexa. Usando pesos iguais.")
            weights = {asset: 1.0 / len(self.ativos_selecionados) for asset in self.ativos_selecionados}
            self.metodo_alocacao_atual = 'PESOS IGUAIS (Dados insuficientes)'
        else:
            # Use GARCH volatilities if available, otherwise fallback to historical std dev
            garch_vols_selecionados = {}
            for s in self.ativos_selecionados:
                if s in self.volatilidades_garch and not np.isnan(self.volatilidades_garch[s]):
                    garch_vols_selecionados[s] = self.volatilidades_garch[s]
                elif s in final_returns_df.columns and final_returns_df[s].std() > 0:
                    garch_vols_selecionados[s] = final_returns_df[s].std() * np.sqrt(252) # Fallback to historical std dev
                    print(f"  ⚠️ GARCH vol missing for {s}, using historical std dev.")
            
            # Ensure all selected assets are present in the returns dataframe and garch_vols dictionary
            # Filter garch_vols to match the columns in final_returns_df
            garch_vols_filtered = {asset: vol for asset, vol in garch_vols_selecionados.items() if asset in final_returns_df.columns}

            # Pass fundamental and ML data to optimizer if needed for future strategies
            optimizer = OtimizadorPortfolioAvancado(final_returns_df, garch_vols=garch_vols_filtered,
                                                    fundamental_data=self.dados_fundamentalistas.loc[self.ativos_selecionados] if self.dados_fundamentalistas is not None else None,
                                                    ml_predictions=self.predicoes_ml)
            
            strategy = 'MaxSharpe' # Default strategy
            if 'CONSERVADOR' in nivel_risco or 'INTERMEDIÁRIO' in nivel_risco:
                strategy = 'MinVolatility'
            elif 'AVANÇADO' in nivel_risco:
                strategy = 'CVaR' # CVaR for aggressive investors
                
            weights = optimizer.otimizar(estrategia=strategy)
            self.metodo_alocacao_atual = f'{strategy} ({optimizer.cov_matrix.name if hasattr(optimizer.cov_matrix, "name") else "Covariance"})' # Capture strategy and matrix type
            
        # Normalize weights to ensure they sum to 1, handle potential issues from optimization
        if weights and sum(weights.values()) > 0:
            total_weight = sum(weights.values())
            weights = {k: v / total_weight for k, v in weights.items()}
        elif self.ativos_selecionados: # Fallback to equal weights if weights are invalid
            weights = {asset: 1.0 / len(self.ativos_selecionados) for asset in self.ativos_selecionados}
            self.metodo_alocacao_atual += " | FALLBACK Pesos Iguais"
        else: # No assets selected, return empty
            weights = {}

        # Apply minimum/maximum weight constraints after normalization if they weren't strictly enforced
        # This is a post-hoc adjustment, ideally handled by the optimizer itself.
        # For simplicity, we'll rely on the optimizer's bounds and normalization.

        self.alocacao_portfolio = {
            s: {
                'weight': w,
                'amount': self.valor_investimento * w
            }
            for s, w in weights.items() if s in self.ativos_selecionados # Ensure only selected assets get allocation
        }
        
        # Adjust allocation if some assets were excluded (e.g., due to data issues)
        allocated_assets = list(self.alocacao_portfolio.keys())
        if len(allocated_assets) < len(self.ativos_selecionados):
            print(f"  ⚠️ Some selected assets were not allocated: {set(self.ativos_selecionados) - set(allocated_assets)}")
            # Option: Distribute remaining capital or acknowledge missing allocation. Here, we just note it.
        
        return self.alocacao_portfolio
    
    def calcular_metricas_portfolio(self):
        """Calcula métricas consolidadas do portfólio"""
        
        if not self.ativos_selecionados or not self.alocacao_portfolio:
            return {}
        
        # Filter returns data to include only allocated assets and ensure they exist
        allocated_assets = list(self.alocacao_portfolio.keys())
        valid_returns_data = {
            s: self.dados_por_ativo[s]['returns']
            for s in allocated_assets
            if s in self.dados_por_ativo and 'returns' in self.dados_por_ativo[s]
        }
        
        returns_df = pd.DataFrame(valid_returns_data).dropna()
        
        if returns_df.empty:
            print("  ✗ No valid returns data for calculating portfolio metrics.")
            return {}
        
        # Ensure weights match the columns in returns_df
        weights_dict = {s: self.alocacao_portfolio[s]['weight'] for s in returns_df.columns}
        weights = np.array([weights_dict[s] for s in returns_df.columns])
        
        # Normalize weights again in case of missing assets
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            print("  ✗ Weights sum to zero, cannot calculate portfolio metrics.")
            return {}

        portfolio_returns = (returns_df * weights).sum(axis=1)
        
        # Calculate annualized metrics
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - TAXA_LIVRE_RISCO) / annual_volatility if annual_volatility > 0 else 0
        
        # Calculate Max Drawdown from the calculated portfolio returns
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
        """Gera justificativas textuais para seleção"""
        
        for simbolo in self.ativos_selecionados:
            if simbolo not in self.dados_por_ativo: continue # Skip if no data for this asset

            justification = []
            
            # Performance metrics
            if simbolo in self.dados_performance.index:
                perf = self.dados_performance.loc[simbolo]
                justification.append(
                    f"Perf: Sharpe {perf.get('sharpe', np.nan):.3f}, "
                    f"Retorno Anual {perf.get('retorno_anual', np.nan)*100:.2f}%, "
                    f"Vol. Anual {perf.get('volatilidade_anual', np.nan)*100:.2f}%"
                )
            
            # Fundamental metrics
            if simbolo in self.dados_fundamentalistas.index:
                fund = self.dados_fundamentalistas.loc[simbolo]
                # Select relevant fundamental features for justification
                justification.append(
                    f"Fund: P/L {fund.get('pe_ratio', np.nan):.2f}, ROE {fund.get('roe', np.nan):.2f}%, "
                    f"Margem Liq. {fund.get('profit_margin', np.nan):.2f}%"
                )
            
            # ML predictions
            if simbolo in self.predicoes_ml:
                ml = self.predicoes_ml[simbolo]
                proba_up = ml.get('predicted_proba_up', 0.5)
                auc_score = ml.get('auc_roc_score', np.nan)
                auc_str = f"{auc_score:.3f}" if not pd.isna(auc_score) else "N/A"
                
                justification.append(
                    f"ML: Prob. Alta {proba_up*100:.1f}% (AUC {auc_str})"
                )
            
            # Volatility
            if simbolo in self.volatilidades_garch and not np.isnan(self.volatilidades_garch[simbolo]):
                vol_garch = self.volatilidades_garch[simbolo]
                justification.append(f"Vol. GARCH: {vol_garch*100:.2f}%")
            elif simbolo in self.dados_performance.index: # Fallback to historical volatility
                vol_hist = self.dados_performance.loc[simbolo].get('volatilidade_anual', np.nan)
                if not np.isnan(vol_hist):
                    justification.append(f"Vol. Histórica: {vol_hist*100:.2f}%")

            # Statistical Model Prediction
            if simbolo in self.predicoes_estatisticas:
                stat_pred = self.predicoes_estatisticas[simbolo]
                forecast_price = stat_pred.get('forecast')
                current_price = stat_pred.get('current_price')
                
                if forecast_price is not None and current_price is not None and not np.isnan(forecast_price) and not np.isnan(current_price) and current_price != 0:
                    pred_change_pct = ((forecast_price - current_price) / current_price) * 100
                    justification.append(f"Estatístico: Prev. Preço R$ {forecast_price:.2f} ({pred_change_pct:.2f}%)")
                elif stat_pred.get('predicted_direction') is not None:
                    direction_str = "Alta" if stat_pred['predicted_direction'] == 1 else "Baixa" if stat_pred['predicted_direction'] == 0 else "Neutro"
                    justification.append(f"Estatístico: Prev. Direção ({direction_str})")
            
            self.justificativas_selecao[simbolo] = " | ".join(justification)
        
        return self.justificativas_selecao
    
    def executar_pipeline(self, simbolos_customizados, perfil_investidor, otimizar_ml=False):
        """Executa pipeline completo"""
        
        self.perfil = perfil_investidor # Store profile
        ml_lookback_days = perfil_investidor.get('ml_lookback_days', LOOKBACK_ML)
        nivel_risco = perfil_investidor.get('risk_level', 'MODERADO')
        horizonte_tempo = perfil_investidor.get('time_horizon', 'MÉDIO PRAZO')
        
        # Etapa 1: Coleta de dados
        if not self.coletar_e_processar_dados(simbolos_customizados):
            return False
        
        # Etapa 2: Volatilidades GARCH
        self.calcular_volatilidades_garch()
        
        # Etapa 3: Treinamento ML e Estatístico
        self.treinar_modelos_ensemble(dias_lookback_ml=ml_lookback_days, otimizar=otimizar_ml)
        
        # Etapa 4: Pontuação e seleção
        self.pontuar_e_selecionar_ativos(horizonte_tempo=horizonte_tempo)
        
        # Etapa 5: Otimização de alocação
        self.otimizar_alocacao(nivel_risco=nivel_risco)
        
        # Etapa 6: Métricas do portfólio
        self.calcular_metricas_portfolio()
        
        # Etapa 7: Justificativas
        self.gerar_justificativas()
        
        return True

# =============================================================================
# CLASSE: ANALISADOR INDIVIDUAL DE ATIVOS
# =============================================================================

class AnalisadorIndividualAtivos:
    """Análise completa de ativos individuais com máximo de features"""
    
    @staticmethod
    def calcular_todos_indicadores_tecnicos(hist):
        """Calcula TODOS os indicadores técnicos possíveis"""
        df = hist.copy()
        
        # Retornos e volatilidade
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        df['volatility_60'] = df['returns'].rolling(window=60).std() * np.sqrt(252)
        
        # Médias móveis (SMA)
        for periodo in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{periodo}'] = SMAIndicator(close=df['Close'], window=periodo).sma_indicator()
        
        # Médias móveis exponenciais (EMA)
        for periodo in [9, 12, 26, 50, 200]:
            df[f'ema_{periodo}'] = EMAIndicator(close=df['Close'], window=periodo).ema_indicator()
        
        # RSI (Relative Strength Index)
        for periodo in [7, 14, 21, 28]:
            df[f'rsi_{periodo}'] = RSIIndicator(close=df['Close'], window=periodo).rsi()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        df['williams_r'] = WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close'], lbp=14).williams_r()
        
        # MACD (Moving Average Convergence Divergence)
        macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # MACD alternativo (5, 35, 5)
        macd_alt = MACD(close=df['Close'], window_slow=35, window_fast=5, window_sign=5)
        df['macd_alt'] = macd_alt.macd()
        
        # Bollinger Bands
        bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_pband'] = bb.bollinger_pband()
        
        # ATR (Average True Range)
        atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
        df['atr'] = atr.average_true_range()
        df['atr_percent'] = (df['atr'] / df['Close']) * 100
        
        # ADX (Average Directional Index)
        adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        
        # CCI (Commodity Channel Index)
        df['cci'] = CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=20).cci()
        
        # ROC (Rate of Change)
        df['roc_12'] = ROCIndicator(close=df['Close'], window=12).roc()
        df['roc_20'] = ROCIndicator(close=df['Close'], window=20).roc()
        
        # Momentum
        df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        # Ichimoku Cloud
        high_9 = df['High'].rolling(window=9).max()
        low_9 = df['Low'].rolling(window=9).min()
        df['tenkan_sen'] = (high_9 + low_9) / 2
        
        high_26 = df['High'].rolling(window=26).max()
        low_26 = df['Low'].rolling(window=26).min()
        df['kijun_sen'] = (high_26 + low_26) / 2
        
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        
        high_52 = df['High'].rolling(window=52).max()
        low_52 = df['Low'].rolling(window=52).min()
        df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
        
        df['chikou_span'] = df['Close'].shift(-26)
        
        # VWAP (Volume Weighted Average Price)
        df['vwap'] = VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).volume_weighted_average_price()
        
        # Chaikin Money Flow
        df['cmf_20'] = ChaikinMoneyFlowIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=20).chaikin_money_flow()
        
        # Money Flow Index (MFI)
        df['mfi_14'] = MFIIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=14).money_flow_index()
        
        # On-Balance Volume (OBV)
        df['obv'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
        
        # Parabolic SAR (simplified) - More complex SAR involves trailing stops
        # Using a rolling mean as a proxy for now, a full SAR implementation would be complex
        df['sar_proxy'] = df['Close'].rolling(window=5).mean()
        
        # Return NaN for any column that is entirely NaN after calculations (e.g., short history indicators)
        return df.dropna(axis=1, how='all')
    
    @staticmethod
    def calcular_features_fundamentalistas_expandidas(ticker_obj):
        """Extrai máximo de features fundamentalistas possíveis"""
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
            
            # Rentabilidade
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
            
            # Crescimento
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
            
            # Preço
            'current_price': info.get('currentPrice', np.nan),
            'target_high_price': info.get('targetHighPrice', np.nan),
            'target_low_price': info.get('targetLowPrice', np.nan),
            'target_mean_price': info.get('targetMeanPrice', np.nan),
            'recommendation': info.get('recommendationKey', 'none')
        }
        
        # Convert potential None values to np.nan
        for key, value in features.items():
            if value is None:
                features[key] = np.nan
        
        return features
    
    @staticmethod
    def realizar_clusterizacao_pca(dados_ativos, n_clusters=5):
        """Realiza clusterização K-means + PCA para análise de similaridade"""
        
        # Prepara dados numéricos
        features_numericas = dados_ativos.select_dtypes(include=[np.number]).copy()
        features_numericas = features_numericas.replace([np.inf, -np.inf], np.nan)
        
        # Impute missing values with median before scaling
        for col in features_numericas.columns:
            if features_numericas[col].isnull().any():
                median_val = features_numericas[col].median()
                features_numericas[col] = features_numericas[col].fillna(median_val)
        
        # Drop columns that are still all NaN or have zero variance after imputation
        features_numericas = features_numericas.dropna(axis=1, how='all')
        features_numericas = features_numericas.loc[:, (features_numericas.std() > 1e-6)] # Keep columns with variance

        if features_numericas.empty or len(features_numericas) < n_clusters:
            print(f"  ⚠️ Insufficient valid numeric features ({len(features_numericas.columns)}) or data points ({len(features_numericas)}) for clustering.")
            return None, None, None
        
        # Normalização
        scaler = StandardScaler()
        dados_normalizados = scaler.fit_transform(features_numericas)
        
        # PCA
        # Reduce to max 3 components or number of features if less than 3
        n_pca_components = min(3, len(features_numericas.columns))
        pca = PCA(n_components=n_pca_components)
        componentes_pca = pca.fit_transform(dados_normalizados)
        
        # K-means
        # Adjust n_clusters if there are fewer data points than requested clusters
        actual_n_clusters = min(n_clusters, len(features_numericas))
        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10) # Added n_init for robustness
        clusters = kmeans.fit_predict(dados_normalizados)
        
        # Cria DataFrame de resultados
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
            "📊 Ibovespa Completo (87 ativos)",
            "🌐 Lista Completa de Ativos (300+ ativos)",
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

def aba_construtor_portfolio():
    """Aba 3: Questionário e Construção de Portfólio"""
    
    if 'ativos_para_analise' not in st.session_state or not st.session_state.ativos_para_analise:
        st.warning("⚠️ Por favor, selecione os ativos na aba **'Seleção de Ativos'** primeiro.")
        return
    
    # Initialize session state variables if they don't exist
    if 'builder' not in st.session_state:
        st.session_state.builder = None
        st.session_state.builder_complete = False
        st.session_state.profile = {}
    
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
            options_liquidity = [
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
                    options=options_liquidity,
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
                otimizar_ml = st.checkbox("Ativar otimização Optuna (mais lento, melhor precisão)", value=False, key='optimize_ml_checkbox')
            
            submitted = st.form_submit_button("🚀 Gerar Portfólio Otimizado", type="primary")
            
            if submitted:
                # Analisa perfil
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
                
                # Cria construtor
                builder = ConstrutorPortfolioAutoML(investment)
                st.session_state.builder = builder
                
                # Executa pipeline
                with st.spinner(f'Criando portfólio para **PERFIL {risk_level}** ({horizon})...'):
                    success = builder.executar_pipeline(
                        simbolos_customizados=st.session_state.ativos_para_analise,
                        perfil_investidor=st.session_state.profile,
                        otimizar_ml=otimizar_ml
                    )
                    
                    if not success:
                        st.error("Falha ao coletar dados suficientes ou processar os ativos. Tente novamente com uma seleção diferente de ativos ou verifique sua conexão.")
                        # Optionally clear builder and profile to allow retry
                        st.session_state.builder = None
                        st.session_state.profile = {}
                        return
                    
                    st.session_state.builder_complete = True
                    st.rerun() # Rerun the app to show results
    
    # FASE 2: RESULTADOS
    else:
        builder = st.session_state.builder
        profile = st.session_state.profile
        assets = builder.ativos_selecionados
        allocation = builder.alocacao_portfolio
        
        st.markdown('## ✅ Portfólio Otimizado Gerado')
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Perfil de Risco", profile.get('risk_level', 'N/A'), f"Score: {profile.get('risk_score', 'N/A')}")
        col2.metric("Horizonte", profile.get('time_horizon', 'N/A'))
        col3.metric("Sharpe Ratio", f"{builder.metricas_portfolio.get('sharpe_ratio', 0):.3f}")
        col4.metric("Estratégia", builder.metodo_alocacao_atual.split('(')[0].strip()) # Extract strategy name
        
        # Button to restart analysis
        if st.button("🔄 Recomeçar Análise", key='recomecar_analysis'):
            # Clear relevant session state variables to restart
            st.session_state.builder_complete = False
            st.session_state.builder = None
            st.session_state.profile = {}
            st.session_state.ativos_para_analise = [] # Clear asset selection as well
            st.session_state.analisar_ativo_triggered = False # Clear analysis trigger too
            st.rerun()
        
        st.markdown("---")
        
        # Dashboard de resultados (código existente mantido e melhorado)
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Alocação", "📈 Performance", "🔬 Análise ML", "📉 Volatilidade GARCH", "❓ Justificativas"
        ])
        
        with tab1:
            col_alloc, col_table = st.columns([1, 2])
            
            with col_alloc:
                st.markdown('#### Alocação de Capital')
                alloc_data = pd.DataFrame([
                    {'Ativo': a, 'Peso (%)': allocation[a]['weight'] * 100}
                    for a in assets if a in allocation and allocation[a]['weight'] > 0.001 # Filter small weights for pie chart
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
                    if asset in allocation and allocation[asset]['weight'] > 0: # Only show allocated assets
                        weight = allocation[asset]['weight']
                        amount = allocation[asset]['amount']
                        sector = builder.dados_fundamentalistas.loc[asset, 'sector'] if asset in builder.dados_fundamentalistas.index and 'sector' in builder.dados_fundamentalistas.columns else 'Unknown'
                        ml_info = builder.predicoes_ml.get(asset, {})
                        stat_info = builder.predicoes_estatisticas.get(asset, {}) # Get statistical info
                        
                        alloc_table.append({
                            'Ativo': asset.replace('.SA', ''), # Clean ticker name
                            'Setor': sector,
                            'Peso (%)': f"{weight * 100:.2f}",
                            'Valor (R$)': f"R$ {amount:,.2f}",
                            'ML Prob. Alta (%)': f"{ml_info.get('predicted_proba_up', 0.5)*100:.1f}",
                            'ML AUC': f"{ml_info.get('auc_roc_score', 0):.3f}" if not pd.isna(ml_info.get('auc_roc_score')) else "N/A",
                            'Estatístico Dir.': f"{stat_info.get('predicted_direction', 0.5)*100:.0f}%" if stat_info.get('predicted_direction') is not None else "N/A", # Display as percentage
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
                        'Nº Modelos': ml_info.get('num_models', 0)
                    })
            
            df_ml = pd.DataFrame(ml_data)
            
            if not df_ml.empty:
                fig_ml = go.Figure()
                
                fig_ml.add_trace(go.Bar(
                    x=df_ml['Ativo'],
                    y=df_ml['Prob. Alta (%)'],
                    marker=dict(
                        color=df_ml['Prob. Alta (%)'],
                        colorscale='RdYlGn', # Green for high probability, Red for low
                        showscale=True,
                        colorbar=dict(title="Prob. (%)")
                    ),
                    text=df_ml['Prob. Alta (%)'].round(1),
                    textposition='outside'
                ))
                
                fig_layout = obter_template_grafico()
                fig_layout['title']['text'] = "Probabilidade de Alta Futura (ML Ensemble)"
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
                # Ensure asset exists in performance metrics and GARCH calculations
                if ativo in builder.dados_performance.index and ativo in builder.volatilidades_garch:
                    vol_hist = builder.dados_performance.loc[ativo, 'volatilidade_anual'] if 'volatilidade_anual' in builder.dados_performance.columns else np.nan
                    vol_garch = builder.volatilidades_garch.get(ativo)
                    
                    # Handle cases where GARCH might have failed or returned NaN
                    if vol_garch is not None and not np.isnan(vol_garch):
                        status = '✓ GARCH Ajustado'
                        vol_display = vol_garch
                    elif vol_hist is not None and not np.isnan(vol_hist): # Fallback to historical if GARCH failed
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
                
                # Filter out NAs for plotting if necessary, or handle directly
                plot_df_garch = df_garch[df_garch['Vol. GARCH (%)'] != 'N/A'].copy() # Filter GARCH adjusted for plotting bars
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

def aba_analise_individual():
    """Aba 4: Análise Individual de Ativos - COM CACHE"""
    
    st.markdown("## 🔍 Análise Individual Completa de Ativos")
    
    # Determine available assets for selection
    if 'ativos_para_analise' in st.session_state and st.session_state.ativos_para_analise:
        ativos_disponiveis = st.session_state.ativos_para_analise
    else:
        ativos_disponiveis = ATIVOS_IBOVESPA # Default to Ibovespa if no selection made yet
        if not ativos_disponiveis:
            ativos_disponiveis = TODOS_ATIVOS # Fallback to all if Ibovespa is empty
            
    if not ativos_disponiveis:
        st.error("Nenhum ativo disponível para análise. Verifique as configurações ou selecione ativos.")
        return

    col1, col2 = st.columns([3, 1])
    
    with col1:
        ativo_selecionado = st.selectbox(
            "Selecione um ativo para análise detalhada:",
            options=ativos_disponiveis,
            format_func=lambda x: x.replace('.SA', '') if isinstance(x, str) else x, # Format ticker names
            key='individual_asset_select'
        )
    
    with col2:
        if st.button("🔄 Analisar Ativo", key='analyze_asset_button', type="primary"):
            st.session_state.analisar_ativo_triggered = True # Flag to trigger analysis
    
    # Check if analysis should be performed
    if 'analisar_ativo_triggered' not in st.session_state or not st.session_state.analisar_ativo_triggered:
        st.info("👆 Selecione um ativo e clique em 'Analisar Ativo' para começar a análise completa.")
        return
    
    # Execute analysis
    @st.cache_data(ttl=timedelta(hours=6))
    def carregar_dados_ativo_individual(ativo_selecionado):
        """Carrega dados de um ativo individual com cache de 6 horas"""
        try:
            ticker = yf.Ticker(ativo_selecionado)
            hist = ticker.history(period='max')
            
            if hist.empty:
                return None, None
            
            return hist, ticker
        except Exception as e:
            print(f"Erro ao carregar {ativo_selecionado}: {str(e)}")
            return None, None

    with st.spinner(f"Analisando {ativo_selecionado}..."):
        hist, ticker = carregar_dados_ativo_individual(ativo_selecionado)
        
        if hist is None:
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
            
            # Display key metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            preco_atual = df_completo['Close'].iloc[-1] if not df_completo.empty and 'Close' in df_completo.columns else np.nan
            variacao_dia = df_completo['returns'].iloc[-1] * 100 if not df_completo.empty and 'returns' in df_completo.columns else np.nan
            volume_medio = df_completo['Volume'].mean() if not df_completo.empty and 'Volume' in df_completo.columns else np.nan
            
            col1.metric("Preço Atual", f"R$ {preco_atual:.2f}" if not np.isnan(preco_atual) else "N/A", f"{variacao_dia:+.2f}%" if not np.isnan(variacao_dia) else "N/A")
            col2.metric("Volume Médio", f"{volume_medio:,.0f}" if not np.isnan(volume_medio) else "N/A")
            col3.metric("Setor", features_fund.get('sector', 'N/A'))
            col4.metric("Indústria", features_fund.get('industry', 'N/A'))
            col5.metric("Beta", f"{features_fund.get('beta', np.nan):.2f}" if not np.isnan(features_fund.get('beta')) else "N/A")
            
            # Candlestick chart with Volume
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
            st.markdown("### Indicadores Técnicos")
            
            # Display key indicators
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            col1.metric("RSI (14)", f"{df_completo['rsi_14'].iloc[-1]:.2f}" if 'rsi_14' in df_completo.columns else "N/A")
            col2.metric("MACD", f"{df_completo['macd'].iloc[-1]:.4f}" if 'macd' in df_completo.columns else "N/A")
            col3.metric("Stoch %K", f"{df_completo['stoch_k'].iloc[-1]:.2f}" if 'stoch_k' in df_completo.columns else "N/A")
            col4.metric("ADX", f"{df_completo['adx'].iloc[-1]:.2f}" if 'adx' in df_completo.columns else "N/A")
            col5.metric("CCI", f"{df_completo['cci'].iloc[-1]:.2f}" if 'cci' in df_completo.columns else "N/A")
            col6.metric("ATR (%)", f"{df_completo['atr_percent'].iloc[-1]:.2f}%" if 'atr_percent' in df_completo.columns else "N/A")

            # RSI and Stochastic Oscillator Plot
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
            
            # Table of all current indicators
            st.markdown("#### Valores Atuais dos Indicadores Técnicos")
            
            # Extract current values for available indicators
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
            st.markdown("### Análise Fundamentalista Expandida")
            
            # Valuation Metrics
            st.markdown("#### Valuation")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            col1.metric("P/L (TTM)", f"{features_fund.get('pe_ratio', np.nan):.2f}" if not pd.isna(features_fund.get('pe_ratio')) else "N/A")
            col2.metric("P/VP", f"{features_fund.get('pb_ratio', np.nan):.2f}" if not pd.isna(features_fund.get('pb_ratio')) else "N/A")
            col3.metric("P/VPA (Vendas)", f"{features_fund.get('ps_ratio', np.nan):.2f}" if not pd.isna(features_fund.get('ps_ratio')) else "N/A")
            col4.metric("PEG", f"{features_fund.get('peg_ratio', np.nan):.2f}" if not pd.isna(features_fund.get('peg_ratio')) else "N/A")
            col5.metric("EV/EBITDA", f"{features_fund.get('ev_to_ebitda', np.nan):.2f}" if not pd.isna(features_fund.get('ev_to_ebitda')) else "N/A")
            
            # Profitability Metrics
            st.markdown("#### Rentabilidade")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            col1.metric("ROE", f"{features_fund.get('roe', np.nan):.2f}%" if not pd.isna(features_fund.get('roe')) else "N/A")
            col2.metric("ROA", f"{features_fund.get('roa', np.nan):.2f}%" if not pd.isna(features_fund.get('roa')) else "N/A")
            col3.metric("ROIC", f"{features_fund.get('roic', np.nan):.2f}%" if not pd.isna(features_fund.get('roic')) else "N/A")
            col4.metric("Margem Operacional", f"{features_fund.get('operating_margin', np.nan):.2f}%" if not pd.isna(features_fund.get('operating_margin')) else "N/A")
            col5.metric("Margem Bruta", f"{features_fund.get('gross_margin', np.nan):.2f}%" if not pd.isna(features_fund.get('gross_margin')) else "N/A")
            
            # Dividend Metrics
            st.markdown("#### Dividendos")
            col1, col2, col3 = st.columns(3)
            
            col1.metric("Dividend Yield", f"{features_fund.get('div_yield', np.nan):.2f}%" if not pd.isna(features_fund.get('div_yield')) else "N/A")
            col2.metric("Payout Ratio", f"{features_fund.get('payout_ratio', np.nan):.2f}%" if not pd.isna(features_fund.get('payout_ratio')) else "N/A")
            col3.metric("DY Médio 5A", f"{features_fund.get('five_year_avg_div_yield', np.nan):.2f}%" if not pd.isna(features_fund.get('five_year_avg_div_yield')) else "N/A")
            
            # Growth Metrics
            st.markdown("#### Crescimento")
            col1, col2, col3 = st.columns(3)
            col1.metric("Cresc. Receita", f"{features_fund.get('revenue_growth', np.nan):.2f}%" if not pd.isna(features_fund.get('revenue_growth')) else "N/A")
            col2.metric("Cresc. Lucros", f"{features_fund.get('earnings_growth', np.nan):.2f}%" if not pd.isna(features_fund.get('earnings_growth')) else "N/A")
            col3.metric("Cresc. Lucros (Q)", f"{features_fund.get('earnings_quarterly_growth', np.nan):.2f}%" if not pd.isna(features_fund.get('earnings_quarterly_growth')) else "N/A")

            # Financial Health
            st.markdown("#### Saúde Financeira")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Dívida/Patrimônio", f"{features_fund.get('debt_to_equity', np.nan):.2f}" if not pd.isna(features_fund.get('debt_to_equity')) else "N/A")
            col2.metric("Current Ratio", f"{features_fund.get('current_ratio', np.nan):.2f}" if not pd.isna(features_fund.get('current_ratio')) else "N/A")
            col3.metric("Quick Ratio", f"{features_fund.get('quick_ratio', np.nan):.2f}" if not pd.isna(features_fund.get('quick_ratio')) else "N/A")
            col4.metric("Fluxo de Caixa Livre", f"R$ {features_fund.get('free_cashflow', np.nan):,.0f}" if not pd.isna(features_fund.get('free_cashflow')) else "N/A")
            
            # Full Fundamental Data Table
            st.markdown("---")
            st.markdown("#### Todos os Fundamentos Disponíveis")
            
            df_fund_display = pd.DataFrame({
                'Métrica': list(features_fund.keys()),
                'Valor': [f"{v:.4f}" if isinstance(v, (int, float)) and not pd.isna(v) else str(v) for v in features_fund.values()]
            })
            
            st.dataframe(df_fund_display, use_container_width=True, hide_index=True)
        
        with tab4:
            st.markdown("### Análise de Machine Learning")
            
            st.info("Treinando modelos ML com um conjunto expandido de features para previsão de direção futura...")
            
            # Prepare data for ML
            # Feature selection: consider only numeric features, technicals, fundamentals, macro, and temporal
            numeric_cols_df = df_completo.select_dtypes(include=[np.number])
            temporal_cols = ['day_of_week', 'month', 'quarter', 'day_of_month', 'week_of_year']
            
            # Combine relevant columns, excluding identifiers and target
            candidate_features = numeric_cols_df.columns.tolist() + temporal_cols
            # Ensure 'Close' is not used as a feature itself unless lagged.
            # Features related to Close price calculation (Open, High, Low, Volume) are usually excluded.
            # Let's select features based on what's likely generated by EngenheiroFeatures and is numeric/temporal.
            
            # Dynamic feature list from EngenheiroFeatures + fundamental + temporal
            features_from_eng = [col for col in df_completo.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'log_returns']] # Exclude price/volume basics
            features_from_eng = [f for f in features_from_eng if pd.api.types.is_numeric_dtype(df_completo[f])] # Keep only numeric ones

            # Include fundamental features if available (assuming they were added to df_completo)
            # This part requires `df_completo` to have fundamental features correctly merged.
            # For now, rely on the features computed by `EngenheiroFeatures` and temporal ones.
            
            all_potential_features = features_from_eng + temporal_cols
            
            # Final feature set: ensure they exist in df_completo and are numeric
            final_features_for_ml = [f for f in all_potential_features if f in df_completo.columns and pd.api.types.is_numeric_dtype(df_completo[f])]
            
            # Add target column
            df_ml_data = df_completo[final_features_for_ml + ['Close']].copy() # Keep Close temporarily for target calc

            # Calculate Future Direction target (using a shorter lookahead for faster individual analysis)
            # Using LOOKBACK_ML from global config, but can be adjusted for individual analysis speed
            prediction_horizon = LOOKBACK_ML # Default from config
            df_ml_data['Future_Direction'] = np.where(
                df_ml_data['Close'].pct_change(prediction_horizon).shift(-prediction_horizon) > 0,
                1, 0
            )
            
            # Drop Close and final columns, drop NaNs
            df_ml_data = df_ml_data.drop(columns=['Close'])
            df_ml_data = df_ml_data.dropna()
            
            # Ensure enough data points for ML training
            if len(df_ml_data) > 100: # Minimum data points
                X = df_ml_data.drop('Future_Direction', axis=1)
                y = df_ml_data['Future_Direction']
                
                if len(np.unique(y)) < 2: # Check for class imbalance
                    st.warning("Dados insuficientes ou classe única encontrada para análise ML.")
                    return

                # Train ensemble models
                modelos = EnsembleML.treinar_ensemble(X, y, otimizar_optuna=False) # Use fixed params for speed in individual analysis
                
                if not modelos:
                    st.warning("Falha ao treinar modelos ML.")
                    return
                
                # Cross-validation with TimeSeriesSplit
                scores = []
                # Adjust n_splits based on data length for meaningful CV
                n_splits_cv = min(5, len(X) // 100) if len(X) > 100 else 1
                tscv = TimeSeriesSplit(n_splits=max(2, n_splits_cv)) # Ensure at least 2 splits
                
                for train_idx, val_idx in tscv.split(X):
                    if len(train_idx) == 0 or len(val_idx) == 0: continue
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2: continue # Skip if classes are not mixed
                    
                    # Retrain models for CV fold (simpler than passing trained models directly)
                    modelos_cv, auc_scores_cv = EnsembleML.treinar_ensemble(X_train, y_train, otimizar_optuna=False)
                    if not modelos_cv: continue

                    # NEW: Usa previsão ponderada para a validação
                    proba = EnsembleML.prever_ensemble_ponderado(modelos_cv, auc_scores_cv, X_val)
                    
                    if len(np.unique(y_val)) >= 2:
                        score = roc_auc_score(y_val, proba)
                        scores.append(score)
                
                auc_medio = np.mean(scores) if scores else np.nan
                
                # Final prediction using the last relevant data point
                # Use the features corresponding to the time step just before the target prediction
                # The last row of X represents features calculated up to the second to last day.
                # The target 'Future_Direction' is 'prediction_horizon' days ahead.
                # So, to predict for the next 'prediction_horizon' days, we use the features from the very last available row.
                last_features_row = X.iloc[[-1]] # Features from the last available day

                if last_features_row.empty:
                     proba_final = 0.5
                else:
                    # Use the trained models and their AUC scores for weighted prediction
                    proba_final = EnsembleML.prever_ensemble_ponderado(modelos, auc_scores, last_features_row)[0]
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Probabilidade de Alta Futura", f"{proba_final*100:.2f}%")
                col2.metric("AUC-ROC Médio (CV)", f"{auc_medio:.3f}" if not pd.isna(auc_medio) else "N/A")
                col3.metric("Nº Features Usadas", len(X.columns))
                
                # Feature Importance (using XGBoost if available)
                if 'xgboost' in modelos and hasattr(modelos['xgboost'], 'feature_importances_'):
                    st.markdown("#### Feature Importance (XGBoost)")
                    
                    importances = modelos['xgboost'].feature_importances_
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
                        title='Top 20 Features Mais Importantes'
                    )
                    fig_imp.update_layout(**obter_template_grafico())
                    st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.warning("Dados insuficientes para análise ML detalhada.")
        
        with tab5:
            st.markdown("### Clusterização e Análise de Similaridade")
            
            st.info("Comparando este ativo com outros ativos similares usando K-means + PCA...")
            
            # Prepare data for clustering: use a subset of available assets for performance
            # Fetch data for a broader set of assets for comparison
            
            # Limit the number of assets for clustering to avoid performance issues
            max_assets_for_clustering = 50 
            
            # Prioritize assets from the same sector if possible, otherwise sample broadly
            assets_to_cluster = []
            if 'sector' in features_fund and features_fund['sector'] != 'Unknown':
                sector_assets = [a for a in ATIVOS_IBOVESPA if a.split('.')[0].replace('.SA', '') in ATIVOS_POR_SETOR.get(features_fund['sector'], [])] # Match by name without .SA
                # Add the current asset if it's in Ibovespa
                if ativo_selecionado in ATIVOS_IBOVESPA:
                    assets_to_cluster.append(ativo_selecionado)
                
                # Add sector assets, prioritizing those with good performance/fundamentals
                sector_assets_sorted = sorted(sector_assets, key=lambda x: (
                    self.dados_performance.loc[x, 'sharpe'] if x in self.dados_performance.index else -np.inf,
                    self.dados_fundamentalistas.loc[x, 'roe'] if x in self.dados_fundamentalistas.index and 'roe' in self.dados_fundamentalistas.columns else -np.inf
                ), reverse=True)
                
                assets_to_cluster.extend(sector_assets_sorted[:max_assets_for_clustering - len(assets_to_cluster)])
            
            # If not enough sector assets or no sector defined, sample broadly
            if len(assets_to_cluster) < max_assets_for_clustering:
                # Ensure the current asset is included
                if ativo_selecionado not in assets_to_cluster:
                    assets_to_cluster.append(ativo_selecionado)
                
                # Add other diverse assets (e.g., from Ibovespa)
                other_assets = [a for a in ATIVOS_IBOVESPA if a not in assets_to_cluster]
                assets_to_cluster.extend(other_assets[:max_assets_for_clustering - len(assets_to_cluster)])
                
            # Fetch data for the selected comparison assets
            comparison_data = {}
            if len(assets_to_cluster) > 5: # Need a reasonable number for clustering
                with st.spinner(f"Coletando dados para {len(assets_to_cluster)} ativos de comparação..."):
                    for asset_comp in assets_to_cluster:
                        try:
                            ticker_comp = yf.Ticker(asset_comp)
                            # Fetch similar period as used for individual analysis, or a fixed shorter period for clustering consistency
                            hist_comp = ticker_comp.history(period='2y') # Use 2 years for clustering consistency
                            
                            if not hist_comp.empty:
                                # Use a subset of features for clustering: performance & fundamental metrics
                                features_for_cluster = ['retorno_anual', 'volatilidade_anual', 'sharpe', 'max_drawdown', 
                                                        'pe_ratio', 'pb_ratio', 'roe', 'debt_to_equity', 'revenue_growth']
                                
                                # Calculate necessary metrics if not readily available from the main collector
                                # For simplicity here, we'll use a simplified set of metrics calculation
                                returns_comp = hist_comp['Close'].pct_change().dropna()
                                
                                if len(returns_comp) > 50: # Need sufficient data
                                    comp_data = {
                                        'retorno_anual': returns_comp.mean() * 252,
                                        'volatilidade_anual': returns_comp.std() * np.sqrt(252),
                                        'sharpe': (returns_comp.mean() * 252 - TAXA_LIVRE_RISCO) / (returns_comp.std() * np.sqrt(252)) if returns_comp.std() > 0 else 0,
                                        'max_drawdown': ((1 + returns_comp).cumprod() / (1 + returns_comp).cumprod().expanding().max() - 1).min() if not returns_comp.empty else np.nan
                                    }
                                    
                                    # Fetch fundamental data
                                    info_comp = ticker_comp.info
                                    fund_metrics = AnalisadorIndividualAtivos.calcular_features_fundamentalistas_expandidas(ticker_comp)
                                    
                                    for metric in ['pe_ratio', 'pb_ratio', 'roe', 'debt_to_equity', 'revenue_growth']:
                                         comp_data[metric] = fund_metrics.get(metric, np.nan)

                                    comparison_data[asset_comp] = comp_data
                        except Exception as e:
                            print(f"  ⚠️ Error fetching data for {asset_comp} during clustering: {str(e)[:50]}")
                            continue
            
            # Perform clustering if enough data was collected
            if len(comparison_data) > 5:
                df_comparacao = pd.DataFrame(comparison_data).T
                
                # Perform clustering and PCA
                resultado_pca, pca, kmeans = AnalisadorIndividualAtivos.realizar_clusterizacao_pca(
                    df_comparacao,
                    n_clusters=5 # Standard number of clusters
                )
                
                if resultado_pca is not None:
                    # PCA Plot (3D or 2D based on number of components)
                    if 'PC3' in resultado_pca.columns:
                        fig_pca = px.scatter_3d(
                            resultado_pca,
                            x='PC1', y='PC2', z='PC3',
                            color='Cluster',
                            hover_name=resultado_pca.index.str.replace('.SA', ''),
                            title='Clusterização K-means + PCA (3D) - Similaridade de Ativos'
                        )
                    else:
                        fig_pca = px.scatter(
                            resultado_pca,
                            x='PC1', y='PC2',
                            color='Cluster',
                            hover_name=resultado_pca.index.str.replace('.SA', ''),
                            title='Clusterização K-means + PCA (2D) - Similaridade de Ativos'
                        )
                    
                    fig_pca.update_layout(**obter_template_grafico(), height=600)
                    st.plotly_chart(fig_pca, use_container_width=True)
                    
                    # Identify cluster and similar assets for the selected asset
                    if ativo_selecionado in resultado_pca.index:
                        cluster_ativo = resultado_pca.loc[ativo_selecionado, 'Cluster']
                        ativos_similares_df = resultado_pca[resultado_pca['Cluster'] == cluster_ativo]
                        ativos_similares = ativos_similares_df.index.tolist()
                        ativos_similares = [a for a in ativos_similares if a != ativo_selecionado]
                        
                        st.success(f"**{ativo_selecionado.replace('.SA', '')}** pertence ao Cluster {cluster_ativo}")
                        
                        if ativos_similares:
                            st.markdown(f"#### Outros Ativos no Cluster {cluster_ativo}:")
                            st.write(", ".join([a.replace('.SA', '') for a in ativos_similares[:15]])) # Show top 15 similar
                    
                    # Explained Variance Ratio Plot
                    st.markdown("#### Variância Explicada por Componente Principal")
                    var_exp = pca.explained_variance_ratio_ * 100
                    
                    df_var = pd.DataFrame({
                        'Componente': [f'PC{i+1}' for i in range(len(var_exp))],
                        'Variância (%)': var_exp
                    })
                    
                    fig_var = px.bar(
                        df_var,
                        x='Componente',
                        y='Variância (%)',
                        title='Variância Explicada por Componente Principal'
                    )
                    fig_var.update_layout(**obter_template_grafico())
                    st.plotly_chart(fig_var, use_container_width=True)
            else:
                st.warning("Dados insuficientes para realizar a clusterização e análise de similaridade.")
    
        except Exception as e:
            st.error(f"Erro ao analisar o ativo {ativo_selecionado}: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")

# NEW FUNCTION FOR GOVERNANCE TAB
def aba_governanca():
    """
    NEW: Aba 5 - Governança de Modelo
    Exibe métricas de performance, histórico de AUC e alertas
    """
    st.markdown("## 🛡️ Governança de Modelo - Monitoramento de Performance")
    
    if 'builder' not in st.session_state or st.session_state.builder is None:
        st.warning("⚠️ Execute o **Construtor de Portfólio** primeiro para visualizar métricas de governança.")
        return
    
    builder = st.session_state.builder
    
    if not hasattr(builder, 'governanca_por_ativo') or not builder.governanca_por_ativo:
        st.info("📊 Dados de governança serão exibidos após o treinamento dos modelos ML.")
        return
    
    st.markdown("""
    <div class="info-box">
    <h4>📈 Sistema de Governança Elite</h4>
    <p>Monitora continuamente a performance dos modelos de Machine Learning e emite alertas quando:</p>
    <ul>
        <li>AUC-ROC cai abaixo de <strong>0.65</strong> (mínimo aceitável)</li>
        <li>Degradação superior a <strong>5%</strong> em relação ao máximo histórico</li>
        <li>Tendência de queda consistente nos últimos 5 períodos</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    ativos_com_governanca = list(builder.governanca_por_ativo.keys())
    
    if not ativos_com_governanca:
        st.warning("Nenhum ativo com dados de governança disponível.")
        return
    
    ativo_selecionado = st.selectbox(
        "Selecione um ativo para análise de governança:",
        options=ativos_com_governanca,
        format_func=lambda x: x.replace('.SA', '')
    )
    
    governanca = builder.governanca_por_ativo[ativo_selecionado]
    relatorio = governanca.gerar_relatorio()
    
    # Exibe status com cor apropriada
    severidade = relatorio['severidade']
    status_msg = relatorio['status']
    
    if severidade == 'success':
        st.markdown(f'<div class="alert-success"><strong>✅ {status_msg}</strong></div>', unsafe_allow_html=True)
    elif severidade == 'warning':
        st.markdown(f'<div class="alert-warning"><strong>⚠️ {status_msg}</strong></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-error"><strong>🚨 {status_msg}</strong></div>', unsafe_allow_html=True)
    
    # Métricas principais
    st.markdown("### 📊 Métricas de Performance")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metricas = relatorio['metricas']
    
    col1.metric("AUC Atual", f"{metricas['AUC Atual']:.3f}" if not np.isnan(metricas['AUC Atual']) else "N/A")
    col2.metric("AUC Médio", f"{metricas['AUC Médio']:.3f}" if not np.isnan(metricas['AUC Médio']) else "N/A")
    col3.metric("AUC Máximo", f"{metricas['AUC Máximo']:.3f}" if not np.isnan(metricas['AUC Máximo']) else "N/A")
    col4.metric("Precision", f"{metricas['Precision Média']:.3f}" if not np.isnan(metricas['Precision Média']) else "N/A")
    col5.metric("Recall", f"{metricas['Recall Médio']:.3f}" if not np.isnan(metricas['Recall Médio']) else "N/A")
    
    # Gráfico de histórico de AUC
    st.markdown("### 📈 Histórico de AUC-ROC")
    
    historico = relatorio['historico']
    
    if len(historico['AUC']) > 0:
        df_hist = pd.DataFrame({
            'Período': range(1, len(historico['AUC']) + 1),
            'AUC-ROC': historico['AUC'],
            'Precision': historico['Precision'],
            'Recall': historico['Recall'],
            'F1-Score': historico['F1']
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_hist['Período'],
            y=df_hist['AUC-ROC'],
            mode='lines+markers',
            name='AUC-ROC',
            line=dict(color='#2c3e50', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_hline(
            y=AUC_THRESHOLD_MIN,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mínimo Aceitável ({AUC_THRESHOLD_MIN})"
        )
        
        if not np.isnan(metricas['AUC Máximo']):
            fig.add_hline(
                y=metricas['AUC Máximo'],
                line_dash="dot",
                line_color="green",
                annotation_text=f"Máximo Histórico ({metricas['AUC Máximo']:.3f})"
            )
        
        fig.update_layout(
            **obter_template_grafico(),
            title=f"Evolução do AUC-ROC - {ativo_selecionado.replace('.SA', '')}",
            xaxis_title="Período",
            yaxis_title="AUC-ROC",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📋 Ver Dados Detalhados"):
            st.dataframe(df_hist, use_container_width=True)
    else:
        st.info("Histórico insuficiente para exibir gráficos.")

def main():
    """Função principal com estrutura de 5 abas (NEW: adicionada aba de Governança)"""
    
    # Initialize session state variables if they don't exist
    if 'builder' not in st.session_state:
        st.session_state.builder = None
        st.session_state.builder_complete = False
        st.session_state.profile = {}
        st.session_state.ativos_para_analise = [] # Initialize asset selection
        st.session_state.analisar_ativo_triggered = False # Initialize analysis trigger
        
    configurar_pagina()
    
    # Sidebar configuration
    st.sidebar.markdown(
        '<p style="font-size: 26px; font-weight: bold; color: #2c3e50;">📈 AutoML Elite</p>',
        unsafe_allow_html=True
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Recursos Elite")
    st.sidebar.markdown("""
    - **9 Modelos ML**: XGBoost, LightGBM, CatBoost, RF, ET, KNN, SVC, LR, GNB
    - **Ponderação AUC-ROC**: Ensemble inteligente
    - **Governança**: Monitoramento e alertas
    - **GARCH/EGARCH**: Volatilidade avançada
    - **Modelos Estatísticos**: ARIMA, Prophet, VAR
    - **30+ Indicadores Técnicos**: Biblioteca `ta`
    - **Análise Fundamentalista**: Métricas expandidas
    - **Clusterização**: K-means + PCA para similaridade
    - **Otimização CVaR**: Risco condicional
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Versão**: 7.0.0 Elite Final")
    st.sidebar.markdown("**Sistema**: Quantitative AutoML")
    st.sidebar.markdown("[Repositório](https://github.com/seu-usuario/seu-repositorio)") # Placeholder for link
    
    # Main title
    st.markdown('<h1 class="main-header">Sistema AutoML Elite - Otimização Quantitativa de Portfólio</h1>', unsafe_allow_html=True)
    
    # NEW: 5 tabs instead of 4
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📚 Introdução",
        "🎯 Seleção de Ativos",
        "🏗️ Construtor de Portfólio",
        "🔍 Análise Individual",
        "🛡️ Governança de Modelo"  # NEW TAB
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
    
    with tab5:
        aba_governanca()  # NEW FUNCTION

if __name__ == "__main__":
    main()
