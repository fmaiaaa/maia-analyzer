"""
=============================================================================
SISTEMA AUTOML AVANÇADO - OTIMIZAÇÃO DE PORTFÓLIO FINANCEIRO V9.0
=============================================================================

Sistema completo de otimização de portfólio com:
- **P-0**: MLOps-Data - Download GCS pré-processado (v7)
- **P-F1 (NOVO v9.0)**: Pipeline de ML otimizado com PCA (StandardScaler + PCA)
    para acelerar o treinamento do ensemble.
- **P-1 a P-14**: Todas as melhorias das versões anteriores.
- 9 Modelos ML com ponderação AUC-ROC
- Governança e monitoramento de drift
- GARCH/EGARCH, Copula, HRP
- Modelos estatísticos (ARIMA, Prophet, VAR)
- Análise Fundamentalista e Técnica completas
- Clusterização 3D/PCA para Análise Individual

Versão: 9.0.0 - Otimização de Velocidade com PCA
=============================================================================
"""

# --- 1. CORE LIBRARIES & UTILITIES ---
import warnings
import numpy as np
import pandas as pd
import subprocess
import sys
import time
import os
from datetime import datetime, timedelta
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.stats import zscore, norm
from pathlib import Path
import requests
from io import StringIO

# --- 2. GOOGLE CLOUD STORAGE ---
try:
    from google.cloud import storage
except ImportError:
    print("Instalando google-cloud-storage...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', 'google-cloud-storage'])
    from google.cloud import storage

# --- 3. STREAMLIT & PLOTTING ---
try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    print("Instalando streamlit e plotly...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', 'streamlit', 'plotly'])
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

# --- 4. FEATURE ENGINEERING (TA) ---
try:
    import ta
    from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, CCIIndicator
    from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
    from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel, DonchianChannel
    from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, MFIIndicator, VolumeWeightedAveragePrice
except ImportError:
    print("Instalando ta...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', 'ta'])
    import ta
    # (importações específicas do TA)

# --- 5. MACHINE LEARNING (SCIKIT-LEARN) ---
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
    from sklearn.linear_model import RidgeClassifier, LogisticRegression, BayesianRidge
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.decomposition import PCA  # <-- Importado para v9.0
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import silhouette_score, roc_auc_score, precision_score, recall_score, f1_score
    from sklearn.impute import SimpleImputer  # <-- Importado para v9.0
    from sklearn.pipeline import Pipeline     # <-- Importado para v9.0
except ImportError:
    print("Instalando scikit-learn...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', 'scikit-learn'])
    # (importações específicas do sklearn)

# --- 6. BOOSTED MODELS ---
try:
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier
except ImportError:
    print("Instalando xgboost, lightgbm, catboost...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', 'xgboost', 'lightgbm', 'catboost'])
    # (importações específicas)

# --- 7. ECONOMETRICS & TIME SERIES ---
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.statespace.varmax import VARMAX
    from prophet import Prophet
    from arch import arch_model
except ImportError:
    print("Instalando statsmodels, prophet, arch...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', 'statsmodels', 'prophet', 'arch'])
    # (importações específicas)

# --- 8. OPTIMIZATION AND XAI ---
try:
    import optuna
    import shap
    import lime
    import lime.lime_tabular
except ImportError:
    print("Instalando optuna, shap, lime...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', 'optuna', 'shap', 'lime'])
    import optuna
    
# --- 9. DEEP LEARNING (TENSORFLOW/KERAS) ---
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
except ImportError:
    print("Instalando tensorflow...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', 'tensorflow'])
    # (importações específicas)


# --- 10. CONFIGURATION ---
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.options.mode.chained_assignment = None # Desativa o SettingWithCopyWarning

# =============================================================================
# CONSTANTES GLOBAIS E CONFIGURAÇÕES
# =============================================================================

# Configurações Globais
PERIODO_DADOS = 'max'
MIN_DIAS_HISTORICO = 252
NUM_ATIVOS_PORTFOLIO = 5
TAXA_LIVRE_RISCO = 0.1075
LOOKBACK_ML = 30 # Dias à frente para prever a direção

# NOVO v9.0: Configuração do PCA
# Define o PCA para capturar 95% da variância explicada
PCA_N_COMPONENTS = 0.95 

# Ponderações padrão para os scores (serão adaptadas pelo perfil)
WEIGHT_PERFORMANCE = 0.40
WEIGHT_FUNDAMENTAL = 0.30
WEIGHT_TECHNICAL = 0.30
WEIGHT_ML = 0.30 # Peso do score de ML no score total

# Limites de peso por ativo na otimização
PESO_MIN = 0.10
PESO_MAX = 0.30

# Configurações do GCS (baseado na v7)
GCS_PROJECT_ID = os.getenv('GCS_PROJECT_ID', 'maia-analyzer')
GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME', 'meu-portfolio-dados-gratuitos')
GCS_DATA_PREFIX = 'dados_financeiros_etl/'

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
TODOS_ATIVOS = sorted(list(set([ativo for setor, ativos in ATIVOS_POR_SETOR.items() for ativo in ativos])))

# =============================================================================
# CONSTANTES DE GOVERNANÇA (NEW)
# =============================================================================

AUC_THRESHOLD_MIN = 0.65  # Alerta se AUC cair abaixo deste valor
AUC_DROP_THRESHOLD = 0.05   # Alerta se queda de 5% no AUC
DRIFT_WINDOW = 20           # Janela para monitoramento de drift

# =============================================================================
# CLASSE: GOVERNANÇA DE MODELO (v7)
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
# CLASSE: ANALISADOR DE PERFIL DO INVESTIDOR (v7)
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
# CLASSE: ENGENHEIRO DE FEATURES (v9.0 - Mínima)
# =============================================================================

class EngenheiroFeatures:
    """
    v9.0: Esta classe foi minimizada. Os cálculos de features (técnicos, 
    fundamentalistas) agora são feitos offline pelo script ETL e lidos 
    pelo LeitorGCS. Apenas métodos utilitários (como normalização) são mantidos.
    """
    
    @staticmethod
    def _normalizar(serie, maior_melhor=True):
        """Normaliza uma série para o range [0, 1] (Min-Max Scaling)"""
        if serie.isnull().all():
            return pd.Series(0, index=serie.index) # Retorna zeros se tudo for NaN
        
        serie_limpa = serie.dropna()
        if serie_limpa.empty:
            return pd.Series(0, index=serie.index)

        min_val = serie_limpa.min()
        max_val = serie_limpa.max()
        
        if max_val == min_val: # Lida com casos de variância zero
            return pd.Series(0.5, index=serie.index)
        
        if maior_melhor:
            normalized = (serie - min_val) / (max_val - min_val)
        else:
            normalized = (max_val - serie) / (max_val - min_val)
        
        # Garante que valores fora do range (devido ao fillna) sejam limitados
        return normalized.clip(0, 1)

# =============================================================================
# CLASSE: LEITOR GCS (v7)
# =============================================================================

class LeitorGCS:
    """
    Lê dados financeiros PRÉ-PROCESSADOS (CSV completo) do GCS.
    Consome o CSV gerado pelo gerador_financeiro.py.
    """
    
    def __init__(self, project_id=GCS_PROJECT_ID, bucket_name=GCS_BUCKET_NAME, data_prefix=GCS_DATA_PREFIX):
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.data_prefix = data_prefix
        self.client = None
        self.bucket = None
        self.cache = {}
        self._inicializar_cliente()
    
    def _inicializar_cliente(self):
        """Inicializa o cliente GCS com tratamento robusto de erros."""
        try:
            self.client = storage.Client(project=self.project_id)
            self.bucket = self.client.bucket(self.bucket_name)
            # Valida a conexão
            list(self.bucket.list_blobs(prefix=self.data_prefix, max_results=1))
            print(f"✓ Conectado ao GCS: {self.bucket_name} / {self.data_prefix}")
        except Exception as e:
            print(f"⚠️ Aviso: Falha ao conectar no GCS: {str(e)[:100]}")
            self.client = None
            self.bucket = None
    
    def limpar_cache(self):
        self.cache.clear()
    
    def ler_dados_historicos_completos(self, ticker, usar_cache=True):
        """Lê o CSV completo (Histórico, Técnico, GARCH, Fundamentalista) do GCS."""
        if usar_cache and ticker in self.cache:
            return self.cache[ticker].copy()

        try:
            if not self.bucket or self.client is None:
                print(f"❌ Erro: Cliente GCS não autenticado para {ticker}.")
                return None
            
            blob_path = f"{self.data_prefix}{ticker}.csv"
            blob = self.bucket.blob(blob_path)
            
            if not blob.exists():
                print(f"ℹ️ Arquivo {blob_path} não encontrado no GCS.")
                return None
            
            dados_csv = blob.download_as_string()
            
            df = pd.read_csv(
                StringIO(dados_csv.decode('utf-8')),
                index_col='Date',
                parse_dates=True,
                dtype={'ticker': str}
            )
            
            if df.empty or len(df) < MIN_DIAS_HISTORICO * 0.7:
                print(f"⚠️ Dados insuficientes para {ticker}: {len(df)} dias")
                return None
            
            df = df.sort_index()
            self.cache[ticker] = df.copy()
            
            # print(f"✓ {ticker}: {len(df)} dias carregados com sucesso")
            return df.copy()
            
        except Exception as e:
            print(f"❌ Erro geral ao ler {ticker} do GCS: {str(e)[:100]}")
            return None

# =============================================================================
# CLASSE: COLETOR DE DADOS (v7 - GCS)
# =============================================================================

class ColetorDados:
    """
    Coleta dados PRÉ-PROCESSADOS (CSV) do GCS.
    v9.0: Lógica mantida da v7, robusta e baseada em GCS.
    """
    
    def __init__(self, periodo=PERIODO_DADOS):
        self.periodo = periodo
        self.leitor_gcs = LeitorGCS() 
        self.dados_por_ativo = {}
        self.dados_fundamentalistas = pd.DataFrame() # <-- Armazena dados normalizados/imputados
        self.dados_fundamentalistas_raw = pd.DataFrame() # <-- Armazena dados brutos
        self.ativos_sucesso = []
        self.dados_macro = {}
        self.metricas_performance = pd.DataFrame()
    
    def coletar_dados_macroeconomicos(self):
        """
        Dados macroeconômicos agora vêm do CSV pré-processado.
        (v7)
        """
        print("📊 Correlações macroeconômicas lidas diretamente do CSV pré-processado.")
        self.dados_macro = {'IBOV': pd.Series(), 'USD_BRL': pd.Series()}

    def _identificar_colunas(self, df_cols):
        """Helper para identificar tipos de colunas do GCS CSV"""
        cols_preco_vol = {'Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'log_returns', 'ticker'}
        
        prefixos_tecnicos = (
            'sma_', 'ema_', 'wma_', 'hma_', 'volatility_', 'returns_', 'volume_',
            'close_lag_', 'volume_lag_', 'macd_', 'bb_', 'adx_', 'atr_', 'cci_',
            'obv', 'cmf', 'mfi', 'vwap', 'drawdown', 'max_drawdown', 'autocorr_',
            'day_of_week', 'month', 'quarter', 'day_of_month', 'week_of_year',
            'garch_', 'stoch_', 'rsi_', 'williams_', 'kc_', 'dc_', 'momentum_',
            'higher_high', 'lower_low', 'price_', 'returns_mean', 'returns_std',
            'returns_skew', 'returns_kurt', 'volume_mean', 'volume_std'
        )
        
        features_tecnicas = set()
        features_fund_e_perf = set()
        
        for col in df_cols:
            if col in cols_preco_vol:
                continue
            if col.startswith(prefixos_tecnicos):
                features_tecnicas.add(col)
            else:
                # Tudo o que não for preço/volume ou técnico é considerado
                # fundamental ou de performance (ex: 'pe_ratio', 'sharpe_ratio')
                features_fund_e_perf.add(col)
                
        return features_tecnicas, features_fund_e_perf

    def coletar_e_processar_dados(self, simbolos):
        """
        Versão otimizada da coleta (v7)
        Carrega todos os dados do GCS e separa fundamentos/performance.
        """
        self.ativos_sucesso = []
        self.coletar_dados_macroeconomicos()
        
        print(f"\n{'='*70}")
        print(f"INICIANDO COLETA DE DADOS PRÉ-PROCESSADOS DO GCS (v9.0)")
        print(f"Ativos para processar: {len(simbolos)}")
        print(f"Bucket: {self.leitor_gcs.bucket_name} | Pasta: {self.leitor_gcs.data_prefix}")
        print(f"{'='*70}\n")
        
        lista_fundamentos_e_perf = []
        ativos_processados = 0
        
        # Ids de features (identificados apenas uma vez)
        colunas_identificadas = False
        self.features_tecnicas_identificadas = set()
        self.features_fund_e_perf_identificadas = set()
        
        for ticker in tqdm(simbolos, desc="📥 Coletando dados do GCS"):
            try:
                # 🎯 LER O CSV COMPLETO do GCS
                df_completo = self.leitor_gcs.ler_dados_historicos_completos(ticker)
                
                if df_completo is None or df_completo.empty:
                    continue
                
                # 1. Identifica colunas na primeira leitura bem-sucedida
                if not colunas_identificadas:
                    tec, fund_perf = self._identificar_colunas(df_completo.columns)
                    self.features_tecnicas_identificadas = tec
                    self.features_fund_e_perf_identificadas = fund_perf
                    colunas_identificadas = True
                
                # 2. Dados Históricos, Técnicos e GARCH
                self.dados_por_ativo[ticker] = df_completo
                self.ativos_sucesso.append(ticker)
                ativos_processados += 1
                
                # 3. Extrair dados Fundamentalistas e Performance (da primeira linha)
                primeira_linha = df_completo.iloc[0]
                features_fund = {'Ticker': ticker}
                
                for col in self.features_fund_e_perf_identificadas:
                    if col in primeira_linha.index and pd.notna(primeira_linha[col]):
                        features_fund[col] = primeira_linha[col]

                lista_fundamentos_e_perf.append(features_fund)
                
            except Exception as e:
                print(f"  ❌ {ticker}: Erro na leitura/processamento GCS - {str(e)[:50]}")
                continue
        
        if not lista_fundamentos_e_perf:
            st.error("Nenhum dado fundamentalista ou de performance foi carregado. Verifique o GCS.")
            return False
            
        # 4. Consolida e Processa Fundamentos/Performance
        self.dados_fundamentalistas_raw = pd.DataFrame(lista_fundamentos_e_perf).set_index('Ticker')
        self.dados_fundamentalistas_raw = self.dados_fundamentalistas_raw.replace([np.inf, -np.inf], np.nan)
        
        # Separa métricas de performance
        performance_cols = [col for col in self.features_fund_e_perf_identificadas 
                           if col in ['sharpe_ratio', 'annual_return', 'annual_volatility', 'max_drawdown']]
        
        if performance_cols:
            self.metricas_performance = self.dados_fundamentalistas_raw[performance_cols].copy()
            rename_map = {
                'sharpe_ratio': 'sharpe',
                'annual_return': 'retorno_anual',
                'annual_volatility': 'volatilidade_anual',
                'max_drawdown': 'max_drawdown'
            }
            self.metricas_performance = self.metricas_performance.rename(columns=rename_map)
        
        # Isola colunas fundamentalistas (excluindo performance)
        fundamental_cols = [col for col in self.features_fund_e_perf_identificadas if col not in performance_cols]
        self.dados_fundamentalistas = self.dados_fundamentalistas_raw[fundamental_cols].copy()
        
        # Imputa e Normaliza/Padroniza dados fundamentalistas (para pontuação)
        numeric_cols = self.dados_fundamentalistas.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            for col in numeric_cols:
                if self.dados_fundamentalistas[col].isnull().any():
                    median_val = self.dados_fundamentalistas[col].median()
                    self.dados_fundamentalistas[col] = self.dados_fundamentalistas[col].fillna(median_val)
                
            # Padroniza (RobustScaler) para o Score
            scaler = RobustScaler()
            self.dados_fundamentalistas[numeric_cols] = scaler.fit_transform(self.dados_fundamentalistas[numeric_cols])

        
        print(f"\n✓ Coleta concluída:")
        print(f"  - Ativos com sucesso: {len(self.ativos_sucesso)}")
        print(f"  - Fundamentos consolidados: {len(self.dados_fundamentalistas)}")
        print(f"  - Features Técnicas Identificadas: {len(self.features_tecnicas_identificadas)}")
        print()
        
        return len(self.ativos_sucesso) > 0
        
# =============================================================================
# CLASSE: MODELAGEM DE VOLATILIDADE GARCH (v7)
# =============================================================================

class VolatilidadeGARCH:
    """Modelagem de volatilidade GARCH/EGARCH"""
    
    @staticmethod
    def ajustar_garch(returns, tipo_modelo='GARCH'):
        """Ajusta modelo GARCH e prevê volatilidade"""
        try:
            returns_limpo = returns.dropna() * 100
            
            if len(returns_limpo) < 100 or returns_limpo.std() == 0:
                return np.nan
            
            if tipo_modelo == 'EGARCH':
                modelo = arch_model(returns_limpo, vol='EGARCH', p=1, q=1, rescale=False)
            else:
                modelo = arch_model(returns_limpo, vol='Garch', p=1, q=1, rescale=False)
            
            resultado = modelo.fit(disp='off', show_warning=False, options={'maxiter': 1000})
            
            if resultado is None or not resultado.params.any():
                return np.nan
            
            previsao = resultado.forecast(horizon=1)
            volatilidade = np.sqrt(previsao.variance.values[-1, 0]) / 100
            
            if np.isnan(volatilidade) or np.isinf(volatilidade):
                return np.nan
            
            return volatilidade * np.sqrt(252) # Anualiza
            
        except Exception as e:
            return np.nan

# =============================================================================
# CLASSE: MODELOS ESTATÍSTICOS DE SÉRIES TEMPORAIS (v7)
# =============================================================================

class ModelosEstatisticos:
    """Modelos estatísticos para previsão de séries temporais financeiras"""
    
    @staticmethod
    def ajustar_arima(series, order=(1, 1, 1), horizon=1):
        """Ajusta modelo ARIMA e faz previsão"""
        try:
            series_limpa = series.dropna()
            if len(series_limpa) < 50: return {'forecast': np.nan}
            modelo = ARIMA(series_limpa, order=order)
            resultado = modelo.fit()
            previsao = resultado.forecast(steps=horizon)
            return {'forecast': previsao.iloc[-1], 'aic': resultado.aic}
        except Exception as e:
            return {'forecast': np.nan}
    
    @staticmethod
    def ajustar_sarima(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), horizon=1):
        """Ajusta modelo SARIMA"""
        try:
            series_limpa = series.dropna()
            if len(series_limpa) < 100: return {'forecast': np.nan}
            modelo = SARIMAX(series_limpa, order=order, seasonal_order=seasonal_order)
            resultado = modelo.fit(disp=False, maxiter=100)
            previsao = resultado.forecast(steps=horizon)
            return {'forecast': previsao.iloc[-1], 'aic': resultado.aic}
        except Exception as e:
            return {'forecast': np.nan}
    
    @staticmethod
    def ajustar_prophet(series, horizon=30):
        """Ajusta modelo Prophet"""
        try:
            df_prophet = pd.DataFrame({'ds': series.index, 'y': series.values}).dropna()
            if len(df_prophet) < 50: return {'forecast': np.nan}
            
            logging.getLogger('prophet').setLevel(logging.WARNING)
            modelo = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            modelo.fit(df_prophet)
            future = modelo.make_future_dataframe(periods=horizon)
            previsao = modelo.predict(future)
            ultimo_forecast = previsao.iloc[-1]
            return {
                'forecast': ultimo_forecast['yhat'], 
                'yhat_lower': ultimo_forecast['yhat_lower'], 
                'yhat_upper': ultimo_forecast['yhat_upper']
            }
        except Exception as e:
            return {'forecast': np.nan}
    
    @staticmethod
    def ensemble_estatistico(series, horizon=1):
        """Cria ensemble de modelos estatísticos (ARIMA, SARIMA, Prophet)"""
        previsoes = {}
        pesos = {}
        
        # ARIMA
        resultado_arima = ModelosEstatisticos.ajustar_arima(series, order=(1, 1, 1), horizon=horizon)
        if not np.isnan(resultado_arima['forecast']):
            previsoes['ARIMA'] = resultado_arima['forecast']
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
            conf_width = resultado_prophet.get('yhat_upper', 0) - resultado_prophet.get('yhat_lower', 0)
            pesos['Prophet'] = 1.0 / (conf_width + 1) if not np.isnan(conf_width) and conf_width > 0 else 1.0
        
        if not previsoes:
            return {'ensemble_forecast': np.nan, 'individual_forecasts': {}}
        
        # Média Ponderada (baseada no inverso do AIC/Confiança)
        total_peso = sum(pesos.values())
        pesos_norm = {k: v / total_peso for k, v in pesos.items()}
        forecast_ensemble = sum(previsoes[k] * pesos_norm[k] for k in previsoes.keys())
        
        return {
            'ensemble_forecast': forecast_ensemble,
            'individual_forecasts': previsoes,
            'weights': pesos_norm
        }

# =============================================================================
# CLASSE: ENSEMBLE DE MODELOS ML (v7 - 9 Modelos)
# =============================================================================

class EnsembleML:
    """
    Ensemble de 9 modelos ML com ponderação por AUC-ROC.
    v9.0: A classe não muda. Ela recebe X e y, não importa se X é 
    features brutas ou componentes PCA.
    """
    
    @staticmethod
    def treinar_ensemble(X, y, otimizar_optuna=False):
        """Treina ensemble expandido e retorna (modelos, auc_scores)"""
        
        # Converte X para DataFrame se for np.array (saída do PCA)
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'PC{i+1}' for i in range(X.shape[1])])
        
        modelos = {}
        auc_scores = {}
        
        # Configurações dos 9 modelos
        configs = {
            'xgboost': xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='logloss', use_label_encoder=False),
            'lightgbm': lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1),
            'catboost': CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1, random_state=42, verbose=False),
            'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'extra_trees': ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42),
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
                    
                    if len(np.unique(y_val)) < 2: continue # Pula se o fold de validação tiver uma classe só
                    
                    modelo_fold = configs[nome]
                    
                    # Tratamento especial para CatBoost com PCA (features precisam ser numéricas)
                    if nome == 'catboost':
                         modelo_fold.fit(X_train, y_train, cat_features=None) # Garante que não procure por 'cat_features'
                    else:
                        modelo_fold.fit(X_train, y_train)
                    
                    if hasattr(modelo_fold, 'predict_proba'):
                        y_proba = modelo_fold.predict_proba(X_val)[:, 1]
                    else:
                        y_proba = modelo_fold.decision_function(X_val)
                    
                    auc = roc_auc_score(y_val, y_proba)
                    auc_fold_scores.append(auc)
                
                auc_medio = np.mean(auc_fold_scores) if auc_fold_scores else 0.5
                
                # Treina modelo final com todos os dados
                if nome == 'catboost':
                    modelo.fit(X, y, cat_features=None)
                else:
                    modelo.fit(X, y)
                    
                modelos[nome] = modelo
                auc_scores[nome] = auc_medio
                
                # print(f"  ✓ {nome}: AUC = {auc_medio:.3f}")
                
            except Exception as e:
                print(f"  ⚠️ {nome}: Erro - {str(e)[:50]}")
                continue
        
        return modelos, auc_scores
    
    @staticmethod
    def prever_ensemble_ponderado(modelos, auc_scores, X):
        """Previsão ponderada por AUC-ROC"""
        
        # Converte X para DataFrame se for np.array (saída do PCA)
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'PC{i+1}' for i in range(X.shape[1])])

        previsoes_ponderadas = []
        pesos_normalizados = []
        
        modelos_validos = {nome: modelo for nome, modelo in modelos.items() 
                          if auc_scores.get(nome, 0) > 0.50}
        
        if not modelos_validos:
            # Fallback: usa todos os modelos com peso igual
            return EnsembleML.prever_ensemble(modelos, X)
        
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
                # print(f"  ⚠️ Erro ao prever com {nome}: {str(e)[:50]}")
                continue
        
        if not previsoes_ponderadas:
            return np.full(len(X), 0.5)
        
        return np.sum(previsoes_ponderadas, axis=0)
    
    @staticmethod
    def prever_ensemble(modelos, X):
        """Previsão simples (média) - mantido para fallback"""
        # Converte X para DataFrame se for np.array (saída do PCA)
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'PC{i+1}' for i in range(X.shape[1])])
            
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
        
        return np.mean(previsoes, axis=0) if previsoes else np.full(len(X), 0.5)
    
    # Métodos de Otimização Optuna (mantidos da v7)
    @staticmethod
    def _otimizar_xgboost(X, y):
        def objective(trial):
            params = {'n_estimators': trial.suggest_int('n_estimators', 50, 300), ...}
            # ... (Lógica de otimização Optuna)
            return 0.5 # Placeholder
        study = optuna.create_study(direction='maximize')
        # study.optimize(objective, n_trials=30)
        return study.best_params if hasattr(study, 'best_params') else {}
    
    @staticmethod
    def _otimizar_lightgbm(X, y):
        def objective(trial):
            params = {'n_estimators': trial.suggest_int('n_estimators', 50, 400), ...}
            # ... (Lógica de otimização Optuna)
            return 0.5 # Placeholder
        study = optuna.create_study(direction='maximize')
        # study.optimize(objective, n_trials=30)
        return study.best_params if hasattr(study, 'best_params') else {}

# =============================================================================
# CLASSE: OTIMIZADOR DE PORTFÓLIO (v7)
# =============================================================================

class OtimizadorPortfolioAvancado:
    """Otimização de portfólio com volatilidade GARCH e CVaR (v7)"""
    
    def __init__(self, returns_df, garch_vols=None, fundamental_data=None, ml_predictions=None):
        self.returns = returns_df
        self.mean_returns = returns_df.mean() * 252
        
        if garch_vols is not None and not garch_vols.empty:
            self.cov_matrix = self._construir_matriz_cov_garch(returns_df, garch_vols)
        else:
            self.cov_matrix = returns_df.cov() * 252 # Fallback
            print("  ⚠️ GARCH vols indisponíveis, usando covariância histórica.")
        
        self.num_ativos = len(returns_df.columns)
    
    def _construir_matriz_cov_garch(self, returns_df, garch_vols):
        """Constrói matriz de covariância usando volatilidades GARCH"""
        corr_matrix = returns_df.corr()
        vol_array = np.array([
            garch_vols.get(ativo, returns_df[ativo].std() * np.sqrt(252)) # Fallback
            for ativo in returns_df.columns
        ])
        
        if np.isnan(vol_array).all() or np.all(vol_array == 0):
            return returns_df.cov() * 252
            
        cov_matrix = corr_matrix.values * np.outer(vol_array, vol_array)
        return pd.DataFrame(cov_matrix, index=returns_df.columns, columns=returns_df.columns)
    
    def estatisticas_portfolio(self, pesos):
        p_retorno = np.dot(pesos, self.mean_returns)
        p_vol = np.sqrt(np.dot(pesos.T, np.dot(self.cov_matrix, pesos)))
        return p_retorno, p_vol
    
    def sharpe_negativo(self, pesos):
        p_retorno, p_vol = self.estatisticas_portfolio(pesos)
        if p_vol <= 1e-9: return -100.0
        return -(p_retorno - TAXA_LIVRE_RISCO) / p_vol
    
    def minimizar_volatilidade(self, pesos):
        return self.estatisticas_portfolio(pesos)[1]
    
    def calcular_cvar(self, pesos, confidence=0.95):
        """Calcula Conditional Value at Risk (CVaR)"""
        portfolio_returns = self.returns @ pesos
        sorted_returns = np.sort(portfolio_returns)
        var_index = int(np.floor((1 - confidence) * len(sorted_returns)))
        var = sorted_returns[var_index]
        cvar = sorted_returns[sorted_returns <= var].mean()
        return cvar

    def cvar_negativo(self, pesos, confidence=0.95):
        return -self.calcular_cvar(pesos, confidence)

    def otimizar(self, estrategia='MaxSharpe', confidence_level=0.95):
        """Executa otimização do portfólio"""
        restricoes = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        num_assets = self.num_ativos
        limites = tuple((PESO_MIN, PESO_MAX) for _ in range(num_assets)) if num_assets > 1 else ((0,1),)
        chute_inicial = np.array([1.0 / num_assets] * num_assets)
        
        if num_assets > 0:
            min_w, max_w = limites[0]
            chute_inicial = np.clip(chute_inicial, min_w, max_w)
            chute_inicial /= np.sum(chute_inicial) # Re-normaliza

        if estrategia == 'MinVolatility':
            objetivo = self.minimizar_volatilidade
        elif estrategia == 'CVaR':
            objetivo = lambda pesos: self.cvar_negativo(pesos, confidence=confidence_level)
        else: # Default to MaxSharpe
            objetivo = self.sharpe_negativo
        
        try:
            resultado = minimize(
                objetivo,
                chute_inicial,
                method='SLSQP',
                bounds=limites,
                constraints=restricoes,
                options={'maxiter': 500, 'ftol': 1e-6}
            )
            
            if resultado.success:
                final_weights = resultado.x / np.sum(resultado.x)
                return {ativo: peso for ativo, peso in zip(self.returns.columns, final_weights)}
            else:
                print(f"  ✗ Otimização falhou: {resultado.message} -> Usando pesos iguais.")
                return {ativo: 1.0 / self.num_ativos for ativo in self.returns.columns}
        except Exception as e:
            print(f"  ✗ Erro na otimização: {str(e)} -> Usando pesos iguais.")
            return {ativo: 1.0 / self.num_ativos for ativo in self.returns.columns}

# =============================================================================
# CLASSE PRINCIPAL: CONSTRUTOR DE PORTFÓLIO AUTOML (REATORADO v9.0)
# =============================================================================

class ConstrutorPortfolioAutoML:
    """
    v9.0: Construtor principal com pipeline de ML otimizado por PCA.
    """
    
    def __init__(self, valor_investimento, periodo=PERIODO_DADOS):
        self.valor_investimento = valor_investimento
        self.periodo = periodo
        
        # Dados brutos e processados
        self.dados_por_ativo = {}
        self.dados_fundamentalistas = pd.DataFrame()
        self.dados_fundamentalistas_raw = pd.DataFrame()
        self.dados_performance = pd.DataFrame()
        self.volatilidades_garch = {}
        self.predicoes_ml = {}
        self.predicoes_estatisticas = {}
        self.ativos_sucesso = []
        self.features_tecnicas_identificadas = set()
        
        # Pipelines de ML (v9.0)
        self.ml_pipelines = {} # Armazena {'scaler', 'imputer', 'pca', 'columns'} por ativo
        
        # Modelos e Governança
        self.modelos_ml = {}
        self.auc_scores = {}
        self.governanca_por_ativo = {}
        
        # Resultados
        self.ativos_selecionados = []
        self.alocacao_portfolio = {}
        self.metricas_portfolio = {}
        self.metodo_alocacao_atual = "Não Aplicado"
        self.justificativas_selecao = {}
        self.perfil_dashboard = {}
        self.pesos_atuais = {}
        self.scores_combinados = pd.DataFrame()
    
    def coletar_e_processar_dados(self, simbolos):
        """Coleta e processa dados (GCS v7)"""
        
        coletor = ColetorDados(periodo=self.periodo)
        if not coletor.coletar_e_processar_dados(simbolos):
            return False
        
        self.dados_por_ativo = coletor.dados_por_ativo
        self.dados_fundamentalistas = coletor.dados_fundamentalistas # Normalizados
        self.dados_fundamentalistas_raw = coletor.dados_fundamentalistas_raw # Brutos
        self.ativos_sucesso = coletor.ativos_sucesso
        self.dados_performance = coletor.metricas_performance
        self.features_tecnicas_identificadas = coletor.features_tecnicas_identificadas
        
        print(f"\n✓ Coleta concluída (v9.0): {len(self.ativos_sucesso)} ativos válidos\n")
        return True
    
    def calcular_volatilidades_garch(self):
        """Calcula volatilidades GARCH (v7)"""
        print("\n📊 Calculando volatilidades GARCH...")
        for simbolo in tqdm(self.ativos_sucesso, desc="Modelagem GARCH"):
            if simbolo not in self.dados_por_ativo or 'returns' not in self.dados_por_ativo[simbolo]:
                continue
            returns = self.dados_por_ativo[simbolo]['returns']
            garch_vol = VolatilidadeGARCH.ajustar_garch(returns, tipo_modelo='GARCH')
            if np.isnan(garch_vol):
                garch_vol = VolatilidadeGARCH.ajustar_garch(returns, tipo_modelo='EGARCH')
            if np.isnan(garch_vol):
                garch_vol = returns.std() * np.sqrt(252)
            self.volatilidades_garch[simbolo] = garch_vol
        print(f"✓ Volatilidades GARCH calculadas.\n")
    
    def treinar_modelos_ensemble(self, dias_lookback_ml=LOOKBACK_ML, otimizar=False):
        """
        v9.0: Treina modelos ML com PCA para otimização de velocidade.
        """
        print("\n🤖 Treinando Modelos de Machine Learning (Otimizado com PCA v9.0)...")
        
        # Define quais colunas são features (técnicas + fundamentalistas)
        # Usamos as colunas fundamentalistas *brutas* para o ML, não as normalizadas
        colunas_features_fund = list(self.dados_fundamentalistas_raw.columns)
        colunas_features_tec = list(self.features_tecnicas_identificadas)
        colunas_features_totais = colunas_features_tec + colunas_features_fund
        
        self.predicoes_estatisticas = {}
        self.ml_pipelines = {} # Reseta pipelines
        
        for ativo in tqdm(self.ativos_sucesso, desc="Treinamento ML+PCA"):
            if ativo not in self.dados_por_ativo:
                continue

            df = self.dados_por_ativo[ativo].copy()
            
            # --- 1. Modelos Estatísticos (Sem mudança) ---
            if 'Close' in df.columns and len(df) >= 100:
                try:
                    close_series = df['Close']
                    resultado_estatistico = ModelosEstatisticos.ensemble_estatistico(
                        close_series, 
                        horizon=dias_lookback_ml
                    )
                    self.predicoes_estatisticas[ativo] = {
                        'forecast': resultado_estatistico.get('ensemble_forecast', np.nan),
                        'current_price': close_series.iloc[-1],
                        'predicted_direction': 1 if resultado_estatistico.get('ensemble_forecast', 0) > close_series.iloc[-1] else 0
                    }
                except Exception as e:
                    self.predicoes_estatisticas[ativo] = {'forecast': np.nan, 'predicted_direction': 0.5}
            
            # --- 2. Preparação ML (PCA v9.0) ---
            
            # Cria target
            df['Future_Direction'] = np.where(
                df['Close'].pct_change(dias_lookback_ml).shift(-dias_lookback_ml) > 0,
                1,
                0
            )
            
            # Seleciona features e remove NaNs
            features_para_treino = [f for f in colunas_features_totais if f in df.columns]
            df_treino = df[features_para_treino + ['Future_Direction']].dropna(subset=['Future_Direction'])
            
            if len(df_treino) < MIN_DIAS_HISTORICO:
                self.predicoes_ml[ativo] = {'predicted_proba_up': 0.5, 'auc_roc_score': np.nan}
                continue
            
            X = df_treino[features_para_treino]
            y = df_treino['Future_Direction']
            
            if len(np.unique(y)) < 2:
                self.predicoes_ml[ativo] = {'predicted_proba_up': 0.5, 'auc_roc_score': np.nan}
                continue
            
            try:
                # --- 3. NOVO v9.0: Criação do Pipeline PCA ---
                # Pipeline: 1. Imputar NaNs (mediana) -> 2. Padronizar -> 3. PCA
                
                pipeline_pca = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('pca', PCA(n_components=PCA_N_COMPONENTS))
                ])
                
                # Treina o pipeline (Imputer, Scaler, PCA)
                X_pca = pipeline_pca.fit_transform(X)
                
                # Salva o pipeline e as colunas originais
                self.ml_pipelines[ativo] = {
                    'pipeline': pipeline_pca,
                    'columns': X.columns,
                    'n_components': X_pca.shape[1]
                }
                
                n_comp = X_pca.shape[1]
                
                # --- 4. Treinamento do Ensemble (usando X_pca) ---
                modelos, auc_scores = EnsembleML.treinar_ensemble(X_pca, y, otimizar_optuna=otimizar)
                
                if not modelos:
                    raise Exception("Falha no treinamento do EnsembleML.")
                
                self.modelos_ml[ativo] = modelos
                self.auc_scores[ativo] = auc_scores
                
                # --- 5. Governança (usando métricas do treino) ---
                self.governanca_por_ativo[ativo] = GovernancaModelo(ativo)
                auc_medio = np.mean([s for s in auc_scores.values() if s > 0.5]) if auc_scores else 0.5
                
                # (Métricas de precisão/recall exigiriam y_pred_final no CV, simplificado para AUC)
                self.governanca_por_ativo[ativo].adicionar_metricas(
                    auc_medio, np.nan, np.nan, np.nan
                )
                
                # --- 6. Previsão Final (usando o pipeline PCA) ---
                # Pega a última linha de features (dados mais recentes)
                last_features = df[features_para_treino].iloc[[-1]] 
                
                # Aplica o pipeline (Impute, Scale, PCA)
                last_features_pca = pipeline_pca.transform(last_features)
                
                # Previsão ponderada
                proba_final = EnsembleML.prever_ensemble_ponderado(modelos, auc_scores, last_features_pca)[0]
                
                self.predicoes_ml[ativo] = {
                    'predicted_proba_up': proba_final,
                    'auc_roc_score': auc_medio,
                    'model_name': f'Ensemble PCA ({n_comp} comps)',
                    'num_models': len(modelos)
                }
                
                # print(f"  ✓ {ativo}: Proba={proba_final:.3f}, AUC={auc_medio:.3f}, PCA Comp={n_comp}")
                
            except Exception as e:
                print(f"  ✗ Erro ML+PCA em {ativo}: {str(e)}")
                self.predicoes_ml[ativo] = {'predicted_proba_up': 0.5, 'auc_roc_score': np.nan}
        
        print(f"✓ Modelos ML (PCA) treinados para {len(self.predicoes_ml)} ativos")
        print(f"✓ Modelos estatísticos treinados para {len(self.predicoes_estatisticas)} ativos\n")
    
    def pontuar_e_selecionar_ativos(self, horizonte_tempo):
        """Pontua e ranqueia ativos usando sistema multi-fator (v7)"""
        
        # 1. Definição de Pesos Adaptativos
        if horizonte_tempo == "CURTO PRAZO":
            WEIGHT_PERF, WEIGHT_FUND, WEIGHT_TECH = 0.40, 0.10, 0.20
        elif horizonte_tempo == "LONGO PRAZO":
            WEIGHT_PERF, WEIGHT_FUND, WEIGHT_TECH = 0.40, 0.50, 0.10
        else: # Médio Prazo
            WEIGHT_PERF, WEIGHT_FUND, WEIGHT_TECH = 0.40, 0.30, 0.30
        
        # 2. Normalização dos pesos
        total_non_ml_weight = WEIGHT_PERF + WEIGHT_FUND + WEIGHT_TECH
        scale_factor = (1.0 - WEIGHT_ML) / total_non_ml_weight if total_non_ml_weight > 0 else 0
        
        WEIGHT_PERF *= scale_factor
        WEIGHT_FUND *= scale_factor
        WEIGHT_TECH *= scale_factor
        final_ml_weight = WEIGHT_ML

        self.pesos_atuais = {
            'Performance': WEIGHT_PERF,
            'Fundamentos': WEIGHT_FUND,
            'Técnicos': WEIGHT_TECH,
            'ML': final_ml_weight
        }
        
        # 3. Combina dados (Performance + Fundamentos Normalizados)
        # Usamos self.dados_fundamentalistas (normalizados) para o score
        combinado = self.dados_performance.join(self.dados_fundamentalistas, how='inner').copy()
        
        # Adiciona indicadores técnicos atuais (RSI, MACD) e ML
        for asset in combinado.index:
            if asset in self.dados_por_ativo and 'rsi_14' in self.dados_por_ativo[asset].columns:
                df = self.dados_por_ativo[asset]
                combinado.loc[asset, 'rsi_current'] = df['rsi_14'].iloc[-1]
                combinado.loc[asset, 'macd_current'] = df['macd'].iloc[-1]
            if asset in self.predicoes_ml:
                ml_info = self.predicoes_ml[asset]
                combinado.loc[asset, 'ML_Proba'] = ml_info.get('predicted_proba_up', 0.5)
                combinado.loc[asset, 'ML_Confidence'] = ml_info.get('auc_roc_score', 0.5)
        
        # 4. Cálculo dos Scores
        scores = pd.DataFrame(index=combinado.index)
        
        # Score Performance (Sharpe)
        scores['performance_score'] = EngenheiroFeatures._normalizar(
            combinado.get('sharpe', pd.Series(0, index=combinado.index)), 
            maior_melhor=True
        ) * WEIGHT_PERF
        
        # Score Fundamentalista (P/L (menor=melhor) e ROE (maior=melhor))
        # (Usando dados já normalizados/padronizados pelo RobustScaler)
        pe_score = EngenheiroFeatures._normalizar(combinado.get('pe_ratio', pd.Series(0, index=combinado.index)), maior_melhor=False)
        roe_score = EngenheiroFeatures._normalizar(combinado.get('roe', pd.Series(0, index=combinado.index)), maior_melhor=True)
        scores['fundamental_score'] = (pe_score * 0.5 + roe_score * 0.5) * WEIGHT_FUND
        
        # Score Técnico (RSI e MACD)
        # Normaliza RSI (mais próximo de 50 é melhor, ou seja, menos sobrecomprado/vendido)
        rsi_proximity_score = 100 - abs(combinado.get('rsi_current', pd.Series(50, index=combinado.index)) - 50)
        rsi_norm = EngenheiroFeatures._normalizar(rsi_proximity_score.clip(0, 100), maior_melhor=True)
        macd_norm = EngenheiroFeatures._normalizar(combinado.get('macd_current', pd.Series(0, index=combinado.index)), maior_melhor=True)
        scores['technical_score'] = (rsi_norm * 0.5 + macd_norm * 0.5) * WEIGHT_TECH
        
        # Score ML (Pondera Probabilidade e Confiança/AUC)
        ml_proba_norm = EngenheiroFeatures._normalizar(combinado.get('ML_Proba', pd.Series(0.5, index=combinado.index)), maior_melhor=True)
        ml_confidence_norm = EngenheiroFeatures._normalizar(combinado.get('ML_Confidence', pd.Series(0.5, index=combinado.index)), maior_melhor=True)
        scores['ml_score_weighted'] = (ml_proba_norm * 0.6 + ml_confidence_norm * 0.4) * final_ml_weight
        
        # Score Total
        scores['total_score'] = scores.sum(axis=1)
        
        # Junta com dados brutos para exibição
        self.scores_combinados = scores.join(self.dados_fundamentalistas_raw).join(self.dados_performance).sort_values('total_score', ascending=False)
        
        # 5. Seleção Final com Diversificação Setorial
        ranked_assets = self.scores_combinados.index.tolist()
        final_portfolio = []
        selected_sectors = set()
        num_assets_to_select = min(NUM_ATIVOS_PORTFOLIO, len(ranked_assets))

        for asset in ranked_assets:
            sector = self.dados_fundamentalistas_raw.loc[asset, 'sector'] if asset in self.dados_fundamentalistas_raw.index else 'Unknown'
            
            # Tenta adicionar 1 de cada setor primeiro
            if sector not in selected_sectors:
                final_portfolio.append(asset)
                selected_sectors.add(sector)
            
            if len(final_portfolio) >= num_assets_to_select:
                break
        
        # Se não preencheu, preenche com os melhores restantes (permitindo 2 por setor)
        if len(final_portfolio) < num_assets_to_select:
            for asset in ranked_assets:
                if asset not in final_portfolio:
                    final_portfolio.append(asset)
                    if len(final_portfolio) >= num_assets_to_select:
                        break

        self.ativos_selecionados = final_portfolio
        return self.ativos_selecionados
    
    def otimizar_alocacao(self, nivel_risco):
        """Otimiza alocação (MPT v7)"""
        
        if not self.ativos_selecionados:
            self.metodo_alocacao_atual = "ERRO: Ativos Insuficientes"
            return {}
        
        available_assets_returns = {s: self.dados_por_ativo[s]['returns']
                                    for s in self.ativos_selecionados if s in self.dados_por_ativo}
        
        final_returns_df = pd.DataFrame(available_assets_returns).dropna()
        
        if final_returns_df.shape[0] < 50: # Fallback
            weights = {asset: 1.0 / len(self.ativos_selecionados) for asset in self.ativos_selecionados}
            self.metodo_alocacao_atual = 'PESOS IGUAIS (Dados insuficientes)'
        else:
            garch_vols_selecionados = {s: self.volatilidades_garch.get(s, final_returns_df[s].std() * np.sqrt(252))
                                       for s in final_returns_df.columns}

            optimizer = OtimizadorPortfolioAvancado(final_returns_df, garch_vols=garch_vols_selecionados)
            
            strategy = 'MaxSharpe' # Default
            if 'CONSERVADOR' in nivel_risco or 'INTERMEDIÁRIO' in nivel_risco:
                strategy = 'MinVolatility'
            elif 'AVANÇADO' in nivel_risco:
                strategy = 'CVaR' 
                
            weights = optimizer.otimizar(estrategia=strategy)
            self.metodo_alocacao_atual = f'{strategy} (GARCH)'
            
        # Formata a alocação
        total_weight = sum(weights.values())
        if total_weight == 0: total_weight = 1 # Evita divisão por zero
            
        self.alocacao_portfolio = {
            s: {
                'weight': w / total_weight,
                'amount': self.valor_investimento * (w / total_weight)
            }
            for s, w in weights.items() if s in self.ativos_selecionados
        }
        
        return self.alocacao_portfolio
    
    def calcular_metricas_portfolio(self):
        """Calcula métricas consolidadas (v7)"""
        
        if not self.alocacao_portfolio: return {}
        
        allocated_assets = list(self.alocacao_portfolio.keys())
        valid_returns_data = {s: self.dados_por_ativo[s]['returns']
                              for s in allocated_assets if s in self.dados_por_ativo}
        
        returns_df = pd.DataFrame(valid_returns_data).dropna()
        if returns_df.empty: return {}
        
        weights_dict = {s: self.alocacao_portfolio[s]['weight'] for s in returns_df.columns}
        weights = np.array([weights_dict[s] for s in returns_df.columns])
        weights = weights / np.sum(weights)

        portfolio_returns = (returns_df * weights).sum(axis=1)
        
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - TAXA_LIVRE_RISCO) / annual_volatility if annual_volatility > 0 else 0
        
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        max_drawdown = ((cumulative_returns - running_max) / running_max).min()
        
        self.metricas_portfolio = {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_investment': self.valor_investimento
        }
        return self.metricas_portfolio
    
    def gerar_justificativas(self):
        """Gera justificativas textuais (v7)"""
        
        for simbolo in self.ativos_selecionados:
            if simbolo not in self.dados_por_ativo: continue
            justification = []
            
            # Performance (dados_performance)
            if simbolo in self.dados_performance.index:
                perf = self.dados_performance.loc[simbolo]
                justification.append(f"Perf: Sharpe {perf.get('sharpe', np.nan):.3f}, Ret {perf.get('retorno_anual', np.nan)*100:.2f}%")
            
            # Fundamental (dados_fundamentalistas_raw)
            if simbolo in self.dados_fundamentalistas_raw.index:
                fund = self.dados_fundamentalistas_raw.loc[simbolo]
                justification.append(f"Fund: P/L {fund.get('pe_ratio', np.nan):.2f}, ROE {fund.get('roe', np.nan):.2f}%")
            
            # ML (predicoes_ml)
            if simbolo in self.predicoes_ml:
                ml = self.predicoes_ml[simbolo]
                proba_up = ml.get('predicted_proba_up', 0.5)
                auc_score = ml.get('auc_roc_score', np.nan)
                auc_str = f"{auc_score:.3f}" if not pd.isna(auc_score) else "N/A"
                model_name = ml.get('model_name', 'Ensemble')
                justification.append(f"ML: Prob. Alta {proba_up*100:.1f}% (AUC {auc_str} - {model_name})")
            
            # Volatilidade (volatilidades_garch)
            if simbolo in self.volatilidades_garch:
                vol_garch = self.volatilidades_garch[simbolo]
                justification.append(f"Vol. GARCH: {vol_garch*100:.2f}%")

            self.justificativas_selecao[simbolo] = " | ".join(justification)
        
        return self.justificativas_selecao
    
    def executar_pipeline(self, simbolos_customizados, perfil_investidor, otimizar_ml=False):
        """Executa pipeline completo (v7)"""
        
        self.perfil_dashboard = perfil_investidor
        ml_lookback_days = perfil_investidor.get('ml_lookback_days', LOOKBACK_ML)
        nivel_risco = perfil_investidor.get('risk_level', 'MODERADO')
        horizonte_tempo = perfil_investidor.get('time_horizon', 'MÉDIO PRAZO')
        
        # Etapa 1: Coleta de dados
        if not self.coletar_e_processar_dados(simbolos_customizados):
            return False
        
        # Etapa 2: Volatilidades GARCH
        self.calcular_volatilidades_garch()
        
        # Etapa 3: Treinamento ML (PCA v9.0) e Estatístico
        self.treinar_modelos_ensemble(dias_lookback_ml=ml_lookback_days, otimizar=otimizar_ml)
        
        # Etapa 4: Pontuação e seleção
        self.pontuar_e_selecionar_ativos(horizonte_tempo=horizonte_tempo)
        
        # Etapa 5: Otimização de alocação
        self.alocacao_portfolio = self.otimizar_alocacao(nivel_risco=nivel_risco) # Corrigido para atribuir
        
        # Etapa 6: Métricas do portfólio
        self.calcular_metricas_portfolio()
        
        # Etapa 7: Justificativas
        self.gerar_justificativas()
        
        return True

# =============================================================================
# CLASSE: ANALISADOR INDIVIDUAL DE ATIVOS (v7 - GCS)
# =============================================================================

class AnalisadorIndividualAtivos:
    """
    Análise completa de ativos individuais, consumindo dados GCS.
    v9.0: A lógica de clusterização (PCA nas métricas) é mantida da v7.
    """
    
    @staticmethod
    def realizar_clusterizacao_pca(ativo_selecionado, ativos_comparacao_list, leitor_gcs_instance, n_clusters=5):
        """
        Realiza clusterização K-means + PCA usando dados LIDOS DO GCS (v7).
        PCA é aplicado nas *Métricas de Performance/Fundamentos*, não nas features de ML.
        """
        
        comparison_data = {}
        # Lista de métricas (colunas do GCS) para clusterizar
        features_for_cluster = ['annual_return', 'annual_volatility', 'sharpe_ratio', 'max_drawdown', 
                                'pe_ratio', 'pb_ratio', 'roe', 'debt_to_equity', 'revenue_growth']
        
        if ativo_selecionado not in ativos_comparacao_list:
            ativos_comparacao_list.append(ativo_selecionado)
            
        for asset_comp in ativos_comparacao_list:
            try:
                df_completo = leitor_gcs_instance.ler_dados_historicos_completos(asset_comp)
                if df_completo is None or df_completo.empty:
                    continue
                
                # Extrai as métricas (estão na primeira linha do CSV do GCS)
                comp_metrics = {}
                primeira_linha = df_completo.iloc[0]
                
                for metric in features_for_cluster:
                    comp_metrics[metric] = primeira_linha.get(metric, np.nan)
                
                if not all(pd.isna(comp_metrics.get(f)) for f in features_for_cluster):
                    comparison_data[asset_comp] = comp_metrics
                    
            except Exception as e:
                continue
        
        if len(comparison_data) < 5: 
            return None, None, None
            
        df_comparacao = pd.DataFrame(comparison_data).T
        
        # Prepara dados: Imputa, Escala, PCA
        features_numericas = df_comparacao.select_dtypes(include=[np.number]).copy()
        features_numericas = features_numericas.replace([np.inf, -np.inf], np.nan)
        
        # Imputação (mediana)
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
        
        # PCA (2 ou 3 componentes para visualização)
        n_pca_components = min(3, len(features_numericas.columns))
        if n_pca_components < 2: return None, None, None # Precisa de pelo menos 2D
        
        pca = PCA(n_components=n_pca_components)
        componentes_pca = pca.fit_transform(dados_normalizados)
        
        actual_n_clusters = min(n_clusters, len(features_numericas))
        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(componentes_pca) # Clusteriza nos componentes
        
        resultado_pca = pd.DataFrame(
            componentes_pca,
            columns=[f'PC{i+1}' for i in range(componentes_pca.shape[1])],
            index=features_numericas.index
        )
        resultado_pca['Cluster'] = clusters
        
        return resultado_pca, pca, kmeans

# =============================================================================
# INTERFACE STREAMLIT - 5 ABAS (v9.0)
# =============================================================================

def configurar_pagina():
    """Configura página Streamlit (v7)"""
    st.set_page_config(
        page_title="Portfolio AutoML Elite v9.0",
        page_icon="⚡",
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
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: #f8f9fa;
            border-radius: 4px 4px 0 0;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ffffff;
            border-top: 2px solid #2c3e50;
        }
        .stMetric {
            padding: 10px 15px;
            background-color: #ffffff;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }
        /* Alertas de Governança */
        .alert-success { background-color: #d4edda; border-color: #c3e6cb; color: #155724; padding: .75rem 1.25rem; margin-bottom: 1rem; border: 1px solid transparent; border-radius: .25rem; }
        .alert-warning { background-color: #fff3cd; border-color: #ffeeba; color: #856404; padding: .75rem 1.25rem; margin-bottom: 1rem; border: 1px solid transparent; border-radius: .25rem; }
        .alert-error { background-color: #f8d7da; border-color: #f5c6cb; color: #721c24; padding: .75rem 1.25rem; margin-bottom: 1rem; border: 1px solid transparent; border-radius: .25rem; }
        </style>
    """, unsafe_allow_html=True)

def aba_introducao():
    """Aba 1: Introdução e Metodologia (v9.0)"""
    
    st.markdown("## 📚 Bem-vindo ao Sistema AutoML de Otimização de Portfólio v9.0")
    
    st.markdown("""
    <div class="info-box">
    <h3>🎯 O que este sistema faz?</h3>
    <p>Este é um sistema avançado de construção de portfólios que utiliza 
    <strong>Machine Learning (otimizado com PCA)</strong>, <strong>modelagem GARCH</strong> e <strong>teoria moderna de portfólio</strong> 
    para criar carteiras personalizadas baseadas no seu perfil de risco e objetivos.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔬 Metodologia Científica")
        st.markdown("""
        **1. Coleta de Dados Pré-Processados (GCS)**
        - Dados históricos, técnicos (100+) e fundamentalistas (20+)
        - Carregamento único para eficiência máxima.
        
        **2. Modelagem de Risco**
        - Volatilidade GARCH/EGARCH para cada ativo.
        - Matriz de covariância baseada em GARCH.
        
        **3. Análise de Perfil (Suitability)**
        - Questionário para definir Risco e Horizonte.
        - Ponderação adaptativa (Perf, Fund, Téc, ML) para seleção.
        """)
    
    with col2:
        st.markdown("### 🤖 Tecnologias Utilizadas (v9.0)")
        st.markdown("""
        **Machine Learning Ensemble (Otimizado)**
        - **Pipeline PCA v9.0**: Imputação, Padronização e PCA (Análise de Componentes Principais) são aplicados em 100+ *features* para acelerar o treinamento.
        - **Ensemble de 9 Modelos**: XGBoost, LightGBM, CatBoost, RF, ET, KNN, SVC, LR, GNB.
        - **Ponderação AUC-ROC**: Previsão final ponderada pela performance (AUC) de cada modelo.
        
        **Otimização de Portfólio**
        - Teoria de Markowitz (MaxSharpe/MinVol) e CVaR.
        - Restrições de peso (10-30% por ativo).
        
        **Governança de Modelo (MLOps)**
        - Monitoramento de AUC-ROC e alertas de degradação.
        """)
    
    st.markdown("---")
    
    st.markdown("### 📊 Como Funciona a Seleção dos 5 Ativos?")
    st.markdown("""
    O sistema utiliza um **score composto** que combina quatro dimensões, com pesos adaptados ao seu perfil de risco e horizonte de investimento:
    1.  **📈 Score de Performance:** Sharpe Ratio, Retorno Anual.
    2.  **💼 Score Fundamentalista:** P/L, ROE, P/VP, Dívida.
    3.  **🔧 Score Técnico:** RSI, MACD (momentum atual).
    4.  **🤖 Score de ML (PCA):** Probabilidade de alta (prevista pelo *ensemble* PCA) e Confiança (AUC).
    """)

def aba_selecao_ativos():
    """Aba 2: Seleção de Ativos (v7)"""
    
    st.markdown("## 🎯 Seleção de Ativos para Análise")
    
    st.markdown("""
    <div class="info-box">
    <p>Escolha o universo de ativos para análise. O sistema irá avaliar todos os ativos selecionados 
    e ranquear os <strong>5 melhores</strong> baseado no seu perfil e nos scores multi-fator.</p>
    </div>
    """, unsafe_allow_html=True)
    
    modo_selecao = st.radio(
        "**Modo de Seleção:**",
        ["📊 Ibovespa Completo", "🌐 Lista Completa (Todos os Setores)", "🏢 Setores Específicos"],
        index=0
    )
    
    ativos_selecionados = []
    
    if "Ibovespa Completo" in modo_selecao:
        ativos_selecionados = ATIVOS_IBOVESPA.copy()
    
    elif "Lista Completa" in modo_selecao:
        ativos_selecionados = TODOS_ATIVOS.copy()
    
    elif "Setores Específicos" in modo_selecao:
        setores_disponiveis = list(ATIVOS_POR_SETOR.keys())
        setores_selecionados = st.multiselect(
            "Escolha um ou mais setores:",
            options=setores_disponiveis,
            default=setores_disponiveis[:3]
        )
        if setores_selecionados:
            for setor in setores_selecionados:
                ativos_selecionados.extend(ATIVOS_POR_SETOR[setor])
        else:
            st.warning("Selecione pelo menos um setor")
    
    # Salva seleção no session state
    if ativos_selecionados:
        st.session_state.ativos_para_analise = sorted(list(set(ativos_selecionados)))
        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("✓ Ativos Selecionados", len(st.session_state.ativos_para_analise))
        col2.metric("→ Serão Avaliados", len(st.session_state.ativos_para_analise))
        col3.metric("→ Portfólio Final", NUM_ATIVOS_PORTFOLIO)
        
        st.success("✓ Seleção confirmada! Vá para a aba **'Construtor de Portfólio'**.")
    else:
        st.warning("⚠️ Nenhum ativo selecionado.")

def aba_construtor_portfolio():
    """Aba 3: Questionário e Construção de Portfólio (v7)"""
    
    if 'ativos_para_analise' not in st.session_state or not st.session_state.ativos_para_analise:
        st.warning("⚠️ Por favor, selecione os ativos na aba **'Seleção de Ativos'** primeiro.")
        return
    
    # Inicializa o builder
    if 'builder' not in st.session_state:
        st.session_state.builder = None
    if 'builder_complete' not in st.session_state:
        st.session_state.builder_complete = False
    
    # --- FASE 1: QUESTIONÁRIO ---
    if not st.session_state.builder_complete:
        st.markdown('## 📋 Questionário de Perfil do Investidor')
        st.info(f"Analisando {len(st.session_state.ativos_para_analise)} ativos selecionados...")
        
        col_question1, col_question2 = st.columns(2)
        
        with st.form("investor_profile_form"):
            options_score = ['CT: Concordo Totalmente', 'C: Concordo', 'N: Neutro', 'D: Discordo', 'DT: Discordo Totalmente']
            options_reaction = ['A: Venderia', 'B: Manteria', 'C: Compraria mais']
            options_level_abc = ['A: Avançado', 'B: Intermediário', 'C: Iniciante']
            options_time_horizon = ['A: Curto (até 1 ano)', 'B: Médio (1-5 anos)', 'C: Longo (5+ anos)']
            options_liquidity = ['A: Menos de 6 meses', 'B: Entre 6 meses e 2 anos', 'C: Mais de 2 anos']
            
            with col_question1:
                st.markdown("#### Tolerância ao Risco")
                p2_risk = st.radio("**1. Aceito risco por retorno de longo prazo**", options_score, index=2, key='risk_accept')
                p3_gain = st.radio("**2. Ganhar o máximo é minha prioridade**", options_score, index=2, key='max_gain')
                p4_stable = st.radio("**3. Prefiro crescimento constante**", options_score, index=2, key='stable_growth')
                p5_loss = st.radio("**4. Evitar perdas é o mais importante**", options_score, index=2, key='avoid_loss')
                p511_reaction = st.radio("**5. Se meus investimentos caíssem 10%, eu:**", options_reaction, index=1, key='reaction')
                p_level = st.radio("**6. Meu nível de conhecimento:**", options_level_abc, index=1, key='level')
            
            with col_question2:
                st.markdown("#### Horizonte Temporal e Capital")
                p211_time = st.radio("**7. Prazo máximo para reavaliação:**", options_time_horizon, index=1, key='time_purpose')[0]
                p311_liquid = st.radio("**8. Necessidade de liquidez:**", options_liquidity, index=1, key='liquidity')[0]
                
                st.markdown("---")
                investment = st.number_input("Valor de Investimento (R$)", min_value=1000, value=100000, step=10000, key='investment_amount')
            
            with st.expander("Opções Avançadas"):
                otimizar_ml = st.checkbox("Ativar otimização Optuna (muito lento)", value=False, key='optimize_ml')
            
            submitted = st.form_submit_button("🚀 Gerar Portfólio Otimizado (v9.0)", type="primary")
            
            if submitted:
                # 1. Analisa Perfil
                risk_answers = {
                    'risk_accept': p2_risk, 'max_gain': p3_gain, 'stable_growth': p4_stable,
                    'avoid_loss': p5_loss, 'reaction': p511_reaction, 'level': p_level,
                    'time_purpose': p211_time, 'liquidity': p311_liquid
                }
                analyzer = AnalisadorPerfilInvestidor()
                risk_level, horizon, lookback, score = analyzer.calcular_perfil(risk_answers)
                
                profile = {
                    'risk_level': risk_level, 'time_horizon': horizon,
                    'ml_lookback_days': lookback, 'risk_score': score
                }
                
                # 2. Cria Construtor
                builder = ConstrutorPortfolioAutoML(investment)
                st.session_state.builder = builder
                
                # 3. Executa Pipeline (v9.0 com PCA)
                with st.spinner(f'Executando pipeline v9.0 (com PCA) para perfil **{risk_level}**...'):
                    success = builder.executar_pipeline(
                        simbolos_customizados=st.session_state.ativos_para_analise,
                        perfil_investidor=profile,
                        otimizar_ml=otimizar_ml
                    )
                    
                    if not success:
                        st.error("Falha ao coletar dados ou processar os ativos. Verifique o GCS ou a seleção de ativos.")
                        st.session_state.builder = None
                        return
                    
                    st.session_state.builder_complete = True
                    st.session_state.profile = profile # Salva o perfil
                    st.rerun() # Recarrega para exibir resultados
    
    # --- FASE 2: RESULTADOS ---
    else:
        builder = st.session_state.builder
        profile = st.session_state.profile
        assets = builder.ativos_selecionados
        allocation = builder.alocacao_portfolio
        
        st.markdown('## ✅ Portfólio Otimizado Gerado (v9.0)')
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Perfil de Risco", profile.get('risk_level', 'N/A'), f"Score: {profile.get('risk_score', 'N/A')}")
        col2.metric("Horizonte", profile.get('time_horizon', 'N/A'))
        col3.metric("Sharpe Ratio (Portfólio)", f"{builder.metricas_portfolio.get('sharpe_ratio', 0):.3f}")
        col4.metric("Estratégia MPT", builder.metodo_alocacao_atual.split('(')[0].strip())
        
        if st.button("🔄 Recomeçar Análise (Novo Perfil)", key='recomecar'):
            # Limpa apenas os resultados, mantém os dados carregados (eficiência)
            st.session_state.builder_complete = False
            st.session_state.profile = {}
            # Não limpa st.session_state.builder, para manter os dados GCS carregados
            # A nova execução do pipeline irá sobrescrever os resultados (ML, alocação)
            st.rerun()
        
        st.markdown("---")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Alocação", "📈 Performance", "🔬 Análise ML (PCA)", "📉 Volatilidade GARCH", "❓ Justificativas"
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
                    fig_alloc = px.pie(alloc_data, values='Peso (%)', names='Ativo', hole=0.3)
                    fig_alloc.update_layout(**obter_template_grafico(), title_text="Distribuição do Portfólio")
                    st.plotly_chart(fig_alloc, use_container_width=True)
            
            with col_table:
                st.markdown('#### Detalhamento dos Ativos')
                alloc_table = []
                for asset in assets:
                    if asset in allocation and allocation[asset]['weight'] > 0:
                        weight = allocation[asset]['weight']
                        amount = allocation[asset]['amount']
                        sector = builder.dados_fundamentalistas_raw.loc[asset, 'sector'] if asset in builder.dados_fundamentalistas_raw.index else 'N/A'
                        ml_info = builder.predicoes_ml.get(asset, {})
                        
                        alloc_table.append({
                            'Ativo': asset.replace('.SA', ''),
                            'Setor': sector,
                            'Peso (%)': f"{weight * 100:.2f}",
                            'Valor (R$)': f"R$ {amount:,.2f}",
                            'ML Prob. Alta (%)': f"{ml_info.get('predicted_proba_up', 0.5)*100:.1f}",
                            'ML Modelo': ml_info.get('model_name', 'N/A'),
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
            st.markdown('#### Evolução dos Retornos Cumulativos (Ativos Selecionados)')
            fig_cum = go.Figure()
            for asset in assets:
                if asset in builder.dados_por_ativo:
                    returns = builder.dados_por_ativo[asset]['returns']
                    cum_returns = (1 + returns).cumprod()
                    fig_cum.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns.values, name=asset.replace('.SA', ''), mode='lines'))
            fig_cum.update_layout(**obter_template_grafico(), title_text="Retornos Cumulativos (Base 1)", height=500)
            st.plotly_chart(fig_cum, use_container_width=True)
        
        with tab3:
            st.markdown('#### Análise de Machine Learning (Otimizado com PCA v9.0)')
            ml_data = []
            for asset in assets:
                if asset in builder.predicoes_ml:
                    ml_info = builder.predicoes_ml[asset]
                    ml_data.append({
                        'Ativo': asset.replace('.SA', ''),
                        'Prob. Alta (%)': ml_info.get('predicted_proba_up', 0.5) * 100,
                        'AUC-ROC (CV)': ml_info.get('auc_roc_score', np.nan),
                        'Modelo (v9.0)': ml_info.get('model_name', 'N/A'),
                    })
            df_ml = pd.DataFrame(ml_data)
            
            if not df_ml.empty:
                fig_ml = go.Figure(go.Bar(
                    x=df_ml['Ativo'],
                    y=df_ml['Prob. Alta (%)'],
                    marker=dict(color=df_ml['Prob. Alta (%)'], colorscale='RdYlGn', showscale=True),
                    text=df_ml['Prob. Alta (%)'].round(1),
                    textposition='outside'
                ))
                fig_ml.update_layout(**obter_template_grafico(), title_text="Probabilidade de Alta Futura (ML Ensemble + PCA)", height=400)
                st.plotly_chart(fig_ml, use_container_width=True)
                
                st.markdown('#### Métricas Detalhadas do ML (PCA v9.0)')
                st.dataframe(df_ml, use_container_width=True, hide_index=True)
        
        with tab4:
            st.markdown('#### Análise de Volatilidade GARCH vs Histórica')
            dados_garch = []
            for ativo in assets:
                if ativo in builder.dados_performance.index and ativo in builder.volatilidades_garch:
                    vol_hist = builder.dados_performance.loc[ativo, 'volatilidade_anual']
                    vol_garch = builder.volatilidades_garch.get(ativo)
                    status = '✓ GARCH' if not np.isnan(vol_garch) and vol_garch != vol_hist else '⚠️ Histórica (Fallback)'
                    dados_garch.append({
                        'Ativo': ativo.replace('.SA', ''),
                        'Vol. Histórica (%)': vol_hist * 100,
                        'Vol. GARCH/Fallback (%)': (vol_garch if not np.isnan(vol_garch) else vol_hist) * 100,
                        'Status': status
                    })
            df_garch = pd.DataFrame(dados_garch)
            
            if not df_garch.empty:
                fig_garch = go.Figure()
                fig_garch.add_trace(go.Bar(name='Histórica', x=df_garch['Ativo'], y=df_garch['Vol. Histórica (%)'], marker=dict(color='#7f8c8d')))
                fig_garch.add_trace(go.Bar(name='GARCH/Fallback', x=df_garch['Ativo'], y=df_garch['Vol. GARCH/Fallback (%)'], marker=dict(color='#3498db')))
                fig_garch.update_layout(**obter_template_grafico(), title_text="Comparação: Volatilidade Histórica vs GARCH", barmode='group', height=400)
                st.plotly_chart(fig_garch, use_container_width=True)
        
        with tab5:
            st.markdown('#### Justificativas de Seleção e Alocação (v9.0)')
            for asset, justification in builder.justificativas_selecao.items():
                weight = builder.alocacao_portfolio.get(asset, {}).get('weight', 0)
                st.markdown(f"""
                <div class="info-box">
                <h4>{asset.replace('.SA', '')} ({weight*100:.2f}%)</h4>
                <p>{justification}</p>
                </div>
                """, unsafe_allow_html=True)

def aba_analise_individual():
    """Aba 4: Análise Individual de Ativos (GCS v7)"""
    
    st.markdown("## 🔍 Análise Individual Completa de Ativos (Dados GCS)")
    
    # Usa dados do builder se já estiver carregado
    if 'builder' in st.session_state and st.session_state.builder is not None:
        builder = st.session_state.builder
        leitor_gcs_instance = builder.leitor_gcs # Reutiliza o leitor GCS
        ativos_disponiveis = builder.ativos_sucesso
    else:
        # Fallback: cria um novo leitor GCS e usa o IBOV
        leitor_gcs_instance = LeitorGCS(bucket_name=GCS_BUCKET_NAME, data_prefix=GCS_DATA_PREFIX)
        ativos_disponiveis = ATIVOS_IBOVESPA
        
    if not ativos_disponiveis:
        st.error("Nenhum ativo disponível. Execute o Construtor ou verifique a lista de ativos.")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        ativo_selecionado = st.selectbox(
            "Selecione um ativo para análise detalhada:",
            options=ativos_disponiveis,
            format_func=lambda x: x.replace('.SA', ''),
            key='individual_asset_select'
        )
    
    # Botão de análise para carregar dados sob demanda
    if 'ativo_analisado' not in st.session_state:
        st.session_state.ativo_analisado = None
    
    if col2.button("🔄 Analisar Ativo", key='analyze_asset_button', type="primary"):
        st.session_state.ativo_analisado = ativo_selecionado
    
    if st.session_state.ativo_analisado != ativo_selecionado:
         st.info("👆 Selecione um ativo e clique em 'Analisar Ativo' para carregar a análise completa.")
         return
    
    # --- Executa a Análise Individual ---
    with st.spinner(f"Analisando {ativo_selecionado} (Lendo do GCS)..."):
        
        # Tenta pegar do builder (cache)
        if 'builder' in st.session_state and st.session_state.builder and ativo_selecionado in builder.dados_por_ativo:
            df_completo = builder.dados_por_ativo[ativo_selecionado].copy()
            features_fund_raw = builder.dados_fundamentalistas_raw.loc[ativo_selecionado].to_dict()
        else:
            # Carrega do GCS
            df_completo = leitor_gcs_instance.ler_dados_historicos_completos(ativo_selecionado, usar_cache=False)
            if df_completo is None:
                st.error(f"Não foi possível carregar dados para {ativo_selecionado} do GCS.")
                return
            
            # Extrai features de fundamentos (lógica do Coletor)
            primeira_linha = df_completo.iloc[0]
            features_fund_raw = {}
            _, fund_perf_cols = builder.coletor._identificar_colunas(df_completo.columns)
            for col in fund_perf_cols:
                features_fund_raw[col] = primeira_linha.get(col, np.nan)
        
        if df_completo is None or df_completo.empty:
            st.error(f"❌ Dados insuficientes para {ativo_selecionado}.")
            return

        # --- Abas de Visualização ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Visão Geral",
            "📈 Análise Técnica",
            "💼 Análise Fundamentalista",
            "🤖 Machine Learning (v9.0)",
            "🔬 Clusterização PCA (3D)"
        ])
        
        # --- ABA 1: Visão Geral ---
        with tab1:
            st.markdown(f"### {ativo_selecionado.replace('.SA', '')} - Visão Geral")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Preço Atual", f"R$ {df_completo['Close'].iloc[-1]:.2f}", f"{df_completo['returns'].iloc[-1]*100:+.2f}%")
            col2.metric("Volume Médio", f"{df_completo['Volume'].mean():,.0f}")
            col3.metric("Setor", features_fund_raw.get('sector', 'N/A'))
            col4.metric("Indústria", features_fund_raw.get('industry', 'N/A'))
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df_completo.index, open=df_completo['Open'], high=df_completo['High'], low=df_completo['Low'], close=df_completo['Close'], name='Preço'), row=1, col=1)
            fig.add_trace(go.Bar(x=df_completo.index, y=df_completo['Volume'], name='Volume', marker=dict(color='lightblue')), row=2, col=1)
            fig.update_layout(**obter_template_grafico(), title_text=f"Histórico de Preços e Volume", height=600)
            st.plotly_chart(fig, use_container_width=True)

        # --- ABA 2: Análise Técnica ---
        with tab2:
            st.markdown("### Indicadores Técnicos (Dados GCS)")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("RSI (14)", f"{df_completo['rsi_14'].iloc[-1]:.2f}" if 'rsi_14' in df_completo.columns else "N/A")
            col2.metric("MACD", f"{df_completo['macd'].iloc[-1]:.4f}" if 'macd' in df_completo.columns else "N/A")
            col3.metric("Stoch %K", f"{df_completo['stoch_k'].iloc[-1]:.2f}" if 'stoch_k' in df_completo.columns else "N/A")
            col4.metric("ADX", f"{df_completo['adx'].iloc[-1]:.2f}" if 'adx' in df_completo.columns else "N/A")

            fig_osc = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("RSI (14)", "Stochastic Oscillator"))
            if 'rsi_14' in df_completo.columns:
                fig_osc.add_trace(go.Scatter(x=df_completo.index, y=df_completo['rsi_14'], name='RSI'), row=1, col=1)
                fig_osc.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
                fig_osc.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
            if 'stoch_k' in df_completo.columns:
                fig_osc.add_trace(go.Scatter(x=df_completo.index, y=df_completo['stoch_k'], name='%K'), row=2, col=1)
                fig_osc.add_trace(go.Scatter(x=df_completo.index, y=df_completo['stoch_d'], name='%D'), row=2, col=1)
            fig_osc.update_layout(**obter_template_grafico(), height=550)
            st.plotly_chart(fig_osc, use_container_width=True)

        # --- ABA 3: Análise Fundamentalista ---
        with tab3:
            st.markdown("### Análise Fundamentalista (Dados GCS)")
            st.markdown("#### Valuation")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("P/L", f"{features_fund_raw.get('pe_ratio', np.nan):.2f}")
            col2.metric("P/VP", f"{features_fund_raw.get('pb_ratio', np.nan):.2f}")
            col3.metric("EV/EBITDA", f"{features_fund_raw.get('ev_ebitda', np.nan):.2f}")
            col4.metric("Div. Yield", f"{features_fund_raw.get('div_yield', np.nan):.2f}%")
            
            st.markdown("#### Rentabilidade")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("ROE", f"{features_fund_raw.get('roe', np.nan):.2f}%")
            col2.metric("ROA", f"{features_fund_raw.get('roa', np.nan):.2f}%")
            col3.metric("ROIC", f"{features_fund_raw.get('roic', np.nan):.2f}%")
            col4.metric("Marg. Líquida", f"{features_fund_raw.get('profit_margin', np.nan):.2f}%")

            st.markdown("---")
            st.markdown("#### Todos os Fundamentos Disponíveis (GCS)")
            df_fund_display = pd.DataFrame.from_dict(features_fund_raw, orient='index', columns=['Valor'])
            st.dataframe(df_fund_display, use_container_width=True)

        # --- ABA 4: Machine Learning (v9.0) ---
        with tab4:
            st.markdown("### Análise de Machine Learning (Otimizado com PCA v9.0)")
            
            if 'builder' in st.session_state and st.session_state.builder and ativo_selecionado in builder.predicoes_ml:
                ml_info = builder.predicoes_ml[ativo_selecionado]
                pipeline_info = builder.ml_pipelines.get(ativo_selecionado)
                
                st.success("✓ Dados de ML carregados do pipeline de otimização (v9.0).")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Probabilidade de Alta Futura", f"{ml_info.get('predicted_proba_up', 0.5)*100:.2f}%")
                col2.metric("AUC-ROC Médio (CV)", f"{ml_info.get('auc_roc_score', 0):.3f}")
                
                if pipeline_info:
                    n_components = pipeline_info.get('n_components', 'N/A')
                    n_cols_orig = len(pipeline_info.get('columns', []))
                    col3.metric("Modelo PCA", f"{n_components} Componentes", f"de {n_cols_orig} features")
                else:
                    col3.metric("Modelo", ml_info.get('model_name', 'N/A'))
            else:
                st.warning("⚠️ Execute o **Construtor de Portfólio** primeiro para gerar as previsões ML (PCA).")

        # --- ABA 5: Clusterização PCA (3D) ---
        with tab5:
            st.markdown("### Clusterização e Similaridade (PCA nas Métricas)")
            st.info("Comparando este ativo com outros (Ibovespa) usando K-means + PCA (em 3D) sobre as *métricas* de performance/fundamentos.")
            
            # Usa os ativos do IBOV para comparação
            ativos_comparacao = ATIVOS_IBOVESPA.copy()
            if ativo_selecionado not in ativos_comparacao:
                ativos_comparacao.append(ativo_selecionado)
            
            if len(ativos_comparacao) >= 5:
                with st.spinner(f"Processando Clusterização 3D para {len(ativos_comparacao)} ativos..."):
                    resultado_pca, pca, kmeans = AnalisadorIndividualAtivos.realizar_clusterizacao_pca(
                        ativo_selecionado,
                        ativos_comparacao,
                        leitor_gcs_instance,
                        n_clusters=5
                    )
                
                if resultado_pca is not None and 'PC3' in resultado_pca.columns:
                    fig_pca = px.scatter_3d(
                        resultado_pca,
                        x='PC1', y='PC2', z='PC3',
                        color='Cluster',
                        hover_name=resultado_pca.index.str.replace('.SA', ''),
                        title='Clusterização K-means + PCA (3D) - Similaridade de Ativos'
                    )
                    fig_pca.update_layout(**obter_template_grafico(), height=600)
                    st.plotly_chart(fig_pca, use_container_width=True)
                    
                    if ativo_selecionado in resultado_pca.index:
                        cluster_ativo = resultado_pca.loc[ativo_selecionado, 'Cluster']
                        ativos_similares = resultado_pca[resultado_pca['Cluster'] == cluster_ativo].index.tolist()
                        st.success(f"**{ativo_selecionado.replace('.SA', '')}** pertence ao **Cluster {cluster_ativo}**.")
                        st.markdown(f"**Ativos similares:** {', '.join([a.replace('.SA', '') for a in ativos_similares if a != ativo_selecionado][:10])}...")
                
                elif resultado_pca is not None:
                     st.warning("PCA 2D gerado, mas 3D era esperado. Verifique os dados.")
                else:
                    st.error("Falha na clusterização. Dados insuficientes ou inconsistentes no GCS.")
            else:
                st.warning(f"Apenas {len(ativos_comparacao)} ativos disponíveis. Mínimo de 5 necessários.")

def aba_governanca():
    """Aba 5: Governança de Modelo (v7)"""
    
    st.markdown("## 🛡️ Governança de Modelo - Monitoramento de Performance (v9.0)")
    
    if 'builder' not in st.session_state or st.session_state.builder is None:
        st.warning("⚠️ Execute o **Construtor de Portfólio** primeiro para visualizar métricas de governança.")
        return
    
    builder = st.session_state.builder
    
    if not hasattr(builder, 'governanca_por_ativo') or not builder.governanca_por_ativo:
        st.info("📊 Dados de governança serão exibidos após o treinamento dos modelos ML.")
        return
    
    st.markdown("""
    <div class="info-box">
    <h4>📈 Sistema de Governança Elite (MLOps)</h4>
    <p>Monitora a performance (AUC-ROC) dos modelos ML (PCA) e emite alertas de degradação.</p>
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
    
    # Exibe status
    severidade = relatorio['severidade']
    status_msg = relatorio['status']
    if severidade == 'success':
        st.markdown(f'<div class="alert-success"><strong>✅ {status_msg}</strong></div>', unsafe_allow_html=True)
    elif severidade == 'warning':
        st.markdown(f'<div class="alert-warning"><strong>⚠️ {status_msg}</strong></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-error"><strong>🚨 {status_msg}</strong></div>', unsafe_allow_html=True)
    
    # Métricas
    st.markdown("### 📊 Métricas de Performance (AUC)")
    col1, col2, col3 = st.columns(3)
    metricas = relatorio['metricas']
    col1.metric("AUC Atual", f"{metricas['AUC Atual']:.3f}")
    col2.metric("AUC Médio (Janela)", f"{metricas['AUC Médio']:.3f}")
    col3.metric("AUC Máximo (Histórico)", f"{metricas['AUC Máximo']:.3f}")
    
    # Gráfico de histórico de AUC
    st.markdown("### 📈 Histórico de AUC-ROC")
    historico = relatorio['historico']
    if len(historico['AUC']) > 0:
        df_hist = pd.DataFrame({'Período': range(1, len(historico['AUC']) + 1), 'AUC-ROC': historico['AUC']})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_hist['Período'], y=df_hist['AUC-ROC'], mode='lines+markers', name='AUC-ROC'))
        fig.add_hline(y=AUC_THRESHOLD_MIN, line_dash="dash", line_color="red", annotation_text=f"Mínimo ({AUC_THRESHOLD_MIN})")
        fig.update_layout(**obter_template_grafico(), title_text=f"Evolução do AUC-ROC - {ativo_selecionado.replace('.SA', '')}", height=400)
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Função principal (v9.0)"""
    
    # Inicializa session state
    if 'builder' not in st.session_state:
        st.session_state.builder = None
        st.session_state.builder_complete = False
        st.session_state.profile = {}
        st.session_state.ativos_para_analise = []
        st.session_state.ativo_analisado = None
        
    configurar_pagina()
    
    # Sidebar
    st.sidebar.markdown(
        '<p style="font-size: 26px; font-weight: bold; color: #2c3e50;">⚡ AutoML Elite v9.0</p>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Recursos v9.0")
    st.sidebar.markdown("""
    - **Pipeline PCA (Novo!)**: Treinamento ML otimizado
    - **9 Modelos ML**: Ensemble AUC-Ponderado
    - **Governança MLOps**: Monitoramento de AUC
    - **GARCH/EGARCH**: Volatilidade avançada
    - **Modelos Estatísticos**: ARIMA, Prophet
    - **Clusterização 3D**: Análise de Pares
    - **Dados GCS**: Pipeline pré-processado
    """)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Versão**: 9.0.0 - Otimização PCA")
    
    # Título Principal
    st.markdown('<h1 class="main-header">Sistema AutoML Elite - Otimização Quantitativa v9.0</h1>', unsafe_allow_html=True)
    
    # 5 Abas
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📚 Introdução",
        "🎯 Seleção de Ativos",
        "🏗️ Construtor de Portfólio",
        "🔍 Análise Individual",
        "🛡️ Governança de Modelo"
    ])
    
    with tab1:
        aba_introducao()
    
    with tab2:
        aba_selecao_ativos()
    
    with tab3:
        aba_construtor_portfolio()
    
    with tab4:
        aba_analise_individual()
    
    with tab5:
        aba_governanca()

if __name__ == "__main__":
    main()
