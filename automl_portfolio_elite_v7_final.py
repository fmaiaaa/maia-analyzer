"""
=============================================================================
SISTEMA AUTOML AVANÇADO - OTIMIZAÇÃO DE PORTFÓLIO FINANCEIRO V5.0 ELITE
=============================================================================

Sistema completo de otimização de portfólio com:
- Questionário de perfil de investidor com regras de derrubada
- Seleção de ativos por setor via GCS
- Ensemble de 9 modelos ML ponderado por AUC-ROC
- Modelagem de volatilidade GARCH/EGARCH
- Otimização HRP (Hierarchical Risk Parity) e CVaR com Copula
- Engenharia massiva de features (200+)
- Smart Beta Factors e Elliott Wave
- Governança de modelo com alertas de drift
- Stress Testing e Explicabilidade (SHAP)
- Dashboard interativo completo

Versão: 5.0.0 - Sistema AutoML Elite Quantitativo (GCS + HRP + Copula)
Autor: Sistema AutoML
Data: 2025
=============================================================================
"""

import io
from google.cloud import storage
import warnings
import numpy as np
import pandas as pd
import subprocess
import sys
import time
from datetime import datetime, timedelta
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.stats import zscore, norm, t, multivariate_normal
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

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
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

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
    'keras': 'keras',
    'google-cloud-storage': 'google-cloud-storage'
}

def ensure_package(module_name, package_name):
    """Instala pacote se não estiver disponível"""
    try:
        __import__(module_name)
    except ImportError:
        print(f"Instalando {package_name}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', package_name])
        
        if 'streamlit' in sys.modules:
            import streamlit as st
            st.warning(f"{package_name} foi instalado. Por favor, **reexecute** o servidor Streamlit.")
            st.stop()

# Instala todas as dependências
try:
    for module, package in REQUIRED_PACKAGES.items():
        ensure_package(module.split('.')[0], package)
    
    import streamlit as st
    import yfinance as yf
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
    from sklearn.linear_model import RidgeClassifier, LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, precision_score, recall_score, f1_score
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier
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

PERIODO_DADOS = 'max'
MIN_DIAS_HISTORICO = 252
NUM_ATIVOS_PORTFOLIO = 5
TAXA_LIVRE_RISCO = 0.1075
LOOKBACK_ML = 30

# Ponderações padrão para os scores
WEIGHT_PERFORMANCE = 0.40
WEIGHT_FUNDAMENTAL = 0.30
WEIGHT_TECHNICAL = 0.30
WEIGHT_ML = 0.30

# Limites de peso por ativo na otimização
PESO_MIN = 0.10
PESO_MAX = 0.30

# Constantes de armazenamento GCS
GCS_BUCKET_NAME = 'meu-portfolio-dados-gratuitos'
GCS_MASTER_FILE_PATH = 'dados_consolidados/todos_ativos_master.csv'

# Constantes de governança
AUC_THRESHOLD_MIN = 0.65
AUC_DROP_THRESHOLD = 0.05
DRIFT_WINDOW = 20
STRESS_TEST_SIGMA = 2.0

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

TODOS_ATIVOS = []
for setor, ativos in ATIVOS_POR_SETOR.items():
    TODOS_ATIVOS.extend(ativos)
TODOS_ATIVOS = sorted(list(set(TODOS_ATIVOS)))

# =============================================================================
# CLASSE: GOVERNANÇA DE MODELO
# =============================================================================

class GovernancaModelo:
    """Classe para monitoramento e governança de modelos ML com concept drift"""
    
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
        
        if len(self.historico_auc) > self.max_historico:
            self.historico_auc.pop(0)
            self.historico_precision.pop(0)
            self.historico_recall.pop(0)
            self.historico_f1.pop(0)
        
        if auc > self.auc_maximo:
            self.auc_maximo = auc
    
    def verificar_alertas(self):
        """Verifica se há alertas de degradação de performance e concept drift"""
        if not self.historico_auc:
            return []
        
        alertas = []
        auc_atual = self.historico_auc[-1]
        
        # Alerta crítico: AUC abaixo do mínimo
        if auc_atual < AUC_THRESHOLD_MIN:
            alertas.append({
                'tipo': 'CRÍTICO',
                'mensagem': f'AUC ({auc_atual:.3f}) abaixo do mínimo aceitável ({AUC_THRESHOLD_MIN})'
            })
        
        # Alerta de degradação
        if self.auc_maximo > 0:
            degradacao = (self.auc_maximo - auc_atual) / self.auc_maximo
            if degradacao > AUC_DROP_THRESHOLD:
                alertas.append({
                    'tipo': 'ATENÇÃO',
                    'mensagem': f'Degradação de {degradacao*100:.1f}% em relação ao máximo ({self.auc_maximo:.3f})'
                })
        
        # P-10: Concept Drift - Tendência de queda consistente
        if len(self.historico_auc) >= 5:
            ultimos_5 = self.historico_auc[-5:]
            if all(ultimos_5[i] > ultimos_5[i+1] for i in range(len(ultimos_5)-1)):
                alertas.append({
                    'tipo': 'CONCEPT DRIFT',
                    'mensagem': 'Tendência de queda consistente nos últimos 5 períodos - possível drift de conceito'
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
        
        if any(a['tipo'] == 'CRÍTICO' for a in alertas):
            severidade = 'error'
            status = 'Modelo requer atenção imediata'
        elif any(a['tipo'] in ['ATENÇÃO', 'CONCEPT DRIFT'] for a in alertas):
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
    """Analisa perfil de risco e horizonte temporal do investidor com regras de derrubada"""
    
    def __init__(self):
        self.nivel_risco = ""
        self.horizonte_tempo = ""
        self.dias_lookback_ml = 5
        self.regras_aplicadas = []
    
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
            'A': 5,
            'B': 20,
            'C': 30
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
        """
        P-4, P-5, P-6: Calcula perfil com regras de derrubada e detecção de conflitos
        """
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
        
        # P-4: Regra de Derrubada 1 - Reação à queda de 10%
        if respostas_risco['reaction'] == 'A: Venderia':
            if nivel_risco in ['MODERADO', 'MODERADO-ARROJADO', 'AVANÇADO']:
                nivel_risco_original = nivel_risco
                nivel_risco = 'INTERMEDIÁRIO'
                self.regras_aplicadas.append(
                    f"Perfil reduzido de {nivel_risco_original} para INTERMEDIÁRIO devido à baixa tolerância a perdas"
                )
        
        # P-4: Regra de Derrubada 2 - Conhecimento iniciante
        if respostas_risco['level'] == 'C: Iniciante':
            if nivel_risco in ['MODERADO-ARROJADO', 'AVANÇADO']:
                nivel_risco_original = nivel_risco
                nivel_risco = 'MODERADO'
                self.regras_aplicadas.append(
                    f"Perfil reduzido de {nivel_risco_original} para MODERADO devido ao conhecimento iniciante"
                )
        
        # P-5: Conflito de Horizonte/Risco
        conflito_detectado = False
        if nivel_risco == 'AVANÇADO' and respostas_risco['liquidity'] == 'A':
            conflito_detectado = True
            ml_lookback = 20
            horizonte_tempo = "MÉDIO PRAZO"
            self.regras_aplicadas.append(
                "CONFLITO: Perfil AVANÇADO com liquidez de curto prazo. Horizonte ajustado para MÉDIO PRAZO (20 dias)"
            )
        
        self.nivel_risco = nivel_risco
        self.horizonte_tempo = horizonte_tempo
        self.dias_lookback_ml = ml_lookback
        
        return nivel_risco, horizonte_tempo, ml_lookback, pontuacao, conflito_detectado

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
        df['volatility_252'] = df['returns'].rolling(window=252).std() * np.sqrt(252)
        
        # Médias Móveis (SMA, EMA, WMA, HMA)
        for periodo in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{periodo}'] = SMAIndicator(close=df['Close'], window=periodo).sma_indicator()
            df[f'ema_{periodo}'] = EMAIndicator(close=df['Close'], window=periodo).ema_indicator()
            weights = np.arange(1, periodo + 1)
            df[f'wma_{periodo}'] = df['Close'].rolling(periodo).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        
        # Hull Moving Average (HMA)
        for periodo in [20, 50]:
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
        
        # P-9: Feature Elliott Wave - Ciclo de mercado
        df['elliott_cycle'] = EngenheiroFeatures._calcular_elliott_cycle(df)
        
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
        
        # Temporal encoding
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['day_of_month'] = df.index.day
        df['week_of_year'] = df.index.isocalendar().week
        
        return df.dropna()
    
    @staticmethod
    def _calcular_elliott_cycle(df):
        """
        P-9: Calcula fase do ciclo de mercado baseado em Elliott Wave
        Retorna número de dias desde último pico/vale
        """
        try:
            close_series = df['Close'].copy()
            
            # Identifica picos e vales locais
            window = 20
            picos = (close_series == close_series.rolling(window, center=True).max())
            vales = (close_series == close_series.rolling(window, center=True).min())
            
            # Conta dias desde último pico ou vale
            dias_desde_evento = pd.Series(0, index=df.index)
            
            contador = 0
            for i in range(len(df)):
                if picos.iloc[i] or vales.iloc[i]:
                    contador = 0
                else:
                    contador += 1
                dias_desde_evento.iloc[i] = contador
            
            return dias_desde_evento
        except:
            return pd.Series(0, index=df.index)
    
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
            return pd.Series(0, index=serie.index)
        
        min_val = serie.min()
        max_val = serie.max()
        
        if max_val == min_val:
            return pd.Series(0.5 if maior_melhor else 0.5, index=serie.index)
        
        if maior_melhor:
            return (serie - min_val) / (max_val - min_val)
        else:
            return (max_val - serie) / (max_val - min_val)

# =============================================================================
# CLASSE: COLETOR DE DADOS (COM GCS)
# =============================================================================

class ColetorDados:
    """Coleta e processa dados de mercado via Google Cloud Storage"""
    
    def __init__(self, periodo=PERIODO_DADOS):
        self.periodo = periodo
        self.dados_por_ativo = {}
        self.dados_fundamentalistas = pd.DataFrame()
        self.ativos_sucesso = []
        self.dados_macro = {}
        self.metricas_performance = pd.DataFrame()
        self.df_master = None
    
    def carregar_dados_gcs(self):
        """Carrega dados consolidados do Google Cloud Storage"""
        print("\n📊 Carregando dados do GCS...")
        print(f"Bucket: {GCS_BUCKET_NAME}")
        print(f"Arquivo: {GCS_MASTER_FILE_PATH}")
        
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(GCS_BUCKET_NAME)
            blob = bucket.blob(GCS_MASTER_FILE_PATH)
            
            csv_data_bytes = blob.download_as_bytes()
            csv_data = csv_data_bytes.decode('utf-8')
            
            self.df_master = pd.read_csv(io.StringIO(csv_data))
            self.df_master['Date'] = pd.to_datetime(self.df_master['Date'])
            
            print(f"✓ Dados carregados: {len(self.df_master)} linhas")
            print(f"✓ Ativos únicos: {self.df_master['ticker'].nunique()}")
            print(f"✓ Período: {self.df_master['Date'].min()} até {self.df_master['Date'].max()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados do GCS: {str(e)}")
            st.error(f"Erro ao conectar ao GCS: {str(e)}")
            return False
    
    def obter_dados_ativo(self, simbolo):
        """Obtém dados históricos de um ativo específico do GCS"""
        if self.df_master is None:
            if not self.carregar_dados_gcs():
                return None
        
        df_ativo = self.df_master[self.df_master['ticker'] == simbolo].copy()
        
        if df_ativo.empty:
            return None
        
        df_ativo = df_ativo.set_index('Date').sort_index()
        
        if 'ticker' in df_ativo.columns:
            df_ativo = df_ativo.drop('ticker', axis=1)
        
        return df_ativo
    
    def coletar_dados_macroeconomicos(self):
        """Coleta dados macroeconômicos do GCS"""
        print("\n📊 Coletando dados macroeconômicos do GCS...")
        
        if self.df_master is None:
            if not self.carregar_dados_gcs():
                return
        
        indices = {
            'IBOV': '^BVSP',
            'SP500': '^GSPC',
            'NASDAQ': '^IXIC',
            'VIX': '^VIX',
            'DXY': 'DX-Y.NYB',
            'GOLD': 'GC=F',
            'OIL': 'CL=F'
        }
        
        for nome, simbolo in indices.items():
            try:
                df_macro = self.df_master[self.df_master['ticker'] == simbolo].copy()
                
                if not df_macro.empty:
                    df_macro = df_macro.set_index('Date').sort_index()
                    if 'Close' in df_macro.columns:
                        self.dados_macro[nome] = df_macro['Close'].pct_change()
                        print(f"  ✓ {nome}: {len(df_macro)} dias")
                    else:
                        self.dados_macro[nome] = pd.Series()
                else:
                    self.dados_macro[nome] = pd.Series()

            except Exception as e:
                print(f"  ⚠️ {nome}: Erro - {str(e)[:50]}")
                self.dados_macro[nome] = pd.Series()
        
        print(f"✓ Dados macroeconômicos coletados: {len(self.dados_macro)} indicadores")

    def adicionar_correlacoes_macro(self, df, simbolo):
        """Adiciona correlações com indicadores macroeconômicos"""
        if not self.dados_macro or 'returns' not in df.columns:
            return df
        
        try:
            if df['returns'].isnull().all():
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
        """Coleta dados dos ativos do GCS e os processa com engenharia de features"""
        
        if self.df_master is None:
            if not self.carregar_dados_gcs():
                return False
        
        print(f"\n{'='*60}")
        print(f"FILTRANDO E PROCESSANDO DADOS - {len(simbolos)} ativos solicitados")
        print(f"{'='*60}\n")
        
        self.ativos_sucesso = []
        lista_fundamentalistas = []
        
        for simbolo in tqdm(simbolos, desc="⚙️ Processando ativos"):
            try:
                df_ativo = self.df_master[self.df_master['ticker'] == simbolo].copy()
                
                if df_ativo.empty:
                    print(f"  ⚠️ {simbolo}: Não encontrado no dataset.")
                    continue
                
                df_ativo = df_ativo.set_index('Date').sort_index()
                
                if 'ticker' in df_ativo.columns:
                    df_ativo = df_ativo.drop('ticker', axis=1)
                
                df_features = EngenheiroFeatures.calcular_indicadores_tecnicos(df_ativo)
                
                df_features = df_features.dropna(subset=['Close', 'returns'])
                
                min_dias_flexivel = max(180, int(MIN_DIAS_HISTORICO * 0.7))
                if len(df_features) < min_dias_flexivel:
                    print(f"  ⚠️ {simbolo}: Apenas {len(df_features)} dias após feature engineering (mínimo: {min_dias_flexivel}).")
                    continue
                
                threshold_nan_ratio = 0.5
                df_features = df_features.dropna(axis=0, thresh=len(df_features.columns) * threshold_nan_ratio)
                
                if len(df_features) < min_dias_flexivel:
                    print(f"  ⚠️ {simbolo}: Poucos dados restantes após remoção de NaNs ({len(df_features)} dias).")
                    continue

                self.dados_por_ativo[simbolo] = df_features
                self.ativos_sucesso.append(simbolo)
                
                print(f"  ✓ {simbolo}: {len(df_features)} dias processados com {len(df_features.columns)} features.")
                
                fund_cols = [col for col in df_features.columns if col.startswith('fund_')]
                if fund_cols:
                    features_fund_dict = df_features[fund_cols].iloc[-1].to_dict()
                    features_fund_cleaned = {k.replace('fund_', ''): v for k, v in features_fund_dict.items()}
                    
                    features_fund_cleaned['Ticker'] = simbolo
                    
                    if 'sector' in df_features.columns:
                        features_fund_cleaned['sector'] = df_features['sector'].iloc[-1]
                    if 'industry' in df_features.columns:
                        features_fund_cleaned['industry'] = df_features['industry'].iloc[-1]
                    
                    lista_fundamentalistas.append(features_fund_cleaned)
                else:
                    lista_fundamentalistas.append({'Ticker': simbolo})
            
            except Exception as e:
                print(f"  ❌ {simbolo}: Erro no processamento - {str(e)}")
                continue
        
        if len(self.ativos_sucesso) < NUM_ATIVOS_PORTFOLIO:
            print(f"\n❌ ERRO: Apenas {len(self.ativos_sucesso)} ativos válidos após processamento.")
            print(f"    Necessário: {NUM_ATIVOS_PORTFOLIO} ativos mínimos.")
            return False
        
        if lista_fundamentalistas:
            self.dados_fundamentalistas = pd.DataFrame(lista_fundamentalistas).set_index('Ticker')
            self.dados_fundamentalistas = self.dados_fundamentalistas.replace([np.inf, -np.inf], np.nan)
            
            scaler = RobustScaler()
            numeric_cols = self.dados_fundamentalistas.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if self.dados_fundamentalistas[col].isnull().any():
                    median_val = self.dados_fundamentalistas[col].median()
                    self.dados_fundamentalistas[col] = self.dados_fundamentalistas[col].fillna(median_val)
            
            if len(numeric_cols) > 0 and not self.dados_fundamentalistas.empty:
                self.dados_fundamentalistas[numeric_cols] = scaler.fit_transform(
                    self.dados_fundamentalistas[numeric_cols]
                )
        
        metricas = {}
        for simbolo in self.ativos_sucesso:
            if simbolo in self.dados_por_ativo and 'returns' in self.dados_por_ativo[simbolo]:
                returns = self.dados_por_ativo[simbolo]['returns']
                drawdown = self.dados_por_ativo[simbolo].get('drawdown', pd.Series([np.nan], index=returns.index))
                
                if not returns.empty and not returns.isnull().all():
                    ret_mean = returns.mean()
                    ret_std = returns.std()
                    
                    metricas[simbolo] = {
                        'retorno_anual': ret_mean * 252,
                        'volatilidade_anual': ret_std * np.sqrt(252),
                        'sharpe': (ret_mean * 252 - TAXA_LIVRE_RISCO) / (ret_std * np.sqrt(252)) if ret_std > 0 and not np.isnan(ret_std) else 0,
                        'max_drawdown': drawdown.min() if not drawdown.isnull().all() else np.nan
                    }

        self.metricas_performance = pd.DataFrame(metricas).T
        
        print(f"\n✓ Processamento concluído. {len(self.ativos_sucesso)} ativos válidos.")
        return True

# Arquivo muito extenso - continuando na próxima parte...
