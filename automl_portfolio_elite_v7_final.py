"""
=============================================================================
SISTEMA AUTOML AVANÇADO - OTIMIZAÇÃO DE PORTFÓLIO FINANCEIRO
Elite Quantitative Solution - FGV EPGE Edition
=============================================================================

Sistema completo de otimização de portfólio com:
- Questionário de perfil de investidor
- Seleção de ativos por setor
- Ensemble de modelos ML (XGBoost, LightGBM, RandomForest, CatBoost, Ridge, Neural Networks)
- Modelagem de volatilidade GARCH/EGARCH e DCC-GARCH
- Otimização de hiperparâmetros com Optuna
- Engenharia massiva de features com Smart Beta Factors
- Explicabilidade (SHAP/LIME) e Governança (Drift Monitoring)
- Otimização avançada: HRP, CVaR, Sortino, Restrições de Liquidez
- Teste de Estresse e Validação de Risco
- Dashboard interativo completo com formatação profissional

Versão: 6.0.1 - Elite Quantitative Solution (Fixed)
Baseado nas ementas: GRDECO222 (Machine Learning) e GRDECO203 (Lab Ciência de Dados - Finanças)
=============================================================================
"""

import warnings
import numpy as np
import pandas as pd
import subprocess
import sys
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

PERIODO_DADOS = 'max'
MIN_DIAS_HISTORICO = 252
LOOKBACK_ML = 20
NUM_ATIVOS_PORTFOLIO = 5
PESO_MIN = 0.10
PESO_MAX = 0.30
TAXA_LIVRE_RISCO = 0.1075
WEIGHT_PERF = 0.40
WEIGHT_FUND = 0.30
WEIGHT_TECH = 0.30
WEIGHT_ML = 0.30

WEIGHT_VALUE = 0.33
WEIGHT_QUALITY = 0.33
WEIGHT_MOMENTUM = 0.34

AUC_THRESHOLD_ALERT = 0.65  # Alerta se AUC cair abaixo deste valor
AUC_DROP_THRESHOLD = 0.05   # Alerta se queda de 5% no AUC
DRIFT_WINDOW = 20           # Janela para monitoramento de drift

STRESS_TEST_SIGMA = 2.0     # Número de desvios-padrão para choque

# Updated data collection period to maximum available and adjusted other constants
# Configurações Globais (mantidas do original, mas PERIODO_DADOS já definido acima)
PERIODO_DADOS = 'max'  # Changed from '2y' to 'max' for maximum historical depth
MIN_DIAS_HISTORICO = 252  # Reduced minimum to accommodate max period
NUM_ATIVOS_PORTFOLIO = 5
TAXA_LIVRE_RISCO = 0.1075 # Updated risk-free rate
LOOKBACK_ML = 20  # Extended prediction horizon to 20 days (adjusted by new constant)

# Ponderações padrão para os scores (mantidas, mas ajustadas dinamicamente na função de pontuação)
WEIGHT_PERFORMANCE = 0.40
WEIGHT_FUNDAMENTAL = 0.30
WEIGHT_TECHNICAL = 0.30
WEIGHT_ML = 0.30 # Adicionado peso para ML no score total

# Limites de peso por ativo na otimização (mantidos)
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
    """Calcula indicadores técnicos, fundamentalistas e Smart Beta Factors"""
    
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
        
        df['market_cycle_phase'] = EngenheiroFeatures._calcular_fase_ciclo(df['Close'])
        
        df['sentiment_score_30d'] = EngenheiroFeatures._simular_sentimento(df.index)
        df['esg_score_normalized'] = EngenheiroFeatures._simular_esg(df.index)
        
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
    
    # Calcular Smart Beta Factors explícitos (P-2)
    @staticmethod
    def calcular_smart_beta_factors(df_fundamentalista):
        """
        Calcula Smart Beta Factors explícitos (P-2)
        Value, Quality, Momentum normalizados de 0 a 1
        """
        df = df_fundamentalista.copy()
        
        value_components = []
        if 'pe_ratio' in df.columns:
            value_components.append(EngenheiroFeatures._normalizar(df['pe_ratio'], maior_melhor=False))
        if 'pb_ratio' in df.columns:
            value_components.append(EngenheiroFeatures._normalizar(df['pb_ratio'], maior_melhor=False))
        if 'ps_ratio' in df.columns:
            value_components.append(EngenheiroFeatures._normalizar(df['ps_ratio'], maior_melhor=False))
        
        df['Value_Score'] = pd.concat(value_components, axis=1).mean(axis=1) if value_components else 0.5
        
        quality_components = []
        if 'roe' in df.columns:
            quality_components.append(EngenheiroFeatures._normalizar(df['roe'], maior_melhor=True))
        if 'roic' in df.columns:
            quality_components.append(EngenheiroFeatures._normalizar(df['roic'], maior_melhor=True))
        if 'profit_margin' in df.columns:
            quality_components.append(EngenheiroFeatures._normalizar(df['profit_margin'], maior_melhor=True))
        
        df['Quality_Score'] = pd.concat(quality_components, axis=1).mean(axis=1) if quality_components else 0.5
        
        # Placeholder aqui, será preenchido na etapa de pontuação
        df['Momentum_Score'] = 0.5
        
        return df

    @staticmethod
    def _calcular_fase_ciclo(close_series):
        """Calcula fase do ciclo de mercado baseado em tendência de longo prazo"""
        sma_200 = close_series.rolling(window=200).mean()
        sma_50 = close_series.rolling(window=50).mean()
        
        # Fase: 1 = Expansão, 0.5 = Neutro, 0 = Contração
        fase = pd.Series(0.5, index=close_series.index)
        fase[close_series > sma_200] = 1.0  # Acima da média de longo prazo
        fase[(close_series < sma_200) & (sma_50 < sma_200)] = 0.0  # Tendência de baixa
        
        return fase
    
    @staticmethod
    def _simular_sentimento(index):
        """Simula score de sentimento (rolling average neutro com ruído)"""
        # Valor neutro (0.5) com pequena variação aleatória
        np.random.seed(42)
        sentiment = pd.Series(0.5 + np.random.normal(0, 0.1, len(index)), index=index)
        return sentiment.clip(0, 1).rolling(window=30).mean().fillna(0.5)
    
    @staticmethod
    def _simular_esg(index):
        """Simula score ESG normalizado (constante neutro)"""
        # Valor neutro constante para demonstração
        return pd.Series(0.5, index=index)

    @staticmethod
    def _normalizar(series, maior_melhor=True):
        """Normaliza série para score (0 a 1)"""
        if series.empty or series.std() == 0:
            return pd.Series(0.5, index=series.index) # Retorna neutro se sem variância
        
        # Trata infinitos e NaNs para evitar problemas na normalização
        series_clean = series.replace([np.inf, -np.inf], np.nan).fillna(series.median())
        
        # Min-Max Scaling (normalização)
        min_val = series_clean.min()
        max_val = series_clean.max()
        
        if max_val == min_val:
            return pd.Series(0.5, index=series.index)
        
        normalized = (series_clean - min_val) / (max_val - min_val)
        
        if not maior_melhor:
            return 1 - normalized
        return normalized


# =============================================================================
# CLASSE: COLETOR DE DADOS
# =============================================================================

class ColetorDados:
    """Coleta e processa dados de mercado com profundidade máxima"""
    
    def __init__(self, periodo=PERIODO_DADOS):
        self.periodo = periodo
        self.dados_por_ativo = {}
        self.dados_fundamentalistas = pd.DataFrame()
        self.dados_macro = {}
        self.metricas_performance = pd.DataFrame() # Initialize metric dataframe
        self.ativos_sucesso = []
    
    def coletar_dados_macroeconomicos(self):
        """Coleta dados macroeconômicos para features externas"""
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
                        print(f"  ✓ {nome}: {len(hist)} dias")
                    else:
                        print(f"  ⚠️ {nome}: Sem dados históricos")
                        self.dados_macro[nome] = pd.Series()
                except Exception as e:
                    print(f"  ⚠️ {nome}: Erro - {str(e)[:50]}")
                    self.dados_macro[nome] = pd.Series()
            

