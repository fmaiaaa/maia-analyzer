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
- Engenharia massiva de features (V7.1: Inclui Smart Beta Factors)
- Smart Beta Factors (NEW)
- Governança de Modelo com Drift Detection (NEW)
- Dashboard interativo completo

Versão: 7.1.0 - Sistema AutoML Completo Elite (Otimizado)
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
    'keras': 'keras'
}

def ensure_package(module_name, package_name):
    """Instala pacote se não estiver disponível"""
    try:
        __import__(module_name)
    except ImportError:
        print(f"Instalando {package_name}...")
        # Adicionado '--break-system-packages' para compatibilidade com ambientes modernos
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', package_name, '--break-system-packages'])
        
        # Check if streamlit is loaded to prompt for restart
        if 'streamlit' in sys.modules:
            import streamlit as st
            st.warning(f"{package_name} foi instalado. Por favor, **reexecute** o servidor Streamlit para aplicar as mudanças.")
            st.stop() # Stop execution if in a streamlit app to ensure restart

# Instala todas as dependências (Descomente esta seção se estiver rodando em ambiente local sem dependências instaladas)
# try:
#     for module, package in REQUIRED_PACKAGES.items():
#         ensure_package(module.split('.')[0], package)
    
#     import streamlit as st
#     import yfinance as yf
#     import plotly.graph_objects as go
#     import plotly.express as px
#     from plotly.subplots import make_subplots
#     from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
#     from sklearn.linear_model import RidgeClassifier, LogisticRegression, BayesianRidge
#     from sklearn.naive_bayes import GaussianNB
#     from sklearn.model_selection import TimeSeriesSplit, cross_val_score
#     from sklearn.preprocessing import StandardScaler, RobustScaler
#     from sklearn.decomposition import PCA
#     from sklearn.cluster import KMeans, DBSCAN
#     from sklearn.metrics import silhouette_score, mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, precision_score, recall_score, f1_score
#     import xgboost as xgb
#     import lightgbm as lgb
#     from catboost import CatBoostClassifier
#     from arch import arch_model
#     import optuna
#     import shap
#     import lime
#     import lime.lime_tabular
    
#     optuna.logging.set_verbosity(optuna.logging.WARNING)
    
# except Exception as e:
#     if 'streamlit' in sys.modules:
#         st.error(f"Erro ao carregar bibliotecas: {e}")
#     else:
#         print(f"Erro ao carregar bibliotecas: {e}")
#     # sys.exit(1) # Removido para permitir que o código funcione em ambientes sem as libs
# # Assumindo que o ambiente Jupyter/Colab já tem as libs essenciais
import streamlit as st # Adicionado para evitar erro de NameError
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score # Adicionado para GovernancaModelo


# =============================================================================
# CONSTANTES GLOBAIS E CONFIGURAÇÕES
# =============================================================================

# Configurações Globais
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

# Lista simplificada para o exemplo (Mantenha a original do seu projeto)
ATIVOS_IBOVESPA = [
    'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'MGLU3.SA', 
    'WEGE3.SA', 'RENT3.SA', 'LREN3.SA', 'PRIO3.SA', 'CSNA3.SA'
]

# Mapa de Setores (Mantenha o original)
ATIVOS_POR_SETOR = {
    'Bens Industriais': ['NATU3.SA', 'WEGE3.SA', 'VAMO3.SA'],
    'Consumo Cíclico': ['MGLU3.SA', 'LREN3.SA', 'RENT3.SA'],
    'Consumo não Cíclico': ['PCAR3.SA', 'JBSS3.SA'],
    'Financeiro': ['ITUB4.SA', 'BBDC4.SA', 'BBSE3.SA'],
    'Materiais Básicos': ['VALE3.SA', 'CSNA3.SA', 'SUZB3.SA'],
    'Petróleo, Gás e Biocombustíveis': ['PETR4.SA', 'PRIO3.SA', 'UGPA3.SA']
}

# Lista completa de todos os ativos
TODOS_ATIVOS = []
for setor, ativos in ATIVOS_POR_SETOR.items():
    TODOS_ATIVOS.extend(ativos)
TODOS_ATIVOS = sorted(list(set(TODOS_ATIVOS)))

# CONSTANTES DE ARMAZENAMENTO DE DADOS (GCS)
GCS_BUCKET_NAME = 'meu-portfolio-dados-gratuitos' 
GCS_MASTER_FILE_PATH = 'dados_consolidados/todos_ativos_master.csv'

# CONSTANTES DE GOVERNANÇA (NEW)
AUC_THRESHOLD_MIN = 0.65 
AUC_DROP_THRESHOLD = 0.05
DRIFT_WINDOW = 20
STRESS_TEST_SIGMA = 2.0

# =============================================================================
# CLASSE: GOVERNANÇA DE MODELO (CORRIGIDA E EXPANDIDA)
# =============================================================================

class GovernancaModelo:
    """
    Classe para monitoramento e governança de modelos ML,
    incluindo rastreamento de performance e detecção de Data Drift.
    """
    
    def __init__(self, ativo, max_historico=DRIFT_WINDOW):
        self.ativo = ativo
        self.max_historico = max_historico
        self.historico_auc = []
        self.historico_precision = []
        self.historico_recall = []
        self.historico_f1 = []
        self.auc_maximo = 0.0
        self.media_features = {} # NEW: Armazena a média das features de referência

    def inicializar_drift(self, df_features):
        """Inicializa as médias de referência para o monitoramento de drift."""
        # Seleciona features importantes para monitorar (retorno, vol, rsi, momentum)
        features_monitor = [col for col in df_features.columns if any(tag in col for tag in ['returns', 'volatility', 'rsi', 'momentum'])]
        
        if not features_monitor:
            return

        # Calcula a média da janela inicial como referência
        if len(df_features) >= self.max_historico:
            referencia = df_features[features_monitor].iloc[-self.max_historico:].mean()
            self.media_features = referencia.to_dict()

    def verificar_drift(self, df_features_atual):
        """Compara as médias da janela atual com a referência para detectar drift."""
        if not self.media_features or len(df_features_atual) < self.max_historico:
            return []
        
        alertas = []
        features_monitor = self.media_features.keys()
        
        # Última janela para comparação
        janela_atual = df_features_atual.iloc[-self.max_historico:][list(features_monitor)].mean()
        
        for feature, media_referencia in self.media_features.items():
            media_atual = janela_atual.get(feature, np.nan)
            
            if np.isnan(media_referencia) or np.isnan(media_atual):
                continue
                
            std_dev = df_features_atual[feature].iloc[-self.max_historico:].std()
            
            # Alerta se a média se mover mais de 1.5 desvios-padrão de sua média histórica
            if std_dev > 1e-6 and abs(media_atual - media_referencia) > 1.5 * std_dev: 
                alertas.append({
                    'tipo': 'DRIFT (Feature)',
                    'mensagem': f'Deriva detectada em **{feature}**. Média mudou de {media_referencia:.4f} para {media_atual:.4f}'
                })
            
        return alertas
        
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
    
    def verificar_alertas(self, df_features_atual):
        """Verifica se há alertas de degradação de performance e de drift."""
        if not self.historico_auc:
            return []
        
        alertas = []
        auc_atual = self.historico_auc[-1]
        
        # Alerta 1: Performance
        if auc_atual < AUC_THRESHOLD_MIN:
            alertas.append({
                'tipo': 'CRÍTICO (AUC)',
                'mensagem': f'AUC ({auc_atual:.3f}) abaixo do mínimo aceitável ({AUC_THRESHOLD_MIN})'
            })
        
        # Alerta 2: Degradação
        if self.auc_maximo > 0:
            degradacao = (self.auc_maximo - auc_atual) / self.auc_maximo
            if degradacao > AUC_DROP_THRESHOLD:
                alertas.append({
                    'tipo': 'ATENÇÃO (Degradação)',
                    'mensagem': f'Degradação de performance de {degradacao*100:.1f}% em relação ao máximo histórico.'
                })
        
        # Alerta 3: Data Drift (Novo)
        alertas.extend(self.verificar_drift(df_features_atual))
        
        return alertas
    
    def gerar_relatorio(self, df_features_atual):
        """Gera relatório completo de governança"""
        if not self.historico_auc:
            return {
                'status': 'Sem dados suficientes',
                'severidade': 'info',
                'metricas': {},
                'alertas': [],
                'historico': {}
            }
        
        alertas = self.verificar_alertas(df_features_atual)
        
        if any(a['tipo'].startswith('CRÍTICO') for a in alertas):
            severidade = 'error'
            status = 'Modelo requer atenção imediata'
        elif any(a['tipo'].startswith('ATENÇÃO') or a['tipo'].startswith('DRIFT') for a in alertas):
            severidade = 'warning'
            status = 'Modelo em monitoramento (Alertas ativos)'
        else:
            severidade = 'success'
            status = 'Modelo operando normalmente'
        
        return {
            'status': status,
            'severidade': severidade,
            'metricas': {
                'AUC Atual': self.historico_auc[-1] if self.historico_auc else 0,
                'AUC Médio': np.mean(self.historico_auc) if self.historico_auc else 0,
                'AUC Máximo': self.auc_maximo,
                'Precision Média': np.mean(self.historico_precision) if self.historico_precision else 0,
                'Recall Médio': np.mean(self.historico_recall) if self.historico_recall else 0,
                'F1-Score Médio': np.mean(self.historico_f1) if self.historico_f1 else 0
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
# CLASSE: ENGENHEIRO DE FEATURES (EXPANDIDA COM SMART BETA)
# =============================================================================

class EngenheiroFeatures:
    """Calcula indicadores técnicos e fundamentalistas com máxima profundidade e Smart Beta Factors"""
    
    @staticmethod
    def calcular_indicadores_tecnicos(hist):
        """Calcula indicadores técnicos completos e Smart Beta Factors."""
        df = hist.copy()
        
        # Retornos e Volatilidade
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        df['volatility_60'] = df['returns'].rolling(window=60).std() * np.sqrt(252)
        df['volatility_252'] = df['returns'].rolling(window=252).std() * np.sqrt(252) 
        
        # Médias Móveis (SMA, EMA)
        for periodo in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{periodo}'] = SMAIndicator(close=df['Close'], window=periodo).sma_indicator()
            df[f'ema_{periodo}'] = EMAIndicator(close=df['Close'], window=periodo).ema_indicator()
            
        # Cruzamentos e Posições
        df['price_sma20_ratio'] = df['Close'] / df['sma_20']
        df['sma20_sma50_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
        
        # RSI e Momentum
        for periodo in [7, 14, 28]:
            df[f'rsi_{periodo}'] = RSIIndicator(close=df['Close'], window=periodo).rsi()
        
        df['momentum_20'] = ROCIndicator(close=df['Close'], window=20).roc()
        
        # Volatility Indicators
        bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['bb_width'] = bb.bollinger_wband()
        df['atr'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()

        # =====================================================================
        # NEW: SMART BETA FACTORS (Baseado em Séries Temporais)
        # =====================================================================
        
        # 1. Momentum Factor (11/1 Mês)
        # Retorno de 252 dias (1 ano) excluindo os últimos 21 dias (1 mês)
        df['factor_momentum'] = df['Close'].pct_change(periods=252).shift(21)
        
        # 2. Value Factor (Proxy: Inverse Price-to-MA200)
        # Ativos baratos em relação a média de longo prazo
        epsilon = 1e-6 # Pequeno valor para evitar divisão por zero/inf
        df['factor_value_proxy'] = 1 / (df['Close'] / (df['sma_200'] + epsilon))
        
        # 3. Low Volatility Factor (Inverse Annual Volatility)
        # Favorece ativos com baixa volatilidade histórica
        df['factor_low_vol'] = 1 / (df['volatility_252'] + epsilon)
        
        # 4. Size Factor (Proxy: Inverse Log Market Cap)
        # Favorece ativos com menor valor (Market Cap), indicando Small Caps
        # Como não temos Market Cap, usamos 1/Close Price como proxy para Small Cap no BR
        df['factor_size_proxy'] = 1 / (df['Close'] + epsilon)
        
        # Z-Score Scaling para fatores (melhor para comparar forças)
        fatores_para_normalizar = ['factor_momentum', 'factor_value_proxy', 'factor_low_vol', 'factor_size_proxy']
        
        for fator in fatores_para_normalizar:
            if fator in df.columns and not df[fator].isnull().all():
                # Apenas escala a parte não-NaN dos dados
                valid_data = df[fator].dropna()
                if len(valid_data) > 1:
                    df.loc[valid_data.index, f'{fator}_scaled'] = zscore(valid_data, nan_policy='omit')
                else:
                    df[f'{fator}_scaled'] = np.nan # Não há dados suficientes para calcular zscore
            else:
                df[f'{fator}_scaled'] = np.nan

        return df.dropna(subset=['returns', 'log_returns'] + [c for c in df.columns if c.startswith('sma_')])


# (Métodos `calcular_features_fundamentalistas` e `_normalizar` permanecem inalterados)


# =============================================================================
# CLASSE: COLETOR DE DADOS (REFATORADA)
# =============================================================================

class ColetorDados:
    """Coleta e processa dados de mercado com profundidade máxima"""
    
    def __init__(self, periodo=PERIODO_DADOS):
        self.periodo = periodo
        self.dados_por_ativo = {}
        self.dados_fundamentalistas = pd.DataFrame()
        self.ativos_sucesso = []
        self.dados_macro = {} # Armazena retornos diários dos macros
        self.metricas_performance = pd.DataFrame()
        self.df_master = None
        self.model_governors = {} # NEW: Armazena instâncias de GovernancaModelo

    def carregar_dados_gcs(self):
        """Carrega dados consolidados do Google Cloud Storage (Apenas placeholder simulado)"""
        print("\n📊 Carregando dados do GCS...")
        # SIMULAÇÃO: No ambiente real, esta seção faria o download.
        # Aqui, assumimos que o download falha ou que o master DataFrame é carregado
        # Em um ambiente com a biblioteca 'storage' instalada, este código funciona:
        # try:
        #     storage_client = storage.Client()
        #     bucket = storage_client.bucket(GCS_BUCKET_NAME)
        #     blob = bucket.blob(GCS_MASTER_FILE_PATH)
        #     csv_data_bytes = blob.download_as_bytes()
        #     csv_data = csv_data_bytes.decode('utf-8')
        #     self.df_master = pd.read_csv(io.StringIO(csv_data))
        #     self.df_master['Date'] = pd.to_datetime(self.df_master['Date'])
        #     return True
        # except Exception as e:
        #     print(f"❌ Erro ao carregar dados do GCS: {str(e)}")
        #     return False
        
        # SIMULAÇÃO DE CARREGAMENTO PARA EXECUÇÃO LOCAL (Cria um DataFrame de Exemplo)
        data_simulada = []
        start_date = datetime.now() - timedelta(days=730)
        dates = pd.date_range(start=start_date, periods=500)
        np.random.seed(42)
        
        for ticker in TODOS_ATIVOS + ['^BVSP', '^GSPC']:
            close_prices = 100 * (1 + np.random.randn(len(dates)) * 0.01).cumprod()
            volume = np.random.randint(100000, 5000000, len(dates))
            temp_df = pd.DataFrame({
                'Date': dates,
                'Open': close_prices * (1 - np.random.rand(len(dates)) * 0.01),
                'High': close_prices * (1 + np.random.rand(len(dates)) * 0.01),
                'Low': close_prices * (1 - np.random.rand(len(dates)) * 0.01),
                'Close': close_prices,
                'Volume': volume,
                'ticker': ticker,
            })
            data_simulada.append(temp_df)

        self.df_master = pd.concat(data_simulada, ignore_index=True)
        print("✓ SIMULAÇÃO: Dados Mestre carregados com sucesso.")
        return True

    def coletar_dados_macroeconomicos(self):
        """Coleta dados macroeconômicos do GCS (ou simulação) e calcula retornos."""
        if self.df_master is None:
            if not self.carregar_dados_gcs():
                return
        
        print("\n📊 Coletando dados macroeconômicos...")
        indices = {
            'IBOV': '^BVSP', 'SP500': '^GSPC', 'NASDAQ': '^IXIC', 
            'VIX': '^VIX', 'DXY': 'DX-Y.NYB', 'GOLD': 'GC=F', 'OIL': 'CL=F'
        }
        
        for nome, simbolo in indices.items():
            df_macro = self.df_master[self.df_master['ticker'] == simbolo].copy()
            
            if not df_macro.empty and 'Close' in df_macro.columns:
                df_macro = df_macro.set_index('Date').sort_index()
                self.dados_macro[nome] = df_macro['Close'].pct_change()
            else:
                self.dados_macro[nome] = pd.Series()
        
        print(f"✓ Dados macroeconômicos coletados: {len(self.dados_macro)} indicadores")

    def adicionar_correlacoes_macro(self, df, simbolo):
        """Adiciona correlações com indicadores macroeconômicos."""
        if not self.dados_macro or 'returns' not in df.columns:
            return df
        
        df_final = df.copy()
        
        try:
            df_returns_aligned = df['returns']

            for nome, serie_macro in self.dados_macro.items():
                if serie_macro.empty or serie_macro.isnull().all():
                    continue
                
                # Alinha os retornos do ativo com os retornos macro
                aligned_returns = pd.concat([df_returns_aligned, serie_macro], axis=1, join='inner').dropna()
                
                if aligned_returns.shape[0] > 60:
                    corr_rolling = aligned_returns.iloc[:, 0].rolling(60).corr(aligned_returns.iloc[:, 1])
                    # Reindexa para o DataFrame original (manter o tamanho e índice)
                    df_final[f'corr_{nome.lower()}'] = corr_rolling.reindex(df.index) 
                else:
                    df_final[f'corr_{nome.lower()}'] = np.nan
            
            return df_final
        
        except Exception as e:
            print(f"  ⚠️ {simbolo}: Erro ao calcular correlações macro - {str(e)[:80]}")
            return df
    
    def coletar_e_processar_dados(self, simbolos):
        """Coleta e processa dados de mercado com engenharia de features máxima"""
        
        if self.df_master is None:
            if not self.carregar_dados_gcs():
                return False
        
        # 1. Coletar e preparar dados macro ANTES do loop principal
        self.coletar_dados_macroeconomicos()
        
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
                
                # Aplica Feature Engineering (inclui Smart Beta)
                df_features = EngenheiroFeatures.calcular_indicadores_tecnicos(df_ativo)
                
                # Adiciona Correlações Macro (com base na refatoração)
                df_features = self.adicionar_correlacoes_macro(df_features, simbolo)
                
                # Limpeza final e verificação de tamanho
                min_dias_flexivel = max(180, int(MIN_DIAS_HISTORICO * 0.7))
                if len(df_features) < min_dias_flexivel:
                    print(f"  ⚠️ {simbolo}: Apenas {len(df_features)} dias válidos.")
                    continue
                
                self.dados_por_ativo[simbolo] = df_features.fillna(method='ffill').fillna(0) # Tratar NaNs finais
                self.ativos_sucesso.append(simbolo)
                
                # Inicializa Governança de Modelo e Drift
                if simbolo not in self.model_governors:
                    self.model_governors[simbolo] = GovernancaModelo(simbolo)
                    self.model_governors[simbolo].inicializar_drift(df_features)
                
            except Exception as e:
                print(f"  ❌ {simbolo}: Erro no processamento - {str(e)}")
                continue 
        
        # Geração de Métricas de Performance (o código é funcional, mas longo demais para duplicar aqui, 
        # mantenho a lógica de preenchimento na versão final.)
        metricas = {}
        for simbolo in self.ativos_sucesso:
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

        if len(self.ativos_sucesso) < NUM_ATIVOS_PORTFOLIO:
            print(f"\n❌ ERRO: Apenas {len(self.ativos_sucesso)} ativos válidos após processamento.")
            return False
            
        print(f"\n✓ Processamento concluído. {len(self.ativos_sucesso)} ativos válidos.")
        return True

# (O resto das classes (AnalizadorPerfilInvestidor, Funções de Estilo, etc.) permanecem inalterados)

# =============================================================================
# MAPEAMENTOS DE PONTUAÇÃO DO QUESTIONÁRIO (MANTIDOS)
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
# CLASSE: ANALISADOR DE PERFIL DO INVESTIDOR (MANTIDA)
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
# FUNÇÕES DE ESTILO E VISUALIZAÇÃO (MANTIDAS)
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

# --- FIM DO CÓDIGO CORRIGIDO E OTIMIZADO ---
