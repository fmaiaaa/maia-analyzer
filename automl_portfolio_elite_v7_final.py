"""
=============================================================================
SISTEMA AUTOML AVANÇADO - OTIMIZAÇÃO DE PORTFÓLIO FINANCEIRO
=============================================================================

Sistema completo de otimização de portfólio com:
- Questionário de perfil de investidor
- Seleção de ativos por setor
- Ensemble de modelos ML (Lê Resultados FINAIS do ETL do GCS)
- Modelagem de volatilidade GARCH (Lê a previsão do GCS)
- Otimização de hiperparâmetros com Optuna (Removido o cálculo)
- Engenharia massiva de features (Lê features prontas do GCS)
- Smart Beta Factors
- Dashboard interativo completo

Versão: 6.0.0 - Modo GCS-ETL (Lê dados em 4 CSVs por ativo)
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
import json # Necessário para ler metadados ML

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
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, CCIIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel, DonchianChannel
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, MFIIndicator, VolumeWeightedAveragePrice

# --- 5. MACHINE LEARNING (SCIKIT-LEARN) ---
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
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
# import shap, lime # Removidas

# --- 9. DEEP LEARNING (TENSORFLOW/KERAS) ---
# Removidas para evitar dependências pesadas
# import tensorflow as tf 
# from tensorflow import keras

# Conectividade GCS
from google.cloud import storage 
import gcsfs # Adicionado para garantir a leitura direta

# --- 10. CONFIGURATION ---
try:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except:
    pass 
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURAÇÕES GLOBAIS
# =============================================================================

PERIODO_DADOS = '5y'
MIN_DIAS_HISTORICO = 252 
NUM_ATIVOS_PORTFOLIO = 5
TAXA_LIVRE_RISCO = 0.1075

# 🎯 Constantes do ETL (Nomes de arquivos exportados)
GCS_BUCKET_NAME = 'meu-portfolio-dados-gratuitos'
GCS_FOLDER_PATH = 'dados_financeiros_etl'
GCS_DESTINATION_FOLDER = 'dados_financeiros_etl' 
CSV_TECHNICAL = '{ticker}_tecnicos.csv'
CSV_FUNDAMENTAL = '{ticker}_fundamentos.csv'
CSV_ML_RESULTS = '{ticker}_ml_results.csv'
CSV_ML_METADATA = '{ticker}_ml_metadata.csv'

# Colunas de metadados do ETL
GCS_METADATA_COLS = ['sharpe_ratio', 'annual_return', 'annual_volatility', 'max_drawdown', 'sector', 'industry', 'garch_volatility']

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
# 4. LISTAS DE ATIVOS E SETORES
# =============================================================================
# Mantidas as listas originais para compatibilidade com o questionário
ATIVOS_IBOVESPA = [
    'ALOS3.SA', 'ABEV3.SA', 'ASAI3.SA', 'AESB3.SA', 'AZZA3.SA', 'B3SA3.SA',
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
    'Bens Industriais': ['NATU3.SA', 'ASAI3.SA', 'JSLG3.SA', 'CMIN3.SA', 'VAMO3.SA'],
    'Consumo Cíclico': ['AZZA3.SA', 'ALOS3.SA', 'CEAB3.SA', 'DIRR3.SA', 'CYRE3.SA', 'CVCB3.SA', 'LREN3.SA', 'MGLU3.SA', 'MRVE3.SA', 'RENT3.SA', 'YDUQ3.SA', 'COGN3.SA'],
    'Consumo não Cíclico': ['BEEF3.SA', 'NATU3.SA', 'PCAR3.SA', 'VIVA3.SA'],
    'Financeiro': ['BBSE3.SA', 'BBDC3.SA', 'BBDC4.SA', 'BBAS3.SA', 'BPAC11.SA', 'CXSE3.SA', 'HYPE3.SA', 'IGTI11.SA', 'IRBR3.SA', 'ITSA4.SA', 'ITUB4.SA', 'MULT3.SA', 'PSSA3.SA', 'RADL3.SA', 'RDOR3.SA', 'SANB11.SA'],
    'Materiais Básicos': ['BRAP4.SA', 'BRKM5.SA', 'CSNA3.SA', 'GGBR4.SA', 'GOAU4.SA', 'KLBN11.SA', 'SLCE3.SA', 'SUZB3.SA', 'USIM5.SA', 'VALE3.SA'],
    'Petróleo, Gás e Biocombustíveis': ['ENEV3.SA', 'PRIO3.SA', 'PETR3.SA', 'PETR4.SA', 'RAIZ4.SA', 'RECV3.SA', 'UGPA3.SA', 'VBBR3.SA'],
    'Saúde': ['FLRY3.SA', 'HYPE3.SA', 'RADL3.SA', 'RDOR3.SA'],
    'Tecnologia da Informação': ['TOTS3.SA', 'WEGE3.SA'],
    'Telecomunicações': ['TIMS3.SA', 'VIVT3.SA'],
    'Utilidade Pública': ['AESB3.SA', 'BRAV3.SA', 'CMIG4.SA', 'CPLE6.SA', 'CPFE3.SA', 'EGIE3.SA', 'ELET3.SA', 'ELET6.SA', 'ENGI11.SA', 'EQTL3.SA', 'ISAE4.SA', 'RAIL3.SA', 'SBSP3.SA', 'TAEE11.SA']
}

TODOS_ATIVOS = sorted(list(set([ativo for ativos in ATIVOS_POR_SETOR.values() for ativo in ativos])))


# =============================================================================
# 6. MAPEAMENTOS DE PONTUAÇÃO DO QUESTIONÁRIO (Replicados para auto-contenção)
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
# 2. CLASSE: ANALISADOR DE PERFIL DO INVESTIDOR (Adaptado para ETL Lookbacks)
# =============================================================================

class AnalisadorPerfilInvestidor:
    """
    Analisa perfil de risco e horizonte temporal do investidor com base 
    em um questionário padronizado.
    """
    
    def __init__(self):
        self.nivel_risco = ""
        self.horizonte_tempo = ""
        self.dias_lookback_ml = 126 
    
    def determinar_nivel_risco(self, pontuacao: int) -> str:
        """Traduz a pontuação total do questionário em perfil de risco."""
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
        com base nas respostas de liquidez e objetivo, mapeando para os prazos do ETL.
        """
        # Mapeia as chaves de resposta para os prazos do ETL (Curto: 126, Médio: 252, Longo: 504)
        time_map = {
            'A': 126,  
            'B': 252, 
            'C': 504  
        }
        
        final_lookback = max(
            time_map.get(liquidez_key, 126),
            time_map.get(objetivo_key, 126)
        )
        
        if final_lookback >= 504:
            self.horizonte_tempo = "LONGO PRAZO"
            self.dias_lookback_ml = 504
        elif final_lookback >= 252:
            self.horizonte_tempo = "MÉDIO PRAZO"
            self.dias_lookback_ml = 252
        else:
            self.horizonte_tempo = "CURTO PRAZO"
            self.dias_lookback_ml = 126
        
        return self.horizonte_tempo, self.dias_lookback_ml
    
    def calcular_perfil(self, respostas_risco: dict) -> tuple[str, str, int, int]:
        """Calcula o perfil completo do investidor."""
        
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
# 3. FUNÇÕES DE ESTILO E VISUALIZAÇÃO (INALTERADAS)
# =============================================================================

def obter_template_grafico() -> dict:
    """Retorna um template de layout otimizado para gráficos Plotly com estilo Times New Roman."""
    return {
        'plot_bgcolor': 'white', 'paper_bgcolor': 'white',
        'font': {'family': 'Times New Roman, serif', 'size': 12, 'color': 'black'},
        'title': {'font': {'family': 'Times New Roman, serif', 'size': 16, 'color': '#2c3e50', 'weight': 'bold'}, 'x': 0.5, 'xanchor': 'center'},
        'xaxis': {'showgrid': True, 'gridcolor': 'lightgray', 'showline': True, 'linecolor': 'black', 'linewidth': 1, 'tickfont': {'family': 'Times New Roman, serif', 'color': 'black'}, 'title': {'font': {'family': 'Times New Roman, serif', 'color': 'black'}}, 'zeroline': False},
        'yaxis': {'showgrid': True, 'gridcolor': 'lightgray', 'showline': True, 'linecolor': 'black', 'linewidth': 1, 'tickfont': {'family': 'Times New Roman, serif', 'color': 'black'}, 'title': {'font': {'family': 'Times New Roman, serif', 'color': 'black'}}, 'zeroline': False},
        'legend': {'font': {'family': 'Times New Roman, serif', 'color': 'black'}, 'bgcolor': 'rgba(255, 255, 255, 0.8)', 'bordercolor': 'lightgray', 'borderwidth': 1},
        'colorway': ['#2c3e50', '#7f8c8d', '#3498db', '#e74c3c', '#27ae60']
    }

# =============================================================================
# CLASSE: ENGENHEIRO DE FEATURES (ADAPTADA)
# =============================================================================

class EngenheiroFeatures:
    """Apenas provê utilitários e normalização, o cálculo é do ETL."""
    
    @staticmethod
    def calcular_indicadores_tecnicos(hist: pd.DataFrame) -> pd.DataFrame:
        """Retorna o DataFrame lido do GCS com imputação mínima de NaNs."""
        df = hist.copy()
        df_imputed = df.fillna(method='ffill').fillna(0.0) 
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
             return df.dropna(how='all') 
        
        return df_imputed.dropna(subset=['Close', 'Volume']) 

    @staticmethod
    def calcular_features_fundamentalistas(info: dict) -> dict:
        """Função Dummy, os dados são lidos do GCS."""
        return {}
    
    @staticmethod
    def _normalizar(serie: pd.Series, maior_melhor: bool = True) -> pd.Series:
        """Normaliza uma série de indicadores para o range [0, 1] (Min-Max Scaling)."""
        if serie.isnull().all():
            return pd.Series(0.5, index=serie.index) 
        
        min_val = serie.min()
        max_val = serie.max()
        
        if max_val == min_val:
            return pd.Series(0.5, index=serie.index)
        
        if maior_melhor:
            return (serie - min_val) / (max_val - min_val)
        else:
            return (max_val - serie) / (max_val - min_val)

# =============================================================================
# FUNÇÃO AUXILIAR: COLETA DE DADOS INDIVIDUAIS DO GCS (LER OS 4 ARQUIVOS)
# =============================================================================

def carregar_dados_ativo_gcs_csv(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    """
    CHAVE: Carrega os 4 arquivos CSV (Técnicos, Fundamentos, ML Results, ML Metadata) 
    para um único ativo e consolida.
    Retorna (df_tecnicos, df_fundamentos_df, ml_results_dict, df_ml_metadata).
    """
    
    file_map = {
        'tecnicos': CSV_TECHNICAL.format(ticker=ticker),
        'fundamentos': CSV_FUNDAMENTAL.format(ticker=ticker),
        'ml_results': CSV_ML_RESULTS.format(ticker=ticker),
        'ml_metadata': CSV_ML_METADATA.format(ticker=ticker),
    }
    
    dfs = {}
    
    for key, filename in file_map.items():
        uri = f"gs://{GCS_BUCKET_NAME}/{GCS_FOLDER_PATH}/{filename}"
        try:
            if key == 'tecnicos':
                 df = pd.read_csv(uri, index_col='Date', parse_dates=True)
                 if df.index.tz is not None: df.index = df.index.tz_localize(None) 
            elif key == 'fundamentos' or key == 'ml_results':
                 df = pd.read_csv(uri, index_col='ticker')
            elif key == 'ml_metadata':
                 df = pd.read_csv(uri, index_col=['ticker', 'target_name'])
                 
            dfs[key] = df
                 
        except Exception as e:
            # print(f"❌ Erro ao carregar {key} para {ticker}: {e}")
            return pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame() 

    # 2. Consolidação da saída
    
    df_tecnicos = dfs.get('tecnicos', pd.DataFrame())
    df_fundamentos_df = dfs.get('fundamentos', pd.DataFrame())
    
    if not df_fundamentos_df.empty:
         # O dataframe de fundamentos tem apenas uma linha, mas mantemos como DF para consistência
         pass
    
    ml_results_dict = dfs.get('ml_results', pd.DataFrame()).iloc[0].to_dict() if not dfs.get('ml_results').empty else {}
    df_ml_metadata = dfs.get('ml_metadata', pd.DataFrame())
    
    return df_tecnicos, df_fundamentos_df, ml_results_dict, df_ml_metadata
    
# =============================================================================
# CLASSE: COLETOR DE DADOS GCS (Lê 4 CSVs)
# =============================================================================

class ColetorDadosGCS(object):
    """Coleta dados de mercado lendo os 4 arquivos CSV do ETL no GCS."""
    
    cols_performance_and_meta = GCS_METADATA_COLS + ['ticker'] 
    
    def __init__(self, periodo=PERIODO_DADOS):
        self.periodo = periodo
        self.dados_por_ativo = {} 
        self.dados_fundamentalistas = pd.DataFrame() 
        self.ml_data = {} 
        self.ativos_sucesso = []
        self.metricas_performance = pd.DataFrame()
        self.volatilidades_garch_raw = {}
        self.ml_metadata_raw = {} 

    def coletar_e_processar_dados(self, simbolos: list) -> bool:
        """Carrega os 4 conjuntos de dados para cada ativo do GCS e reestrutura."""
        
        print(f"\n{'='*60}")
        print(f"INICIANDO COLETA E PROCESSAMENTO - LEITURA GCS (4 CSVs por Ativo)")
        print(f"Total de ativos solicitados: {len(simbolos)}")
        print(f"{'='*60}\n")
        
        self.ativos_sucesso = []
        lista_fundamentalistas = []
        metricas_ml = {}
        metricas_perf = {}
        garch_vols = {}

        for simbolo in tqdm(simbolos, desc="📥 Carregando ativos do GCS (4x CSV)"):
            
            df_tech, df_fund_df, ml_results_dict, df_meta = carregar_dados_ativo_gcs_csv(simbolo)
            
            if df_tech.empty or df_fund_df.empty or 'Close' not in df_tech.columns:
                continue

            MIN_DIAS_HISTORICO_FLEXIVEL = max(180, int(MIN_DIAS_HISTORICO * 0.7))
            if len(df_tech) < MIN_DIAS_HISTORICO_FLEXIVEL:
                continue
                
            # --- 1. ARMAZENAMENTO DO HISTÓRICO (Features Temporais e Técnicas) ---
            self.dados_por_ativo[simbolo] = df_tech
            self.ativos_sucesso.append(simbolo)
            
            # --- 2. EXTRAÇÃO DOS FUNDAMENTOS E MÉTRICAS ---
            
            fund_dict = df_fund_df.iloc[0].to_dict() 
            
            # B. Métricas ML (Proba, AUC)
            metricas_ml[simbolo] = ml_results_dict
            
            # C. Métricas de Performance e GARCH (Lidas dos fundamentos)
            metricas_perf[simbolo] = {
                'retorno_anual': fund_dict.get('annual_return', np.nan),
                'volatilidade_anual': fund_dict.get('annual_volatility', np.nan),
                'sharpe': fund_dict.get('sharpe_ratio', np.nan),
                'max_drawdown': fund_dict.get('max_drawdown', np.nan)
            }
            garch_vols[simbolo] = fund_dict.get('garch_volatility', np.nan)

            # Adiciona Ticker e Fundamentos para a lista de Fundamentos
            fund_dict['Ticker'] = simbolo
            lista_fundamentalistas.append(fund_dict)
            
            self.ml_metadata_raw[simbolo] = df_meta
            
        # --- 3. FINALIZAÇÃO ---
        
        if len(self.ativos_sucesso) < NUM_ATIVOS_PORTFOLIO:
            print(f"\n❌ ERRO: Apenas {len(self.ativos_sucesso)} ativos válidos encontrados no GCS. Necessário: {NUM_ATIVOS_PORTFOLIO}.")
            return False
            
        self.dados_fundamentalistas = pd.DataFrame(lista_fundamentalistas).set_index('Ticker')
        self.metricas_performance = pd.DataFrame(metricas_perf).T
        self.volatilidades_garch_raw = garch_vols 
        self.ml_data = metricas_ml
        
        return True

# =============================================================================
# CLASSES: GARCH, ESTATÍSTICOS, ML (REMOVIDAS OU SIMPLIFICADAS)
# =============================================================================

class VolatilidadeGARCH:
     @staticmethod
     def ajustar_garch(returns: pd.Series, tipo_modelo: str = 'GARCH') -> float:
         return np.nan 

class ModelosEstatisticos:
     @staticmethod
     def ensemble_estatistico(series: pd.Series, horizon: int = 1) -> dict:
         return {'ensemble_forecast': np.nan, 'individual_forecasts': {}} 

class EnsembleML:
     @staticmethod
     def _otimizar_lightgbm(X, y):
         return {} 

# =============================================================================
# CLASSE: OTIMIZADOR DE PORTFÓLIO (INALTERADA)
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
            print("  ⚠️ Volatilidades GARCH não disponíveis, usando covariância histórica.")
        
        self.num_ativos = len(returns_df.columns)
        self.fundamental_data = fundamental_data
        self.ml_predictions = ml_predictions

    def _construir_matriz_cov_garch(self, returns_df: pd.DataFrame, garch_vols: dict) -> pd.DataFrame:
        """Constrói matriz de covariância usando correlações históricas e volatilidades GARCH."""
        
        corr_matrix = returns_df.corr()
        
        vol_array = np.array([
            garch_vols.get(ativo, returns_df[ativo].std() * np.sqrt(252))
            for ativo in returns_df.columns
        ])
        
        if np.isnan(vol_array).all() or np.all(vol_array <= 1e-9):
            print("  ⚠️ Volatilidades GARCH inválidas. Voltando para covariância histórica.")
            return returns_df.cov() * 252
            
        cov_matrix = corr_matrix.values * np.outer(vol_array, vol_array)
        
        return pd.DataFrame(cov_matrix, index=returns_df.columns, columns=returns_df.columns)
    
    def estatisticas_portfolio(self, pesos: np.ndarray) -> tuple[float, float]:
        """Calcula Retorno (anualizado) e Volatilidade (anualizada) do portfólio."""
        p_retorno = np.dot(pesos, self.mean_returns)
        p_vol = np.sqrt(np.dot(pesos.T, np.dot(self.cov_matrix, pesos)))
        return p_retorno, p_vol
    
    def sharpe_negativo(self, pesos: np.ndarray) -> float:
        """Objetivo: Minimizar o Sharpe Ratio Negativo (Equivale a Maximizar o Sharpe Ratio)."""
        p_retorno, p_vol = self.estatisticas_portfolio(pesos)
        
        if p_vol <= 1e-9: 
            return -100.0 
            
        return -(p_retorno - TAXA_LIVRE_RISCO) / p_vol
    
    def minimizar_volatilidade(self, pesos: np.ndarray) -> float:
        """Objetivo: Minimizar a Volatilidade do Portfólio."""
        return self.estatisticas_portfolio(pesos)[1]
    
    def calcular_cvar(self, pesos: np.ndarray, confidence: float = 0.95) -> float:
        """Calcula Conditional Value at Risk (CVaR) diário, NÃO anualizado."""
        portfolio_returns = self.returns @ pesos
        sorted_returns = np.sort(portfolio_returns)
        
        var_index = int(np.floor((1 - confidence) * len(sorted_returns)))
        var = sorted_returns[var_index]
        
        cvar = sorted_returns[sorted_returns <= var].mean()
        return cvar

    def cvar_negativo(self, pesos: np.ndarray, confidence: float = 0.95) -> float:
        """Objetivo: Minimizar o CVaR Negativo (Equivale a Minimizar o CVaR Positivo)."""
        return -self.calcular_cvar(pesos, confidence)

    def otimizar(self, estrategia: str = 'MaxSharpe', confidence_level: float = 0.95) -> dict:
        """
        Executa otimização do portfólio para a estratégia selecionada.
        """
        if self.num_ativos == 0:
            return {}

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
                print(f"  ✗ Otimização falhou ({estrategia}): {resultado.message}")
                return {ativo: 1.0 / self.num_ativos for ativo in self.returns.columns}
        
        except Exception as e:
            print(f"  ✗ Erro fatal na otimização ({estrategia}): {str(e)}")
            return {ativo: 1.0 / self.num_ativos for ativo in self.returns.columns}
    
# =============================================================================
# CLASSE PRINCIPAL: CONSTRUTOR DE PORTFÓLIO AUTOML (VERSÃO GCS-ETL)
# =============================================================================

class ConstrutorPortfolioAutoML:
    """
    Orquestrador principal para construção de portfólio AutoML
    Coordena coleta (GCS), Modelagem (Lê resultados do GCS) e otimização.
    """
    
    def __init__(self, valor_investimento: float, periodo: str = PERIODO_DADOS):
        self.valor_investimento = valor_investimento
        self.periodo = periodo
        
        self.dados_por_ativo = {}
        self.dados_fundamentalistas = pd.DataFrame()
        self.dados_performance = pd.DataFrame()
        self.volatilidades_garch = {}
        self.ml_data = {} # Novo: Dados ML lidos do GCS
        self.ativos_sucesso = []
        self.metricas_performance = pd.DataFrame()
        self.ml_metadata_raw = {} # Novo: Metadados ML brutos
        
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
        """[MODIFICADO] Coleta e processa dados de mercado (VIA GCS/4 CSVs)."""
        
        coletor = ColetorDadosGCS(periodo=self.periodo) 
        
        if not coletor.coletar_e_processar_dados(simbolos): return False
        
        self.dados_por_ativo = coletor.dados_por_ativo
        self.dados_fundamentalistas = coletor.dados_fundamentalistas
        self.ativos_sucesso = coletor.ativos_sucesso
        self.dados_performance = coletor.metricas_performance
        self.volatilidades_garch = coletor.volatilidades_garch_raw 
        self.ml_data = coletor.ml_data 
        self.ml_metadata_raw = coletor.ml_metadata_raw
        
        print(f"\n✓ Coleta (GCS/4 CSVs) concluída: {len(self.ativos_sucesso)} ativos válidos\n")
        return True

    def calcular_volatilidades_garch(self):
        """[MODIFICADO] Apenas valida se as volatilidades GARCH foram carregadas do GCS."""
        valid_vols = len([k for k, v in self.volatilidades_garch.items() if not np.isnan(v)])
        print(f"✓ Volatilidades GARCH/Histórica carregadas para {valid_vols} ativos (Leitura GCS)")
        
    def ler_modelos_ensemble(self, dias_lookback_ml: int = 126):
        """
        🎯 NOVO: Lê os resultados de ML (Proba e AUC) do GCS, baseados no horizonte 
        de tempo (dias_lookback_ml) definido pelo perfil.
        """
        
        ml_proba_col = f'ml_proba_{dias_lookback_ml}d'
        
        self.predicoes_ml = {}
        self.predicoes_estatisticas = {} 
        
        print(f"\n🧠 Lendo Previsões ML (Horizonte: {dias_lookback_ml} dias)...")

        for simbolo in self.ativos_sucesso:
            
            # --- 1. LÊ PROBABILIDADE FINAL ---
            proba_final = self.ml_data.get(simbolo, {}).get(ml_proba_col, np.nan)

            # --- 2. LÊ CONFIANÇA (AUC do LGBM - do arquivo _ml_metadata.csv) ---
            auc_score = np.nan
            if simbolo in self.ml_metadata_raw:
                target_name = [k for k, v in {'curto_prazo': 126, 'medio_prazo': 252, 'longo_prazo': 504}.items() if v == dias_lookback_ml]
                
                if target_name and target_name[0] in self.ml_metadata_raw[simbolo].index.get_level_values('target_name'):
                    try:
                        metadata_row = self.ml_metadata_raw[simbolo].loc[simbolo, target_name[0]]
                        models_data = json.loads(metadata_row['models_data'])
                        # Usamos o AUC médio do LGBM no WFCV como proxy para Confiança
                        auc_score = models_data.get('lgbm', {}).get('wfc_auc_mean', np.nan) 
                    except Exception:
                        auc_score = np.nan
            
            # Armazenamento da Previsão ML
            self.predicoes_ml[simbolo] = {
                'predicted_proba_up': proba_final if not np.isnan(proba_final) else 0.5,
                'auc_roc_score': auc_score if not np.isnan(auc_score) else 0.5,
                'model_name': 'Ensemble GCS'
            }
            
            # --- 3. DUMMY para Estatísticos (Previsão de preço futuro removida, pois não está no ETL) ---
            self.predicoes_estatisticas[simbolo] = {'forecast': np.nan, 'predicted_direction': np.nan}


        print(f"✓ Resultados ML lidos para {len(self.predicoes_ml)} ativos")

    # 🎯 REMOVIDO: O método treinar_modelos_ensemble foi substituído por ler_modelos_ensemble.
    # O restante do código (pontuar_e_selecionar_ativos, otimizar_alocacao, etc.)
    # usa as variáveis agora preenchidas pelo GCS (ml_data, volatilidades_garch).
    
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
            # 🎯 FEATURE TÉCNICA (Lê do df_tech cacheado)
            if asset in self.dados_por_ativo and 'rsi_14' in self.dados_por_ativo[asset].columns:
                df = self.dados_por_ativo[asset]
                combinado.loc[asset, 'rsi_current'] = df['rsi_14'].iloc[-1]
                combinado.loc[asset, 'macd_current'] = df['macd'].iloc[-1]
                combinado.loc[asset, 'bb_position_current'] = df['bb_position'].iloc[-1]

            # 🎯 RESULTADOS ML (Lê do novo cache self.predicoes_ml)
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

        # 🎯 Volatilidade: Usa o GARCH lido do GCS, com fallback para Histórico
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
            fund = self.dados_fundamentalistas.loc[simbolo] if simbolo in self.dados_fundamentalistas.index else {}
            
            # Metricas de Performance/Volatilidade
            justification.append(f"Perf: Sharpe {perf.get('sharpe', np.nan):.3f}, Retorno {perf.get('retorno_anual', np.nan)*100:.2f}%, Vol. {self.volatilidades_garch.get(simbolo, perf.get('volatilidade_anual', np.nan))*100:.2f}% (GARCH/Hist.)")
            
            # Fundamentos
            justification.append(f"Fund: P/L {fund.get('pe_ratio', np.nan):.2f}, ROE {fund.get('roe', np.nan):.2f}%")
            
            # Resultados ML
            if simbolo in self.predicoes_ml:
                ml = self.predicoes_ml[simbolo]
                proba_up = ml.get('predicted_proba_up', 0.5)
                auc_score = ml.get('auc_roc_score', np.nan)
                auc_str = f"{auc_score:.3f}" if not pd.isna(auc_score) else "N/A"
                justification.append(f"ML: Prob. Alta {proba_up*100:.1f}% (AUC {auc_str})")
            
            # Estatístico (Dummy: Apenas indica N/A se for usado)
            if simbolo in self.predicoes_estatisticas:
                stat_pred = self.predicoes_estatisticas[simbolo]
                if stat_pred.get('forecast') is not None and not np.isnan(stat_pred.get('forecast')):
                    justification.append("Estatístico: Previsão disponível (Via GCS)")
                else:
                    justification.append("Estatístico: Previsão indisponível (Removido do ETL)")
                
            self.justificativas_selecao[simbolo] = " | ".join(justification)
        
        return self.justificativas_selecao
        
    def executar_pipeline(self, simbolos_customizados: list, perfil_inputs: dict, otimizar_ml: bool = False) -> bool:
        """Executa pipeline completo: Coleta -> Leitura ML -> Pontuação -> Otimização."""
        
        self.perfil_dashboard = perfil_inputs
        ml_lookback_days = perfil_inputs.get('ml_lookback_days', 126)
        nivel_risco = perfil_inputs.get('risk_level', 'MODERADO')
        horizonte_tempo = perfil_inputs.get('time_horizon', 'MÉDIO PRAZO')
        
        # 1. Coleta de dados (LÊ TODOS OS CSVs DO GCS)
        if not self.coletar_e_processar_dados(simbolos_customizados): return False
        
        # 2. Volatilidades GARCH (Apenas valida)
        self.calcular_volatilidades_garch()
        
        # 3. Leitura dos Resultados ML (Substitui o Treinamento)
        self.ler_modelos_ensemble(dias_lookback_ml=ml_lookback_days)
        
        # 4. Pontuação e seleção
        self.pontuar_e_selecionar_ativos(horizonte_tempo=horizonte_tempo)
        
        # 5. Otimização de alocação
        self.otimizar_alocacao(nivel_risco=nivel_risco)
        
        # 6. Métricas e Justificativas
        self.calcular_metricas_portfolio()
        self.gerar_justificativas()
        
        print("\n✅ Pipeline de Otimização AutoML Concluído!")
        
        return True

class AnalisadorIndividualAtivos:
    """Análise completa de ativos individuais com máximo de features"""
    
    @staticmethod
    def calcular_todos_indicadores_tecnicos(hist: pd.DataFrame) -> pd.DataFrame:
        """Retorna o histórico lido do GCS (já com os indicadores)."""
        df = hist.copy()
        
        if 'Open' not in df.columns or 'High' not in df.columns or 'Low' not in df.columns or 'Close' not in df.columns or 'Volume' not in df.columns:
            df_reconstructed = df.filter(regex='^Open$|^High$|^Low$|^Close$|^Volume$').copy()
            return df_reconstructed
        
        return df

    @staticmethod
    def calcular_features_fundamentalistas_expandidas(info: dict) -> dict:
        """Função Dummy."""
        return {} 
    
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
            print(f"  ⚠️ Dados insuficientes para clustering: {len(features_numericas)} pontos.")
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
# INTERFACE STREAMLIT - FUNÇÕES DE ABA (AJUSTADAS PARA LEITURA GCS)
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
        .stMetric { 
            padding: 10px 15px;
            background-color: #ffffff;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
            margin-bottom: 10px;
        }
        .stMetric delta { font-weight: bold; color: #28a745; } 
        .stMetric delta[style*="color: red"] { color: #dc3545 !important; } 
        .info-box {
            background-color: #f8f9fa;
            border-left: 4px solid #2c3e50;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
        }
        .stTabs [data-baseweb="tab-list"] { gap: 20px; }
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
        .stTabs [aria-selected="true"] span { font-weight: bold; }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            color: #2c3e50; 
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

def aba_introducao():
    """Aba 1: Introdução e Metodologia"""
    
    st.markdown("## 📚 Bem-vindo ao Sistema AutoML de Otimização de Portfólio")
    
    st.markdown("""
    <div class="info-box">
    <h3>🎯 O que este sistema faz?</h3>
    <p>Este é um sistema avançado de construção e otimização de portfólios que utiliza 
    <strong>Machine Learning</strong>, <strong>modelagem estatística</strong> e <strong>teoria moderna de portfólio</strong> 
    para criar carteiras personalizadas.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🤖 Tecnologias Utilizadas (Modo GCS-ETL)")
    st.markdown("""
    Este sistema opera lendo <strong>4 arquivos CSV pré-calculados</strong> por ativo do Google Cloud Storage (GCS). O treinamento de modelos é feito off-line (ETL) para garantir velocidade e consistência.
    
    * **Coleta:** Leitura direta dos arquivos `_tecnicos.csv`, `_fundamentos.csv`, `_ml_results.csv`, `_ml_metadata.csv`.
    * **Machine Learning:** Utiliza a <strong>Probabilidade Final do Ensemble</strong> e o <strong>AUC do LGBM (WFCV)</strong> lidos do GCS.
    * **GARCH:** A volatilidade GARCH ajustada é lida do arquivo `_fundamentos.csv`.
    * **Seleção:** Score Multi-Fator Ponderado por perfil.
    * **Otimização:** Maximização de Sharpe / Minimização de CVaR usando a matriz de covariância baseada em GARCH.
    """)
    
    st.markdown("---")
    
    st.markdown("### ⚖️ Ponderação Adaptativa por Perfil")
    
    perfil_table = pd.DataFrame({
        'Perfil': ['Conservador', 'Intermediário', 'Moderado', 'Moderado-Arrojado', 'Avançado'],
        'Horizonte': ['Longo Prazo', 'Longo Prazo', 'Médio Prazo', 'Curto Prazo', 'Curto Prazo'],
        'Performance': ['40%', '40%', '40%', '40%', '40%'],
        'Fundamentos': ['50%', '40%', '30%', '20%', '10%'],
        'Técnicos': ['10%', '20%', '30%', '40%', '50%'],
        'ML': ['30%', '30%', '30%', '30%', '30%'], 
        'Foco': ['Qualidade e Estabilidade', 'Equilíbrio com Foco em Fundamentos', 'Equilíbrio Geral', 'Momentum e Curto Prazo', 'Visão de Curto Prazo e Momentum']
    })
    
    st.table(perfil_table)
    
    st.markdown("---")
    
    st.info("""
    **Próximos Passos:**
    1. **Seleção de Ativos**: Escolha os ativos que foram processados pelo seu ETL.
    2. **Construtor de Portfólio**: Responda o questionário para ler os resultados do ML do GCS e gerar seu portfólio.
    """)

def aba_selecao_ativos():
    """Aba 2: Seleção de Ativos - Enhanced with 4 selection modes (Inalterada)"""
    
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
        
        ativos_com_setor = {}
        for setor, ativos in ATIVOS_POR_SETOR.items():
            for ativo in ativos:
                ativos_com_setor[ativo] = setor
        
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
        
        novos_ativos = []
        if novos_ativos_input.strip():
            linhas = novos_ativos_input.strip().split('\n')
            for linha in linhas:
                ticker = linha.strip().upper()
                if ticker:
                    if not ticker.endswith('.SA'):
                        ticker = f"{ticker}.SA"
                    novos_ativos.append(ticker)
        
        ativos_selecionados = list(set(ativos_da_lista + novos_ativos))
        
        if ativos_selecionados:
            st.success(f"✓ **{len(ativos_selecionados)} ativos** selecionados")
            
            with st.expander("📋 Ver ativos selecionados"):
                df_selecionados = pd.DataFrame({
                    'Ticker': [a.replace('.SA', '') for a in ativos_selecionados],
                    'Código Completo': ativos_selecionados,
                    'Setor': [ativos_com_setor.get(a, setor_customizado or setor_novos_ativos or 'Não especificado') 
                             for a in ativos_selecionados]
                })
                st.dataframe(df_selecionados, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ Nenhum ativo selecionado. Por favor, faça uma seleção.")
    
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
        
        col_question1, col_question2 = st.columns(2)
        
        with st.form("investor_profile_form"):
            options_score = [
                'CT: Concordo Totalmente', 'C: Concordo', 'N: Neutro', 'D: Discordo', 'DT: Discordo Totalmente'
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
                    options=options_score, index=2, key='risk_accept_radio'
                )
                p3_gain = st.radio(
                    "**2. Ganhar o máximo é minha prioridade, mesmo com risco**",
                    options=options_score, index=2, key='max_gain_radio'
                )
                p4_stable = st.radio(
                    "**3. Prefiro crescimento constante, sem volatilidade**",
                    options=options_score, index=2, key='stable_growth_radio'
                )
                p5_loss = st.radio(
                    "**4. Evitar perdas é mais importante que crescimento**",
                    options=options_score, index=2, key='avoid_loss_radio'
                )
                p511_reaction = st.radio(
                    "**5. Se meus investimentos caíssem 10%, eu:**",
                    options=options_reaction, index=1, key='reaction_radio'
                )
                p_level = st.radio(
                    "**6. Meu nível de conhecimento em investimentos:**",
                    options=options_level_abc, index=1, key='level_radio'
                )
            
            with col_question2:
                st.markdown("#### Horizonte Temporal e Capital")
                p211_time = st.radio(
                    "**7. Prazo máximo para reavaliação de estratégia:**",
                    options=options_time_horizon, index=2, key='time_purpose_radio'
                )[0] 
                
                p311_liquid = st.radio(
                    "**8. Necessidade de liquidez (prazo mínimo para resgate):**",
                    options=options_liquidez, index=2, key='liquidity_radio'
                )[0] 
                
                st.markdown("---")
                investment = st.number_input(
                    "Valor de Investimento (R$)",
                    min_value=1000, max_value=10000000, value=100000, step=10000, key='investment_amount'
                )
            
            with st.expander("Opções Avançadas"):
                # 🎯 REMOVIDO: Optuna não é mais uma opção, pois o ML é pré-calculado
                st.info("Otimização de Machine Learning (Optuna) está **desabilitada**. O sistema usará os resultados do Ensemble de ML que foram pré-calculados e salvos no GCS.")
                otimizar_ml = False # Força false
            
            submitted = st.form_submit_button("🚀 Gerar Portfólio Otimizado", type="primary")
            
            if submitted:
                analyzer = AnalisadorPerfilInvestidor()
                risk_answers = {
                    'risk_accept': p2_risk, 'max_gain': p3_gain, 'stable_growth': p4_stable,
                    'avoid_loss': p5_loss, 'reaction': p511_reaction, 'level': p_level,
                    'time_purpose': p211_time, 'liquidity': p311_liquid
                }
                
                risk_level, horizon, lookback, score = analyzer.calcular_perfil(risk_answers)
                
                st.session_state.profile = {
                    'risk_level': risk_level, 'time_horizon': horizon,
                    'ml_lookback_days': lookback, 'risk_score': score
                }
                
                try:
                    builder_local = ConstrutorPortfolioAutoML(investment)
                    st.session_state.builder = builder_local
                except Exception as e:
                    st.error(f"Erro fatal ao inicializar o construtor do portfólio: {e}")
                    return

                with st.spinner(f'Criando portfólio para **PERFIL {risk_level}** ({horizon})...'):
                    try:
                        # OTIMIZAR_ML é forçado a False
                        success = builder_local.executar_pipeline(
                            simbolos_customizados=st.session_state.ativos_para_analise,
                            perfil_inputs=st.session_state.profile,
                            otimizar_ml=False 
                        )
                    except AttributeError as e:
                        st.error(f"Erro de Atributo: Verifique se a classe ConstrutorPortfolioAutoML foi definida corretamente. Erro: {e}")
                        return
                    
                    if not success:
                        st.error("Falha ao coletar dados suficientes ou processar os ativos. Certifique-se de que os dados (4 CSVs por ativo) estão no GCS.")
                        st.session_state.builder = None
                        st.session_state.profile = {}
                        return
                    
                    st.session_state.builder_complete = True
                    st.rerun() 
    
    # FASE 2: RESULTADOS
    else:
        builder = st.session_state.builder
        if builder is None:
            st.error("Objeto construtor não encontrado. Recomece a análise.")
            st.session_state.builder_complete = False
            return
            
        profile = st.session_state.profile
        assets = builder.ativos_selecionados
        allocation = builder.alocacao_portfolio
        
        st.markdown('## ✅ Portfólio Otimizado Gerado')
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Perfil de Risco", profile.get('risk_level', 'N/A'), f"Score: {profile.get('risk_score', 'N/A')}")
        col2.metric("Horizonte", profile.get('time_horizon', 'N/A'))
        col3.metric("Sharpe Ratio", f"{builder.metricas_portfolio.get('sharpe_ratio', 0):.3f}")
        col4.metric("Estratégia", builder.metodo_alocacao_atual.split('(')[0].strip())
        
        if st.button("🔄 Recomeçar Análise", key='recomecar_analysis'):
            st.session_state.builder_complete = False
            st.session_state.builder = None
            st.session_state.profile = {}
            st.session_state.ativos_para_analise = [] 
            st.rerun()
        
        st.markdown("---")
        
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
                    fig_alloc = px.pie(values='Peso (%)', names='Ativo', hole=0.3, data_frame=alloc_data)
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
                        
                        # 🎯 REMOVIDO: forecast e predicted_direction (Não lido do ETL)
                        
                        alloc_table.append({
                            'Ativo': asset.replace('.SA', ''), 
                            'Setor': sector,
                            'Peso (%)': f"{weight * 100:.2f}",
                            'Valor (R$)': f"R$ {amount:,.2f}",
                            'ML Prob. Alta (%)': f"{ml_info.get('predicted_proba_up', 0.5)*100:.1f}",
                            'ML AUC': f"{ml_info.get('auc_roc_score', 0):.3f}" if not pd.isna(ml_info.get('auc_roc_score')) else "N/A",
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
                        x=cum_returns.index, y=cum_returns.values, name=asset.replace('.SA', ''), mode='lines'
                    ))
            
            fig_layout = obter_template_grafico()
            fig_layout['title']['text'] = "Evolução dos Retornos Cumulativos dos Ativos Selecionados"
            fig_layout['yaxis']['title'] = "Retorno Acumulado (Base 1)"
            fig_layout['xaxis']['title'] = "Data"
            fig_cum.update_layout(**fig_layout, height=500)
            
            st.plotly_chart(fig_cum, use_container_width=True)
        
        with tab3:
            st.markdown('#### Análise de Machine Learning (Resultados do ETL)')
            
            ml_data = []
            for asset in assets:
                if asset in builder.predicoes_ml:
                    ml_info = builder.predicoes_ml[asset]
                    ml_data.append({
                        'Ativo': asset.replace('.SA', ''),
                        'Prob. Alta (%)': ml_info.get('predicted_proba_up', 0.5) * 100,
                        'AUC-ROC (WFCV)': ml_info.get('auc_roc_score', np.nan),
                        'Modelo': ml_info.get('model_name', 'N/A'),
                        'Nº Modelos': 5 # O Ensemble completo
                    })
            
            df_ml = pd.DataFrame(ml_data)
            
            if not df_ml.empty:
                fig_ml = go.Figure()
                
                fig_ml.add_trace(go.Bar(
                    x=df_ml['Ativo'], y=df_ml['Prob. Alta (%)'],
                    marker=dict(color=df_ml['Prob. Alta (%)'], colorscale='RdYlGn', showscale=True, colorbar=dict(title="Prob. (%)")),
                    text=df_ml['Prob. Alta (%)'].round(1), textposition='outside'
                ))
                
                fig_layout = obter_template_grafico()
                fig_layout['title']['text'] = f"Probabilidade de Alta Futura (Ensemble Ponderado, Horizonte: {profile.get('ml_lookback_days', 'N/A')}d)"
                fig_layout['yaxis']['title'] = "Probabilidade (%)"
                fig_layout['xaxis']['title'] = "Ativo"
                fig_ml.update_layout(**fig_layout, height=400)
                
                st.plotly_chart(fig_ml, use_container_width=True)
                
                st.markdown("---")
                st.markdown('#### Métricas Detalhadas do ML (AUC e Probabilidades dos Modelos Individuais)')
                
                # 🎯 Tenta mostrar o AUC e Proba de CADA modelo (LGBM, XGB, etc) do Ensemble
                ml_details_list = []
                for asset in assets:
                    if asset in builder.ml_metadata_raw:
                        metadata_df = builder.ml_metadata_raw[asset]
                        target_name_key = [k for k, v in {'curto_prazo': 126, 'medio_prazo': 252, 'longo_prazo': 504}.items() if v == profile.get('ml_lookback_days', 126)]
                        
                        if target_name_key:
                            try:
                                metadata_row = metadata_df.loc[(asset, target_name_key[0])]
                                models_data = json.loads(metadata_row['models_data'])
                                
                                for model_name, data in models_data.items():
                                    ml_details_list.append({
                                        'Ativo': asset.replace('.SA', ''),
                                        'Modelo': model_name.upper(),
                                        'WFCV AUC': f"{data.get('wfc_auc_mean', np.nan):.3f}",
                                        'Proba Final (%)': f"{data.get('final_proba', np.nan)*100:.2f}",
                                        'Peso WFCV': f"{data.get('wfc_weight', np.nan)*100:.2f}%"
                                    })
                            except:
                                pass # Ignora se a linha de metadados estiver faltando

                if ml_details_list:
                    df_ml_details = pd.DataFrame(ml_details_list)
                    st.dataframe(df_ml_details, use_container_width=True, hide_index=True)
                else:
                    st.info("Metadados detalhados do Ensemble (AUC/Proba Individual) indisponíveis no GCS.")
            
            with tab4:
                st.markdown('#### Análise de Volatilidade GARCH (Lida do GCS)')
                
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
                        name='Volatilidade Histórica', x=plot_df_garch['Ativo'], y=plot_df_garch['Vol. Histórica (%)'], marker=dict(color='#7f8c8d'), opacity=0.7
                    ))
                    fig_garch.add_trace(go.Bar(
                        name='Volatilidade GARCH Ajustada', x=plot_df_garch['Ativo'], y=plot_df_garch['Vol. GARCH (%)'], marker=dict(color='#3498db')
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
# FUNÇÃO AUXILIAR: COLETAR DADOS DE UM ÚNICO ATIVO DO GCS (CORRIGIDO PARA O FORMATO 4 CSVs)
# =============================================================================

def coletar_ativo_unico_gcs(ativo_selecionado: str):
    """Carrega dados de um único ativo do GCS, separa histórico e fundamentos."""
    
    df_tech, df_fund_df, ml_results_dict, df_meta = carregar_dados_ativo_gcs_csv(ativo_selecionado)
    
    MIN_DIAS_HISTORICO_FLEXIVEL = max(180, int(MIN_DIAS_HISTORICO * 0.7))
    
    if df_tech.empty or df_fund_df.empty or 'Close' not in df_tech.columns or len(df_tech) < MIN_DIAS_HISTORICO_FLEXIVEL:
        return None, None
        
    # 1. Histórico/Features
    # df_tech é o histórico completo (OHLCV + Features Técnicas + Targets)
    hist = df_tech.copy()
    
    # 2. Fundamentos (Lidos do df_fund_df)
    fund_data = df_fund_df.iloc[0].to_dict()
    
    # Adicionar dados ML para o Analisador Individual (apenas o AUC)
    target_126d = 'curto_prazo'
    auc_score = np.nan
    
    if not df_meta.empty and target_126d in df_meta.index.get_level_values('target_name'):
        try:
            metadata_row = df_meta.loc[(ativo_selecionado, target_126d)]
            models_data = json.loads(metadata_row['models_data'])
            auc_score = models_data.get('lgbm', {}).get('wfc_auc_mean', np.nan) 
        except:
            pass
            
    fund_data['ML_Proba'] = ml_results_dict.get('ml_proba_126d', np.nan)
    fund_data['ML_AUC'] = auc_score
    
    return hist, fund_data
    

# =============================================================================
# FUNÇÃO: INTERFACE STREAMLIT - ABA ANÁLISE INDIVIDUAL (AJUSTADA)
# =============================================================================

def aba_analise_individual():
    """Aba 4: Análise Individual de Ativos - Otimizada para usar SOMENTE dados do GCS."""
    
    st.markdown("## 🔍 Análise Individual Completa de Ativos (Lê GCS)")
    
    if 'ativos_para_analise' in st.session_state and st.session_state.ativos_para_analise:
        ativos_disponiveis = st.session_state.ativos_para_analise
    else:
        ativos_disponiveis = ATIVOS_IBOVESPA 
            
    if not ativos_disponiveis:
        st.error("Nenhum ativo disponível para análise. Verifique as configurações ou selecione ativos.")
        return

    if 'individual_asset_select' not in st.session_state and ativos_disponiveis:
        st.session_state.individual_asset_select = ativos_disponiveis[0]

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
    
    with st.spinner(f"Analisando {ativo_selecionado} (Dados do GCS)..."):
        try:
            # 🎯 CHAVE: Coleta Sob Demanda (Lê os 4 CSVs e consolida)
            df_completo, features_fund = coletar_ativo_unico_gcs(ativo_selecionado)
            
            if df_completo is None or df_completo.empty or 'Close' not in df_completo.columns:
                st.error(f"❌ Não foi possível obter dados (Histórico/Features) válidos do GCS para **{ativo_selecionado.replace('.SA', '')}**. Verifique a URL do GCS.")
                return

            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Visão Geral", "📈 Análise Técnica", "💼 Análise Fundamentalista",
                "🤖 Machine Learning", "🔬 Clusterização e Similaridade"
            ])
            
            # --- Tab 1: Visão Geral ---
            with tab1:
                st.markdown(f"### {ativo_selecionado.replace('.SA', '')} - Visão Geral")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                preco_atual = df_completo['Close'].iloc[-1] if not df_completo.empty and 'Close' in df_completo.columns else np.nan
                variacao_dia = df_completo['returns'].iloc[-1] * 100 if not df_completo.empty and 'returns' in df_completo.columns else np.nan
                volume_medio = df_completo['Volume'].mean() if not df_completo.empty and 'Volume' in df_completo.columns else np.nan
                beta_val = features_fund.get('beta', np.nan)
                
                col1.metric("Preço Atual", f"R$ {preco_atual:.2f}" if not np.isnan(preco_atual) else "N/A", f"{variacao_dia:+.2f}%" if not np.isnan(variacao_dia) else "N/A")
                col2.metric("Volume Médio", f"{volume_medio:,.0f}" if not np.isnan(volume_medio) else "N/A")
                col3.metric("Setor", features_fund.get('sector', 'N/A'))
                col4.metric("Indústria", features_fund.get('industry', 'N/A'))
                col5.metric("Beta", f"{beta_val:.2f}" if not pd.isna(beta_val) else "N/A")
                
                if not df_completo.empty and 'Open' in df_completo.columns and 'Volume' in df_completo.columns:
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                    fig.add_trace(go.Candlestick(x=df_completo.index, open=df_completo['Open'], high=df_completo['High'], low=df_completo['Low'], close=df_completo['Close'], name='Preço'), row=1, col=1)
                    fig.add_trace(go.Bar(x=df_completo.index, y=df_completo['Volume'], name='Volume', marker=dict(color='lightblue'), opacity=0.7), row=2, col=1)
                    fig_layout = obter_template_grafico()
                    fig_layout['title']['text'] = f"Histórico de Preços e Volume - {ativo_selecionado.replace('.SA', '')}"
                    fig_layout['height'] = 600
                    fig.update_layout(**fig_layout)
                    st.plotly_chart(fig, use_container_width=True)
                else: st.warning("Dados de histórico (OHLCV) incompletos para gráfico.")

            # --- Tab 2: Análise Técnica ---
            with tab2:
                st.markdown("### Indicadores Técnicos")
                
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                
                col1.metric("RSI (14)", f"{df_completo['rsi_14'].iloc[-1]:.2f}" if 'rsi_14' in df_completo.columns and not df_completo['rsi_14'].empty else "N/A")
                col2.metric("MACD", f"{df_completo['macd'].iloc[-1]:.4f}" if 'macd' in df_completo.columns and not df_completo['macd'].empty else "N/A")
                col3.metric("Stoch %K", f"{df_completo['stoch_k'].iloc[-1]:.2f}" if 'stoch_k' in df_completo.columns and not df_completo['stoch_k'].empty else "N/A")
                col4.metric("ADX", f"{df_completo['adx'].iloc[-1]:.2f}" if 'adx' in df_completo.columns and not df_completo['adx'].empty else "N/A")
                col5.metric("CCI", f"{df_completo['cci'].iloc[-1]:.2f}" if 'cci' in df_completo.columns and not df_completo['cci'].empty else "N/A")
                col6.metric("ATR (%)", f"{df_completo['atr_percent'].iloc[-1]:.2f}%" if 'atr_percent' in df_completo.columns and not df_completo['atr_percent'].empty else "N/A")

                st.markdown("#### RSI e Stochastic Oscillator")
                
                fig_osc = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("RSI (14)", "Stochastic Oscillator (%K & %D)"))
                
                if 'rsi_14' in df_completo.columns:
                    fig_osc.add_trace(go.Scatter(x=df_completo.index, y=df_completo['rsi_14'], name='RSI', line=dict(color='#3498db')), row=1, col=1)
                    fig_osc.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1, annotation_text="Overbought (70)")
                    fig_osc.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1, annotation_text="Oversold (30)")
                
                if 'stoch_k' in df_completo.columns and 'stoch_d' in df_completo.columns:
                    fig_osc.add_trace(go.Scatter(x=df_completo.index, y=df_completo['stoch_k'], name='Stochastic %K', line=dict(color='#e74c3c')), row=2, col=1)
                    fig_osc.add_trace(go.Scatter(x=df_completo.index, y=df_completo['stoch_d'], name='Stochastic %D', line=dict(color='#7f8c8d')), row=2, col=1)
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
                else: st.warning("Nenhum indicador técnico com dados atuais disponível.")

            # --- Tab 3: Análise Fundamentalista ---
            with tab3:
                st.markdown("### Análise Fundamentalista Expandida")
                
                st.markdown("#### Valuation")
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("P/L (TTM)", f"{features_fund.get('pe_ratio', np.nan):.2f}" if not pd.isna(features_fund.get('pe_ratio')) else "N/A")
                col2.metric("P/VP", f"{features_fund.get('pb_ratio', np.nan):.2f}" if not pd.isna(features_fund.get('pb_ratio')) else "N/A")
                col3.metric("P/VPA (Vendas)", f"{features_fund.get('ps_ratio', np.nan):.2f}" if not pd.isna(features_fund.get('ps_ratio')) else "N/A")
                col4.metric("PEG", f"{features_fund.get('peg_ratio', np.nan):.2f}" if not pd.isna(features_fund.get('peg_ratio')) else "N/A")
                col5.metric("EV/EBITDA", f"{features_fund.get('ev_ebitda', np.nan):.2f}" if not pd.isna(features_fund.get('ev_ebitda')) else "N/A")
                
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
                col3.metric("DY Médio 5A", "N/A" ) 
                
                st.markdown("#### Crescimento")
                col1, col2, col3 = st.columns(3)
                col1.metric("Cresc. Receita", f"{features_fund.get('revenue_growth', np.nan):.2f}%" if not pd.isna(features_fund.get('revenue_growth')) else "N/A")
                col2.metric("Cresc. Lucros (Anual)", f"{features_fund.get('earnings_growth', np.nan):.2f}%" if not pd.isna(features_fund.get('earnings_growth')) else "N/A")
                col3.metric("Cresc. Lucros (Q)", "N/A") 

                st.markdown("#### Saúde Financeira")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Dívida/Patrimônio", f"{features_fund.get('debt_to_equity', np.nan):.2f}" if not pd.isna(features_fund.get('debt_to_equity')) else "N/A")
                col2.metric("Current Ratio", f"{features_fund.get('current_ratio', np.nan):.2f}" if not pd.isna(features_fund.get('current_ratio')) else "N/A")
                col3.metric("Quick Ratio", f"{features_fund.get('quick_ratio', np.nan):.2f}" if not pd.isna(features_fund.get('quick_ratio')) else "N/A")
                col4.metric("Fluxo de Caixa Livre", "N/A")
                
                st.markdown("---")
                st.markdown("#### Todos os Fundamentos Disponíveis")
                
                df_fund_display = pd.DataFrame({
                    'Métrica': list(features_fund.keys()),
                    'Valor': [f"{v:.4f}" if isinstance(v, (int, float)) and not pd.isna(v) else str(v) for v in features_fund.values()]
                })
                
                st.dataframe(df_fund_display, use_container_width=True, hide_index=True)
            
            # --- Tab 4: Machine Learning ---
            with tab4:
                st.markdown("### Análise de Machine Learning (Resultados do ETL)")
                
                proba_final = features_fund.get('ML_Proba', np.nan)
                auc_medio = features_fund.get('ML_AUC', np.nan)
                
                st.info("A previsão ML (Prob. Alta e AUC) é lida diretamente dos arquivos `_ml_results.csv` e `_ml_metadata.csv` do GCS (Ensemble completo de 5 modelos).")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Probabilidade de Alta Futura", f"{proba_final*100:.2f}%" if not pd.isna(proba_final) else "N/A")
                col2.metric("AUC-ROC Médio (WFCV - LGBM)", f"{auc_medio:.3f}" if not pd.isna(auc_medio) else "N/A")
                col3.metric("Horizonte de Previsão", f"{st.session_state.builder.perfil_dashboard.get('ml_lookback_days', 126)} dias" if 'builder' in st.session_state and st.session_state.builder else "126 dias (Padrão)")

                st.markdown("---")
                st.markdown("#### Feature Importance (Indisponível no modo GCS-ETL)")
                st.warning("O gráfico de Feature Importance e a predição sob demanda foram removidos, pois os modelos não são treinados em tempo real no Streamlit. Utilize o *notebook* do ETL para visualizar a importância das features.")
                
            # --- Tab 5: Clusterização ---
            with tab5:
                st.markdown("### Clusterização e Análise de Similaridade")
                
                if 'builder' not in st.session_state or st.session_state.builder is None:
                    st.warning("A Clusterização está desabilitada. Por favor, execute a aba **'Construtor de Portfólio'** primeiro para carregar os dados de múltiplos ativos.")
                    return
                
                builder = st.session_state.builder
                
                assets_to_cluster = []
                
                # Coleta ativos do mesmo setor para comparação
                sector = features_fund.get('sector', 'Unknown')
                if sector != 'Unknown':
                    sector_assets = [a for a in builder.ativos_sucesso if builder.dados_fundamentalistas.loc[a, 'sector'] == sector]
                    assets_to_cluster.extend(sector_assets)
                    assets_to_cluster = list(set(assets_to_cluster)) # Remove duplicatas
                
                if len(assets_to_cluster) < 5:
                    st.warning("Dados insuficientes para realizar a clusterização e análise de similaridade (mínimo de 5 ativos no mesmo setor, lidos do GCS).")
                    return

                # Cria o DF de comparação apenas com Métricas e Fundamentos
                comparison_data_rows = []
                for asset_comp in assets_to_cluster:
                    comp_data = {}
                    
                    # Performance e GARCH (do dados_fundamentalistas)
                    fund_metrics = builder.dados_fundamentalistas.loc[asset_comp].to_dict()
                    for metric in ['sharpe_ratio', 'annual_return', 'annual_volatility', 'garch_volatility', 'pe_ratio', 'pb_ratio', 'roe', 'debt_to_equity']:
                        comp_data[metric] = fund_metrics.get(metric, np.nan)
                    
                    comparison_data_rows.append(comp_data)

                df_comparacao = pd.DataFrame(comparison_data_rows, index=assets_to_cluster)
                df_comparacao = df_comparacao.select_dtypes(include=[np.number]).dropna(how='all')
                
                if len(df_comparacao) > 5:
                    resultado_pca, pca, kmeans = AnalisadorIndividualAtivos.realizar_clusterizacao_pca(
                        df_comparacao, n_clusters=min(5, len(df_comparacao) - 1)
                    )
                    
                    if resultado_pca is not None:
                        # Gráficos PCA e Cluster
                        if 'PC3' in resultado_pca.columns:
                            fig_pca = px.scatter_3d(resultado_pca, x='PC1', y='PC2', z='PC3', color='Cluster', hover_name=resultado_pca.index.str.replace('.SA', ''), title='Clusterização K-means + PCA (3D) - Similaridade de Ativos')
                        else:
                            fig_pca = px.scatter(resultado_pca, x='PC1', y='PC2', color='Cluster', hover_name=resultado_pca.index.str.replace('.SA', ''), title='Clusterização K-means + PCA (2D) - Similaridade de Ativos')
                        
                        fig_pca.update_layout(**obter_template_grafico(), height=600)
                        st.plotly_chart(fig_pca, use_container_width=True)
                        
                        if ativo_selecionado in resultado_pca.index:
                            cluster_ativo = resultado_pca.loc[ativo_selecionado, 'Cluster']
                            ativos_similares_df = resultado_pca[resultado_pca['Cluster'] == cluster_ativo]
                            ativos_similares = [a for a in ativos_similares_df.index.tolist() if a != ativo_selecionado]
                            
                            st.success(f"**{ativo_selecionado.replace('.SA', '')}** pertence ao Cluster {cluster_ativo}")
                            
                            if ativos_similares:
                                st.markdown(f"#### Outros Ativos no Cluster {cluster_ativo} (Setor: {sector}):")
                                st.write(", ".join([a.replace('.SA', '') for a in ativos_similares[:15]]))
                        
                        st.markdown("#### Variância Explicada por Componente Principal")
                        var_exp = pca.explained_variance_ratio_ * 100
                        df_var = pd.DataFrame({'Componente': [f'PC{i+1}' for i in range(len(var_exp))], 'Variância (%)': var_exp})
                        fig_var = px.bar(df_var, x='Componente', y='Variância (%)', title='Variância Explicada por Componente Principal')
                        fig_var.update_layout(**obter_template_grafico())
                        st.plotly_chart(fig_var, use_container_width=True)

                    else:
                        st.warning("Falha ao rodar PCA/Cluster. Verifique a qualidade dos dados numéricos.")

                else:
                    st.warning(f"Menos de 5 ativos ({len(df_comparacao)}) com dados numéricos completos para realizar a clusterização.")

        except Exception as e:
            st.error(f"Erro ao analisar o ativo {ativo_selecionado}: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
def main():
    """Função principal com estrutura de 4 abas (Sidebar removida)"""
    
    if 'builder' not in st.session_state:
        st.session_state.builder = None
        st.session_state.builder_complete = False
        st.session_state.profile = {}
        st.session_state.ativos_para_analise = []
        st.session_state.analisar_ativo_triggered = False
        
    configurar_pagina()
    
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
