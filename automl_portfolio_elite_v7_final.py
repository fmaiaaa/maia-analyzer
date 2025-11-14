# -*- coding: utf-8 -*-
"""
=============================================================================
SISTEMA DE PORTFÓLIOS ADAPTATIVOS - OTIMIZAÇÃO QUANTITATIVA
=============================================================================

Adaptação do Sistema AutoML para usar dados pré-processados (CSV/GCS)
gerados pelo gerador_financeiro.py, eliminando a dependência do yfinance
na interface Streamlit e adotando uma linguagem profissional.

Versão: 8.6.1 (CORRIGIDA) - Ajuste na leitura de CSVs do GCS (index_col=0)
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
# REMOVIDAS: O cálculo em si é assumido nos dados do GCS.

# --- 5. MACHINE LEARNING (SCIKIT-LEARN) ---
# MANTIDAS: Usadas para a função de clusterização (PCA/KMeans) local.
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# --- 6. BOOSTED MODELS & OPTIMIZATION ---
# REMOVIDAS: O treinamento e otimização são feitos no gerador_financeiro.py (Optuna/XGBoost/LightGBM/CatBoost).

# --- 7. SPECIALIZED TIME SERIES & ECONOMETRICS ---
# MANTIDA: Usada como dependência lógica, embora o resultado (volatilidade) venha do GCS.
from arch import arch_model

# --- 8. CONFIGURATION (NOVO NOME PROFISSIONAL) ---
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURAÇÕES E CONSTANTES GLOBAIS
# =============================================================================

PERIODO_DADOS = '5y'
MIN_DIAS_HISTORICO = 252
NUM_ATIVOS_PORTFOLIO = 5
TAXA_LIVRE_RISCO = 0.1075
LOOKBACK_ML = 30
SCORE_PERCENTILE_THRESHOLD = 0.85 # Percentil 85 (Ajustado do 90/95 para garantir seleção)

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
# 4. LISTAS DE ATIVOS E SETORES (AJUSTADAS SOMENTE PARA IBOVESPA)
# =============================================================================

# Lista ATUALIZADA de ativos do Ibovespa (baseada no arquivo gerador_financeiro.py)
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

# Mapeamento setorial (MANTIDO, mas apenas para os ativos acima, usando a lista original como referência)
ATIVOS_POR_SETOR_IBOV = {
    'Bens Industriais': ['EMBR3.SA', 'VAMO3.SA', 'WEGE3.SA', 'VIVA3.SA', 'ASAI3.SA', 'SMFT3.SA', 'CMIN3.SA', 'SLCE3.SA'],
    'Consumo Cíclico': ['AZZA3.SA', 'ALOS3.SA', 'CEAB3.SA', 'COGN3.SA', 'CURY3.SA', 'CVCB3.SA', 'CYRE3.SA', 'DIRR3.SA', 'LREN3.SA', 'MGLU3.SA', 'MRVE3.SA', 'RENT3.SA', 'YDUQ3.SA'],
    'Consumo não Cíclico': ['BEEF3.SA', 'NATU3.SA', 'PCAR3.SA', 'VIVA3.SA'], # Algumas duplicatas são resolvidas no ALL_ASSETS
    'Financeiro': ['B3SA3.SA', 'BBSE3.SA', 'BBDC3.SA', 'BBDC4.SA', 'BBAS3.SA', 'BPAC11.SA', 'CXSE3.SA', 'HYPE3.SA', 'IGTI11.SA', 'IRBR3.SA', 'ITSA4.SA', 'ITUB4.SA', 'MULT3.SA', 'PSSA3.SA', 'RDOR3.SA', 'SANB11.SA'],
    'Materiais Básicos': ['BRAP4.SA', 'BRKM5.SA', 'CSNA3.SA', 'GGBR4.SA', 'GOAU4.SA', 'KLBN11.SA', 'POMO4.SA', 'SUZB3.SA', 'USIM5.SA', 'VALE3.SA'],
    'Petróleo, Gás e Biocombustíveis': ['ENEV3.SA', 'PETR3.SA', 'PETR4.SA', 'PRIO3.SA', 'RAIZ4.SA', 'RECV3.SA', 'UGPA3.SA', 'VBBR3.SA'],
    'Saúde': ['FLRY3.SA', 'HAPV3.SA', 'RADL3.SA'],
    'Tecnologia da Informação': ['TOTS3.SA'],
    'Telecomunicações': ['TIMS3.SA', 'VIVT3.SA'],
    'Utilidade Pública': ['AESB3.SA', 'AURE3.SA', 'BRAV3.SA', 'CMIG4.SA', 'CPLE6.SA', 'CPFE3.SA', 'EGIE3.SA', 'ELET3.SA', 'ELET6.SA', 'ENGI11.SA', 'EQTL3.SA', 'ISAE4.SA', 'RAIL3.SA', 'SBSP3.SA', 'TAEE11.SA']
}

# Garante que as listas de ativos e setores estejam sincronizadas (apenas IBOVESPA)
TODOS_ATIVOS = sorted(list(set(ATIVOS_IBOVESPA)))

# Remove setores vazios que poderiam surgir na filtragem
ATIVOS_POR_SETOR = {
    setor: [ativo for ativo in ativos if ativo in ATIVOS_IBOVESPA] 
    for setor, ativos in ATIVOS_POR_SETOR_IBOV.items()
    if any(ativo in ATIVOS_IBOVESPA for ativo in ativos)
}


# =============================================================================
# 5. MAPEAMENTOS DE PONTUAÇÃO DO QUESTIONÁRIO (LÓGICA INALTERADA)
# =============================================================================

# Mapas de pontuação ORIGINAIS (NÃO MUDAR A LÓGICA)
SCORE_MAP_ORIGINAL = {
    'CT: Concordo Totalmente': 5, 'C: Concordo': 4, 'N: Neutro': 3, 'D: Discordo': 2, 'DT: Discordo Totalmente': 1
}
SCORE_MAP_INV_ORIGINAL = {
    'CT: Concordo Totalmente': 1, 'C: Concordo': 2, 'N: Neutro': 3, 'D: Discordo': 4, 'DT: Discordo Totalmente': 5
}
SCORE_MAP_CONHECIMENTO_ORIGINAL = {
    'A: Avançado (Análise fundamentalista, macro e técnica)': 5, 
    'B: Intermediário (Conhecimento básico sobre mercados e ativos)': 3, 
    'C: Iniciante (Pouca ou nenhuma experiência em investimentos)': 1
}
SCORE_MAP_REACTION_ORIGINAL = {
    'A: Venderia imediatamente': 1, 
    'B: Manteria e reavaliaria a tese': 3, 
    'C: Compraria mais para aproveitar preços baixos': 5
}

# --- NOVOS MAPEAMENTOS (Textos Descritivos -> Chaves Originais) ---

# Mapeamento para perguntas de Concordância (1, 2)
OPTIONS_CONCORDA = [
    "CT: (Concordo Totalmente) - Estou confortável com altas flutuações, pois entendo que são o preço para retornos potencialmente maiores.",
    "C: (Concordo) - Aceito alguma volatilidade, mas espero que os ganhos compensem o risco assumido de forma clara.",
    "N: (Neutro) - Tenho dificuldade em opinar; minha decisão dependeria do momento e do ativo específico.",
    "D: (Discordo) - Prefiro estratégias mais cautelosas, mesmo que isso signifique um potencial de retorno menor.",
    "DT: (Discordo Totalmente) - Não estou disposto a ver meu patrimônio flutuar significativamente; prefiro segurança absoluta."
]
MAP_CONCORDA = {
    OPTIONS_CONCORDA[0]: 'CT: Concordo Totalmente',
    OPTIONS_CONCORDA[1]: 'C: Concordo',
    OPTIONS_CONCORDA[2]: 'N: Neutro',
    OPTIONS_CONCORDA[3]: 'D: Discordo',
    OPTIONS_CONCORDA[4]: 'DT: Discordo Totalmente'
}

# Mapeamento para perguntas de Discordância (Invertido) (3, 4)
OPTIONS_DISCORDA = [
    "CT: (Concordo Totalmente) - A preservação do capital é minha prioridade máxima, acima de qualquer ganho potencial.",
    "C: (Concordo) - É muito importante para mim evitar perdas, mesmo que isso limite o crescimento do meu portfólio.",
    "N: (Neutro) - Busco um equilíbrio; não quero perdas excessivas, mas sei que algum risco é necessário para crescer.",
    "D: (Discordo) - Estou focado no crescimento de longo prazo e entendo que perdas de curto prazo fazem parte do processo.",
    "DT: (Discordo Totalmente) - Meu foco é maximizar o retorno; perdas de curto prazo são irrelevantes se a tese de longo prazo for válida."
]
MAP_DISCORDA = {
    OPTIONS_DISCORDA[0]: 'CT: Concordo Totalmente',
    OPTIONS_DISCORDA[1]: 'C: Concordo',
    OPTIONS_DISCORDA[2]: 'N: Neutro',
    OPTIONS_DISCORDA[3]: 'D: Discordo',
    OPTIONS_DISCORDA[4]: 'DT: Discordo Totalmente'
}

# Mapeamento para Reação à Perda (5)
OPTIONS_REACTION_DETALHADA = [
    "A: (Vender Imediatamente) - Venderia a posição para evitar perdas maiores; prefiro realizar o prejuízo e reavaliar.",
    "B: (Manter e Reavaliar) - Manteria a calma, reavaliaria os fundamentos do ativo e o cenário macro para tomar uma decisão.",
    "C: (Comprar Mais) - Encararia como uma oportunidade de compra, aumentando a posição a um preço menor, se os fundamentos estiverem intactos."
]
MAP_REACTION = {
    OPTIONS_REACTION_DETALHADA[0]: 'A: Venderia imediatamente',
    OPTIONS_REACTION_DETALHADA[1]: 'B: Manteria e reavaliaria a tese',
    OPTIONS_REACTION_DETALHADA[2]: 'C: Compraria mais para aproveitar preços baixos'
}

# Mapeamento para Nível de Conhecimento (6)
OPTIONS_CONHECIMENTO_DETALHADA = [
    "A: (Avançado) - Sinto-me confortável analisando balanços (fundamentalista), gráficos (técnica) e cenários macroeconômicos.",
    "B: (Intermediário) - Entendo os conceitos básicos (Renda Fixa vs. Variável, risco vs. retorno) e acompanho o mercado.",
    "C: (Iniciante) - Tenho pouca ou nenhuma experiência prática em investimentos além da poupança ou produtos bancários simples."
]
MAP_CONHECIMENTO = {
    OPTIONS_CONHECIMENTO_DETALHADA[0]: 'A: Avançado (Análise fundamentalista, macro e técnica)',
    OPTIONS_CONHECIMENTO_DETALHADA[1]: 'B: Intermediário (Conhecimento básico sobre mercados e ativos)',
    OPTIONS_CONHECIMENTO_DETALHADA[2]: 'C: Iniciante (Pouca ou nenhuma experiência em investimentos)'
}

# Mapeamento para Horizonte e Liquidez (Chave é a Letra Inicial)
OPTIONS_TIME_HORIZON_DETALHADA = [
    'A: Curto (até 1 ano) - Meu objetivo é preservar capital ou realizar um ganho rápido, com alta liquidez.', 
    'B: Médio (1-5 anos) - Busco um crescimento balanceado e posso tolerar alguma flutuação neste período.', 
    'C: Longo (5+ anos) - Meu foco é a acumulação de patrimônio; flutuações de curto/médio prazo não me afetam.'
]
OPTIONS_LIQUIDEZ_DETALHADA = [
    'A: Menos de 6 meses - Posso precisar resgatar o valor a qualquer momento (ex: reserva de emergência).', 
    'B: Entre 6 meses e 2 anos - Não preciso do dinheiro imediatamente, mas tenho um objetivo de curto/médio prazo.', 
    'C: Mais de 2 anos - Este é um investimento de longo prazo; não tenho planos de resgatar nos próximos anos.'
]


# =============================================================================
# 6. CLASSE: ANALISADOR DE PERFIL DO INVESTIDOR (LÓGICA CALIBRADA - V8.3.0)
# =============================================================================

class AnalisadorPerfilInvestidor:
    """Analisa perfil de risco e horizonte temporal do investidor."""
    
    def __init__(self):
        self.nivel_risco = ""
        self.horizonte_tempo = ""
        self.dias_lookback_ml = 5
    
    def determinar_nivel_risco(self, pontuacao: int) -> str:
        """
        Determina o nível de risco com base na pontuação total (min 26, max 130).
        Faixas igualmente espaçadas (aprox. 21 pontos por nível).
        """
        # Faixas de pontuação calibradas:
        # CONSERVADOR: 26 - 46
        # INTERMEDIÁRIO: 47 - 67
        # MODERADO: 68 - 88
        # MODERADO-ARROJADO: 89 - 109
        # AVANÇADO: 110 - 130
        
        if pontuacao <= 46: return "CONSERVADOR"
        elif pontuacao <= 67: return "INTERMEDIÁRIO"
        elif pontuacao <= 88: return "MODERADO"
        elif pontuacao <= 109: return "MODERADO-ARROJADO"
        else: return "AVANÇADO"
    
    def determinar_horizonte_ml(self, liquidez_key: str, objetivo_key: str) -> tuple[str, int]:
        # A chave original (A, B, C) é extraída da resposta descritiva
        time_map = { 'A': 5, 'B': 20, 'C': 30 }
        final_lookback = max( time_map.get(liquidez_key, 5), time_map.get(objetivo_key, 5) )
        
        if final_lookback >= 30:
            self.horizonte_tempo = "LONGO PRAZO"; self.dias_lookback_ml = 30
        elif final_lookback >= 20:
            self.horizonte_tempo = "MÉDIO PRAZO"; self.dias_lookback_ml = 20
        else:
            self.horizonte_tempo = "CURTO PRAZO"; self.dias_lookback_ml = 5
        
        return self.horizonte_tempo, self.dias_lookback_ml
    
    def calcular_perfil(self, respostas_risco_originais: dict) -> tuple[str, str, int, int]:
        
        # Mapeamento do texto da resposta para o score numérico (USANDO OS MAPAS ORIGINAIS)
        score_risk_accept = SCORE_MAP_ORIGINAL.get(respostas_risco_originais['risk_accept'], 3)
        score_max_gain = SCORE_MAP_ORIGINAL.get(respostas_risco_originais['max_gain'], 3)
        score_stable_growth = SCORE_MAP_INV_ORIGINAL.get(respostas_risco_originais['stable_growth'], 3)
        score_avoid_loss = SCORE_MAP_INV_ORIGINAL.get(respostas_risco_originais['avoid_loss'], 3)
        score_level = SCORE_MAP_CONHECIMENTO_ORIGINAL.get(respostas_risco_originais['level'], 3)
        score_reaction = SCORE_MAP_REACTION_ORIGINAL.get(respostas_risco_originais['reaction'], 3)

        pontuacao = (
            score_risk_accept * 5 +
            score_max_gain * 5 +
            score_stable_growth * 5 +
            score_avoid_loss * 5 +
            score_level * 3 +
            score_reaction * 3
        )
        nivel_risco = self.determinar_nivel_risco(pontuacao)
        
        # Extrai apenas a chave (A, B ou C) para determinar o horizonte ML
        liquidez_key = respostas_risco_originais['liquidity'][0] if isinstance(respostas_risco_originais['liquidity'], str) and respostas_risco_originais['liquidity'] else 'C'
        objetivo_key = respostas_risco_originais['time_purpose'][0] if isinstance(respostas_risco_originais['time_purpose'], str) and respostas_risco_originais['time_purpose'] else 'C'
        
        horizonte_tempo, ml_lookback = self.determinar_horizonte_ml(
            liquidez_key, objetivo_key
        )
        return nivel_risco, horizonte_tempo, ml_lookback, pontuacao

# =============================================================================
# 7. FUNÇÕES DE ESTILO E VISUALIZAÇÃO (Aprimoradas)
# =============================================================================

def obter_template_grafico() -> dict:
    """Retorna um template de layout otimizado para gráficos Plotly com estilo Limpo/Neutro (Preto/Cinza)."""
    return {
        'plot_bgcolor': '#f8f9fa', # Cinza muito claro
        'paper_bgcolor': 'white',
        'font': {
            'family': 'Arial, sans-serif', # Fonte mais limpa
            'size': 12,
            'color': '#343a40' # Cinza escuro
        },
        'title': {
            'font': {
                'family': 'Arial, sans-serif',
                'size': 16,
                'color': '#212529', # Preto/Cinza muito escuro
                'weight': 'bold'
            },
            'x': 0.5,
            'xanchor': 'center'
        },
        'xaxis': {
            'showgrid': True, 'gridcolor': '#e9ecef', 'showline': True, 'linecolor': '#ced4da', 'linewidth': 1,
            'tickfont': {'family': 'Arial, sans-serif', 'color': '#343a40'}, 'title': {'font': {'family': 'Arial, sans-serif', 'color': '#343a40'}}, 'zeroline': False
        },
        'yaxis': {
            'showgrid': True, 'gridcolor': '#e9ecef', 'showline': True, 'linecolor': '#ced4da', 'linewidth': 1,
            'tickfont': {'family': 'Arial, sans-serif', 'color': '#343a40'}, 'title': {'font': {'family': 'Arial, sans-serif', 'color': '#343a40'}}, 'zeroline': False
        },
        'legend': {
            'font': {'family': 'Arial, sans-serif', 'color': '#343a40'},
            'bgcolor': 'rgba(255, 255, 255, 0.8)',
            'bordercolor': '#e9ecef',
            'borderwidth': 1
        },
        # Paleta de cores Neutra/Monocromática
        'colorway': ['#212529', '#495057', '#6c757d', '#adb5bd', '#ced4da']
    }

# =============================================================================
# 8. CLASSE: ENGENHEIRO DE FEATURES (Mantido)
# =============================================================================

class EngenheiroFeatures:
    """Funções utilitárias de features e normalização."""

    @staticmethod
    def _normalizar(serie: pd.Series, maior_melhor: bool = True) -> pd.Series:
        """Normaliza uma série de indicadores para o range [0, 1] (Min-Max Scaling)."""
        if serie.isnull().all():
            return pd.Series(0, index=serie.index)
        
        # Trata outliers extremos (ex: P/L de 5000) usando clipping no percentil 98 e 2
        q_low = serie.quantile(0.02)
        q_high = serie.quantile(0.98)
        serie_clipped = serie.clip(q_low, q_high)
        
        min_val = serie_clipped.min()
        max_val = serie_clipped.max()
        
        if max_val == min_val:
            return pd.Series(0.5, index=serie.index)
        
        if maior_melhor:
            return (serie_clipped - min_val) / (max_val - min_val)
        else:
            return (max_val - serie_clipped) / (max_val - min_val)

# =============================================================================
# 9. FUNÇÕES DE COLETA DE DADOS GCS (CORRIGIDO)
# =============================================================================

@st.cache_data(ttl=3600) # Cache de 1 hora para dados GCS
def carregar_dados_ativo_gcs_csv(base_url: str, ticker: str, file_suffix: str) -> pd.DataFrame:
    """
    Carrega o DataFrame de um único ativo via URL pública do GCS (formato CSV).
    *** CORREÇÃO v8.6.1: Usa index_col=0 para ler o índice salvo pelo gerador_financeiro.py ***
    """
    file_name = f"{ticker}{file_suffix}" 
    full_url = f"{base_url}{file_name}"
    
    try:
        # --- ALTERAÇÃO PRINCIPAL ---
        # Carrega o CSV, assumindo que a primeira coluna (0) é o índice
        # Isso corrige o problema de leitura, pois o gerador_financeiro.py salva com index=True
        df_ativo = pd.read_csv(full_url, index_col=0)
        
        # 1. Pós-processamento para _tecnicos.csv
        if file_suffix == '_tecnicos.csv':
            df_ativo.index.name = 'Date'
            # 2. Conversão para Datetime e remoção de timezone
            if df_ativo.index.dtype == object:
                df_ativo.index = pd.to_datetime(df_ativo.index)
            
            if df_ativo.index.tz is not None:
                 df_ativo.index = df_ativo.index.tz_convert(None) 

        # 2. Pós-processamento para _fundamentos.csv
        elif file_suffix == '_fundamentos.csv':
            df_ativo.index.name = 'Ticker'
        
        # 3. Pós-processamento para _ml_results.csv
        elif file_suffix == '_ml_results.csv':
            df_ativo.index.name = 'Ticker'
            
        # 4. Conversão de colunas numéricas (para todos)
        # Garante que colunas de ID não sejam convertidas
        id_cols = ['ticker', 'sector', 'industry', 'recommendation', 'Date', 'index', 'Ticker']
        for col in df_ativo.columns:
            if col not in id_cols:
                # Força a conversão para float, ignorando erros
                df_ativo[col] = pd.to_numeric(df_ativo[col], errors='coerce')
        
        return df_ativo.sort_index()

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
        # Filtra colunas que começam com 'fund_'
        fund_data = fund_row.filter(regex='^fund_').to_dict()
        fund_data = {k.replace('fund_', ''): v for k, v in fund_data.items()}
        
        # Adiciona colunas de performance estáticas
        fund_data['sharpe_ratio'] = fund_row.get('sharpe_ratio', np.nan)
        fund_data['annual_return'] = fund_row.get('annual_return', np.nan)
        fund_data['annual_volatility'] = fund_row.get('annual_volatility', np.nan)
        fund_data['max_drawdown'] = fund_row.get('max_drawdown', np.nan)
        fund_data['garch_volatility'] = fund_row.get('garch_volatility', np.nan)
        fund_data['sector'] = fund_row.get('sector', 'Unknown')
        fund_data['industry'] = fund_row.get('industry', 'Unknown')
        
        # Renomeia para P/L e ROE simples, se existirem
        if 'pe_ratio' in fund_data: fund_data['pe_ratio'] = fund_data['pe_ratio'] 
        if 'roe' in fund_data: fund_data['roe'] = fund_data['roe'] 
        
        return fund_data

    # O cache é aplicado na função de carregamento (carregar_dados_ativo_gcs_csv)
    def coletar_e_processar_dados(self, simbolos: list) -> bool:
        """Carrega os DataFrames para todos os ativos no pipeline."""
        
        self.ativos_sucesso = []
        lista_fundamentalistas = []
        garch_vols = {}
        metricas_simples_list = []

        MIN_DIAS_HISTORICO_FLEXIVEL = max(180, int(MIN_DIAS_HISTORICO * 0.7))

        # A barra de progresso do Streamlit será controlada na 'aba_construtor_portfolio'
        # A 'tqdm' aqui serve apenas como log de console, se necessário, mas removemos 
        # para não poluir o log.
        for simbolo in simbolos:
            
            # 1. Carrega os 3 arquivos essenciais
            df_tecnicos = carregar_dados_ativo_gcs_csv(GCS_BASE_URL, simbolo, file_suffix='_tecnicos.csv')
            df_fundamentos = carregar_dados_ativo_gcs_csv(GCS_BASE_URL, simbolo, file_suffix='_fundamentos.csv')
            df_ml_results = carregar_dados_ativo_gcs_csv(GCS_BASE_URL, simbolo, file_suffix='_ml_results.csv')
            
            # 2. Validação Mínima
            if (df_tecnicos.empty or 'Close' not in df_tecnicos.columns or 
                len(df_tecnicos.dropna(subset=['Close', 'returns'])) < MIN_DIAS_HISTORICO_FLEXIVEL or 
                df_fundamentos.empty): 
                continue

            # --- 3. Extração e Preparação ---
            
            # Armazena o DF temporal (Close, returns, indicadores, targets, etc.)
            self.dados_por_ativo[simbolo] = df_tecnicos.dropna(how='all')
            self.ativos_sucesso.append(simbolo)
            
            # --- ALTERAÇÃO v8.6.1 ---
            # Extrai a linha única de fundamentos usando .loc[simbolo]
            # (O DataFrame agora é indexado por 'Ticker' (simbolo))
            if simbolo not in df_fundamentos.index:
                continue # Pula se o símbolo não for encontrado no índice
            fund_row = df_fundamentos.loc[simbolo] 
            fund_data = self._get_fundamental_metrics_from_df(fund_row)
            
            # 4. Adiciona dados ML à última linha do DataFrame Temporal
            if not df_ml_results.empty and simbolo in df_ml_results.index:
                # --- ALTERAÇÃO v8.6.1 ---
                # Usa .loc[simbolo]
                ml_row = df_ml_results.loc[simbolo] 
                # Tenta encontrar a coluna de proba mais longa (252d é o padrão de longo prazo)
                ml_proba_col = next((c for c in ml_row.index if c.startswith('ml_proba_') and not pd.isna(ml_row[c])), 'ml_proba_252d')
                
                # Assume a última data válida no DF técnico (garantindo que existe)
                if not self.dados_por_ativo[simbolo].empty:
                    last_index = self.dados_por_ativo[simbolo].index[-1]
                    
                    self.dados_por_ativo[simbolo].loc[last_index, 'ML_Proba'] = ml_row.get(ml_proba_col, 0.5)
                    
                    # CORRIGIDO: Não usar valor fixo (0.7). Usar np.nan (que será tratado como neutro/médio no score).
                    # A confiança AUC-ROC deve ser carregada do GCS, mas se não existir, usamos np.nan.
                    self.dados_por_ativo[simbolo].loc[last_index, 'ML_Confidence'] = ml_row.get('auc_roc_score_best_model', np.nan)
            
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
        
        # Anexa os dados estáticos (fundamentos) à última linha dos DFs temporais para fácil acesso
        for simbolo in self.ativos_sucesso:
            if simbolo in self.dados_fundamentalistas.index and not self.dados_por_ativo[simbolo].empty:
                last_index = self.dados_por_ativo[simbolo].index[-1]
                for col, value in self.dados_fundamentalistas.loc[simbolo].items():
                    # Garante que a coluna 'returns' não seja sobrescrita por 'annual_return'
                    if col not in ['annual_return', 'annual_volatility', 'max_drawdown']: 
                        # Garante que colunas existentes (como 'pe_ratio' série) não sejam sobrescritas
                        if col not in self.dados_por_ativo[simbolo].columns:
                            self.dados_por_ativo[simbolo].loc[last_index, col] = value
                        else:
                            # Se a coluna existe mas está toda nula, preenche o último valor
                            if self.dados_por_ativo[simbolo][col].isnull().all():
                                self.dados_por_ativo[simbolo].loc[last_index, col] = value
                
        return True

    def coletar_ativo_unico_gcs(self, ativo_selecionado: str) -> tuple[pd.DataFrame | None, dict | None]:
        """Coleta e retorna dados de um único ativo sob demanda (Aba 4)."""
        
        df_tecnicos = carregar_dados_ativo_gcs_csv(GCS_BASE_URL, ativo_selecionado, file_suffix='_tecnicos.csv')
        df_fundamentos = carregar_dados_ativo_gcs_csv(GCS_BASE_URL, ativo_selecionado, file_suffix='_fundamentos.csv')
        
        # Validação: Garante que o ticker carregado está no índice
        if df_tecnicos.empty or df_fundamentos.empty or 'Close' not in df_tecnicos.columns or ativo_selecionado not in df_fundamentos.index:
            return None, None
        
        # --- ALTERAÇÃO v8.6.1 ---
        # Usa .loc[ativo_selecionado] para buscar a linha de fundamentos
        fund_row = df_fundamentos.loc[ativo_selecionado]
        features_fund = self._get_fundamental_metrics_from_df(fund_row)
        
        df_ml_results = carregar_dados_ativo_gcs_csv(GCS_BASE_URL, ativo_selecionado, file_suffix='_ml_results.csv')
        
        if not df_tecnicos.empty:
            last_index = df_tecnicos.index[-1]
            
            # Adiciona features estáticas
            for key, value in features_fund.items():
                # Evita sobrescrever séries temporais com valores estáticos
                if key not in df_tecnicos.columns or df_tecnicos[key].isnull().all():
                     df_tecnicos.loc[last_index, key] = value
                
            # Adiciona ML
            if not df_ml_results.empty and ativo_selecionado in df_ml_results.index:
                # --- ALTERAÇÃO v8.6.1 ---
                # Usa .loc[ativo_selecionado]
                ml_row = df_ml_results.loc[ativo_selecionado] 
                ml_proba_col = next((c for c in ml_row.index if c.startswith('ml_proba_') and not pd.isna(ml_row[c])), 'ml_proba_252d')
                
                df_tecnicos.loc[last_index, 'ML_Proba'] = ml_row.get(ml_proba_col, 0.5)
                # CORRIGIDO: Não usar mock, usar np.nan se a coluna não for carregada.
                df_tecnicos.loc[last_index, 'ML_Confidence'] = ml_row.get('auc_roc_score_best_model', np.nan) 
        
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
        
        # Filtra os símbolos para garantir que são apenas os do IBOVESPA
        simbolos_filtrados = [s for s in simbolos if s in TODOS_ATIVOS]
        
        if not simbolos_filtrados: return False
        
        # A coleta de dados agora é feita sem a barra 'tqdm' interna 
        # para ser controlada pelo Streamlit
        if not coletor.coletar_e_processar_dados(simbolos_filtrados):
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
                'auc_roc_score': last_row.get('ML_Confidence', np.nan), # CORRIGIDO: usa np.nan se não carregado
                'model_name': 'Ensemble GCS (Pré-calculado)'
            }

    def pontuar_e_selecionar_ativos(self, horizonte_tempo: str):
        """
        Pontua e ranqueia ativos usando sistema multi-fator (Perf, Fund, Tech, ML) 
        e seleciona usando Clusterização KMeans/PCA.
        """
        
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
            # Usa a última linha válida
            last_valid_index = df.last_valid_index()
            if last_valid_index is not None:
                last_row = df.loc[last_valid_index].to_dict()
                last_metrics[asset] = last_row
            
        combinado = pd.DataFrame(last_metrics).T
        
        # 3. Garantia de Colunas
        # Colunas fundamentais e de performance que o GCS_Coletor já padronizou
        # (ex: 'pe_ratio', 'roe', 'sharpe_ratio', 'sector')
        # Colunas técnicas (ex: 'rsi_14', 'macd_diff')
        # Colunas ML (ex: 'ML_Proba', 'ML_Confidence')
        required_cols = [
            'sharpe_ratio', 'pe_ratio', 'roe', 'rsi_14', 'macd_diff', 
            'ML_Proba', 'ML_Confidence', 'sector'
        ]
        
        for col in required_cols:
             if col not in combinado.columns: 
                 # Tenta buscar do GCS (ex: 'fund_pe_ratio' se 'pe_ratio' falhou)
                 if col == 'pe_ratio' and 'fund_pe_ratio' in combinado.columns: 
                     combinado[col] = combinado['fund_pe_ratio']
                 elif col == 'roe' and 'fund_roe' in combinado.columns:
                     combinado[col] = combinado['fund_roe']
                 else:
                     combinado[col] = np.nan # Preenche com NaN se ausente
        
        # 4. Cálculo dos Scores (Com Explicações Detalhadas)
        
        # 4.1. Score de Performance (Sharpe)
        # Lógica: O Sharpe Ratio mede o retorno ajustado ao risco (Retorno / Volatilidade).
        # Um Sharpe maior é melhor (maior_melhor=True).
        # Este score é ponderado por WEIGHT_PERF.
        scores['performance_score'] = EngenheiroFeatures._normalizar(combinado['sharpe_ratio'], maior_melhor=True) * WEIGHT_PERF
        
        # 4.2. Score Fundamentalista (P/L e ROE)
        # Lógica: P/L (Preço/Lucro) é um múltiplo de valuation.
        # Um P/L mais baixo é geralmente considerado 'mais barato' (maior_melhor=False).
        pe_score = EngenheiroFeatures._normalizar(combinado['pe_ratio'].replace([np.inf, -np.inf], np.nan), maior_melhor=False)
        
        # Lógica: ROE (Return on Equity) é uma métrica de qualidade/rentabilidade.
        # Um ROE maior indica que a empresa gera mais lucro com seu patrimônio (maior_melhor=True).
        roe_score = EngenheiroFeatures._normalizar(combinado['roe'].replace([np.inf, -np.inf], np.nan), maior_melhor=True)
        
        # O score fundamentalista é a média do score de 'valor' (P/L) e 'qualidade' (ROE).
        scores['fundamental_score'] = (pe_score * 0.5 + roe_score * 0.5) * WEIGHT_FUND
        
        # 4.3. Score Técnico (RSI e MACD)
        # Lógica: RSI (Índice de Força Relativa) é um oscilador de momentum.
        # Valores altos (>70) indicam sobrecompra (sinal de venda/reversão).
        # Estamos buscando ativos *não* sobrecomprados, logo, um RSI menor é 'melhor' (maior_melhor=False).
        rsi_norm = EngenheiroFeatures._normalizar(combinado['rsi_14'], maior_melhor=False)
        
        # Lógica: MACD Histograma (macd_diff) mede a força da tendência.
        # Um histograma positivo e crescente indica momentum de alta (maior_melhor=True).
        macd_norm = EngenheiroFeatures._normalizar(combinado['macd_diff'], maior_melhor=True)
        
        # O score técnico é a média dos dois indicadores de momentum.
        scores['technical_score'] = (rsi_norm * 0.5 + macd_norm * 0.5) * WEIGHT_TECH

        # 4.4. Score de Machine Learning (Proba e Confiança)
        # Lógica: ML_Proba é a probabilidade do ativo superar sua mediana de retornos (gerada pelo 'gerador_financeiro.py').
        # Uma probabilidade maior é melhor (maior_melhor=True).
        ml_proba_norm = EngenheiroFeatures._normalizar(combinado['ML_Proba'], maior_melhor=True)
        
        # Lógica: ML_Confidence é o AUC-ROC do modelo, medindo sua capacidade de distinguir "altas" de "baixas".
        # Uma confiança maior (AUC > 0.5) é melhor (maior_melhor=True).
        # Se for NaN (não carregado), a normalização resultará em 0.5 (peso neutro).
        ml_confidence_norm = EngenheiroFeatures._normalizar(combinado['ML_Confidence'], maior_melhor=True)
        
        # O score de ML é uma média ponderada da *previsão* (60%) e da *confiança* no modelo (40%).
        scores['ml_score_weighted'] = (ml_proba_norm * 0.6 + ml_confidence_norm * 0.4) * final_ml_weight
        
        # 4.5. Score Total
        scores['total_score'] = scores['performance_score'] + scores['fundamental_score'] + scores['technical_score'] + scores['ml_score_weighted']
        
        # Junta os scores com os dados originais para a clusterização
        self.scores_combinados = scores.join(combinado, rsuffix='_combined').sort_values('total_score', ascending=False)
        
        
        # --- 5. SELEÇÃO FINAL (ESTRATÉGIA DE CLUSTERIZAÇÃO KMeans/PCA) ---
        
        # 5.1. Preparar Dados para Clusterização
        # Usando métricas de performance e fundamentos
        features_cluster = [
            'sharpe_ratio', 'annual_return', 'annual_volatility', 'max_drawdown', 
            'pe_ratio', 'pb_ratio', 'roe', 'debt_to_equity', 'revenue_growth',
            'div_yield', 'beta', 'garch_volatility'
        ]
        # Garante que as colunas existam no dataframe 'combinado'
        features_disponiveis = [col for col in features_cluster if col in self.scores_combinados.columns]
        df_para_cluster = self.scores_combinados[features_disponiveis]

        # 5.2. Executar Clusterização (encontrando K ótimo)
        if len(df_para_cluster) < 10: # Fallback se houver poucos ativos
            st.warning("Poucos ativos para clusterização. Usando seleção por Score Total.")
            self.ativos_selecionados = self.scores_combinados.head(NUM_ATIVOS_PORTFOLIO).index.tolist()
            return self.ativos_selecionados
            
        resultado_pca, pca, kmeans, optimal_k = AnalisadorIndividualAtivos.realizar_clusterizacao_pca(
            df_para_cluster, 
            max_clusters=min(10, len(df_para_cluster) - 1) # Limita o K máximo
        )
        
        if resultado_pca is None: # Fallback se clusterização falhar
            st.warning("Falha na clusterização (PCA/KMeans). Usando seleção por Score Total.")
            self.ativos_selecionados = self.scores_combinados.head(NUM_ATIVOS_PORTFOLIO).index.tolist()
            return self.ativos_selecionados

        # 5.3. Combinar Scores e Clusters
        df_final = self.scores_combinados.join(resultado_pca['Cluster'])
        
        # 5.4. Definir Threshold (Percentil 85, conforme solicitado)
        score_threshold = df_final['total_score'].quantile(SCORE_PERCENTILE_THRESHOLD)
        
        # 5.5. Filtrar Candidatos
        candidates_df = df_final[
            (df_final['total_score'] >= score_threshold) & 
            (df_final['Cluster'].notna())
        ].copy()
        
        if candidates_df.empty: # Fallback se o threshold for muito alto
            st.warning(f"Nenhum ativo atingiu o percentil {SCORE_PERCENTILE_THRESHOLD*100:.0f} de score. Usando o Top 5 geral.")
            self.ativos_selecionados = self.scores_combinados.head(NUM_ATIVOS_PORTFOLIO).index.tolist()
            return self.ativos_selecionados

        # 5.6. Ranquear por Cluster (Implementação da Lógica de Seleção)
        # Cria uma coluna 'cluster_rank' (0 para o melhor de cada cluster, 1 para o segundo, etc.)
        candidates_df['cluster_rank'] = candidates_df.groupby('Cluster')['total_score'].rank(method='first', ascending=False) - 1
        
        # Ordena primeiro pelo 'cluster_rank' (pegando todos os rank 0, depois todos os rank 1...)
        # e desempata pelo 'total_score' (o rank 0 do cluster A com score 0.9 vem antes do rank 0 do cluster B com score 0.8)
        candidates_df_sorted = candidates_df.sort_values(
            by=['cluster_rank', 'total_score'], 
            ascending=[True, False]
        )
        
        # 5.7. Selecionar o Top N
        self.ativos_selecionados = candidates_df_sorted.head(NUM_ATIVOS_PORTFOLIO).index.tolist()
        
        # 5.8. Fallback (Garantir que o portfólio tenha N ativos)
        if len(self.ativos_selecionados) < NUM_ATIVOS_PORTFOLIO:
            st.warning(f"Seleção por Cluster retornou {len(self.ativos_selecionados)} ativos. Completando com o Top Score geral.")
            ativos_faltantes = NUM_ATIVOS_PORTFOLIO - len(self.ativos_selecionados)
            
            # Pega os melhores do ranking geral que ainda não foram selecionados
            top_geral = self.scores_combinados.head(NUM_ATIVOS_PORTFOLIO * 2).index
            
            for ativo in top_geral:
                if ativo not in self.ativos_selecionados:
                    self.ativos_selecionados.append(ativo)
                    ativos_faltantes -= 1
                if ativos_faltantes == 0:
                    break
        
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
            
            # Usando os nomes das colunas após o processamento no ColetorDadosGCS
            pe_ratio = fund.get('pe_ratio', fund.get('fund_pe_ratio', np.nan))
            roe = fund.get('roe', fund.get('fund_roe', np.nan))
            justification.append(f"Fundamentos: P/L {pe_ratio:.2f}, ROE {roe*100:.2f}%") # ROE vem como decimal do GCS
            
            proba_up = self.predicoes_ml.get(simbolo, {}).get('predicted_proba_up', 0.5)
            # CORRIGIDO: usa np.nan se não carregado, o que será exibido como N/A
            auc_score = self.predicoes_ml.get(simbolo, {}).get('auc_roc_score', np.nan) 
            auc_str = f"{auc_score:.3f}" if not pd.isna(auc_score) else "N/A"
            justification.append(f"ML: Prob. Alta {proba_up*100:.1f}% (Confiança: {auc_str})")
            
            self.justificativas_selecao[simbolo] = " | ".join(justification)
        
        return self.justificativas_selecao
        
    def executar_pipeline(self, simbolos_customizados: list, perfil_inputs: dict, progress_bar=None) -> bool:
        """
        Executa o pipeline completo de construção de portfólio, 
        atualizando uma barra de progresso do Streamlit.
        """
        
        self.perfil_dashboard = perfil_inputs
        ml_lookback_days = perfil_inputs.get('ml_lookback_days', LOOKBACK_ML)
        nivel_risco = perfil_inputs.get('risk_level', 'MODERADO')
        horizonte_tempo = perfil_inputs.get('time_horizon', 'MÉDIO PRAZO')
        
        try:
            if progress_bar: progress_bar.progress(10, text="Coletando e processando dados do GCS...")
            if not self.coletar_e_processar_dados(simbolos_customizados): 
                st.error("Falha na coleta de dados. (executar_pipeline)")
                return False
            
            if progress_bar: progress_bar.progress(30, text="Calculando volatilidades (GARCH)...")
            self.calcular_volatilidades_garch()
            
            if progress_bar: progress_bar.progress(50, text="Carregando predições de Machine Learning...")
            # Otimização do ML removida do front-end
            self.treinar_modelos_ensemble(dias_lookback_ml=ml_lookback_days, otimizar=False) 
            
            if progress_bar: progress_bar.progress(70, text="Ranqueando e selecionando ativos (Cluster)...")
            self.pontuar_e_selecionar_ativos(horizonte_tempo=horizonte_tempo) 
            
            if progress_bar: progress_bar.progress(85, text="Otimizando alocação (MPT)...")
            self.alocacao_portfolio = self.otimizar_alocacao(nivel_risco=nivel_risco)
            
            if progress_bar: progress_bar.progress(95, text="Calculando métricas finais...")
            self.calcular_metricas_portfolio()
            self.gerar_justificativas()
            
            if progress_bar: progress_bar.progress(100, text="Pipeline concluído!")
            if progress_bar: time.sleep(1) # Pausa para ver a mensagem
        
        except Exception as e:
            st.error(f"Erro durante a execução do pipeline: {e}")
            import traceback
            st.code(traceback.format_exc())
            return False
            
        return True

# =============================================================================
# 12. CLASSE: ANALISADOR INDIVIDUAL DE ATIVOS (ADAPTADA COM PCA/KMEANS OTIMIZADO)
# =============================================================================

class AnalisadorIndividualAtivos:
    """Análise completa de ativos individuais com máximo de features."""
    
    @staticmethod
    def realizar_clusterizacao_pca(dados_ativos: pd.DataFrame, max_clusters: int = 10) -> tuple[pd.DataFrame | None, PCA | None, KMeans | None, int | None]:
        """
        Realiza clusterização K-means após PCA, encontrando o número ótimo de clusters (K)
        via Silhouette Score.
        """
        
        # Filtra as features importantes para a clusterização
        features_cluster = [
            'sharpe', 'retorno_anual', 'volatilidade_anual', 'max_drawdown', 
            'pe_ratio', 'pb_ratio', 'roe', 'debt_to_equity', 'revenue_growth',
            'div_yield', 'beta', 'garch_volatility'
        ]
        
        # Garante que as colunas existam e não sejam vazias
        features_numericas = dados_ativos.filter(items=features_cluster).select_dtypes(include=[np.number]).copy()
        
        # Limpeza de dados
        features_numericas = features_numericas.replace([np.inf, -np.inf], np.nan)
        features_numericas.rename(columns={'pe_ratio': 'P/L', 'roe': 'ROE'}, inplace=True) # Para visualização
        
        # Preenchimento de NaNs (Estratégia: Mediana)
        for col in features_numericas.columns:
            if features_numericas[col].isnull().any():
                median_val = features_numericas[col].median()
                features_numericas[col] = features_numericas[col].fillna(median_val)
        
        features_numericas = features_numericas.dropna(axis=1, how='all')
        # Remove colunas com variação próxima de zero
        features_numericas = features_numericas.loc[:, (features_numericas.std() > 1e-6)]

        # Condição de Falha: Precisa de pelo menos 3 ativos para clusterizar (K=2)
        if features_numericas.empty or len(features_numericas) < 3:
            return None, None, None, None
            
        scaler = StandardScaler()
        dados_normalizados = scaler.fit_transform(features_numericas)
        
        # PCA
        n_pca_components = min(3, len(features_numericas.columns))
        pca = PCA(n_components=n_pca_components)
        componentes_pca = pca.fit_transform(dados_normalizados)
        
        # --- Encontrar K Ótimo (Silhouette Score) ---
        best_score = -1
        optimal_k = 2 # K mínimo é 2
        
        # O K máximo não pode ser maior que N-1 amostras
        k_range = range(2, min(max_clusters + 1, len(features_numericas))) 

        if not k_range: # Se len(features_numericas) for 2 ou 3, k_range pode ser vazio ou pequeno
             if len(features_numericas) >= 2:
                 optimal_k = 2
             else:
                 return None, None, None, None # Impossível clusterizar
        else:
            for k in k_range:
                try:
                    kmeans_k = KMeans(n_clusters=k, random_state=42, n_init='auto')
                    clusters_k = kmeans_k.fit_predict(componentes_pca)
                    score = silhouette_score(componentes_pca, clusters_k)
                    
                    if score > best_score:
                        best_score = score
                        optimal_k = k
                except Exception:
                    continue # Ignora falhas em K específico
        
        # --- KMeans Final com K Ótimo ---
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto') 
        clusters = kmeans.fit_predict(componentes_pca)
        
        resultado_pca = pd.DataFrame(
            componentes_pca,
            columns=[f'PC{i+1}' for i in range(componentes_pca.shape[1])],
            index=features_numericas.index
        )
        resultado_pca['Cluster'] = clusters
        
        return resultado_pca, pca, kmeans, optimal_k

# =============================================================================
# 13. INTERFACE STREAMLIT - REESTRUTURADA COM NOVO TEMA
# =============================================================================

def configurar_pagina():
    """Configura página Streamlit com novo título e estilo (Neutro/Preto)."""
    st.set_page_config(
        page_title="Sistema de Portfólios Adaptativos",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Estilos CSS com tema Preto/Branco/Cinza e Abas Centralizadas
    st.markdown("""
        <style>
        /* Paleta de Cores (Tema Neutro/Preto) */
        :root {
            --primary-color: #000000; /* Preto */
            --secondary-color: #6c757d; /* Cinza secundário */
            --background-light: #ffffff; /* Fundo branco */
            --background-dark: #f8f9fa; /* Cinza muito claro */
            --text-color: #212529; /* Preto/Cinza escuro para texto */
            --text-color-light: #ffffff; /* Texto claro (para botões pretos) */
            --border-color: #dee2e6; /* Cinza claro para bordas */
        }
        
        body {
            background-color: var(--background-light);
            color: var(--text-color);
        }

        .main-header {
            font-family: 'Arial', sans-serif;
            color: var(--primary-color); /* Preto */
            text-align: center;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 10px;
            font-size: 2.2rem !important;
            margin-bottom: 20px;
            font-weight: 600;
        }

        /* Aplica a fonte Arial */
        .stButton button, .stDownloadButton button, .stFormSubmitButton button, 
        .stTabs [data-baseweb="tab"], .stMetric label, .main-header, .info-box,
        h1, h2, h3, h4, h5, p, body {
             font-family: 'Arial', sans-serif !important;
        }
        
        /* Botões Padrão (Contorno Preto) */
        .stButton button, .stDownloadButton button {
            border: 1px solid var(--primary-color) !important;
            color: var(--primary-color) !important;
            border-radius: 6px;
            padding: 8px 16px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            transition: all 0.3s ease;
            background-color: transparent !important;
        }
        .stButton button:hover, .stDownloadButton button:hover {
            background-color: var(--primary-color) !important;
            color: var(--text-color-light) !important;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        
        /* Botões Primários (Preenchimento Preto) */
        .stButton button[kind="primary"], .stFormSubmitButton button {
            background-color: var(--primary-color) !important;
            color: var(--text-color-light) !important;
            border: none !important;
        }
        .stButton button[kind="primary"]:hover, .stFormSubmitButton button:hover {
            background-color: #333333 !important; /* Um pouco mais claro no hover */
            color: var(--text-color-light) !important;
        }
        
        /* Estilo para TABS (Centralizadas) */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px; /* Espaçamento entre abas */
            border-bottom: 2px solid var(--border-color);
            display: flex; /* Obrigatório para centralizar */
            justify-content: center; /* CENTRALIZA AS ABAS */
            width: 100%; /* Ocupa a largura total */
        }
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            background-color: transparent;
            border-radius: 4px 4px 0 0;
            padding-top: 5px;
            padding-bottom: 5px;
            color: var(--secondary-color);
            font-weight: 500;
            flex-grow: 0 !important; /* Impede que as abas estiquem */
        }
        .stTabs [aria-selected="true"] {
            background-color: transparent;
            border-bottom: 2px solid var(--primary-color); /* Destaque Preto */
            color: var(--primary-color); /* Texto Preto */
            font-weight: 700;
        }
        
        /* Estilo para caixas de informação e métricas */
        .info-box {
            background-color: var(--background-dark);
            border-left: 4px solid var(--primary-color); /* Destaque Preto */
            padding: 15px;
            margin: 10px 0;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }
        .info-box h3 { margin-top: 0; }
        .info-box h4 { margin-top: 0; }
        
        .stMetric {
            padding: 10px 15px;
            background-color: var(--background-dark);
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 10px;
        }
        .stMetric label { font-weight: 600; color: var(--text-color); }
        .stMetric delta { font-weight: 700; color: #28a745; }
        .stMetric delta[style*="color: red"] { color: #dc3545 !important; }
        
        /* Barra de Progresso */
        .stProgress > div > div > div > div {
            background-color: var(--primary-color); /* Cor da barra preta */
        }
        
        /* Estilo para referências bibliográficas */
        .reference-block {
            background-color: #fdfdfd;
            border: 1px solid var(--border-color);
            padding: 12px;
            margin-bottom: 12px;
            border-radius: 6px;
        }
        .reference-block p {
            margin-bottom: 5px;
        }
        .reference-block .explanation {
            font-style: italic;
            color: var(--secondary-color);
            font-size: 0.95em;
            border-top: 1px dashed #e0e0e0;
            padding-top: 8px;
            margin-top: 8px;
        }
        
        </style>
    """, unsafe_allow_html=True)

def aba_introducao():
    """Aba 1: Introdução Metodológica Extensiva (v8.4.2)"""
    
    st.markdown("## 📚 Metodologia Quantitativa e Arquitetura do Sistema")
    
    st.markdown("""
    <div class="info-box">
    <h3>🎯 Visão Geral do Ecossistema</h3>
    <p>Este sistema opera em duas fases distintas e complementares: um <b>"Motor" de processamento de dados (offline)</b> e este <b>"Painel" de otimização (online)</b>. Esta arquitetura garante que o aplicativo Streamlit seja leve, rápido e não dependa de APIs de mercado em tempo real (como o `yfinance`), que podem ser lentas ou instáveis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # --- Coluna 1: O MOTOR ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('### 1. O "Motor" de Dados (`gerador_financeiro.py`)')
        st.markdown("""
        Este script Python (executado separadamente, ex: no Google Colab) é o responsável por todo o trabalho pesado de coleta, processamento e modelagem.
        """)
        
        with st.expander("Etapa 1.1: Coleta e Engenharia de Features"):
            st.markdown("""
            Para cada ativo do Ibovespa, o motor executa uma análise multifacetada:
            
            - **Análise Técnica:** Cálculo de dezenas de indicadores de *momentum* e tendência (RSI, MACD, Bandas de Bollinger, Médias Móveis, ADX, etc.).
            - **Análise Fundamentalista:** Coleta de métricas estáticas de *valuation* e *qualidade* (P/L, P/VP, ROE, ROIC, Margens, Crescimento de Receita).
            - **Análise Estatística:** Cálculo de métricas de risco/retorno (Sharpe Ratio, Max Drawdown) e, crucialmente, a volatilidade condicional futura através de modelos **GARCH(1,1)**, que capturam a "memória" da volatilidade de forma mais eficaz que o desvio padrão histórico.
            """)

        with st.expander("Etapa 1.2: Treinamento de Modelos de Machine Learning (ML)"):
            st.markdown("""
            O coração do sistema é um *Ensemble* de 5 modelos (LightGBM, XGBoost, CatBoost, RandomForest, SVC) treinado para cada ativo.
            
            **Objetivo (Target):**
            O modelo *não* tenta prever o preço (uma tarefa notoriamente difícil e ruidosa). Em vez disso, ele prevê a probabilidade do ativo ter um **desempenho superior à mediana móvel** dos seus próprios retornos futuros (ex: nos próximos 252 dias). Isso transforma o problema em uma classificação binária (performou acima/abaixo da mediana), que é mais robusta e menos suscetível a *outliers* extremos.
            
            **Prevenção de Overfitting (Sobreajuste):**
            - **Ensemble:** A média das previsões de 5 modelos diversos reduz a variância e previne que o sistema dependa do "viés" de um único algoritmo.
            - **Regularização:** A Otimização de Hiperparâmetros (HPO) com Optuna ajusta parâmetros de regularização (L1/L2) e restringe a complexidade (ex: `max_depth` baixo) para evitar que o modelo decore o ruído dos dados de treino.
            - **Feature Selection:** Um RandomForest inicial seleciona apenas as features mais importantes, reduzindo o ruído e a "maldição da dimensionalidade".
            
            **Prevenção de Underfitting (Subajuste):**
            - O HPO (Optuna) garante que os modelos tenham capacidade suficiente (ex: `num_leaves` ou `C` do SVC) para capturar os padrões reais, evitando uma simplificação excessiva que ignore sinais válidos.
            
            **Prevenção de Lookahead Bias e Data Leakage:**
            - **Viés de Olhar Futuro (Lookahead):** É evitado ao garantir que, no dia "D", o modelo só utilize dados disponíveis até "D". O *target* (calculado com dados futuros) é corretamente "deslocado" para o passado para servir como rótulo de treino, sem nunca ser usado como *feature*.
            - **Vazamento de Dados (Leakage):** A validação é feita com **Validação Cruzada Walk-Forward (WFCV)**, específica para séries temporais.
            
            *Exemplo de WFCV (Janela Deslizante):*
            | Fold | Dados de Treino | Dados de Teste |
            | :--- | :--- | :--- |
            | 1 | Dias 1-500 | Dias 501-560 |
            | 2 | Dias 61-560 | Dias 561-620 |
            | 3 | Dias 121-620 | Dias 621-680 |
            
            Isso simula realisticamente o ato de treinar o modelo em dados passados e prever o futuro imediato, sem nunca "contaminar" o treino com informações futuras.
            """)
        
        with st.expander("Etapa 1.3: Exportação (GCS)"):
             st.markdown(f"""
            O script salva os resultados em 4 arquivos CSV distintos por ativo no bucket `{GCS_BUCKET_NAME}`:
            
            - **`[TICKER]_tecnicos.csv`**: A série temporal completa (OHLCV, Retornos, RSI, MACD, etc.).
            - **`[TICKER]_fundamentos.csv`**: Uma *única linha* estática (P/L, ROE, Sharpe, Setor, Vol. GARCH).
            - **`[TICKER]_ml_results.csv`**: Uma *única linha* com a probabilidade final do *Ensemble* de ML.
            - **`[TICKER]_ml_metadata.csv`**: Logs detalhados do treino (não usados pelo painel).
            """)

    # --- Coluna 2: O PAINEL ---
    with col2:
        st.markdown('### 2. O "Painel" de Otimização (Este Aplicativo)')
        st.markdown("""
        Este painel Streamlit consome os dados pré-processados pelo "Motor" para montar o portfólio ideal para o perfil do usuário.
        """)
        
        with st.expander("Etapa 2.1: Definição do Perfil e Carga de Dados"):
            st.markdown("""
            O aplicativo primeiro lê os arquivos CSV do GCS (com base na seleção de ativos da Aba 2) e, em seguida, solicita que o usuário preencha o questionário (Aba 3). As respostas definem duas variáveis críticas:
            
            1.  **Nível de Risco:** (Conservador, Moderado, etc.)
            2.  **Horizonte Temporal:** (Curto, Médio, Longo Prazo)
            """)

        with st.expander("Etapa 2.2: Ranqueamento Multi-Fatorial"):
            st.markdown("""
            O sistema calcula um **Score Total** para cada ativo combinando quatro pilares. O **Horizonte Temporal** do seu perfil define os pesos de cada pilar:
            
            | Pilar | O que mede? | Peso (Longo Prazo) | Peso (Curto Prazo) |
            | :--- | :--- | :--- | :--- |
            | **Performance** | Sharpe Ratio (Risco/Retorno) | Médio | Alto |
            | **Fundamentos** | P/L e ROE (Valor/Qualidade) | **Alto** | Baixo |
            | **Técnicos** | RSI e MACD (Momentum) | Baixo | **Alto** |
            | **Machine Learning**| Probabilidade de Alta (Sinal) | Médio | Médio |
            
            Os ativos são então filtrados, mantendo-se apenas aqueles com **Score Total acima do Percentil 85** (ou seja, os 15% melhores do universo analisado).
            """)
            
        with st.expander("Etapa 2.3: Seleção por Cluster (KMeans/PCA)"):
            st.markdown("""
            Para garantir a diversificação, os ativos filtrados (acima do Percentil 85) são agrupados por similaridade usando **KMeans/PCA**.
            
            1.  **PCA (Principal Component Analysis):** Reduz dezenas de métricas (P/L, ROE, Sharpe, Volatilidade) em 2 ou 3 "Componentes Principais", capturando a "essência" financeira de cada ativo.
            2.  **KMeans:** Agrupa (clusteriza) os ativos com base nesses componentes. O número de clusters é definido dinamicamente (usando *Silhouette Score*) para encontrar os agrupamentos mais naturais.
            
            O sistema então seleciona os 5 ativos finais (para o portfólio) seguindo esta regra:
            
            - **Rank 1:** Pega o ativo com **maior Score Total** de cada cluster.
            - **Rank 2:** Se ainda faltarem ativos para completar os 5, ele pega o **segundo melhor** ativo de cada cluster (priorizando pelo maior score).
            - **...e assim por diante** até 5 ativos serem selecionados.
            
            Essa abordagem garante um portfólio diversificado, selecionando os "campeões" de diferentes perfis financeiros (ex: 1 ativo de "Crescimento", 1 de "Valor", 1 de "Baixa Volatilidade", etc.).
            """)
            
        with st.expander("Etapa 2.4: Otimização (Teoria Moderna de Portfólio - MPT)"):
            st.markdown("""
            Após selecionar os 5 melhores e mais diversos ativos, o sistema usa a **Teoria Moderna de Portfólio (MPT)** de Harry Markowitz para definir o peso (percentual) de cada ativo na carteira.
            
            O objetivo é encontrar a carteira na "Fronteira Eficiente" – o ponto que oferece o maior retorno esperado para um dado nível de risco (volatilidade).
            
            O **Nível de Risco** do seu perfil determina *qual* ponto da fronteira o sistema irá buscar:
            
            - **Conservador/Intermediário:** Busca a carteira de **Mínima Volatilidade (MinVolatility)**. Foco total em reduzir o risco.
            - **Moderado:** Busca a carteira com o **Máximo Sharpe Ratio (MaxSharpe)**. O melhor equilíbrio entre risco e retorno.
            - **Avançado:** Otimiza usando **CVaR (Conditional Value at Risk)**. Foca em minimizar a perda média nos piores cenários (ex: 5% piores dias), buscando proteção contra "eventos de cauda".
            """)

    st.markdown("---")
    st.info("""
    **Próxima Etapa:**
    Utilize o menu de abas para navegar até **'Seleção de Ativos'** e, em seguida, **'Construtor de Portfólio'** para gerar sua alocação otimizada.
    """)

def aba_selecao_ativos():
    """Aba 2: Seleção de Ativos (Design e Textos ajustados)"""
    
    st.markdown("## 🎯 Definição do Universo de Análise")
    
    st.markdown("""
    <div class="info-box">
    <p>O universo de análise está restrito ao **Índice Ibovespa**. O sistema utiliza todos os ativos selecionados para realizar o ranqueamento multi-fatorial e otimizar a carteira.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Opções de Seleção
    modo_selecao = st.radio(
        "**Modo de Seleção:**",
        [
            "📊 Índice de Referência (Todos do Ibovespa)",
            "🏢 Seleção Setorial",
            "✍️ Seleção Individual"
        ],
        index=0,
        key='selection_mode_radio_v8' # Chave única
    )
    
    ativos_selecionados = []
    
    if "Índice de Referência" in modo_selecao:
        ativos_selecionados = TODOS_ATIVOS.copy()
        st.success(f"✓ **{len(ativos_selecionados)} ativos** (Ibovespa completo) definidos para análise.")
        
        with st.expander("📋 Visualizar Tickers"):
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
    
    elif "Seleção Setorial" in modo_selecao:
        st.markdown("### 🏢 Seleção por Setor")
        setores_disponiveis = sorted(list(ATIVOS_POR_SETOR.keys()))
        col1, col2 = st.columns([2, 1])
        
        with col1:
            setores_selecionados = st.multiselect(
                "Escolha um ou mais setores:",
                options=setores_disponiveis,
                default=setores_disponiveis[:3] if setores_disponiveis else [],
                key='setores_multiselect_v8' # Chave única
            )
        
        if setores_selecionados:
            for setor in setores_selecionados: ativos_selecionados.extend(ATIVOS_POR_SETOR[setor])
            ativos_selecionados = list(set(ativos_selecionados))
            
            with col2:
                st.metric("Setores", len(setores_selecionados))
                st.metric("Total de Ativos", len(ativos_selecionados))
            
            with st.expander("📋 Visualizar Ativos por Setor"):
                for setor in setores_selecionados:
                    ativos_do_setor = ATIVOS_POR_SETOR.get(setor, [])
                    st.markdown(f"**{setor}** ({len(ativos_do_setor)} ativos)")
                    st.write(", ".join([a.replace('.SA', '') for a in ativos_do_setor]))
        else:
            st.warning("⚠️ Selecione pelo menos um setor.")
    
    elif "Seleção Individual" in modo_selecao:
        st.markdown("### ✍️ Seleção Individual de Tickers")
        
        ativos_com_setor = {}
        for setor, ativos in ATIVOS_POR_SETOR.items():
            for ativo in ativos: ativos_com_setor[ativo] = setor
        
        todos_tickers_ibov = sorted(list(ativos_com_setor.keys()))
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("#### 📝 Selecione Tickers (Ibovespa)")
            ativos_selecionados = st.multiselect(
                "Pesquise e selecione os tickers:",
                options=todos_tickers_ibov,
                format_func=lambda x: f"{x.replace('.SA', '')} - {ativos_com_setor.get(x, 'Desconhecido')}",
                key='ativos_individuais_multiselect_v8' # Chave única
            )
        
        with col2:
            st.metric("Tickers Selecionados", len(ativos_selecionados))

        if not ativos_selecionados:
            st.warning("⚠️ Nenhum ativo definido.")
    
    if ativos_selecionados:
        st.session_state.ativos_para_analise = ativos_selecionados
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Tickers Definidos", len(ativos_selecionados))
        col2.metric("Para Ranqueamento", len(ativos_selecionados))
        col3.metric("Carteira Final", NUM_ATIVOS_PORTFOLIO)
        
        st.success("✓ Definição concluída. Prossiga para a aba **'Construtor de Portfólio'**.")
    else:
        st.warning("⚠️ O universo de análise está vazio.")

# =============================================================================
# Aba 3: Questionário e Construção de Portfólio (COM BARRA DE PROGRESSO)
# =============================================================================

def aba_construtor_portfolio():
    """Aba 3: Questionário e Construção de Portfólio"""
    
    if 'ativos_para_analise' not in st.session_state or not st.session_state.ativos_para_analise:
        st.warning("⚠️ Por favor, defina o universo de análise na aba **'Seleção de Ativos'** primeiro.")
        return
    
    if 'builder' not in st.session_state: st.session_state.builder = None
    if 'profile' not in st.session_state: st.session_state.profile = {}
    if 'builder_complete' not in st.session_state: st.session_state.builder_complete = False
    
    # Placeholder para a barra de progresso
    progress_bar_placeholder = st.empty()
    
    # FASE 1: QUESTIONÁRIO
    if not st.session_state.builder_complete:
        st.markdown('## 📋 Calibração do Perfil de Risco')
        
        st.info(f"✓ **{len(st.session_state.ativos_para_analise)} ativos** prontos. Responda o questionário para calibrar a otimização.")
        
        col_question1, col_question2 = st.columns(2)
        
        with st.form("investor_profile_form_v8_6", clear_on_submit=False): 
            
            with col_question1:
                st.markdown("#### Tolerância ao Risco")
                
                # Pergunta 1 (Concorda)
                p2_risk_desc = st.radio(
                    "**1. Tolerância à Volatilidade:** Como você se sente sobre flutuações significativas (ex: quedas de 15-20%) no valor do seu portfólio em um único ano?", 
                    options=OPTIONS_CONCORDA, index=2, key='risk_accept_radio_v8_q1'
                )
                
                # Pergunta 2 (Concorda)
                p3_gain_desc = st.radio(
                    "**2. Foco em Retorno Máximo:** Meu objetivo principal é maximizar o retorno, mesmo que isso signifique assumir riscos substancialmente maiores.", 
                    options=OPTIONS_CONCORDA, index=2, key='max_gain_radio_v8_q2'
                )
                
                # Pergunta 3 (Discorda - Invertido)
                p4_stable_desc = st.radio(
                    "**3. Prioridade de Estabilidade:** Priorizo a estabilidade e a preservação do meu capital acima do potencial de grandes ganhos.", 
                    options=OPTIONS_DISCORDA, index=2, key='stable_growth_radio_v8_q3'
                )
                
                # Pergunta 4 (Discorda - Invertido)
                p5_loss_desc = st.radio(
                    "**4. Aversão à Perda:** A prevenção de perdas de curto prazo é mais crítica para mim do que a busca por crescimento acelerado no longo prazo.", 
                    options=OPTIONS_DISCORDA, index=2, key='avoid_loss_radio_v8_q4'
                )
                
                # Pergunta 5 (Reação)
                p511_reaction_desc = st.radio(
                    "**5. Reação a Queda de 10%:** Se um ativo em sua carteira caísse 10% rapidamente, qual seria sua reação mais provável?", 
                    options=OPTIONS_REACTION_DETALHADA, index=1, key='reaction_radio_v8_q5'
                )
                
                # Pergunta 6 (Conhecimento)
                p_level_desc = st.radio(
                    "**6. Nível de Conhecimento:** Qual seu nível de conhecimento sobre o mercado financeiro e tipos de investimento?", 
                    options=OPTIONS_CONHECIMENTO_DETALHADA, index=1, key='level_radio_v8_q6'
                )
            
            with col_question2:
                st.markdown("#### Horizonte e Capital")
                
                # Pergunta 7 (Horizonte)
                p211_time_desc = st.radio(
                    "**7. Horizonte de Investimento:** Por quanto tempo você pretende manter este investimento antes de precisar de uma reavaliação estratégica ou do capital?", 
                    options=OPTIONS_TIME_HORIZON_DETALHADA, index=2, key='time_purpose_radio_v8_q7'
                )
                
                # Pergunta 8 (Liquidez)
                p311_liquid_desc = st.radio(
                    "**8. Necessidade de Liquidez:** Qual é o prazo mínimo que você pode garantir que *não* precisará resgatar este capital?", 
                    options=OPTIONS_LIQUIDEZ_DETALHADA, index=2, key='liquidity_radio_v8_q8'
                )
                
                st.markdown("---")
                investment = st.number_input(
                    "Capital Total a ser Alocado (R$)",
                    min_value=1000, max_value=10000000, value=100000, step=10000, key='investment_amount_input_v8'
                )
            
            submitted = st.form_submit_button("🚀 Gerar Alocação Otimizada", type="primary", key='submit_optimization_button_v8')
            
            if submitted:
                # 1. Mapeia respostas descritivas para as chaves de pontuação ORIGINAIS
                risk_answers_originais = {
                    'risk_accept': MAP_CONCORDA.get(p2_risk_desc, 'N: Neutro'),
                    'max_gain': MAP_CONCORDA.get(p3_gain_desc, 'N: Neutro'),
                    'stable_growth': MAP_DISCORDA.get(p4_stable_desc, 'N: Neutro'),
                    'avoid_loss': MAP_DISCORDA.get(p5_loss_desc, 'N: Neutro'),
                    'reaction': MAP_REACTION.get(p511_reaction_desc, 'B: Manteria e reavaliaria a tese'),
                    'level': MAP_CONHECIMENTO.get(p_level_desc, 'B: Intermediário (Conhecimento básico sobre mercados e ativos)'),
                    'time_purpose': p211_time_desc, # A lógica de horizonte já usa a string (A, B, C)
                    'liquidity': p311_liquid_desc,  # A lógica de horizonte já usa a string (A, B, C)
                }
                
                # 2. Analisa perfil (usando a lógica de pontuação original e inalterada)
                analyzer = AnalisadorPerfilInvestidor()
                risk_level, horizon, lookback, score = analyzer.calcular_perfil(risk_answers_originais)
                
                st.session_state.profile = {
                    'risk_level': risk_level, 'time_horizon': horizon, 'ml_lookback_days': lookback, 'risk_score': score
                }
                
                # 3. Cria construtor
                try:
                    builder_local = ConstrutorPortfolioAutoML(investment)
                    st.session_state.builder = builder_local
                except Exception as e:
                    st.error(f"Erro fatal ao inicializar o construtor do portfólio: {e}")
                    return

                # 4. Executa pipeline com BARRA DE PROGRESSO
                # Substitui o st.spinner
                progress_widget = progress_bar_placeholder.progress(0, text=f"Iniciando pipeline para PERFIL {risk_level}...")
                
                success = builder_local.executar_pipeline(
                    simbolos_customizados=st.session_state.ativos_para_analise,
                    perfil_inputs=st.session_state.profile,
                    progress_bar=progress_widget # Passa o objeto da barra
                )
                
                # Limpa a barra de progresso
                progress_bar_placeholder.empty()
                    
                if not success:
                    st.error("Falha na aquisição ou processamento dos dados. Verifique a disponibilidade dos arquivos CSV no GCS ou se há ativos suficientes.")
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
        
        if st.button("🔄 Recalibrar Perfil e Otimizar", key='recomecar_analysis_button_v8'):
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
                    {'Ativo': a.replace('.SA', ''), 'Peso (%)': allocation[a]['weight'] * 100}
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
                        colorscale='Greys', # Escala de cinza
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

                # Usando cores do template (preto e cinza)
                template_colors = obter_template_grafico()['colorway']
                
                fig_garch.add_trace(go.Bar(name='Volatilidade Histórica', x=plot_df_garch['Ticker'], y=plot_df_garch['Vol. Histórica (%)'], marker=dict(color=template_colors[2]), opacity=0.7)) # Cinza
                fig_garch.add_trace(go.Bar(name='Volatilidade Condicional', x=plot_df_garch['Ticker'], y=plot_df_garch['Vol. Condicional (%)'], marker=dict(color=template_colors[0]))) # Preto
                
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
            st.markdown(f"**Estratégia de Seleção:** Clusterização (KMeans/PCA) com threshold de Score no Percentil {SCORE_PERCENTILE_THRESHOLD*100:.0f}.")
            st.markdown("---")
            
            # Ajusta colunas para usar macd_diff (diferença) e pe/roe simples
            cols_to_display_scores = [
                'total_score', 'performance_score', 'fundamental_score', 'technical_score', 'ml_score_weighted', 
                'sharpe_ratio', 'pe_ratio', 'roe', 'rsi_14', 'macd_diff', 'ML_Proba'
            ]
            # Garante que apenas colunas existentes sejam selecionadas
            cols_existentes = [col for col in cols_to_display_scores if col in builder.scores_combinados.columns]
            
            df_scores_display = builder.scores_combinados[cols_existentes].copy()
            df_scores_display.columns = [
                'Score Total', 'Score Perf.', 'Score Fund.', 'Score Téc.', 'Score ML', 
                'Sharpe', 'P/L', 'ROE', 'RSI 14', 'MACD Hist.', 'Prob. Alta ML'
            ]
            # Converte ROE para percentual
            if 'ROE' in df_scores_display.columns:
                 df_scores_display['ROE'] = df_scores_display['ROE'] * 100
                 
            df_scores_display = df_scores_display.iloc[:15] # Mostra o Top 15
            
            st.markdown("##### Ranqueamento Ponderado Multi-Fatorial (Top 15 Tickers do Universo Analisado)")
            st.dataframe(df_scores_display.style.format(
                {
                    'Score Total': '{:.3f}', 'Score Perf.': '{:.3f}', 'Score Fund.': '{:.3f}', 'Score Téc.': '{:.3f}', 'Score ML': '{:.3f}',
                    'Sharpe': '{:.3f}', 'P/L': '{:.2f}', 'ROE': '{:.2f}%', 'RSI 14': '{:.2f}', 'MACD Hist.': '{:.4f}', 'Prob. Alta ML': '{:.2f}'
                }
            ).background_gradient(cmap='Greys', subset=['Score Total']), use_container_width=True)
            
            st.markdown("---")
            st.markdown('##### Resumo da Seleção de Ativos (Portfólio Final)')
            
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
# Aba 4: Análise Individual (Mantido)
# =============================================================================

def aba_analise_individual():
    """Aba 4: Análise Individual de Ativos - Usando dados pré-carregados ou sob demanda do GCS."""
    
    st.markdown("## 🔍 Análise de Fatores por Ticker")
    
    if 'ativos_para_analise' in st.session_state and st.session_state.ativos_para_analise:
        ativos_disponiveis = sorted(list(set(st.session_state.ativos_para_analise)))
    else:
        ativos_disponiveis = TODOS_ATIVOS 
            
    if not ativos_disponiveis:
        st.error("Nenhum ativo disponível. Verifique a seleção ou o universo padrão.")
        return

    # Garante que o estado individual_asset_select é válido
    if 'individual_asset_select_v8' not in st.session_state or st.session_state.individual_asset_select_v8 not in ativos_disponiveis:
        st.session_state.individual_asset_select_v8 = ativos_disponiveis[0] if ativos_disponiveis else None

    col1, col2 = st.columns([3, 1])
    
    with col1:
        ativo_selecionado = st.selectbox(
            "Selecione um ticker para análise detalhada:",
            options=ativos_disponiveis,
            format_func=lambda x: x.replace('.SA', '') if isinstance(x, str) else x,
            key='individual_asset_select_v8' # Chave única
        )
    
    with col2:
        # Botão de análise agora é "primary" (preenchimento preto)
        if st.button("🔄 Executar Análise", key='analyze_asset_button_v8', type="primary"):
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
                
                # Puxa dos dados fundamentalistas consolidados do builder
                if ativo_selecionado in builder.dados_fundamentalistas.index:
                    features_fund = builder.dados_fundamentalistas.loc[ativo_selecionado].to_dict()
                else:
                    # Fallback se não estiver nos dados consolidados (deve estar)
                    _, features_fund = ColetorDadosGCS().coletar_ativo_unico_gcs(ativo_selecionado)

            # 2. Se falhar ou não houver cache, coleta sob demanda
            if df_completo is None or df_completo.empty or features_fund is None:
                df_completo, features_fund = ColetorDadosGCS().coletar_ativo_unico_gcs(ativo_selecionado)
                if df_completo is not None: df_completo = df_completo.dropna(how='all')

            if df_completo is None or df_completo.empty or 'Close' not in df_completo.columns or features_fund is None:
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
                variacao_dia = df_completo['returns'].iloc[-1] * 100 if 'returns' in df_completo.columns and not df_completo['returns'].empty else 0.0
                volume_medio = df_completo['Volume'].mean() if 'Volume' in df_completo.columns else 0.0
                
                col1.metric("Preço de Fechamento", f"R$ {preco_atual:.2f}", f"{variacao_dia:+.2f}%")
                col2.metric("Volume Médio Recente", f"{volume_medio:,.0f}")
                col3.metric("Setor", features_fund.get('sector', 'N/A'))
                col4.metric("Indústria", features_fund.get('industry', 'N/A'))
                col5.metric("Vol. Anualizada", f"{features_fund.get('annual_volatility', np.nan)*100:.2f}%" if not pd.isna(features_fund.get('annual_volatility')) else "N/A")
                
                if not df_completo.empty and 'Open' in df_completo.columns and 'Volume' in df_completo.columns:
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                    
                    fig.add_trace(go.Candlestick(x=df_completo.index, open=df_completo['Open'], high=df_completo['High'], low=df_completo['Low'], close=df_completo['Close'], name='Preço'), row=1, col=1)
                    
                    # Volume com cor cinza
                    template_colors = obter_template_grafico()['colorway']
                    fig.add_trace(go.Bar(x=df_completo.index, y=df_completo['Volume'], name='Volume', marker=dict(color=template_colors[2]), opacity=0.7), row=2, col=1)
                    
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
                col3.metric("ROE (Rentabilidade)", f"{features_fund.get('roe', np.nan)*100:.2f}%" if not pd.isna(features_fund.get('roe')) else "N/A") # GCS ROE é decimal
                col4.metric("Margem Operacional", f"{features_fund.get('operating_margin', np.nan)*100:.2f}%" if not pd.isna(features_fund.get('operating_margin')) else "N/A") # GCS Margem é decimal
                col5.metric("Cresc. Receita Anual", f"{features_fund.get('revenue_growth', np.nan)*100:.2f}%" if not pd.isna(features_fund.get('revenue_growth')) else "N/A") # GCS Cresc. é decimal
                
                st.markdown("#### Saúde Financeira e Dividendo")
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Dívida/Patrimônio", f"{features_fund.get('debt_to_equity', np.nan):.2f}" if not pd.isna(features_fund.get('debt_to_equity')) else "N/A")
                col2.metric("Current Ratio", f"{features_fund.get('current_ratio', np.nan):.2f}" if not pd.isna(features_fund.get('current_ratio')) else "N/A")
                col3.metric("Dividend Yield", f"{features_fund.get('div_yield', np.nan):.2f}%" if not pd.isna(features_fund.get('div_yield')) else "N/A") # GCS DY é percentual
                col4.metric("Beta (Risco Sistêmico)", f"{features_fund.get('beta', np.nan):.2f}" if not pd.isna(features_fund.get('beta')) else "N/A")
                
                st.markdown("---")
                st.markdown("#### Tabela de Fatores Fundamentais (GCS)")
                
                # Exclui chaves que são mapeadas para nomes simples
                keys_to_exclude = ['pe_ratio', 'roe'] 
                df_fund_display = pd.DataFrame({
                    'Métrica': [k for k in features_fund.keys() if k not in keys_to_exclude],
                    'Valor': [f"{v:.4f}" if isinstance(v, (int, float)) and not pd.isna(v) else str(v) 
                              for k, v in features_fund.items() if k not in keys_to_exclude]
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
                template_colors = obter_template_grafico()['colorway']

                if 'rsi_14' in df_completo.columns:
                    fig_osc.add_trace(go.Scatter(x=df_completo.index, y=df_completo['rsi_14'], name='RSI', line=dict(color=template_colors[0])), row=1, col=1) # Cor Primária (Preto)
                    fig_osc.add_hline(y=70, line_dash="dash", line_color="#dc3545", row=1, col=1)
                    fig_osc.add_hline(y=30, line_dash="dash", line_color="#28a745", row=1, col=1)
                
                if 'macd' in df_completo.columns and 'macd_signal' in df_completo.columns:
                    fig_osc.add_trace(go.Scatter(x=df_completo.index, y=df_completo['macd'], name='MACD', line=dict(color=template_colors[1])), row=2, col=1) # Cor Cinza Escuro
                    fig_osc.add_trace(go.Scatter(x=df_completo.index, y=df_completo['macd_signal'], name='Signal', line=dict(color=template_colors[3])), row=2, col=1) # Cor Cinza Claro
                    # Coluna macd_diff (Histograma)
                    if 'macd_diff' in df_completo.columns:
                        fig_osc.add_trace(go.Bar(x=df_completo.index, y=df_completo['macd_diff'], name='Histograma', marker=dict(color='#e9ecef')), row=2, col=1)
                
                fig_layout = obter_template_grafico()
                fig_layout['height'] = 550
                fig_osc.update_layout(**fig_layout)
                
                st.plotly_chart(fig_osc, use_container_width=True)

            with tab4:
                st.markdown("### Análise de Similaridade e Clusterização")
                
                if not builder_existe or builder.metricas_performance.empty or builder.dados_fundamentalistas.empty:
                    st.warning("A Clusterização está desabilitada. É necessário executar o **'Construtor de Portfólio'** (Aba 3) para carregar os dados de comparação de múltiplos ativos.")
                    return
                
                # Combina métricas de performance e fundamentos
                df_comparacao = builder.metricas_performance.join(builder.dados_fundamentalistas, how='inner', rsuffix='_fund')
                
                # Renomeia as colunas de P/L e ROE para usar as chaves simples
                if 'pe_ratio' not in df_comparacao.columns and 'fund_pe_ratio' in df_comparacao.columns:
                    df_comparacao['pe_ratio'] = df_comparacao['fund_pe_ratio']
                if 'roe' not in df_comparacao.columns and 'fund_roe' in df_comparacao.columns:
                    df_comparacao['roe'] = df_comparacao['fund_roe']
                
                if len(df_comparacao) < 10:
                    st.warning("Dados insuficientes para realizar a clusterização (menos de 10 ativos com métricas completas).")
                    return
                    
                # Chama a função de clusterização (que agora usa a lógica completa)
                resultado_pca, pca, kmeans, optimal_k = AnalisadorIndividualAtivos.realizar_clusterizacao_pca(
                    df_comparacao, 
                    max_clusters=min(10, len(df_comparacao) - 1)
                )
                
                if resultado_pca is not None:
                    st.info(f"Análise de clusterização encontrou **{optimal_k} clusters** ótimos (via Silhouette Score).")
                    
                    # Garantir que o nome do ativo seja desformatado para hover
                    hover_names = resultado_pca.index.str.replace('.SA', '')

                    # Usando a paleta de cores neutra
                    template_colors = obter_template_grafico()['colorway']

                    if 'PC3' in resultado_pca.columns:
                        fig_pca = px.scatter_3d(
                            resultado_pca, 
                            x='PC1', y='PC2', z='PC3', 
                            color=resultado_pca['Cluster'].astype(str), # Converte cluster para string para cor categórica
                            hover_name=hover_names, 
                            title='Similaridade de Tickers (PCA/K-means - 3D)',
                            color_discrete_sequence=template_colors # Paleta Neutra
                        )
                    else:
                        fig_pca = px.scatter(
                            resultado_pca, 
                            x='PC1', y='PC2', 
                            color=resultado_pca['Cluster'].astype(str), # Converte cluster para string para cor categórica
                            hover_name=hover_names, 
                            title='Similaridade de Tickers (PCA/K-means - 2D)',
                            color_discrete_sequence=template_colors # Paleta Neutra
                        )
                    
                    fig_pca.update_layout(**obter_template_grafico(), height=600)
                    st.plotly_chart(fig_pca, use_container_width=True)
                    
                    if ativo_selecionado in resultado_pca.index:
                        cluster_ativo = resultado_pca.loc[ativo_selecionado, 'Cluster']
                        ativos_similares_df = resultado_pca[resultado_pca['Cluster'] == cluster_ativo]
                        ativos_similares = [a for a in ativos_similares_df.index.tolist() if a != ativo_selecionado]
                        
                        st.success(f"**{ativo_selecionado.replace('.SA', '')}** pertence ao **Cluster {cluster_ativo}**")
                        
                        if ativos_similares:
                            st.markdown(f"#### Tickers Similares no Cluster {cluster_ativo}:")
                            st.write(", ".join([a.replace('.SA', '') for a in ativos_similares]))
                        else:
                            st.info("Nenhum outro ticker similar encontrado neste cluster.")

                else:
                    st.warning("Não foi possível realizar a clusterização (erro de dimensionalidade ou dados nulos).")
        
        except Exception as e:
            st.error(f"Erro ao analisar o ticker {ativo_selecionado}: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# =============================================================================
# Aba 5: Referências (NOVA)
# =============================================================================

def aba_referencias():
    """Aba 5: Referências Bibliográficas das Ementas."""
    
    st.markdown("## 📚 Referências e Bibliografia")
    st.markdown("Esta seção consolida as referências bibliográficas indicadas nas ementas das disciplinas relacionadas (GRDECO222 e GRDECO203).")

    st.markdown("---")
    
    st.markdown("### GRDECO222: Machine Learning (Prof. Rafael Martins de Souza)")
    
    st.markdown("#### Bibliografia Obrigatória")
    st.markdown(
        """
        <div class="reference-block">
            <p><strong>1. Jupter Notebooks apresentados em sala de aula.</strong></p>
            <p class="explanation">
            Explicação: O material principal do curso é prático, baseado nos códigos e exemplos desenvolvidos
            pelo professor durante as aulas.
            </p>
        </div>
        <div class="reference-block">
            <p><strong>2. Géron, A. Mãos à Obra: Aprendizado de Máquina com Scikit-Learn, Keras e TensorFlow.</strong></p>
            <p class="explanation">
            Explicação: Considerado um dos principais livros-texto práticos sobre Machine Learning.
            Cobre desde os fundamentos (Regressão, SVMs, Árvores de Decisão) até tópicos avançados
            de Deep Learning, com foco na implementação usando bibliotecas Python populares.
            </p>
        </div>
        """, unsafe_allow_html=True
    )
    
    st.markdown("#### Bibliografia Complementar")
    st.markdown(
        """
        <div class="reference-block">
            <p><strong>1. Coleman, C., Spencer Lyon, S., Jesse Perla, J. QuantEcon Data Science, Introduction to Economic Modeling and Data Science. (https://datascience.quantecon.org/)</strong></p>
            <p class="explanation">
            Explicação: Um recurso online focado na aplicação de Ciência de Dados especificamente
            para modelagem econômica, alinhado com os objetivos da disciplina.
            </p>
        </div>
        <div class="reference-block">
            <p><strong>2. Sargent, T. J., Stachurski, J., Quantitative Economics with Python. (https://python.quantecon.org/)</strong></p>
            <p class="explanation">
            Explicação: Outro projeto da QuantEcon, focado em métodos quantitativos e economia computacional
            usando Python. É uma referência padrão para economistas que programam.
            </p>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown("---")
    
    st.markdown("### GRDECO203: Laboratório de Ciência de Dados Aplicados à Finanças (Prof. Diogo Tavares Robaina)")

    st.markdown("#### Bibliografia Básica")
    st.markdown(
        """
        <div class="reference-block">
            <p><strong>1. HILPISCH, Y. J. Python for finance: analyze big financial dat. O'Reilly Media, 2015.</strong></p>
            <p class="explanation">
            Explicação: Uma referência clássica para finanças quantitativas em Python. Cobre manipulação
            de dados financeiros (séries temporais), análise de risco, e implementação de estratégias
            de trading e precificação de derivativos.
            </p>
        </div>
        <div class="reference-block">
            <p><strong>2. ARRATIA, A. Computational finance an introductory course with R. Atlantis, 2014.</strong></p>
            <p class="explanation">
            Explicação: Focado em finanças computacionais usando a linguagem R, abordando conceitos
            introdutórios e modelagem.
            </p>
        </div>
        <div class="reference-block">
            <p><strong>3. RASCHKA, S. Python machine learning: unlock deeper insights... Packt Publishing, 2015.</strong></p>
            <p class="explanation">
            Explicação: Um guia popular focado na aplicação prática de algoritmos de Machine Learning
            com Scikit-Learn em Python, similar ao livro de Géron.
            </p>
        </div>
        <div class="reference-block">
            <p><strong>4. MAINDONALD, J., and Braun, J. Data analysis and graphics using R: an example-based approach. Cambridge University Press, 2006.</strong></p>
            <p class="explanation">
            Explicação: Livro focado em análise de dados e visualização gráfica utilizando a linguagem R.
            </p>
        </div>
        <div class="reference-block">
            <p><strong>5. REYES, J. M. M. Introduction to Data Science for Social and Policy Research. Cambridge University Press, 2017.</strong></p>
            <p class="explanation">
            Explicação: Aborda a aplicação de Ciência de Dados no contexto de ciências sociais e pesquisa
            de políticas públicas, relevante para a análise econômica.
            </p>
        </div>
        """, unsafe_allow_html=True
    )
    
    st.markdown("#### Bibliografia Complementar")
    st.markdown(
        """
        <div class="reference-block">
            <p><strong>1. TEAM, R. Core. "R language definition." R foundation for statistical computing (2000).</strong></p>
            <p class="explanation">Explicação: A documentação oficial da linguagem R.</p>
        </div>
        <div class="reference-block">
            <p><strong>2. MISHRA, R.; RAM, B. Portfolio Selection Using R. Yugoslav Journal of Operations Research, 2020.</strong></p>
            <p class="explanation">Explicação: Um artigo de pesquisa focado especificamente na aplicação da
            linguagem R para otimização e seleção de portfólios, muito relevante para a disciplina.
            </p>
        </div>
        <div class="reference-block">
            <p><strong>3. WICKHAM, H., et al. (dplyr, Tidy data, Advanced R, ggplot2, R for data science).</strong></p>
            <p class="explanation">
            Explicação: Múltiplas referências de Hadley Wickham, o criador do "Tidyverse" em R.
            São os pacotes e livros fundamentais para a manipulação de dados moderna (dplyr),
            organização (Tidy data) e visualização (ggplot2) na linguagem R.
            </p>
        </div>
        """, unsafe_allow_html=True
    )

# =============================================================================
# 14. FUNÇÃO PRINCIPAL
# =============================================================================

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
    st.markdown('<h1 class="main-header">Sistema de Portfólios Adaptativos (v8.6)</h1>', unsafe_allow_html=True)
    
    # As abas agora serão centralizadas pelo CSS injetado em configurar_pagina()
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📚 Metodologia",
        "🎯 Seleção de Ativos",
        "🏗️ Construtor de Portfólio",
        "🔍 Análise Individual",
        "📖 Referências" # NOVA ABA
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
        aba_referencias() # NOVA FUNÇÃO

if __name__ == "__main__":
    main()
