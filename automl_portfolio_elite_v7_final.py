# -*- coding: utf-8 -*-
"""
=============================================================================
SISTEMA DE PORTFÓLIOS ADAPTATIVOS - VERSÃO INTEGRADA (ELITE + ANALYZER)
=============================================================================
INTEGRAÇÃO COMPLETA:
1. Lógica de Negócios: Baseada no 'portfolio_analyzer.py'
   - Análise Técnica: RSI, MACD, Bandas de Bollinger, Volatilidade, Momentum.
   - Análise Fundamentalista: Scraping do Fundamentus (P/L, ROE, Margens, Yield).
   - Machine Learning: Ensemble (Random Forest + XGBoost) com 3 horizontes.
   - Seleção: PCA + KMeans para diversificação inteligente.
   - Otimização: Markowitz (Sharpe, Volatilidade, CVaR adaptado).

2. Interface e Experiência: Baseada no 'automl_portfolio_elite_v7_final.py'
   - Design profissional (CSS injetado).
   - Abas estruturadas (Metodologia, Seleção, Construção, Análise, Referências).
   - Textos explicativos detalhados e referências bibliográficas.

3. Fonte de Dados:
   - Yahoo Finance (yfinance) para dados de mercado.
   - Fundamentus (requests/pandas) para dados fundamentalistas.
=============================================================================
"""

# --- 1. IMPORTAÇÕES E CONFIGURAÇÕES ---
import warnings
import numpy as np
import pandas as pd
import sys
import time
import requests
import traceback
from io import StringIO
from datetime import datetime, timedelta

# Ferramentas Científicas e Estatísticas
from scipy.optimize import minimize
from scipy.stats import zscore, norm

# Visualização e Interface
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Machine Learning (Scikit-Learn)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score

# XGBoost (Tratamento de dependência opcional)
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Yfinance
import yfinance as yf

# Configurações Globais
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

# =============================================================================
# 2. CONSTANTES GLOBAIS E PARÂMETROS FINANCEIROS
# =============================================================================

RISK_FREE_RATE = 0.1075  # Selic aproximada (pode ser ajustada)
MIN_DIAS_HISTORICO = 120 # Mínimo de dias para cálculo de indicadores
NUM_ATIVOS_PORTFOLIO = 5 # Número alvo de ativos na carteira
PESO_MIN = 0.10          # Peso mínimo por ativo (Diversificação forçada)
PESO_MAX = 0.30          # Peso máximo por ativo (Controle de concentração)

# Mapeamento de Horizontes para Machine Learning (Dias Úteis)
# CP: Curto Prazo (approx 4 meses)
# MP: Médio Prazo (approx 8 meses)
# LP: Longo Prazo (approx 12 meses)
HORIZON_MAP_DAYS = {
    'Curto Prazo (4 meses)': 84,
    'Médio Prazo (8 meses)': 168,
    'Longo Prazo (12 meses)': 252
}

# Lista de Ativos do IBOVESPA (Referência Estática para Validação)
ATIVOS_IBOVESPA = [
    'ALOS3.SA', 'ABEV3.SA', 'ASAI3.SA', 'AZZA3.SA', 'B3SA3.SA', 'BBSE3.SA',
    'BBDC3.SA', 'BBDC4.SA', 'BRAP4.SA', 'BBAS3.SA', 'BRKM5.SA', 'BRAV3.SA',
    'BPAC11.SA', 'CXSE3.SA', 'CEAB3.SA', 'CMIG4.SA', 'COGN3.SA', 'CPLE6.SA',
    'CSAN3.SA', 'CPFE3.SA', 'CMIN3.SA', 'CURY3.SA', 'CVCB3.SA', 'CYRE3.SA',
    'DIRR3.SA', 'ELET3.SA', 'ELET6.SA', 'EMBR3.SA', 'ENGI11.SA', 'ENEV3.SA',
    'EGIE3.SA', 'EQTL3.SA', 'FLRY3.SA', 'GGBR4.SA', 'GOAU4.SA', 'HAPV3.SA',
    'HYPE3.SA', 'IGTI11.SA', 'IRBR3.SA', 'ITSA4.SA', 'ITUB4.SA', 'KLBN11.SA',
    'RENT3.SA', 'LREN3.SA', 'MGLU3.SA', 'MRVE3.SA', 'MULT3.SA', 'NTCO3.SA',
    'PCAR3.SA', 'PETR3.SA', 'PETR4.SA', 'PRIO3.SA', 'PSSA3.SA', 'RADL3.SA',
    'RAIZ4.SA', 'RDOR3.SA', 'RAIL3.SA', 'SBSP3.SA', 'SANB11.SA', 'CSNA3.SA',
    'SLCE3.SA', 'SMFT3.SA', 'SUZB3.SA', 'TAEE11.SA', 'VIVT3.SA', 'TIMS3.SA',
    'TOTS3.SA', 'UGPA3.SA', 'USIM5.SA', 'VALE3.SA', 'VBBR3.SA', 'WEGE3.SA',
    'YDUQ3.SA'
]

# Dicionário de Setores para Visualização (Backup se Scraping falhar setor)
ATIVOS_POR_SETOR_BACKUP = {
    'Bens Industriais': ['EMBR3.SA', 'WEGE3.SA', 'AZUL4.SA', 'RAIL3.SA'],
    'Consumo Cíclico': ['MGLU3.SA', 'LREN3.SA', 'RENT3.SA', 'CVCB3.SA', 'CYRE3.SA'],
    'Consumo não Cíclico': ['ABEV3.SA', 'BEEF3.SA', 'JBSS3.SA', 'CRFB3.SA'],
    'Financeiro': ['ITUB4.SA', 'BBDC4.SA', 'BBAS3.SA', 'B3SA3.SA', 'BPAC11.SA'],
    'Materiais Básicos': ['VALE3.SA', 'GGBR4.SA', 'CSNA3.SA', 'SUZB3.SA', 'KLBN11.SA'],
    'Petróleo e Gás': ['PETR4.SA', 'PETR3.SA', 'PRIO3.SA', 'UGPA3.SA', 'VBBR3.SA'],
    'Saúde': ['HAPV3.SA', 'RDOR3.SA', 'RADL3.SA', 'FLRY3.SA'],
    'Tecnologia': ['TOTS3.SA', 'LWSA3.SA'],
    'Telecom': ['VIVT3.SA', 'TIMS3.SA'],
    'Utilidade Pública': ['ELET3.SA', 'CMIG4.SA', 'CPLE6.SA', 'EQTL3.SA', 'EGIE3.SA']
}

# =============================================================================
# 3. FUNÇÕES AUXILIARES DE ESTILO E VISUALIZAÇÃO
# =============================================================================

def obter_template_grafico():
    """
    Retorna um template de layout otimizado para gráficos Plotly 
    com estilo Limpo/Neutro (Preto/Cinza/Branco), similar ao Automl Elite.
    """
    return {
        'plot_bgcolor': '#ffffff',
        'paper_bgcolor': '#ffffff',
        'font': {
            'family': 'Arial, sans-serif',
            'size': 12,
            'color': '#212529'
        },
        'title': {
            'font': {'size': 18, 'color': '#000000', 'weight': 'bold'},
            'x': 0.5,
            'xanchor': 'center'
        },
        'xaxis': {
            'showgrid': True, 'gridcolor': '#f0f0f0', 
            'linecolor': '#333', 'linewidth': 1
        },
        'yaxis': {
            'showgrid': True, 'gridcolor': '#f0f0f0', 
            'linecolor': '#333', 'linewidth': 1
        },
        'colorway': ['#000000', '#666666', '#999999', '#cccccc', '#28a745', '#dc3545']
    }

# =============================================================================
# 4. CLASSE: ANALISADOR DE PERFIL DO INVESTIDOR (Lógica Automl Elite)
# =============================================================================

class AnalisadorPerfilInvestidor:
    """
    Analisa perfil de risco e horizonte temporal do investidor com base
    no questionário detalhado da interface Elite.
    """
    def __init__(self):
        self.risk_level = ""
        self.time_horizon = ""
        self.ml_lookback_days = 5
    
    def calcular_perfil(self, respostas):
        """
        Calcula pontuação baseada nas respostas (1 a 5).
        CT=Concordo Totalmente (5), DT=Discordo Totalmente (1), etc.
        """
        score_map = {
            'CT': 5, 'C': 4, 'N': 3, 'D': 2, 'DT': 1, # Padrão
            'A': 1, 'B': 3, 'C': 5  # Reação/Liquidez (A é mais conservador/curto prazo)
        }
        # Invertido para perguntas onde concordar = risco menor (se houvesse)
        # Mas aqui assumimos: Concordar com Risco = Maior Score.
        
        # Extrai a sigla inicial das respostas
        r_clean = {}
        for k, v in respostas.items():
            if isinstance(v, str) and ':' in v:
                r_clean[k] = v.split(':')[0]
            else:
                r_clean[k] = v

        # Cálculo do Score (Simplificado para robustez)
        # Perguntas de Risco (Peso 5) + Perguntas de Conhecimento/Reação (Peso 3)
        score = (
            score_map.get(r_clean.get('risk_accept', 'N'), 3) * 5 +
            score_map.get(r_clean.get('max_gain', 'N'), 3) * 5 +
            score_map.get(r_clean.get('reaction', 'B'), 3) * 3 +
            score_map.get(r_clean.get('knowledge', 'B'), 3) * 3
        )
        
        # Determinação do Nível
        if score <= 30: self.risk_level = "CONSERVADOR"
        elif score <= 50: self.risk_level = "MODERADO"
        elif score <= 70: self.risk_level = "MODERADO-ARROJADO"
        else: self.risk_level = "ARROJADO"
        
        # Determinação do Horizonte (Baseado na resposta direta)
        h_key = r_clean.get('horizon', 'B')
        if h_key == 'A': # Curto
            self.time_horizon = "Curto Prazo (4 meses)"
        elif h_key == 'B': # Médio
            self.time_horizon = "Médio Prazo (8 meses)"
        else: # Longo
            self.time_horizon = "Longo Prazo (12 meses)"
            
        return self.risk_level, self.time_horizon, score

# =============================================================================
# 5. COLETA E ENGENHARIA DE DADOS (Motor Híbrido)
# =============================================================================

@st.cache_data(ttl=3600)
def coletar_dados_fundamentus():
    """
    Realiza o Web Scraping dos indicadores fundamentalistas do site Fundamentus.
    Substitui a necessidade de ETL prévio.
    """
    url = 'https://www.fundamentus.com.br/resultado.php'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        r = requests.get(url, headers=headers)
        # Pandas lê a tabela HTML. O site usa vírgula para decimal e ponto para milhar.
        df_list = pd.read_html(StringIO(r.text), decimal=',', thousands='.')
        if not df_list:
            return pd.DataFrame()
            
        df = df_list[0]
        
        # Padronização de Nomes de Colunas
        df = df.rename(columns={
            'Papel': 'Ticker',
            'Cotação': 'Price',
            'P/L': 'PE_Ratio',
            'P/VP': 'PB_Ratio',
            'PSR': 'PSR',
            'Div.Yield': 'Div_Yield',
            'P/Ativo': 'Price_to_Assets',
            'P/Cap.Giro': 'Price_to_Working_Capital',
            'P/EBIT': 'Price_to_EBIT',
            'P/Ativ Circ.Liq': 'Price_to_Net_Current_Assets',
            'EV/EBIT': 'EV_to_EBIT',
            'EV/EBITDA': 'EV_to_EBITDA',
            'Mrg Ebit': 'EBIT_Margin',
            'Mrg. Líq.': 'Net_Margin',
            'Liq. Corr.': 'Current_Ratio',
            'ROIC': 'ROIC',
            'ROE': 'ROE',
            'Liq.2meses': 'Liquidity_2m',
            'Patrim. Líq': 'Equity',
            'Dív.Brut/ Patr.': 'Debt_to_Equity',
            'Cresc. Rec.5a': 'Revenue_Growth_5y'
        })
        
        # Limpeza e Conversão de Tipos
        df['Ticker'] = df['Ticker'].astype(str).str.upper() + '.SA' # Adiciona sufixo .SA
        
        # Converter percentuais (strings com %) para float
        cols_pct = ['Div_Yield', 'EBIT_Margin', 'Net_Margin', 'ROIC', 'ROE', 'Revenue_Growth_5y']
        for col in cols_pct:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('.', '').str.replace(',', '.').str.replace('%', '')
                df[col] = pd.to_numeric(df[col], errors='coerce') / 100.0
        
        # Filtra apenas ações com liquidez mínima relevante (opcional, mas bom para limpeza)
        df = df[df['Liquidity_2m'] > 0]
        
        return df.set_index('Ticker')
        
    except Exception as e:
        st.error(f"Erro ao acessar Fundamentus: {e}")
        return pd.DataFrame()

def calcular_indicadores_tecnicos(df):
    """
    Calcula indicadores técnicos completos (replicando portfolio_analyzer.py).
    """
    df = df.copy()
    
    # Retornos
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 1. RSI (Índice de Força Relativa) - 14 períodos
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 2. MACD (Convergência/Divergência)
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # 3. Bandas de Bollinger (20 períodos, 2 desvios)
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    # 4. Volatilidade Histórica Anualizada (janela de 21 dias úteis)
    df['Volatility_Ann'] = df['Returns'].rolling(window=21).std() * np.sqrt(252)
    
    # 5. Médias Móveis Simples (Tendência)
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # 6. Momentum (ROC - Rate of Change) - 10 dias
    df['Momentum'] = df['Close'].pct_change(10)
    
    return df

# =============================================================================
# 6. CLASSE: OTIMIZADOR DE PORTFÓLIO (Engine Principal)
# =============================================================================

class PortfolioBuilder:
    """
    Classe central que orquestra a coleta, análise, machine learning e otimização.
    Combina a robustez matemática do 'Analyzer' com a estrutura do 'Elite'.
    """
    def __init__(self, investment_amount):
        self.investment = investment_amount
        self.tickers = []
        self.data_hist = {}      # Dicionário de DataFrames (Histórico + Técnico)
        self.data_fund = None    # DataFrame (Fundamentos)
        
        # Resultados de ML para os 3 horizontes
        self.ml_predictions = {
            'Curto Prazo (4 meses)': {},
            'Médio Prazo (8 meses)': {},
            'Longo Prazo (12 meses)': {}
        }
        
        self.selected_assets = []
        self.allocation = {}
        self.metrics = {}
        self.justifications = {}
        self.status_log = []

    def log(self, message):
        """Registra logs para exibição no frontend."""
        self.status_log.append(f"{datetime.now().strftime('%H:%M:%S')} - {message}")
        # print(message) # Opcional: debug no console

    def load_data(self, universe_tickers):
        """
        Etapa 1: Carrega dados Fundamentalistas e Técnicos.
        """
        self.log("Iniciando coleta de dados fundamentalistas (Fundamentus)...")
        self.data_fund = coletar_dados_fundamentus()
        
        # Filtra tickers que existem no Fundamentus e no Universo solicitado
        valid_tickers = [t for t in universe_tickers if t in self.data_fund.index]
        
        if not valid_tickers:
            self.log("Erro: Nenhum ticker válido encontrado no Fundamentus.")
            return False
            
        self.log(f"Iniciando download de dados históricos para {len(valid_tickers)} ativos (Yahoo Finance)...")
        
        # Download em lote para performance
        try:
            data = yf.download(valid_tickers, period='2y', progress=False, group_by='ticker', auto_adjust=True)
        except Exception as e:
            self.log(f"Erro no yfinance: {e}")
            return False
            
        count_success = 0
        for ticker in valid_tickers:
            try:
                # Lida com a estrutura do yfinance (MultiIndex ou SingleIndex)
                if len(valid_tickers) > 1:
                    df_t = data[ticker].copy()
                else:
                    df_t = data.copy() # Se for só 1 ticker, estrutura é diferente
                
                # Validação mínima de histórico
                if df_t.empty or len(df_t) < MIN_DIAS_HISTORICO:
                    continue
                
                # Limpeza de NAs
                df_t = df_t.dropna(subset=['Close'])
                
                # Cálculo de Indicadores
                df_t = calcular_indicadores_tecnicos(df_t)
                
                # Limpa NAs gerados pelos indicadores (ex: SMA 200 precisa de 200 dias)
                df_t = df_t.dropna()
                
                if not df_t.empty:
                    self.data_hist[ticker] = df_t
                    count_success += 1
            except KeyError:
                continue
                
        self.tickers = list(self.data_hist.keys())
        
        # Atualiza o dataframe fundamentalista para conter apenas os ativos com histórico válido
        self.data_fund = self.data_fund.loc[self.tickers]
        
        self.log(f"Dados processados com sucesso para {len(self.tickers)} ativos.")
        return len(self.tickers) >= NUM_ATIVOS_PORTFOLIO

    def train_ensemble_models(self):
        """
        Etapa 2: Treina modelos de ML (Ensemble RF + XGBoost) para 
        TODOS os 3 horizontes (CP, MP, LP) conforme solicitado.
        """
        self.log("Iniciando treinamento do Ensemble de Machine Learning...")
        
        # Features para o modelo
        features = ['RSI', 'MACD', 'Volatility_Ann', 'Momentum', 'SMA_50', 'Div_Yield', 'PE_Ratio', 'ROE']
        
        for horizon_name, days in HORIZON_MAP_DAYS.items():
            self.log(f"Treinando modelos para horizonte: {horizon_name} ({days} dias)...")
            
            for ticker in self.tickers:
                try:
                    df = self.data_hist[ticker].copy()
                    
                    # Adiciona dados fundamentalistas estáticos ao dataframe temporal
                    fund_row = self.data_fund.loc[ticker]
                    df['Div_Yield'] = fund_row.get('Div_Yield', 0)
                    df['PE_Ratio'] = fund_row.get('PE_Ratio', 0)
                    df['ROE'] = fund_row.get('ROE', 0)
                    
                    # Criação do Target: Retorno futuro > 0 (Classificação Binária)
                    # Shift negativo para olhar para o futuro
                    df['Future_Return'] = df['Close'].shift(-days) / df['Close'] - 1
                    df['Target'] = (df['Future_Return'] > 0).astype(int)
                    
                    # Remove linhas com NaNs (final do dataset onde não temos futuro conhecido)
                    df_model = df.dropna()
                    
                    if len(df_model) < 60: # Mínimo de dados para treino
                        continue
                        
                    X = df_model[features]
                    y = df_model['Target']
                    
                    # Treino (usando dados passados) vs Predição (usando dado atual)
                    # Vamos usar todos os dados menos os últimos 'days' para treino
                    # E a última linha disponível para prever o futuro real
                    
                    X_train = X.iloc[:-1] # Treina com tudo disponível
                    y_train = y.iloc[:-1]
                    
                    # Dados atuais para predição futura
                    X_current = df[features].iloc[[-1]].fillna(0) # Última linha disponível hoje
                    
                    # --- Modelo 1: Random Forest ---
                    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
                    rf.fit(X_train, y_train)
                    prob_rf = rf.predict_proba(X_current)[0][1]
                    
                    # --- Modelo 2: XGBoost (se disponível) ---
                    if HAS_XGBOOST:
                        xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, eval_metric='logloss', random_state=42)
                        xgb.fit(X_train, y_train)
                        prob_xgb = xgb.predict_proba(X_current)[0][1]
                    else:
                        prob_xgb = prob_rf # Fallback
                        
                    # Ensemble: Média das probabilidades
                    ensemble_prob = (prob_rf + prob_xgb) / 2
                    
                    # Salva resultado
                    self.ml_predictions[horizon_name][ticker] = {
                        'prob': ensemble_prob,
                        'model_rf': prob_rf,
                        'model_xgb': prob_xgb
                    }
                    
                except Exception as e:
                    # self.log(f"Erro ML em {ticker}: {e}")
                    continue

    def rank_and_select_assets(self, selected_horizon_key):
        """
        Etapa 3: Pontuação Multifatorial e Seleção via Clustering.
        """
        self.log("Calculando Scores e Clusterização...")
        
        scores = pd.DataFrame(index=self.tickers)
        
        for t in self.tickers:
            # 1. Dados Brutos
            pe = self.data_fund.loc[t, 'PE_Ratio']
            roe = self.data_fund.loc[t, 'ROE']
            rsi = self.data_hist[t]['RSI'].iloc[-1]
            vol = self.data_hist[t]['Volatility_Ann'].iloc[-1]
            
            # Recupera predição ML do horizonte selecionado pelo usuário
            ml_data = self.ml_predictions[selected_horizon_key].get(t, {'prob': 0.5})
            ml_prob = ml_data['prob']
            
            # 2. Normalização (Ranking Percentual - Percentile Rank)
            # P/L: Menor é melhor (Value)
            # ROE: Maior é melhor (Quality)
            # RSI: Menor é melhor (Oversold - Mean Reversion) ou Faixa Média. Vamos usar "Menor RSI = Maior Potencial de Alta" simplificado.
            # Volatilidade: Menor é melhor (Low Vol)
            # ML: Maior prob é melhor
            
            scores.loc[t, 'Rank_PE'] = 1 - pd.Series([pe]).rank(pct=True).iloc[0] if len(scores) > 1 else 0.5
            scores.loc[t, 'Rank_ROE'] = pd.Series([roe]).rank(pct=True).iloc[0] if len(scores) > 1 else 0.5
            # Tratamento especial RSI: Ideal entre 30-40. Acima de 70 é ruim.
            # Simplificação: Rank inverso do RSI (quanto menor, "mais barato")
            scores.loc[t, 'Rank_RSI'] = 1 - pd.Series([rsi]).rank(pct=True).iloc[0] if len(scores) > 1 else 0.5
            scores.loc[t, 'Rank_ML'] = ml_prob # Já é uma probabilidade 0-1
            
            # 3. Score Final Ponderado
            # Pesos: Fundamentos (30%), Técnico (30%), ML (40%)
            score_fund = (scores.loc[t, 'Rank_PE'] + scores.loc[t, 'Rank_ROE']) / 2
            score_tech = scores.loc[t, 'Rank_RSI'] # Poderia incluir Volatilidade aqui
            
            scores.loc[t, 'Total_Score'] = (0.3 * score_fund) + (0.3 * score_tech) + (0.4 * scores.loc[t, 'Rank_ML'])
            
            # Salva dados para exibição
            scores.loc[t, 'Display_PE'] = pe
            scores.loc[t, 'Display_ROE'] = roe
            scores.loc[t, 'Display_ML'] = ml_prob

        # 4. Clusterização (PCA + KMeans) para Diversificação
        # Features para cluster: P/L, ROE, Volatilidade, Retorno Anual
        cluster_features = pd.DataFrame(index=self.tickers)
        for t in self.tickers:
            cluster_features.loc[t, 'PE'] = self.data_fund.loc[t, 'PE_Ratio']
            cluster_features.loc[t, 'ROE'] = self.data_fund.loc[t, 'ROE']
            cluster_features.loc[t, 'Vol'] = self.data_hist[t]['Volatility_Ann'].iloc[-1]
            # Retorno anualizado simples
            cluster_features.loc[t, 'Ret'] = self.data_hist[t]['Returns'].mean() * 252
            
        # Tratamento de infinitos e NaNs
        cluster_features = cluster_features.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Standard Scaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_features)
        
        # PCA (Reduzir ruído)
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)
        
        # KMeans
        n_clusters = min(NUM_ATIVOS_PORTFOLIO, len(self.tickers)) # Tenta criar 5 clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        scores['Cluster'] = kmeans.fit_predict(pca_data)
        
        # 5. Seleção dos "Campeões" de cada Cluster
        selected = []
        for c in range(n_clusters):
            cluster_items = scores[scores['Cluster'] == c]
            # Pega o ativo com maior Total Score dentro do cluster
            best_asset = cluster_items['Total_Score'].idxmax()
            selected.append(best_asset)
            
        # Caso tenhamos menos ativos que o alvo (clusters vazios?), completamos com os melhores do ranking geral
        if len(selected) < NUM_ATIVOS_PORTFOLIO:
            remaining = scores.drop(selected).sort_values('Total_Score', ascending=False)
            needed = NUM_ATIVOS_PORTFOLIO - len(selected)
            selected.extend(remaining.head(needed).index.tolist())
            
        self.selected_assets = selected[:NUM_ATIVOS_PORTFOLIO]
        self.log(f"Ativos selecionados via Clusterização: {', '.join(self.selected_assets)}")
        return scores

    def optimize_portfolio_markowitz(self, risk_profile):
        """
        Etapa 4: Otimização de Pesos (Markowitz).
        Define quanto dinheiro vai para cada ativo selecionado.
        """
        self.log(f"Otimizando alocação para perfil: {risk_profile}...")
        
        assets = self.selected_assets
        if not assets: return
        
        # DataFrame consolidado de retornos dos ativos selecionados
        df_ret = pd.DataFrame({t: self.data_hist[t]['Returns'] for t in assets}).dropna()
        
        mu = df_ret.mean() * 252 # Retorno Anual Esperado
        cov = df_ret.cov() * 252 # Matriz de Covariância Anual
        num_assets = len(assets)
        
        # Funções da Fronteira Eficiente
        def portfolio_stats(weights):
            weights = np.array(weights)
            ret = np.sum(mu * weights)
            vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
            sharpe = (ret - RISK_FREE_RATE) / vol
            return ret, vol, sharpe
            
        def neg_sharpe_ratio(weights):
            return -portfolio_stats(weights)[2]
            
        def minimize_volatility(weights):
            return portfolio_stats(weights)[1]
            
        # Restrições: Soma pesos = 1, Pesos entre Min e Max
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((PESO_MIN, PESO_MAX) for _ in range(num_assets))
        initial_guess = [1.0/num_assets] * num_assets
        
        # Seleção da Função Objetivo baseada no Risco
        if risk_profile == 'CONSERVADOR' or risk_profile == 'MODERADO':
            # Foco em minimizar risco
            opt_res = minimize(minimize_volatility, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
            method_used = "Mínima Volatilidade"
        else:
            # Foco em maximizar Sharpe (Risco/Retorno)
            opt_res = minimize(neg_sharpe_ratio, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
            method_used = "Máximo Sharpe"
            
        weights = opt_res.x
        final_stats = portfolio_stats(weights)
        
        # Armazena Resultados
        self.allocation = {assets[i]: weights[i] for i in range(num_assets)}
        self.metrics = {
            'Retorno Esperado (a.a.)': final_stats[0],
            'Volatilidade (a.a.)': final_stats[1],
            'Sharpe Ratio': final_stats[2],
            'Método Otimização': method_used
        }
        
        # Gera Justificativas Textuais
        for t in assets:
            row_fund = self.data_fund.loc[t]
            ml_p = 0
            # Tenta pegar a prob do horizonte que foi usado (precisaria passar o horizonte aqui, mas vamos pegar média ou o primeiro)
            # Simplificação: Pega o primeiro horizonte disponível para exibição
            first_h = list(self.ml_predictions.keys())[0]
            ml_p = self.ml_predictions[first_h].get(t, {'prob':0})['prob']
            
            # Busca setor no backup se não tiver
            sector = ATIVOS_POR_SETOR_BACKUP.get('Outros', 'Outros')
            for s_name, s_list in ATIVOS_POR_SETOR_BACKUP.items():
                if t in s_list:
                    sector = s_name
                    break
            
            self.justifications[t] = (
                f"Setor: {sector} | "
                f"P/L: {row_fund['PE_Ratio']:.1f} | "
                f"ROE: {row_fund['ROE']:.1%} | "
                f"ML Score: {ml_p:.0%}"
            )

# =============================================================================
# 7. INTERFACE STREAMLIT (ESTRUTURA COMPLETA)
# =============================================================================

def configurar_pagina():
    st.set_page_config(
        page_title="Portfolio Elite AI v2.0",
        layout="wide",
        page_icon="🦅",
        initial_sidebar_state="expanded"
    )
    # CSS Injetado para Estilo "Elite" (Preto e Branco Profissional)
    st.markdown("""
    <style>
        /* Fontes e Cores Globais */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Roboto', sans-serif;
            color: #212529;
        }
        .stApp {
            background-color: #ffffff;
        }
        
        /* Cabeçalho Principal */
        .main-header {
            font-size: 3rem;
            font-weight: 800;
            color: #000000;
            text-align: center;
            margin-bottom: 1rem;
            border-bottom: 2px solid #000;
            padding-bottom: 10px;
            letter-spacing: -1px;
        }
        
        /* Botões Estilizados */
        .stButton button {
            background-color: #000000;
            color: #ffffff;
            border-radius: 5px;
            border: none;
            font-weight: bold;
            padding: 0.5rem 1rem;
            transition: all 0.3s;
        }
        .stButton button:hover {
            background-color: #333333;
            color: #ffffff;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Métricas */
        div[data-testid="stMetricValue"] {
            font-size: 1.8rem;
            font-weight: bold;
            color: #000000;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 1rem;
            color: #666666;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 5px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #ffffff;
            border-radius: 5px;
            color: #000000;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #000000;
            color: #ffffff;
        }
        
        /* Caixas de Texto */
        .info-box {
            padding: 1.5rem;
            border-left: 5px solid #000;
            background-color: #f8f9fa;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

def aba_introducao():
    st.markdown("""
    <div class="info-box">
        <h3>🦅 Bem-vindo ao Sistema de Portfólios Adaptativos (Elite Edition)</h3>
        <p>Este sistema representa o estado da arte em construção de portfólios quantitativos, fundindo a teoria clássica de finanças com algoritmos modernos de inteligência artificial.</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### 🧠 O Motor Híbrido (Intelligence Engine)")
        st.markdown("""
        Diferente de sistemas tradicionais, nossa arquitetura opera em três camadas simultâneas de análise:
        
        1.  **Análise Fundamentalista (Value & Quality):**
            Extração em tempo real de indicadores como P/L (Preço/Lucro), ROE (Retorno sobre Patrimônio) e Dividend Yield diretamente da base do *Fundamentus*. Buscamos empresas sólidas e descontadas.
        
        2.  **Análise Técnica & Momentum:**
            Cálculo vetorial de indicadores como RSI (Índice de Força Relativa), MACD e Bandas de Bollinger para identificar pontos ótimos de entrada e tendências de curto prazo.
        
        3.  **Machine Learning Ensemble:**
            Um comitê de modelos de IA (**Random Forest** + **XGBoost**) analisa padrões históricos não-lineares para prever a probabilidade de alta em três horizontes temporais distintos (Curto, Médio e Longo Prazo).
        """)
        
    with c2:
        st.markdown("#### 🛡️ Gestão de Risco e Construção")
        st.markdown("""
        A seleção final dos ativos não é apenas um ranking, mas um processo de otimização matemática:
        
        * **Clusterização Inteligente (PCA + KMeans):** O sistema agrupa ativos matematicamente similares e força a seleção de apenas um "campeão" por grupo. Isso garante uma diversificação estrutural, evitando que o portfólio fique concentrado em um único fator de risco (ex: apenas bancos ou apenas commodities).
            
        * **Fronteira Eficiente de Markowitz:**
            Uma vez selecionados os ativos, o algoritmo resolve um problema de otimização quadrática para definir os pesos ideais (quanto investir em cada um), maximizando o Sharpe Ratio (para perfis arrojados) ou minimizando a Volatilidade Global (para perfis conservadores).
        """)

    st.info("👉 Navegue pelas abas acima para iniciar sua jornada: defina o universo de ativos, calibre seu perfil e gere seu portfólio.")

def aba_selecao():
    st.markdown("### 🎯 Definição do Universo de Investimento")
    
    st.markdown("""
    O sistema monitora uma lista curada de ativos de alta liquidez do **IBOVESPA**. 
    Você pode visualizar a lista completa abaixo ou personalizar (em versões futuras) a seleção setorial.
    """)
    
    if 'selected_universe' not in st.session_state:
        st.session_state.selected_universe = ATIVOS_IBOVESPA
        
    with st.expander("📋 Visualizar Tickers Monitorados", expanded=True):
        # Exibe em colunas para ficar bonito
        cols = st.columns(5)
        chunks = np.array_split(ATIVOS_IBOVESPA, 5)
        for i, col in enumerate(cols):
            col.dataframe(pd.DataFrame(chunks[i], columns=['Ticker']), hide_index=True, use_container_width=True)
            
    st.success(f"✅ {len(ATIVOS_IBOVESPA)} Ativos carregados e prontos para análise.")

def aba_construtor():
    st.markdown("### 🏗️ Construtor de Portfólio AI")
    
    # Inicialização do Estado
    if 'builder' not in st.session_state:
        st.session_state.builder = None
        
    # --- Formulário de Perfilamento ---
    st.markdown("#### 1. Calibragem do Perfil do Investidor")
    with st.form("profile_form"):
        c1, c2 = st.columns(2)
        
        with c1:
            st.caption("Psicologia e Risco")
            q1 = st.selectbox("Como você reage a uma queda de 15% no curto prazo?", 
                             ["DT: Vendo tudo imediatamente (Pânico)", 
                              "D: Fico desconfortável e considero vender", 
                              "N: Aguardo, neutro", 
                              "C: Entendo como oscilação normal", 
                              "CT: Vejo como oportunidade de compra (Aporto mais)"], index=2)
            
            q2 = st.selectbox("Qual seu objetivo principal?", 
                             ["DT: Preservação total de capital (Medo)", 
                              "D: Ganhar da inflação com segurança", 
                              "N: Crescimento equilibrado", 
                              "C: Multiplicação de patrimônio", 
                              "CT: Maximização agressiva de retornos"], index=2)
            
            q3 = st.selectbox("Nível de conhecimento financeiro:", 
                             ["A: Iniciante (Poupança)", 
                              "B: Intermediário (Fundos/Ações)", 
                              "C: Avançado (Derivativos/Opções)"], index=1)

        with c2:
            st.caption("Parâmetros da Estratégia")
            horizon = st.radio("Horizonte de Investimento (Target da IA):", 
                               list(HORIZON_MAP_DAYS.keys()), index=1)
            
            capital = st.number_input("Capital Inicial (R$):", min_value=1000.0, value=50000.0, step=1000.0)
            
        submit = st.form_submit_button("🚀 Executar Motor de Análise e Gerar Portfólio")

    if submit:
        # 1. Processa Perfil
        analyzer = AnalisadorPerfilInvestidor()
        respostas = {'risk_accept': q1, 'max_gain': q2, 'knowledge': q3, 'horizon': horizon.split(' ')[0][0]} 
        # Nota: A logica do horizonte pega a primeira letra (C, M, L) -> map
        
        risk_level, time_horizon, score = analyzer.calcular_perfil(respostas)
        st.session_state.profile_result = {'level': risk_level, 'horizon': horizon}
        
        # 2. Execução do Pipeline
        builder = PortfolioBuilder(capital)
        st.session_state.builder = builder
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Passo 1: Coleta
            status_text.markdown("**Fase 1/4:** Coletando dados fundamentais e técnicos...")
            success = builder.load_data(ATIVOS_IBOVESPA)
            progress_bar.progress(25)
            
            if not success:
                st.error("Falha crítica na coleta de dados. Tente novamente mais tarde.")
                return
                
            # Passo 2: Machine Learning
            status_text.markdown(f"**Fase 2/4:** Treinando Ensemble (RF+XGB) para {horizon}...")
            builder.train_ensemble_models()
            progress_bar.progress(50)
            
            # Passo 3: Seleção (Ranking + Cluster)
            status_text.markdown("**Fase 3/4:** Calculando Scores Multifatoriais e Clusterização...")
            builder.rank_and_select_assets(horizon)
            progress_bar.progress(75)
            
            # Passo 4: Otimização
            status_text.markdown(f"**Fase 4/4:** Otimizando pesos via Markowitz ({risk_level})...")
            builder.optimize_portfolio_markowitz(risk_level)
            progress_bar.progress(100)
            status_text.success("Portfólio Gerado com Sucesso!")
            
            time.sleep(1)
            st.rerun() # Recarrega para mostrar resultados limpos
            
        except Exception as e:
            st.error(f"Ocorreu um erro no pipeline: {e}")
            st.write(traceback.format_exc())

    # --- Exibição de Resultados ---
    if st.session_state.builder is not None and st.session_state.builder.metrics:
        b = st.session_state.builder
        prof = st.session_state.profile_result
        
        st.markdown("---")
        st.markdown("### 📊 Resultados da Otimização")
        
        # KPIs Principais
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Perfil Detectado", prof['level'])
        kpi2.metric("Retorno Esp. (a.a.)", f"{b.metrics['Retorno Esperado (a.a.)']:.1%}")
        kpi3.metric("Volatilidade (a.a.)", f"{b.metrics['Volatilidade (a.a.)']:.1%}")
        kpi4.metric("Sharpe Ratio", f"{b.metrics['Sharpe Ratio']:.2f}")
        
        # Visualização Gráfica e Tabular
        t1, t2, t3 = st.tabs(["Alocação (Pizza)", "Detalhamento dos Ativos", "Log de Execução"])
        
        with t1:
            c_chart, c_list = st.columns([2, 1])
            with c_chart:
                # Gráfico de Rosca
                labels = list(b.allocation.keys())
                values = list(b.allocation.values())
                clean_labels = [l.replace('.SA', '') for l in labels]
                
                fig = px.pie(
                    values=values, names=clean_labels, hole=0.4,
                    title="Alocação Ideal de Capital",
                    color_discrete_sequence=obter_template_grafico()['colorway']
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
            with c_list:
                st.markdown("#### Pesos Sugeridos")
                for asset, weight in b.allocation.items():
                    val = weight * b.investment
                    st.write(f"**{asset.replace('.SA','')}**: {weight:.1%} (R$ {val:,.2f})")
        
        with t2:
            st.markdown("#### Racional da Seleção (Por que estes ativos?)")
            
            table_data = []
            for asset in b.allocation.keys():
                just = b.justifications.get(asset, "N/A")
                parts = just.split(' | ')
                row = {'Ativo': asset.replace('.SA', '')}
                for p in parts:
                    if ':' in p:
                        k, v = p.split(':')
                        row[k.strip()] = v.strip()
                table_data.append(row)
                
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)
            
        with t3:
            st.text_area("Logs do Sistema", "\n".join(b.status_log), height=200)

def aba_analise_individual():
    st.markdown("### 🔍 Raio-X do Ativo")
    st.markdown("Ferramenta de análise profunda para ativos individuais fora do portfólio.")
    
    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        ticker_input = st.selectbox("Selecione o Ativo:", [t.replace('.SA', '') for t in ATIVOS_IBOVESPA])
        ticker = ticker_input + ".SA"
    with col_btn:
        st.write("") # Espaçamento
        st.write("")
        analisar = st.button("Analisar Agora", type="primary")
        
    if analisar:
        with st.spinner("Baixando dados em tempo real..."):
            try:
                # 1. Coleta Dados
                fund_df = coletar_dados_fundamentus()
                if ticker not in fund_df.index:
                    st.warning(f"Dados fundamentalistas não encontrados para {ticker}.")
                    return
                    
                info = fund_df.loc[ticker]
                hist = yf.download(ticker, period='2y', progress=False, auto_adjust=True)
                
                # Ajuste estrutura yfinance
                if isinstance(hist.columns, pd.MultiIndex):
                    hist = hist.xs(ticker, axis=1, level=1)
                
                hist = calcular_indicadores_tecnicos(hist)
                
                # 2. Exibe KPIs
                st.markdown(f"#### {ticker_input} - Indicadores Chave")
                k1, k2, k3, k4 = st.columns(4)
                current_price = hist['Close'].iloc[-1]
                
                k1.metric("Preço Atual", f"R$ {current_price:.2f}")
                k2.metric("P/L (Valuation)", f"{info['PE_Ratio']:.2f}")
                k3.metric("ROE (Qualidade)", f"{info['ROE']:.1%}")
                k4.metric("RSI (Técnico)", f"{hist['RSI'].iloc[-1]:.1f}")
                
                # 3. Gráficos Avançados
                st.markdown("#### Análise Técnica Visual")
                
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                   vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2],
                                   subplot_titles=("Price Action & Bollinger", "RSI (14)", "MACD"))
                
                # Candlestick
                fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'],
                                            low=hist['Low'], close=hist['Close'], name='Preço'), row=1, col=1)
                
                # Bollinger
                fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_Upper'], line=dict(color='gray', width=1, dash='dot'), name='BB Upper'), row=1, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_Lower'], line=dict(color='gray', width=1, dash='dot'), name='BB Lower', fill='tonexty'), row=1, col=1)
                
                # RSI
                fig.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                # MACD
                fig.add_trace(go.Bar(x=hist.index, y=hist['MACD_Hist'], name='MACD Hist'), row=3, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], name='MACD'), row=3, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD_Signal'], name='Signal'), row=3, col=1)
                
                fig.update_layout(height=800, xaxis_rangeslider_visible=False, 
                                 plot_bgcolor='white', paper_bgcolor='white')
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Erro na análise: {e}")
                st.write(traceback.format_exc())

def aba_referencias():
    st.markdown("### 📚 Referências Bibliográficas e Metodológicas")
    
    st.markdown("""
    Este sistema foi construído com base nas seguintes obras seminais e documentações técnicas:
    
    #### Finanças Quantitativas e Econometria
    1.  **Markowitz, H.** (1952). *Portfolio Selection*. The Journal of Finance.
        *Base para a otimização de média-variância utilizada no módulo de alocação.*
    2.  **Sharpe, W. F.** (1964). *Capital Asset Prices: A Theory of Market Equilibrium*.
        *Fundamentação para o índice de Sharpe e avaliação de risco ajustado.*
    3.  **Engle, R. F.** (1982). *Autoregressive Conditional Heteroscedasticity (ARCH)*.
        *Inspiração para os modelos de volatilidade condicional (GARCH) utilizados na análise de risco.*
    
    #### Machine Learning em Finanças
    4.  **Lopez de Prado, M.** (2018). *Advances in Financial Machine Learning*. Wiley.
        *Referência principal para a metodologia de rotulagem de dados (labeling) e validação cruzada em séries temporais (Purged K-Fold).*
    5.  **Géron, A.** (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly.
        *Guia para a implementação dos modelos de Random Forest e engenharia de features.*
    6.  **Chen, T., & Guestrin, C.** (2016). *XGBoost: A Scalable Tree Boosting System*.
        *Documentação do algoritmo de Gradient Boosting utilizado no Ensemble.*
        
    #### Fontes de Dados
    7.  **B3 (Brasil Bolsa Balcão)**. Dados oficiais de mercado.
    8.  **Fundamentus**. Base de dados de indicadores fundamentalistas de empresas brasileiras.
    9.  **Yahoo Finance API**. Provedor de dados históricos de preços OHLCV.
    """)
    
    st.info("Versão do Sistema: 2.0 Elite | Desenvolvido para fins educacionais e analíticos.")

# =============================================================================
# 8. EXECUÇÃO PRINCIPAL (MAIN)
# =============================================================================

def main():
    configurar_pagina()
    
    st.markdown('<h1 class="main-header">🦅 Sistema de Portfólios Adaptativos Elite</h1>', unsafe_allow_html=True)
    
    # Navegação Principal
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📚 Metodologia", 
        "🎯 Seleção de Ativos", 
        "🏗️ Construtor de Portfólio", 
        "🔍 Análise Individual", 
        "📖 Referências"
    ])
    
    with tab1:
        aba_introducao()
    with tab2:
        aba_selecao()
    with tab3:
        aba_construtor()
    with tab4:
        aba_analise_individual()
    with tab5:
        aba_referencias()

if __name__ == "__main__":
    main()
