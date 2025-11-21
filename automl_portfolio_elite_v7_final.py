# -*- coding: utf-8 -*-
"""
=============================================================================
SISTEMA DE PORTFÓLIOS ADAPTATIVOS - VERSÃO INTEGRADA (ELITE + ANALYZER) V3
=============================================================================
INTEGRAÇÃO COMPLETA COM CORREÇÕES DE ROBUSTEZ (YFINANCE/DATAFRAME VAZIO)

1. Lógica de Negócios: Baseada no 'portfolio_analyzer.py'
   - Análise Técnica: RSI, MACD, Bandas de Bollinger, Volatilidade, Momentum.
   - Análise Fundamentalista: Scraping do Fundamentus (P/L, ROE, Margens, Yield).
   - Machine Learning: Ensemble (Random Forest + XGBoost) com 3 horizontes.
   - Seleção: PCA + KMeans para diversificação inteligente.
   - Otimização: Markowitz (Sharpe, Volatilidade, CVaR adaptado).

2. Interface e Experiência: Baseada no 'automl_portfolio_elite_v7_final.py'
   - Design profissional (CSS injetado).
   - Abas estruturadas.
   - Textos explicativos detalhados.

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
    com estilo Limpo/Neutro (Preto/Cinza/Branco).
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
# 4. CLASSE: ANALISADOR DE PERFIL DO INVESTIDOR
# =============================================================================

class AnalisadorPerfilInvestidor:
    """
    Analisa perfil de risco e horizonte temporal do investidor.
    """
    def __init__(self):
        self.risk_level = ""
        self.time_horizon = ""
    
    def calcular_perfil(self, respostas):
        """
        Calcula pontuação baseada nas respostas (1 a 5).
        """
        score_map = {
            'CT': 5, 'C': 4, 'N': 3, 'D': 2, 'DT': 1, # Padrão
            'A': 1, 'B': 3, 'C': 5  # Reação/Liquidez
        }
        
        r_clean = {}
        for k, v in respostas.items():
            if isinstance(v, str) and ':' in v:
                r_clean[k] = v.split(':')[0]
            else:
                r_clean[k] = v

        score = (
            score_map.get(r_clean.get('risk_accept', 'N'), 3) * 5 +
            score_map.get(r_clean.get('max_gain', 'N'), 3) * 5 +
            score_map.get(r_clean.get('reaction', 'B'), 3) * 3 +
            score_map.get(r_clean.get('knowledge', 'B'), 3) * 3
        )
        
        if score <= 30: self.risk_level = "CONSERVADOR"
        elif score <= 50: self.risk_level = "MODERADO"
        elif score <= 70: self.risk_level = "MODERADO-ARROJADO"
        else: self.risk_level = "ARROJADO"
        
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
        
        df = df.rename(columns={
            'Papel': 'Ticker', 'Cotação': 'Price', 'P/L': 'PE_Ratio', 'P/VP': 'PB_Ratio',
            'Div.Yield': 'Div_Yield', 'ROE': 'ROE', 'Mrg Ebit': 'EBIT_Margin',
            'Mrg. Líq.': 'Net_Margin', 'Liq. Corr.': 'Current_Ratio',
            'Liq.2meses': 'Liquidity_2m', 'Dív.Brut/ Patr.': 'Debt_to_Equity',
            'Cresc. Rec.5a': 'Revenue_Growth_5y'
        })
        
        # Limpeza
        df['Ticker'] = df['Ticker'].astype(str).str.upper() + '.SA'
        
        cols_pct = ['Div_Yield', 'EBIT_Margin', 'Net_Margin', 'ROE', 'Revenue_Growth_5y']
        for col in cols_pct:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('.', '').str.replace(',', '.').str.replace('%', '')
                df[col] = pd.to_numeric(df[col], errors='coerce') / 100.0
        
        df = df[df['Liquidity_2m'] > 0]
        
        return df.set_index('Ticker')
        
    except Exception as e:
        st.error(f"Erro ao acessar Fundamentus: {e}")
        return pd.DataFrame()

def calcular_indicadores_tecnicos(df):
    """Calcula indicadores técnicos completos."""
    df = df.copy()
    
    # Validação se o dataframe não está vazio
    if df.empty or 'Close' not in df.columns:
        return df
        
    # Retornos
    df['Returns'] = df['Close'].pct_change()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Bandas de Bollinger
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])
    
    # Volatilidade
    df['Volatility_Ann'] = df['Returns'].rolling(window=21).std() * np.sqrt(252)
    
    # Médias Móveis
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Momentum
    df['Momentum'] = df['Close'].pct_change(10)
    
    return df

# =============================================================================
# 6. CLASSE: OTIMIZADOR DE PORTFÓLIO
# =============================================================================

class PortfolioBuilder:
    """
    Classe central que orquestra a coleta, análise, machine learning e otimização.
    """
    def __init__(self, investment_amount):
        self.investment = investment_amount
        self.tickers = []
        self.data_hist = {}      
        self.data_fund = None    
        
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
        self.status_log.append(f"{datetime.now().strftime('%H:%M:%S')} - {message}")

    def load_data(self, universe_tickers):
        """Carrega dados Fundamentalistas e Técnicos."""
        self.log("Iniciando coleta de dados fundamentalistas (Fundamentus)...")
        self.data_fund = coletar_dados_fundamentus()
        
        valid_tickers = [t for t in universe_tickers if t in self.data_fund.index]
        
        if not valid_tickers:
            self.log("Erro: Nenhum ticker válido encontrado no Fundamentus.")
            return False
            
        self.log(f"Iniciando download de dados históricos para {len(valid_tickers)} ativos (Yahoo Finance)...")
        
        try:
            data = yf.download(valid_tickers, period='2y', progress=False, group_by='ticker', auto_adjust=True)
        except Exception as e:
            self.log(f"Erro no yfinance: {e}")
            return False
            
        count_success = 0
        for ticker in valid_tickers:
            try:
                if len(valid_tickers) > 1:
                    # Acesso robusto a colunas quando temos múltiplos ativos
                    if ticker in data.columns.levels[0] if isinstance(data.columns, pd.MultiIndex) else False:
                        df_t = data[ticker].copy()
                    else:
                        # Tenta acessar diretamente caso a estrutura seja diferente
                        try:
                            df_t = data.xs(ticker, axis=1, level=0).copy()
                        except:
                            continue
                else:
                    df_t = data.copy()
                
                # Remove colunas vazias se houver
                df_t = df_t.dropna(axis=1, how='all')

                if df_t.empty or len(df_t) < MIN_DIAS_HISTORICO:
                    continue
                
                if 'Close' not in df_t.columns:
                    continue

                # Limpeza e Indicadores
                df_t = df_t.dropna(subset=['Close'])
                df_t = calcular_indicadores_tecnicos(df_t)
                df_t = df_t.dropna()
                
                if not df_t.empty:
                    self.data_hist[ticker] = df_t
                    count_success += 1
            except Exception as e:
                continue
                
        self.tickers = list(self.data_hist.keys())
        self.data_fund = self.data_fund.loc[self.tickers]
        
        self.log(f"Dados processados com sucesso para {len(self.tickers)} ativos.")
        return len(self.tickers) >= NUM_ATIVOS_PORTFOLIO

    def train_ensemble_models(self):
        """Treina modelos de ML (Ensemble RF + XGBoost)."""
        self.log("Iniciando treinamento do Ensemble de Machine Learning...")
        
        features = ['RSI', 'MACD', 'Volatility_Ann', 'Momentum', 'SMA_50', 'Div_Yield', 'PE_Ratio', 'ROE']
        
        for horizon_name, days in HORIZON_MAP_DAYS.items():
            self.log(f"Treinando modelos para horizonte: {horizon_name} ({days} dias)...")
            
            for ticker in self.tickers:
                try:
                    df = self.data_hist[ticker].copy()
                    
                    fund_row = self.data_fund.loc[ticker]
                    df['Div_Yield'] = fund_row.get('Div_Yield', 0)
                    df['PE_Ratio'] = fund_row.get('PE_Ratio', 0)
                    df['ROE'] = fund_row.get('ROE', 0)
                    
                    df['Future_Return'] = df['Close'].shift(-days) / df['Close'] - 1
                    df['Target'] = (df['Future_Return'] > 0).astype(int)
                    
                    df_model = df.dropna()
                    if len(df_model) < 60: continue
                        
                    X = df_model[features]
                    y = df_model['Target']
                    
                    X_train = X.iloc[:-1] 
                    y_train = y.iloc[:-1]
                    X_current = df[features].iloc[[-1]].fillna(0)
                    
                    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
                    rf.fit(X_train, y_train)
                    prob_rf = rf.predict_proba(X_current)[0][1]
                    
                    if HAS_XGBOOST:
                        xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, eval_metric='logloss', random_state=42)
                        xgb.fit(X_train, y_train)
                        prob_xgb = xgb.predict_proba(X_current)[0][1]
                    else:
                        prob_xgb = prob_rf
                        
                    ensemble_prob = (prob_rf + prob_xgb) / 2
                    
                    self.ml_predictions[horizon_name][ticker] = {
                        'prob': ensemble_prob,
                        'model_rf': prob_rf,
                        'model_xgb': prob_xgb
                    }
                except Exception:
                    continue

    def rank_and_select_assets(self, selected_horizon_key):
        """Pontuação Multifatorial e Seleção via Clustering."""
        self.log("Calculando Scores e Clusterização...")
        
        scores = pd.DataFrame(index=self.tickers)
        
        for t in self.tickers:
            pe = self.data_fund.loc[t, 'PE_Ratio']
            roe = self.data_fund.loc[t, 'ROE']
            rsi = self.data_hist[t]['RSI'].iloc[-1]
            
            ml_data = self.ml_predictions[selected_horizon_key].get(t, {'prob': 0.5})
            ml_prob = ml_data['prob']
            
            scores.loc[t, 'Rank_PE'] = 1 - pd.Series([pe]).rank(pct=True).iloc[0] if len(scores) > 1 else 0.5
            scores.loc[t, 'Rank_ROE'] = pd.Series([roe]).rank(pct=True).iloc[0] if len(scores) > 1 else 0.5
            scores.loc[t, 'Rank_RSI'] = 1 - pd.Series([rsi]).rank(pct=True).iloc[0] if len(scores) > 1 else 0.5
            scores.loc[t, 'Rank_ML'] = ml_prob
            
            score_fund = (scores.loc[t, 'Rank_PE'] + scores.loc[t, 'Rank_ROE']) / 2
            score_tech = scores.loc[t, 'Rank_RSI']
            
            scores.loc[t, 'Total_Score'] = (0.3 * score_fund) + (0.3 * score_tech) + (0.4 * scores.loc[t, 'Rank_ML'])
            
            scores.loc[t, 'Display_PE'] = pe
            scores.loc[t, 'Display_ROE'] = roe
            scores.loc[t, 'Display_ML'] = ml_prob

        cluster_features = pd.DataFrame(index=self.tickers)
        for t in self.tickers:
            cluster_features.loc[t, 'PE'] = self.data_fund.loc[t, 'PE_Ratio']
            cluster_features.loc[t, 'ROE'] = self.data_fund.loc[t, 'ROE']
            cluster_features.loc[t, 'Vol'] = self.data_hist[t]['Volatility_Ann'].iloc[-1]
            cluster_features.loc[t, 'Ret'] = self.data_hist[t]['Returns'].mean() * 252
            
        cluster_features = cluster_features.replace([np.inf, -np.inf], 0).fillna(0)
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_features)
        
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)
        
        n_clusters = min(NUM_ATIVOS_PORTFOLIO, len(self.tickers))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        scores['Cluster'] = kmeans.fit_predict(pca_data)
        
        selected = []
        for c in range(n_clusters):
            cluster_items = scores[scores['Cluster'] == c]
            best_asset = cluster_items['Total_Score'].idxmax()
            selected.append(best_asset)
            
        if len(selected) < NUM_ATIVOS_PORTFOLIO:
            remaining = scores.drop(selected).sort_values('Total_Score', ascending=False)
            needed = NUM_ATIVOS_PORTFOLIO - len(selected)
            selected.extend(remaining.head(needed).index.tolist())
            
        self.selected_assets = selected[:NUM_ATIVOS_PORTFOLIO]
        self.log(f"Ativos selecionados via Clusterização: {', '.join(self.selected_assets)}")
        return scores

    def optimize_portfolio_markowitz(self, risk_profile):
        """Otimização de Pesos (Markowitz)."""
        self.log(f"Otimizando alocação para perfil: {risk_profile}...")
        
        assets = self.selected_assets
        if not assets: return
        
        df_ret = pd.DataFrame({t: self.data_hist[t]['Returns'] for t in assets}).dropna()
        
        mu = df_ret.mean() * 252 
        cov = df_ret.cov() * 252
        num_assets = len(assets)
        
        def portfolio_stats(weights):
            weights = np.array(weights)
            ret = np.sum(mu * weights)
            vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
            sharpe = (ret - RISK_FREE_RATE) / vol
            return ret, vol, sharpe
            
        def neg_sharpe_ratio(weights): return -portfolio_stats(weights)[2]
        def minimize_volatility(weights): return portfolio_stats(weights)[1]
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((PESO_MIN, PESO_MAX) for _ in range(num_assets))
        initial_guess = [1.0/num_assets] * num_assets
        
        if risk_profile == 'CONSERVADOR' or risk_profile == 'MODERADO':
            opt_res = minimize(minimize_volatility, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
            method_used = "Mínima Volatilidade"
        else:
            opt_res = minimize(neg_sharpe_ratio, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
            method_used = "Máximo Sharpe"
            
        weights = opt_res.x
        final_stats = portfolio_stats(weights)
        
        self.allocation = {assets[i]: weights[i] for i in range(num_assets)}
        self.metrics = {
            'Retorno Esperado (a.a.)': final_stats[0],
            'Volatilidade (a.a.)': final_stats[1],
            'Sharpe Ratio': final_stats[2],
            'Método Otimização': method_used
        }
        
        for t in assets:
            row_fund = self.data_fund.loc[t]
            first_h = list(self.ml_predictions.keys())[0]
            ml_p = self.ml_predictions[first_h].get(t, {'prob':0})['prob']
            
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

def setup_page():
    st.set_page_config(
        page_title="Portfolio Elite AI v3.0",
        layout="wide",
        page_icon="🦅",
        initial_sidebar_state="expanded"
    )
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        html, body, [class*="css"] {font-family: 'Roboto', sans-serif; color: #212529;}
        .stApp {background-color: #ffffff;}
        .main-header {font-size: 3rem; font-weight: 800; color: #000000; text-align: center; margin-bottom: 1rem; border-bottom: 2px solid #000; padding-bottom: 10px;}
        .stButton button {background-color: #000000; color: #ffffff; border-radius: 5px; border: none; font-weight: bold; padding: 0.5rem 1rem;}
        div[data-testid="stMetricValue"] {font-size: 1.8rem; font-weight: bold; color: #000000;}
        .info-box {padding: 1.5rem; border-left: 5px solid #000; background-color: #f8f9fa; margin-bottom: 1rem;}
    </style>
    """, unsafe_allow_html=True)

def aba_introducao():
    st.markdown("""
    <div class="info-box">
        <h3>🦅 Bem-vindo ao Sistema de Portfólios Adaptativos (Elite Edition)</h3>
        <p>Este sistema representa o estado da arte em construção de portfólios quantitativos.</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🧠 Motor Híbrido")
        st.write("- Análise Fundamentalista (Fundamentus)")
        st.write("- Análise Técnica (Yfinance)")
        st.write("- Machine Learning (Ensemble)")
    with c2:
        st.markdown("#### 🛡️ Otimização")
        st.write("- Clusterização (Diversificação)")
        st.write("- Markowitz (Pesos)")

    st.info("👉 Navegue pelas abas para iniciar.")

def aba_selecao():
    st.markdown("### 🎯 Universo de Investimento")
    if 'selected_universe' not in st.session_state:
        st.session_state.selected_universe = ATIVOS_IBOVESPA
        
    with st.expander("📋 Visualizar Tickers Monitorados", expanded=True):
        cols = st.columns(5)
        chunks = np.array_split(ATIVOS_IBOVESPA, 5)
        for i, col in enumerate(cols):
            col.dataframe(pd.DataFrame(chunks[i], columns=['Ticker']), hide_index=True, use_container_width=True)

def aba_construtor():
    st.markdown("### 🏗️ Construtor de Portfólio AI")
    
    if 'builder' not in st.session_state:
        st.session_state.builder = None
        
    with st.form("profile_form"):
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Psicologia e Risco")
            q1 = st.selectbox("Reação a queda de 15%?", 
                             ["DT: Vendo tudo", "D: Desconfortável", "N: Aguardo", "C: Oscilação normal", "CT: Aporto mais"], index=2)
            q2 = st.selectbox("Objetivo?", 
                             ["DT: Preservação", "D: Inflação", "N: Crescimento", "C: Multiplicação", "CT: Agressivo"], index=2)
            q3 = st.selectbox("Conhecimento:", ["A: Iniciante", "B: Intermediário", "C: Avançado"], index=1)

        with c2:
            st.caption("Parâmetros")
            horizon = st.radio("Horizonte ML:", list(HORIZON_MAP_DAYS.keys()), index=1)
            capital = st.number_input("Capital (R$):", min_value=1000.0, value=50000.0, step=1000.0)
            
        submit = st.form_submit_button("🚀 Executar Motor de Análise")

    if submit:
        analyzer = AnalisadorPerfilInvestidor()
        respostas = {'risk_accept': q1, 'max_gain': q2, 'knowledge': q3, 'horizon': horizon.split(' ')[0][0]} 
        risk_level, time_horizon, score = analyzer.calcular_perfil(respostas)
        st.session_state.profile_result = {'level': risk_level, 'horizon': horizon}
        
        builder = PortfolioBuilder(capital)
        st.session_state.builder = builder
        
        progress = st.progress(0)
        status = st.empty()
        
        try:
            status.markdown("1/4: Coletando dados...")
            success = builder.load_data(ATIVOS_IBOVESPA)
            progress.progress(25)
            
            if not success:
                st.error("Falha na coleta.")
                return
                
            status.markdown(f"2/4: Treinando ML ({horizon})...")
            builder.train_ensemble_models()
            progress.progress(50)
            
            status.markdown("3/4: Seleção e Clusterização...")
            builder.rank_and_select_assets(horizon)
            progress.progress(75)
            
            status.markdown("4/4: Otimização Markowitz...")
            builder.optimize_portfolio_markowitz(risk_level)
            progress.progress(100)
            status.success("Concluído!")
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            st.error(f"Erro: {e}")

    if st.session_state.builder is not None and st.session_state.builder.metrics:
        b = st.session_state.builder
        prof = st.session_state.profile_result
        
        st.markdown("---")
        st.markdown("### 📊 Resultados")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Perfil", prof['level'])
        k2.metric("Retorno Esp.", f"{b.metrics['Retorno Esperado (a.a.)']:.1%}")
        k3.metric("Volatilidade", f"{b.metrics['Volatilidade (a.a.)']:.1%}")
        k4.metric("Sharpe", f"{b.metrics['Sharpe Ratio']:.2f}")
        
        t1, t2, t3 = st.tabs(["Alocação", "Detalhes", "Logs"])
        with t1:
            fig = px.pie(values=list(b.allocation.values()), names=[k.replace('.SA','') for k in b.allocation.keys()], 
                         title="Alocação", hole=0.4, color_discrete_sequence=obter_template_grafico()['colorway'])
            st.plotly_chart(fig, use_container_width=True)
        with t2:
            data = [{'Ativo': k.replace('.SA',''), 'Peso': f"{v:.1%}", 'Justificativa': b.justifications.get(k)} for k, v in b.allocation.items()]
            st.dataframe(pd.DataFrame(data), use_container_width=True)
        with t3:
            st.text_area("Logs", "\n".join(b.status_log))

def aba_analise_individual():
    st.markdown("### 🔍 Raio-X do Ativo")
    
    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        ticker_input = st.selectbox("Selecione o Ativo:", [t.replace('.SA', '') for t in ATIVOS_IBOVESPA])
        ticker = ticker_input + ".SA"
    with col_btn:
        st.write("")
        st.write("")
        analisar = st.button("Analisar Agora", type="primary")
        
    if analisar:
        with st.spinner("Baixando dados em tempo real..."):
            try:
                fund_df = coletar_dados_fundamentus()
                if ticker not in fund_df.index:
                    st.warning(f"Dados fundamentalistas não encontrados para {ticker}.")
                    return
                    
                info = fund_df.loc[ticker]
                hist = yf.download(ticker, period='2y', progress=False, auto_adjust=True)
                
                # --- FIX FOR EMPTY DATAFRAME CRASH ---
                if hist.empty:
                    st.error(f"Não foi possível obter dados históricos para {ticker}. Tente outro ativo.")
                    return

                # --- FIX FOR MULTI-INDEX CRASH ---
                # Se o yfinance retornar MultiIndex (Ticker, Price), precisamos achatar
                if isinstance(hist.columns, pd.MultiIndex):
                    try:
                        hist = hist.xs(ticker, axis=1, level=1)
                    except KeyError:
                        # Fallback genérico: drop nível 0
                        hist = hist.droplevel(0, axis=1)
                
                # Validação final de colunas
                if 'Close' not in hist.columns:
                     st.error(f"Estrutura de dados inválida recebida para {ticker}.")
                     return

                hist = calcular_indicadores_tecnicos(hist)
                
                # Check if indicators cleaned everything
                if hist.empty:
                     st.error(f"Histórico insuficiente para cálculo de indicadores de {ticker}.")
                     return
                
                st.markdown(f"#### {ticker_input} - Indicadores Chave")
                k1, k2, k3, k4 = st.columns(4)
                current_price = hist['Close'].iloc[-1] # Now safe to access
                
                k1.metric("Preço Atual", f"R$ {current_price:.2f}")
                k2.metric("P/L", f"{info['PE_Ratio']:.2f}")
                k3.metric("ROE", f"{info['ROE']:.1%}")
                k4.metric("RSI", f"{hist['RSI'].iloc[-1]:.1f}")
                
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2])
                fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='Preço'), row=1, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_Upper'], line=dict(color='gray', dash='dot'), name='BB Upper'), row=1, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_Lower'], line=dict(color='gray', dash='dot'), name='BB Lower', fill='tonexty'), row=1, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", row=2, col=1); fig.add_hline(y=30, line_dash="dash", row=2, col=1)
                fig.add_trace(go.Bar(x=hist.index, y=hist['MACD_Hist'], name='MACD'), row=3, col=1)
                fig.update_layout(height=800, xaxis_rangeslider_visible=False, plot_bgcolor='white')
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Erro na análise: {e}")
                st.write(traceback.format_exc())

def aba_referencias():
    st.markdown("### 📚 Referências")
    st.markdown("""
    1. **Markowitz, H.** (1952). *Portfolio Selection*.
    2. **Géron, A.** (2019). *Hands-On Machine Learning*.
    3. **Fundamentus** & **B3**.
    """)

# =============================================================================
# 8. MAIN
# =============================================================================

def main():
    setup_page()
    st.markdown('<h1 class="main-header">🦅 Sistema de Portfólios Adaptativos Elite</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Introdução", "🎯 Seleção", "🏗️ Construtor", "🔍 Análise Individual", "📖 Referências"])
    
    with tab1: aba_introducao()
    with tab2: aba_selecao()
    with tab3: aba_construtor()
    with tab4: aba_analise_individual()
    with tab5: aba_referencias()

if __name__ == "__main__":
    main()
