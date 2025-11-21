# -*- coding: utf-8 -*-
"""
=============================================================================
SISTEMA DE PORTFÓLIOS ADAPTATIVOS - ELITE LIVE EDITION
=============================================================================

Integração Total:
- Design & UX: "automl_portfolio_elite_v7_final.py" (Visual Premium, Abas, CSS).
- Lógica Analítica: "portfolio_analyzer.py" (yfinance, Fundamentus, Live ML).

Funcionalidades:
1. Coleta em Tempo Real (yfinance + WebScraping).
2. Machine Learning On-the-Fly (Random Forest + XGBoost).
3. Engenharia de Features Técnica e Fundamentalista.
4. Otimização de Markowitz baseada no Perfil do Investidor (Questionário V7).

Versão: 2.0.0 (Final Fusion)
=============================================================================
"""

# --- 1. CORE LIBRARIES & UTILITIES ---
import warnings
import numpy as np
import pandas as pd
import subprocess
import sys
import time
import traceback
import requests
import json
from io import StringIO
from datetime import datetime, timedelta

# --- 2. SCIENTIFIC / STATISTICAL TOOLS ---
from scipy.optimize import minimize
from scipy.stats import zscore

# --- 3. STREAMLIT & VISUALIZATION ---
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- 4. MACHINE LEARNING ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

# Instalação automática de dependências críticas
def ensure_package(package_name):
    try:
        __import__(package_name)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])

# Garante XGBoost
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
except ImportError:
    ensure_package('xgboost')
    import xgboost as xgb
    from xgboost import XGBClassifier

# --- 5. CONFIGURAÇÃO ---
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURAÇÕES GLOBAIS
# =============================================================================

PERIODO_DADOS = '5y'
MIN_DIAS_HISTORICO = 252
NUM_ATIVOS_PORTFOLIO = 5
TAXA_LIVRE_RISCO = 0.1075
SCORE_PERCENTILE_THRESHOLD = 0.70 

# Pesos de Alocação (Constraints)
PESO_MIN = 0.10
PESO_MAX = 0.30

# Horizontes ML
HORIZONTES_ML = {
    'Curto Prazo (4m)': 84,
    'Médio Prazo (8m)': 168,
    'Longo Prazo (12m)': 252
}

# Lista de Ativos (Ibovespa)
ATIVOS_IBOVESPA = [
    'ALOS3.SA', 'ABEV3.SA', 'ASAI3.SA', 'AESB3.SA', 'AZZA3.SA', 'B3SA3.SA',
    'BBSE3.SA', 'BBDC3.SA', 'BBDC4.SA', 'BRAP4.SA', 'BBAS3.SA', 'BRKM5.SA',
    'BRAV3.SA', 'BPAC11.SA', 'CXSE3.SA', 'CMIG4.SA', 'COGN3.SA',
    'CPLE6.SA', 'CSAN3.SA', 'CPFE3.SA', 'CMIN3.SA', 'CURY3.SA', 'CVCB3.SA',
    'CYRE3.SA', 'DIRR3.SA', 'ELET3.SA', 'ELET6.SA', 'EMBR3.SA', 'ENGI11.SA',
    'ENEV3.SA', 'EGIE3.SA', 'EQTL3.SA', 'FLRY3.SA', 'GGBR4.SA', 'GOAU4.SA',
    'HAPV3.SA', 'HYPE3.SA', 'IGTI11.SA', 'IRBR3.SA', 'ITSA4.SA',
    'ITUB4.SA', 'KLBN11.SA', 'RENT3.SA', 'LREN3.SA', 'MGLU3.SA', 'POMO4.SA',
    'BEEF3.SA', 'MRVE3.SA', 'MULT3.SA', 'NATU3.SA',
    'PCAR3.SA', 'PETR3.SA', 'PETR4.SA', 'RECV3.SA', 'PRIO3.SA', 'PSSA3.SA',
    'RADL3.SA', 'RAIZ4.SA', 'RDOR3.SA', 'RAIL3.SA', 'SBSP3.SA', 'SANB11.SA',
    'CSNA3.SA', 'SLCE3.SA', 'SMFT3.SA', 'SUZB3.SA', 'TAEE11.SA', 'VIVT3.SA',
    'TIMS3.SA', 'TOTS3.SA', 'UGPA3.SA', 'USIM5.SA', 'VALE3.SA', 'VAMO3.SA',
    'VBBR3.SA', 'VIVA3.SA', 'WEGE3.SA', 'YDUQ3.SA'
]

ATIVOS_POR_SETOR = {
    'Financeiro': ['BBAS3.SA', 'BBDC4.SA', 'ITUB4.SA', 'B3SA3.SA', 'BPAC11.SA', 'SANB11.SA', 'BBSE3.SA'],
    'Materiais Básicos': ['VALE3.SA', 'GGBR4.SA', 'CSNA3.SA', 'SUZB3.SA', 'KLBN11.SA', 'BRAP4.SA'],
    'Petróleo e Gás': ['PETR4.SA', 'PETR3.SA', 'PRIO3.SA', 'UGPA3.SA', 'RAIZ4.SA', 'VBBR3.SA'],
    'Utilidade Pública': ['ELET3.SA', 'ELET6.SA', 'EQTL3.SA', 'CMIG4.SA', 'CPLE6.SA', 'SBSP3.SA', 'TAEE11.SA'],
    'Consumo e Varejo': ['MGLU3.SA', 'LREN3.SA', 'RENT3.SA', 'ASAI3.SA', 'RDOR3.SA', 'HAPV3.SA'],
    'Industrial e Bens': ['WEGE3.SA', 'EMBR3.SA', 'AZZA3.SA', 'RAIL3.SA']
}
TODOS_ATIVOS = sorted(list(set(ATIVOS_IBOVESPA)))

# =============================================================================
# 2. CLASSES LÓGICAS (Backend / Engine)
# =============================================================================

class FundamentusScraper:
    """WebScraping do site Fundamentus (Backend Lógico)."""
    @staticmethod
    def obter_dados_consolidados():
        url = 'https://www.fundamentus.com.br/resultado.php'
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            r = requests.get(url, headers=headers)
            r.raise_for_status()
            df = pd.read_html(StringIO(r.text), decimal=',', thousands='.')[0]
            
            mapper = {
                'Papel': 'ticker', 'Cotação': 'price', 'P/L': 'pe_ratio', 'P/VP': 'pb_ratio',
                'Div.Yield': 'div_yield', 'ROE': 'roe', 'Mrg. Líq.': 'net_margin',
                'Dív.Brut/ Patrim.': 'debt_to_equity', 'Cresc. Rec.5a': 'revenue_growth'
            }
            df.rename(columns=mapper, inplace=True)
            
            for col in df.columns:
                if df[col].dtype == object and col != 'ticker':
                    df[col] = df[col].str.replace('%', '').str.replace('.', '').str.replace(',', '.').astype(float)
                    if col in ['div_yield', 'roe', 'net_margin', 'revenue_growth']:
                        df[col] = df[col] / 100.0
            
            return df.set_index('ticker')
        except Exception as e:
            # st.error(f"Erro Fundamentus: {e}") # Silencioso para não quebrar layout
            return pd.DataFrame()

class EngenheiroFeatures:
    """Cálculo de Indicadores Técnicos."""
    @staticmethod
    def calcular_indicadores_tecnicos(df):
        if df.empty: return df
        df = df.copy()
        df['returns'] = df['Close'].pct_change()
        
        # RSI 14
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # Volatilidade, BBands e Momentum
        df['volatility_20d'] = df['returns'].rolling(20).std() * np.sqrt(252)
        df['momentum_60'] = df['Close'].pct_change(60)
        df['bb_width'] = (df['Close'].rolling(20).mean() + 2*df['Close'].rolling(20).std() - (df['Close'].rolling(20).mean() - 2*df['Close'].rolling(20).std())) / df['Close'].rolling(20).mean()
        
        return df

    @staticmethod
    def normalizar_score(serie, maior_melhor=True):
        if serie.isnull().all(): return pd.Series(0.5, index=serie.index)
        q_low, q_high = serie.quantile(0.05), serie.quantile(0.95)
        serie_clipped = serie.clip(q_low, q_high)
        min_v, max_v = serie_clipped.min(), serie_clipped.max()
        if max_v == min_v: return pd.Series(0.5, index=serie.index)
        norm = (serie_clipped - min_v) / (max_v - min_v)
        return norm if maior_melhor else 1 - norm

class EngineML:
    """Ensemble de Machine Learning (RF + XGB) em Tempo Real."""
    def __init__(self):
        self.resultados = {}
        self.feature_importances = {} 

    def treinar_ensemble(self, df_historico, ticker):
        if len(df_historico) < MIN_DIAS_HISTORICO: return None
        
        features_col = ['rsi_14', 'macd', 'volatility_20d', 'momentum_60', 'returns']
        df_ml = df_historico[features_col].dropna().copy()
        resultados_ticker = {}
        
        for horizonte_nome, dias_target in HORIZONTES_ML.items():
            df_ml['target'] = (df_historico['Close'].shift(-dias_target) > df_historico['Close']).astype(int)
            data_train = df_ml.dropna()
            
            if len(data_train) < 100 or len(data_train['target'].unique()) < 2:
                resultados_ticker[horizonte_nome] = {'proba': 0.5, 'auc': 0.5}
                continue
                
            X = data_train[features_col]
            y = data_train['target']
            
            # Divisão Temporal
            split = int(len(X) * 0.85)
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]
            
            # Modelos
            rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
            rf.fit(X_train, y_train)
            
            xgb_model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, eval_metric='logloss')
            xgb_model.fit(X_train, y_train)
            
            # Avaliação
            try:
                pred_ens = (rf.predict_proba(X_test)[:, 1] + xgb_model.predict_proba(X_test)[:, 1]) / 2
                auc = roc_auc_score(y_test, pred_ens)
            except:
                auc = 0.5
            
            # Predição Atual
            last_feat = df_ml[features_col].iloc[[-1]]
            final_proba = (rf.predict_proba(last_feat)[0][1] + xgb_model.predict_proba(last_feat)[0][1]) / 2
            
            resultados_ticker[horizonte_nome] = {'proba': final_proba, 'auc': auc}
            
            # Salva importância para visualização (apenas do Longo Prazo para simplificar UX)
            if horizonte_nome == 'Longo Prazo (12m)':
                self.feature_importances[ticker] = dict(zip(features_col, rf.feature_importances_))

        return resultados_ticker

class ConstrutorPortfolio:
    """Orquestrador Lógico."""
    def __init__(self, investimento_inicial):
        self.investimento = investimento_inicial
        self.dados_mercado = {}
        self.dados_fundamentos = pd.DataFrame()
        self.scores = pd.DataFrame()
        self.metricas_performance = pd.DataFrame()
        self.ml_results = {}
        self.ativos_selecionados = []
        self.alocacao = {}
        self.metricas_finais = {}
        self.justificativas = {}
        self.pesos_atuais = {} # Para mostrar na UI
        self.engine_ml = EngineML()
        import yfinance as yf
        self.yf = yf

    def coletar_dados(self, tickers, progress_bar=None):
        if progress_bar: progress_bar.progress(10, text="📡 Conectando ao Fundamentus (WebScraping)...")
        df_fund = FundamentusScraper.obter_dados_consolidados()
        
        valid_tickers = []
        perf_list = []
        total = len(tickers)
        
        for i, ticker in enumerate(tickers):
            if progress_bar:
                p = 20 + int((i/total)*60)
                progress_bar.progress(p, text=f"🧠 Processando {ticker} (yfinance + ML Ensemble)...")
            
            try:
                hist = self.yf.download(ticker, period=PERIODO_DADOS, progress=False)
                if len(hist) < MIN_DIAS_HISTORICO: continue
                if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
                
                hist = EngenheiroFeatures.calcular_indicadores_tecnicos(hist)
                self.dados_mercado[ticker] = hist
                valid_tickers.append(ticker)
                
                # Métricas Performance
                ret = hist['returns'].mean() * 252
                vol = hist['returns'].std() * np.sqrt(252)
                sharpe = (ret - TAXA_LIVRE_RISCO) / vol if vol > 0 else 0
                cum_ret = (1 + hist['returns']).cumprod()
                dd = (cum_ret / cum_ret.expanding().max()) - 1
                max_dd = dd.min()
                
                perf_list.append({'ticker': ticker, 'sharpe': sharpe, 'retorno_anual': ret, 'volatilidade_anual': vol, 'max_drawdown': max_dd})
                
                # ML Live Train
                ml_res = self.engine_ml.treinar_ensemble(hist, ticker)
                if ml_res: self.ml_results[ticker] = ml_res
                
            except: continue
        
        if not valid_tickers: return False
        
        self.metricas_performance = pd.DataFrame(perf_list).set_index('ticker')
        
        # Cruzamento Fundamentalista
        tickers_clean = [t.replace('.SA', '') for t in valid_tickers]
        fund_sub = df_fund[df_fund.index.isin(tickers_clean)].copy()
        fund_sub.index = [t + '.SA' for t in fund_sub.index]
        
        # Merge dados setoriais manuais
        fund_sub['sector'] = 'Unknown'
        for t in fund_sub.index:
            for s, lista in ATIVOS_POR_SETOR.items():
                if t in lista: fund_sub.loc[t, 'sector'] = s; break
                
        self.dados_fundamentos = fund_sub
        return True

    def calcular_scores(self, perfil_risco, horizonte):
        # Pesos Dinâmicos
        if 'Curto' in horizonte: w_perf, w_fund, w_tech, w_ml = 0.3, 0.1, 0.3, 0.3; target_ml = 'Curto Prazo (4m)'
        elif 'Médio' in horizonte: w_perf, w_fund, w_tech, w_ml = 0.3, 0.3, 0.2, 0.2; target_ml = 'Médio Prazo (8m)'
        else: w_perf, w_fund, w_tech, w_ml = 0.2, 0.5, 0.1, 0.2; target_ml = 'Longo Prazo (12m)'
        
        self.pesos_atuais = {'Performance': w_perf, 'Fundamentos': w_fund, 'Técnicos': w_tech, 'ML': w_ml}
        
        df = self.metricas_performance.join(self.dados_fundamentos, how='left')
        
        # Fill Current Tech & ML
        for t in df.index:
            if t in self.dados_mercado:
                df.loc[t, 'rsi_curr'] = self.dados_mercado[t]['rsi_14'].iloc[-1]
                df.loc[t, 'macd_curr'] = self.dados_mercado[t]['macd_diff'].iloc[-1]
            res = self.ml_results.get(t, {}).get(target_ml, {'proba': 0.5, 'auc': 0.5})
            df.loc[t, 'ml_proba'] = res['proba']
            df.loc[t, 'ml_conf'] = res['auc']
            
        scores = pd.DataFrame(index=df.index)
        
        # Scoring
        scores['s_perf'] = EngenheiroFeatures.normalizar_score(df['sharpe'], True) * w_perf
        scores['s_fund'] = ((EngenheiroFeatures.normalizar_score(df['pe_ratio'], False) + EngenheiroFeatures.normalizar_score(df['roe'], True))/2) * w_fund
        scores['s_tech'] = ((EngenheiroFeatures.normalizar_score(df['rsi_curr'], False) + EngenheiroFeatures.normalizar_score(df['macd_curr'], True))/2) * w_tech
        scores['s_ml'] = (EngenheiroFeatures.normalizar_score(df['ml_proba'], True) * df['ml_conf']) * w_ml
        
        scores['total_score'] = scores.sum(axis=1)
        self.scores = scores.join(df).sort_values('total_score', ascending=False)
        
        # Clusterização e Seleção
        feat_clust = ['sharpe', 'volatilidade_anual', 'pe_ratio', 'roe', 'ml_proba']
        data_clust = self.scores[feat_clust].fillna(0)
        if len(data_clust) > 5:
            pca = PCA(n_components=min(3, len(data_clust))).fit_transform(StandardScaler().fit_transform(data_clust))
            kmeans = KMeans(n_clusters=min(5, len(data_clust)), random_state=42).fit_predict(pca)
            self.scores['Cluster'] = kmeans
            
            sel = []
            for c in np.unique(kmeans):
                sel.append(self.scores[self.scores['Cluster'] == c].index[0])
            while len(sel) < NUM_ATIVOS_PORTFOLIO:
                rem = [x for x in self.scores.index if x not in sel]
                if not rem: break
                sel.append(rem[0])
            self.ativos_selecionados = sel[:NUM_ATIVOS_PORTFOLIO]
        else:
            self.ativos_selecionados = self.scores.head(NUM_ATIVOS_PORTFOLIO).index.tolist()

    def otimizar(self, risco):
        ativos = self.ativos_selecionados
        if not ativos: return
        
        ret_df = pd.DataFrame({t: self.dados_mercado[t]['returns'] for t in ativos}).dropna()
        mu, cov = ret_df.mean()*252, ret_df.cov()*252
        
        def get_stats(w): 
            return np.sum(mu*w), np.sqrt(np.dot(w.T, np.dot(cov, w)))
        
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x)-1})
        bnds = tuple((PESO_MIN, PESO_MAX) for _ in range(len(ativos)))
        init = [1/len(ativos)] * len(ativos)
        
        # Estratégia baseada no questionário
        if 'CONSERVADOR' in risco or 'INTERMEDIÁRIO' in risco:
            res = minimize(lambda w: get_stats(w)[1], init, method='SLSQP', bounds=bnds, constraints=cons)
            metodo = f"Mínima Volatilidade ({risco})"
        else:
            res = minimize(lambda w: -(get_stats(w)[0]-TAXA_LIVRE_RISCO)/get_stats(w)[1], init, method='SLSQP', bounds=bnds, constraints=cons)
            metodo = f"Máximo Sharpe ({risco})"
            
        pesos = res.x
        for i, t in enumerate(ativos):
            self.alocacao[t] = {'weight': pesos[i], 'amount': self.investimento * pesos[i]}
            
        rp, vp = get_stats(pesos)
        self.metricas_finais = {'annual_return': rp, 'annual_volatility': vp, 'sharpe_ratio': (rp-TAXA_LIVRE_RISCO)/vp, 'max_drawdown': 0, 'metodo': metodo}
        
        # Justificativas
        for t in ativos:
            row = self.scores.loc[t]
            self.justificativas[t] = f"Score Total: {row['total_score']:.3f} | ML: {row['ml_proba']*100:.1f}% (Conf: {row['ml_conf']:.2f}) | Fund: P/L {row['pe_ratio']:.1f}, ROE {row['roe']*100:.1f}%"

# =============================================================================
# 3. INTERFACE VISUAL (Cópia Fiel do V7)
# =============================================================================

def obter_template_grafico():
    """Template 'Elite' V7."""
    return {
        'plot_bgcolor': '#f8f9fa',
        'paper_bgcolor': 'white',
        'font': {'family': 'Arial, sans-serif', 'size': 12, 'color': '#343a40'},
        'title': {'font': {'family': 'Arial, sans-serif', 'size': 16, 'color': '#212529', 'weight': 'bold'}, 'x': 0.5, 'xanchor': 'center'},
        'xaxis': {'showgrid': True, 'gridcolor': '#e9ecef', 'showline': True, 'linecolor': '#ced4da'},
        'yaxis': {'showgrid': True, 'gridcolor': '#e9ecef', 'showline': True, 'linecolor': '#ced4da'},
        'legend': {'bgcolor': 'rgba(255, 255, 255, 0.8)', 'bordercolor': '#e9ecef', 'borderwidth': 1},
        'colorway': ['#212529', '#495057', '#6c757d', '#adb5bd', '#ced4da']
    }

def configurar_pagina():
    st.set_page_config(page_title="Sistema de Portfólios Adaptativos", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
        <style>
        :root { --primary-color: #000000; --secondary-color: #6c757d; --background-light: #ffffff; --background-dark: #f8f9fa; --text-color: #212529; --text-color-light: #ffffff; --border-color: #dee2e6; }
        body { background-color: var(--background-light); color: var(--text-color); }
        .main-header { font-family: 'Arial', sans-serif; color: var(--primary-color); text-align: center; border-bottom: 2px solid var(--border-color); padding-bottom: 10px; font-size: 2.2rem !important; margin-bottom: 20px; font-weight: 600; }
        .stButton button, .stDownloadButton button { border: 1px solid var(--primary-color) !important; color: var(--primary-color) !important; border-radius: 6px; padding: 8px 16px; transition: all 0.3s ease; background-color: transparent !important; }
        .stButton button:hover, .stDownloadButton button:hover { background-color: var(--primary-color) !important; color: var(--text-color-light) !important; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); }
        .stButton button[kind="primary"], .stFormSubmitButton button { background-color: var(--primary-color) !important; color: var(--text-color-light) !important; border: none !important; }
        .stTabs [data-baseweb="tab-list"] { gap: 24px; border-bottom: 2px solid var(--border-color); display: flex; justify-content: center; width: 100%; }
        .stTabs [data-baseweb="tab"] { height: 40px; background-color: transparent; padding-top: 5px; padding-bottom: 5px; color: var(--secondary-color); font-weight: 500; flex-grow: 0 !important; }
        .stTabs [aria-selected="true"] { border-bottom: 2px solid var(--primary-color); color: var(--primary-color); font-weight: 700; }
        .info-box { background-color: var(--background-dark); border-left: 4px solid var(--primary-color); padding: 15px; margin: 10px 0; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
        .stMetric { padding: 10px 15px; background-color: var(--background-dark); border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 10px; }
        .stMetric label { font-weight: 600; color: var(--text-color); }
        .stProgress > div > div > div > div { background-color: var(--primary-color); }
        </style>
    """, unsafe_allow_html=True)

class AnalisadorPerfilInvestidor:
    """Lógica do V7 para converter respostas em Perfil."""
    def calcular_perfil(self, resp):
        # Scores baseados no V7
        s_risk = {'CT':5, 'C':4, 'N':3, 'D':2, 'DT':1}.get(resp['risk_accept'].split(':')[0], 3)
        s_gain = {'CT':5, 'C':4, 'N':3, 'D':2, 'DT':1}.get(resp['max_gain'].split(':')[0], 3)
        s_stab = {'CT':1, 'C':2, 'N':3, 'D':4, 'DT':5}.get(resp['stable_growth'].split(':')[0], 3) # Invertido
        s_loss = {'CT':1, 'C':2, 'N':3, 'D':4, 'DT':5}.get(resp['avoid_loss'].split(':')[0], 3) # Invertido
        s_react = {'A':1, 'B':3, 'C':5}.get(resp['reaction'][0], 3)
        
        pts = s_risk*5 + s_gain*5 + s_stab*5 + s_loss*5 + s_react*3
        
        if pts <= 46: nivel = "CONSERVADOR"
        elif pts <= 67: nivel = "INTERMEDIÁRIO"
        elif pts <= 88: nivel = "MODERADO"
        elif pts <= 109: nivel = "MODERADO-ARROJADO"
        else: nivel = "AVANÇADO"
        
        # Horizonte
        liq = {'A':5, 'B':20, 'C':30}.get(resp['liquidity'][0], 20)
        time = {'A':5, 'B':20, 'C':30}.get(resp['time_purpose'][0], 20)
        final_lb = max(liq, time)
        
        if final_lb >= 30: hz = "LONGO PRAZO"
        elif final_lb >= 20: hz = "MÉDIO PRAZO"
        else: hz = "CURTO PRAZO"
        
        return nivel, hz, pts

# --- ABAS CONTEÚDO ---

def aba_introducao():
    st.markdown("## 📚 Metodologia Quantitativa e Arquitetura do Sistema")
    st.markdown("""
    <div class="info-box">
    <h3>🎯 Visão Geral (Live Engine)</h3>
    <p>Este sistema opera em <b>Tempo Real</b>. Diferente de versões anteriores que dependiam de processamento offline (GCS), esta versão "Elite Live" conecta-se diretamente às fontes de dados (B3/Fundamentus) e treina modelos de IA sob demanda.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('### 1. O "Motor" de Dados (Live)')
        with st.expander("Etapa 1.1: Coleta e Engenharia Híbrida"):
            st.markdown("""
            - **Técnica:** `yfinance` baixa OHLCV em tempo real. RSI, MACD e Volatilidade calculados na hora.
            - **Fundamentalista:** WebScraping proprietário acessa `Fundamentus` para P/L, ROE, Margens atualizados.
            """)
        with st.expander("Etapa 1.2: Machine Learning (Ensemble On-the-Fly)"):
            st.markdown("""
            O sistema treina um **Ensemble (Random Forest + XGBoost)** para cada ativo no momento da execução.
            - **Target:** Retorno positivo no horizonte selecionado (4, 8 ou 12 meses).
            - **Validação:** Walk-Forward Analysis para evitar *data leakage*.
            """)
    
    with col2:
        st.markdown('### 2. O "Painel" de Otimização')
        with st.expander("Etapa 2.1: Ranqueamento Multi-Fatorial"):
            st.markdown("Score composto por: **Performance (Sharpe)**, **Fundamentos (Valor/Qualidade)**, **Técnicos (Momentum)** e **ML (Probabilidade + Confiança)**.")
        with st.expander("Etapa 2.2: Clusterização e Otimização"):
            st.markdown("""
            - **Diversificação:** PCA + KMeans agrupam ativos similares.
            - **Alocação:** Algoritmo de Markowitz (SLSQP) define os pesos ideais (Min Volatilidade ou Max Sharpe) baseado no seu perfil.
            """)

def aba_selecao_ativos():
    st.markdown("## 🎯 Definição do Universo de Análise")
    st.markdown("""<div class="info-box"><p>O universo de análise está restrito ao **Índice Ibovespa**. O sistema utiliza todos os ativos selecionados para realizar o ranqueamento multi-fatorial e otimizar a carteira.</p></div>""", unsafe_allow_html=True)
    
    modo = st.radio("**Modo de Seleção:**", ["📊 Índice de Referência (Todos do Ibovespa)", "🏢 Seleção Setorial", "✍️ Seleção Individual"])
    
    selecionados = []
    if "Índice" in modo:
        selecionados = ATIVOS_IBOVESPA
        st.success(f"✓ **{len(selecionados)} ativos** (Ibovespa completo) definidos.")
    elif "Setorial" in modo:
        setores = st.multiselect("Escolha setores:", list(ATIVOS_POR_SETOR.keys()))
        for s in setores: selecionados.extend(ATIVOS_POR_SETOR[s])
        selecionados = list(set(selecionados))
    else:
        selecionados = st.multiselect("Selecione Tickers:", TODOS_ATIVOS)
        
    if selecionados:
        st.session_state.ativos_analise = selecionados
        col1, col2, col3 = st.columns(3)
        col1.metric("Tickers", len(selecionados))
        col2.metric("Para Ranqueamento", len(selecionados))
        col3.metric("Carteira Final", NUM_ATIVOS_PORTFOLIO)
    else:
        st.warning("⚠️ Selecione ativos.")

def aba_construtor_portfolio():
    if 'ativos_analise' not in st.session_state:
        st.warning("⚠️ Defina o universo na aba 'Seleção de Ativos'.")
        return

    if not st.session_state.builder_complete:
        st.markdown('## 📋 Calibração do Perfil de Risco')
        st.info(f"✓ **{len(st.session_state.ativos_analise)} ativos** prontos. Responda o questionário.")
        
        with st.form("profile_form"):
            c1, c2 = st.columns(2)
            opts_conc = ["CT: Concordo Totalmente", "C: Concordo", "N: Neutro", "D: Discordo", "DT: Discordo Totalmente"]
            opts_react = ["A: Venderia", "B: Manteria", "C: Compraria Mais"]
            opts_know = ["A: Avançado", "B: Intermediário", "C: Iniciante"]
            opts_time = ["A: Curto (<1 ano)", "B: Médio (1-5 anos)", "C: Longo (>5 anos)"]
            opts_liq = ["A: <6 meses", "B: 6m-2anos", "C: >2 anos"]
            
            with c1:
                st.markdown("#### Tolerância ao Risco")
                ra = st.radio("1. Aceito volatilidade por retorno?", opts_conc, index=2)
                mg = st.radio("2. Retorno máximo é prioridade?", opts_conc, index=2)
                sg = st.radio("3. Prefiro estabilidade a ganhos?", opts_conc, index=2) # Inv
                al = st.radio("4. Evitar perdas é crucial?", opts_conc, index=2) # Inv
                re = st.radio("5. Reação a queda de 10%?", opts_react, index=1)
                kn = st.radio("6. Conhecimento?", opts_know, index=1)
                
            with c2:
                st.markdown("#### Horizonte e Capital")
                tp = st.radio("7. Horizonte de Investimento?", opts_time, index=1)
                lq = st.radio("8. Necessidade de Liquidez?", opts_liq, index=1)
                inv = st.number_input("Capital (R$)", 1000, 10000000, 10000)
                
            sub = st.form_submit_button("🚀 Gerar Alocação Otimizada", type="primary")
            
            if sub:
                resp = {'risk_accept': ra, 'max_gain': mg, 'stable_growth': sg, 'avoid_loss': al, 'reaction': re, 'level': kn, 'time_purpose': tp, 'liquidity': lq}
                nivel, hz, pts = AnalisadorPerfilInvestidor().calcular_perfil(resp)
                
                st.session_state.profile = {'nivel': nivel, 'horizonte': hz, 'score': pts}
                builder = ConstrutorPortfolio(inv)
                st.session_state.builder = builder
                
                pb = st.progress(0, text=f"Iniciando Engine para Perfil {nivel}...")
                
                succ = builder.coletar_dados(st.session_state.ativos_analise, pb)
                if not succ: st.error("Falha na coleta."); return
                
                pb.progress(80, text="Calculando Scores e Clusterização...")
                builder.calcular_scores(nivel, hz)
                
                pb.progress(90, text="Otimizando Markowitz...")
                builder.otimizar(nivel)
                
                pb.progress(100, text="Finalizado!")
                st.session_state.builder_complete = True
                st.rerun()

    else: # Resultados
        b = st.session_state.builder
        p = st.session_state.profile
        
        st.markdown('## ✅ Relatório de Alocação Otimizada')
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Perfil", p['nivel'], f"Score: {p['score']}")
        c2.metric("Horizonte", p['horizonte'])
        c3.metric("Sharpe (Portfólio)", f"{b.metricas_finais['sharpe_ratio']:.2f}")
        c4.metric("Estratégia", b.metricas_finais['metodo'])
        
        if st.button("🔄 Recalibrar"): 
            st.session_state.builder_complete = False; st.rerun()
            
        st.markdown("---")
        t1, t2, t3, t4 = st.tabs(["📊 Alocação", "📈 Performance", "🤖 ML Factor", "❓ Justificativas"])
        
        with t1:
            ca, cb = st.columns([1, 2])
            df_a = pd.DataFrame([{'Ativo': k.replace('.SA',''), 'Peso': v['weight']*100} for k, v in b.alocacao.items()])
            if not df_a.empty:
                fig = px.pie(df_a, values='Peso', names='Ativo', hole=0.3)
                fig.update_layout(obter_template_grafico())
                ca.plotly_chart(fig, use_container_width=True)
                
            rows = []
            for k, v in b.alocacao.items():
                rows.append({'Ticker': k, 'Peso (%)': f"{v['weight']*100:.1f}%", 'Valor (R$)': f"R$ {v['amount']:,.2f}"})
            cb.dataframe(pd.DataFrame(rows), use_container_width=True)
            
        with t2:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Retorno Anual", f"{b.metricas_finais['annual_return']*100:.1f}%")
            c2.metric("Volatilidade", f"{b.metricas_finais['annual_volatility']*100:.1f}%")
            c3.metric("Sharpe", f"{b.metricas_finais['sharpe_ratio']:.2f}")
            c4.metric("Max Drawdown", "N/A") # Simplificação
            
            fig = go.Figure()
            for a in b.alocacao.keys():
                if a in b.dados_mercado:
                    r = (1+b.dados_mercado[a]['returns']).cumprod()
                    fig.add_trace(go.Scatter(x=r.index, y=r.values, name=a.replace('.SA','')))
            fig.update_layout(title="Retorno Acumulado", **obter_template_grafico())
            st.plotly_chart(fig, use_container_width=True)
            
        with t3:
            ml_d = []
            for a in b.alocacao.keys():
                score_row = b.scores.loc[a]
                ml_d.append({'Ticker': a, 'Prob. Alta (%)': score_row['ml_proba']*100, 'Confiança': score_row['ml_conf']})
            df_ml = pd.DataFrame(ml_d).sort_values('Prob. Alta (%)', ascending=False)
            
            fig = px.bar(df_ml, x='Ticker', y='Prob. Alta (%)', color='Prob. Alta (%)', color_continuous_scale='Greys')
            fig.update_layout(obter_template_grafico())
            st.plotly_chart(fig, use_container_width=True)
            
        with t4:
            st.markdown(f"**Pesos Usados:** {b.pesos_atuais}")
            for a, just in b.justificativas.items():
                st.markdown(f"""<div class="info-box"><h4>{a} ({b.alocacao[a]['weight']*100:.1f}%)</h4><p>{just}</p></div>""", unsafe_allow_html=True)

def aba_analise_individual():
    st.markdown("## 🔍 Análise de Fatores por Ticker")
    if 'builder' not in st.session_state:
        st.warning("Execute o construtor primeiro.")
        return
        
    b = st.session_state.builder
    ativos = list(b.dados_mercado.keys())
    sel = st.selectbox("Selecione Ticker:", ativos)
    
    if st.button("🔄 Executar Análise", type="primary"):
        df = b.dados_mercado[sel]
        fund = b.dados_fundamentos[b.dados_fundamentos.index == sel]
        
        t1, t2, t3 = st.tabs(["📊 Histórico", "💼 Fundamentos", "🤖 ML Insight"])
        
        with t1:
            c1, c2, c3 = st.columns(3)
            c1.metric("Preço", f"R$ {df['Close'].iloc[-1]:.2f}")
            c2.metric("RSI", f"{df['rsi_14'].iloc[-1]:.1f}")
            c3.metric("Volatilidade", f"{df['volatility_20d'].iloc[-1]*100:.1f}%")
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Preço'), row=1, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['macd_diff'], name='MACD Hist', marker_color='gray'), row=2, col=1)
            fig.update_layout(height=600, **obter_template_grafico())
            st.plotly_chart(fig, use_container_width=True)
            
        with t2:
            if not fund.empty:
                f = fund.iloc[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("P/L", f"{f['pe_ratio']:.2f}")
                c2.metric("ROE", f"{f['roe']*100:.1f}%")
                c3.metric("DY", f"{f['div_yield']*100:.1f}%")
                st.dataframe(fund.T, use_container_width=True)
            else:
                st.warning("Sem dados fundamentais.")
                
        with t3:
            st.markdown("#### Importância das Features (Random Forest - Longo Prazo)")
            imp = b.engine_ml.feature_importances.get(sel, {})
            if imp:
                df_imp = pd.DataFrame(list(imp.items()), columns=['Feature', 'Importância']).sort_values('Importância', ascending=True)
                fig = px.bar(df_imp, x='Importância', y='Feature', orientation='h', color_discrete_sequence=['#212529'])
                fig.update_layout(obter_template_grafico())
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Dados de importância não disponíveis para este ativo.")

def aba_referencias():
    st.markdown("## 📚 Referências e Bibliografia")
    st.markdown("""
    <div class="reference-block"><p><strong>1. Markowitz, H. (1952).</strong> Portfolio Selection.</p></div>
    <div class="reference-block"><p><strong>2. Breiman, L. (2001).</strong> Random Forests. Machine Learning.</p></div>
    <div class="reference-block"><p><strong>3. Chen, T., & Guestrin, C. (2016).</strong> XGBoost: A Scalable Tree Boosting System.</p></div>
    """, unsafe_allow_html=True)

# --- MAIN ---
def main():
    if 'builder' not in st.session_state:
        st.session_state.builder = None
        st.session_state.builder_complete = False
        st.session_state.profile = {}
        st.session_state.ativos_analise = []
        
    configurar_pagina()
    st.markdown('<h1 class="main-header">Sistema de Portfólios Adaptativos (v8.7 Elite Live)</h1>', unsafe_allow_html=True)
    
    t1, t2, t3, t4, t5 = st.tabs(["📚 Metodologia", "🎯 Seleção de Ativos", "🏗️ Construtor de Portfólio", "🔍 Análise Individual", "📖 Referências"])
    with t1: aba_introducao()
    with t2: aba_selecao_ativos()
    with t3: aba_construtor_portfolio()
    with t4: aba_analise_individual()
    with t5: aba_referencias()

if __name__ == "__main__":
    main()
