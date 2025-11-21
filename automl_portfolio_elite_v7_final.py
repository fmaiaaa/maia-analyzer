# -*- coding: utf-8 -*-
"""
=============================================================================
SISTEMA DE PORTFÓLIOS ADAPTATIVOS - INTEGRADO (YFINANCE + FUNDAMENTUS)
=============================================================================

Integração do design "Elite" com a lógica de "Analyzer".
- Fonte de Dados: yfinance (Preços) + Fundamentus (WebScraping).
- ML: Ensemble (Random Forest + XGBoost) para 3 horizontes (4, 8, 12 meses).
- Otimização: Markowitz (Sharpe/Volatilidade).
- Clustering: PCA + KMeans.

Versão: 9.0.0 (Integration Build)
=============================================================================
"""

# --- 1. CORE LIBRARIES & UTILITIES ---
import warnings
import numpy as np
import pandas as pd
import subprocess
import sys
import time
import requests
import io
from datetime import datetime, timedelta
import traceback

# --- 2. SCIENTIFIC / STATISTICAL TOOLS ---
from scipy.optimize import minimize
from scipy.stats import zscore

# --- 3. STREAMLIT & PLOTTING ---
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- 4. MACHINE LEARNING ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Tenta importar XGBoost, instala se não existir
try:
    from xgboost import XGBClassifier
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xgboost'])
    from xgboost import XGBClassifier

# --- 5. CONFIGURATION ---
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURAÇÕES E CONSTANTES GLOBAIS
# =============================================================================

# Configurações de Mercado
TAXA_LIVRE_RISCO = 0.1075  # 10.75% a.a.
PERIODO_DADOS = '5y'       # Histórico total do yfinance para treino
MIN_DIAS_HISTORICO = 252   # Mínimo para análise técnica confiável
NUM_ATIVOS_PORTFOLIO = 5

# Pesos de Otimização (Baseados no Analyzer)
MIN_WEIGHT = 0.10
MAX_WEIGHT = 0.30

# Horizontes de Machine Learning (Dias úteis aproximados)
# 4 meses (~84 dias), 8 meses (~168 dias), 12 meses (~252 dias)
ML_HORIZONS = {
    'CURTO PRAZO': 84,
    'MÉDIO PRAZO': 168,
    'LONGO PRAZO': 252
}

# Lista Base de Ativos (Ibovespa / Principais Líquidos)
ATIVOS_IBOVESPA = [
    'ALOS3.SA', 'ABEV3.SA', 'ASAI3.SA', 'AZZA3.SA', 'B3SA3.SA',
    'BBSE3.SA', 'BBDC3.SA', 'BBDC4.SA', 'BRAP4.SA', 'BBAS3.SA', 'BRKM5.SA',
    'BRAV3.SA', 'BPAC11.SA', 'CXSE3.SA', 'CMIG4.SA', 'COGN3.SA',
    'CPLE6.SA', 'CSAN3.SA', 'CPFE3.SA', 'CMIN3.SA', 'CURY3.SA', 'CVCB3.SA',
    'CYRE3.SA', 'DIRR3.SA', 'ELET3.SA', 'ELET6.SA', 'EMBR3.SA', 'ENGI11.SA',
    'ENEV3.SA', 'EGIE3.SA', 'EQTL3.SA', 'FLRY3.SA', 'GGBR4.SA', 'GOAU4.SA',
    'HAPV3.SA', 'HYPE3.SA', 'IGTI11.SA', 'IRBR3.SA', 'ITSA4.SA', 'ITUB4.SA',
    'KLBN11.SA', 'RENT3.SA', 'LREN3.SA', 'MGLU3.SA', 'POMO4.SA', 'BEEF3.SA',
    'MRVE3.SA', 'MULT3.SA', 'NATU3.SA', 'PCAR3.SA', 'PETR3.SA', 'PETR4.SA',
    'RECV3.SA', 'PRIO3.SA', 'PSSA3.SA', 'RADL3.SA', 'RAIZ4.SA', 'RDOR3.SA',
    'RAIL3.SA', 'SBSP3.SA', 'SANB11.SA', 'CSNA3.SA', 'SLCE3.SA', 'SMFT3.SA',
    'SUZB3.SA', 'TAEE11.SA', 'VIVT3.SA', 'TIMS3.SA', 'TOTS3.SA', 'UGPA3.SA',
    'USIM5.SA', 'VALE3.SA', 'VAMO3.SA', 'VBBR3.SA', 'VIVA3.SA', 'WEGE3.SA', 'YDUQ3.SA'
]

TODOS_ATIVOS = sorted(list(set(ATIVOS_IBOVESPA)))

# Mapeamento Setorial Simplificado (Para Filtros)
ATIVOS_POR_SETOR = {
    'Financeiro': ['BBAS3.SA', 'BBDC4.SA', 'ITUB4.SA', 'B3SA3.SA', 'BBSE3.SA', 'BPAC11.SA', 'SANB11.SA', 'ITSA4.SA'],
    'Materiais Básicos': ['VALE3.SA', 'GGBR4.SA', 'CSNA3.SA', 'USIM5.SA', 'SUZB3.SA', 'KLBN11.SA', 'BRAP4.SA'],
    'Petróleo e Gás': ['PETR4.SA', 'PETR3.SA', 'PRIO3.SA', 'UGPA3.SA', 'VBBR3.SA', 'RAIZ4.SA', 'RECV3.SA'],
    'Utilidade Pública': ['ELET3.SA', 'EQTL3.SA', 'CMIG4.SA', 'CPLE6.SA', 'SBSP3.SA', 'EGIE3.SA', 'TAEE11.SA'],
    'Consumo e Varejo': ['MGLU3.SA', 'LREN3.SA', 'ABEV3.SA', 'JBSS3.SA', 'BEEF3.SA', 'NATU3.SA', 'RDOR3.SA', 'HAPV3.SA', 'RENT3.SA'],
    'Bens Industriais': ['WEGE3.SA', 'EMBR3.SA', 'AZUL4.SA', 'GOLL4.SA', 'RAIL3.SA'],
    'Imobiliário': ['CYRE3.SA', 'MRVE3.SA', 'EZTC3.SA', 'IGTI11.SA', 'MULT3.SA']
}

# =============================================================================
# 2. UTILITÁRIOS E SCRAPING (FUNDAMENTUS)
# =============================================================================

@st.cache_data(ttl=3600)
def obter_dados_fundamentus():
    """
    Realiza WebScraping do site Fundamentus para obter indicadores de todos os ativos.
    Retorna um DataFrame limpo e indexado pelo Ticker (.SA).
    """
    url = 'https://www.fundamentus.com.br/resultado.php'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        r = requests.get(url, headers=headers)
        df = pd.read_html(io.StringIO(r.text), decimal=',', thousands='.')[0]
        
        # Limpeza e Renomeação
        # Papel, Cotação, P/L, P/VP, PSR, Div.Yield, P/Ativo, P/Cap.Giro, P/EBIT, P/Ativ Circ.Liq, EV/EBIT, EV/EBITDA, Mrg Ebit, Mrg. Líq., Liq. Corr., ROIC, ROE, Liq.2meses, Patrim. Líq, Dív.Brut/ Patrim., Cresc. Rec.5a
        
        # Converte percentuais strings para float
        cols_perc = ['Div.Yield', 'Mrg Ebit', 'Mrg. Líq.', 'ROIC', 'ROE', 'Cresc. Rec.5a']
        for col in cols_perc:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('.', '').str.replace(',', '.').str.replace('%', '')
                df[col] = pd.to_numeric(df[col], errors='coerce') / 100.0

        # Renomeia para padrão do sistema
        rename_map = {
            'Papel': 'Ticker',
            'P/L': 'PE_Ratio',
            'P/VP': 'PB_Ratio',
            'Div.Yield': 'Div_Yield',
            'ROE': 'ROE',
            'Mrg. Líq.': 'Net_Margin',
            'Dív.Brut/ Patrim.': 'Debt_to_Equity'
        }
        df = df.rename(columns=rename_map)
        
        # Ajusta Ticker para formato Yahoo (.SA)
        df['Ticker'] = df['Ticker'] + '.SA'
        df = df.set_index('Ticker')
        
        # Filtra apenas colunas úteis e remove duplicatas/nulos
        cols_uteis = ['PE_Ratio', 'PB_Ratio', 'Div_Yield', 'ROE', 'Net_Margin', 'Debt_to_Equity']
        # Garante que colunas existem
        cols_finais = [c for c in cols_uteis if c in df.columns]
        
        return df[cols_finais]
        
    except Exception as e:
        st.error(f"Erro ao acessar Fundamentus: {e}")
        return pd.DataFrame()

# =============================================================================
# 3. MAPEAMENTOS DE PONTUAÇÃO (Do Elite v7)
# =============================================================================

# Mapeamento para perguntas de Concordância
OPTIONS_CONCORDA = [
    "CT: (Concordo Totalmente) - Estou confortável com altas flutuações.",
    "C: (Concordo) - Aceito alguma volatilidade.",
    "N: (Neutro) - Depende do momento.",
    "D: (Discordo) - Prefiro estratégias cautelosas.",
    "DT: (Discordo Totalmente) - Prefiro segurança absoluta."
]
MAP_CONCORDA_SCORES = {
    OPTIONS_CONCORDA[0]: 5, OPTIONS_CONCORDA[1]: 4, OPTIONS_CONCORDA[2]: 3, OPTIONS_CONCORDA[3]: 2, OPTIONS_CONCORDA[4]: 1
}

# Mapeamento para perguntas de Discordância (Invertido)
OPTIONS_DISCORDA = [
    "CT: (Concordo Totalmente) - Preservação é prioridade máxima.",
    "C: (Concordo) - Evitar perdas é muito importante.",
    "N: (Neutro) - Busco equilíbrio.",
    "D: (Discordo) - Foco no crescimento de longo prazo.",
    "DT: (Discordo Totalmente) - Perdas de curto prazo são irrelevantes."
]
MAP_DISCORDA_SCORES = {
    OPTIONS_DISCORDA[0]: 1, OPTIONS_DISCORDA[1]: 2, OPTIONS_DISCORDA[2]: 3, OPTIONS_DISCORDA[3]: 4, OPTIONS_DISCORDA[4]: 5
}

OPTIONS_REACTION = [
    "A: (Vender Imediatamente)", "B: (Manter e Reavaliar)", "C: (Comprar Mais)"
]
MAP_REACTION_SCORES = {OPTIONS_REACTION[0]: 1, OPTIONS_REACTION[1]: 3, OPTIONS_REACTION[2]: 5}

OPTIONS_CONHECIMENTO = [
    "A: (Avançado)", "B: (Intermediário)", "C: (Iniciante)"
]
MAP_CONHECIMENTO_SCORES = {OPTIONS_CONHECIMENTO[0]: 5, OPTIONS_CONHECIMENTO[1]: 3, OPTIONS_CONHECIMENTO[2]: 1}

# Opções de Tempo/Liquidez
OPTIONS_TIME = ['A: Curto (até 1 ano)', 'B: Médio (1-5 anos)', 'C: Longo (5+ anos)']
OPTIONS_LIQ = ['A: Menos de 6 meses', 'B: 6 meses a 2 anos', 'C: Mais de 2 anos']

class AnalisadorPerfilInvestidor:
    def determinar_nivel_risco(self, pontuacao):
        if pontuacao <= 18: return "CONSERVADOR"
        elif pontuacao <= 30: return "INTERMEDIÁRIO"
        elif pontuacao <= 45: return "MODERADO"
        elif pontuacao <= 60: return "MODERADO-ARROJADO"
        else: return "AVANÇADO"

    def calcular_perfil(self, respostas):
        # Extrai letra inicial para horizonte
        time_key = respostas['time_purpose'][0]
        liq_key = respostas['liquidity'][0]
        
        # Lógica de Horizonte e Target ML
        if time_key == 'C' or liq_key == 'C':
            horizonte = "LONGO PRAZO"
            ml_target_days = ML_HORIZONS['LONGO PRAZO']
        elif time_key == 'B' or liq_key == 'B':
            horizonte = "MÉDIO PRAZO"
            ml_target_days = ML_HORIZONS['MÉDIO PRAZO']
        else:
            horizonte = "CURTO PRAZO"
            ml_target_days = ML_HORIZONS['CURTO PRAZO']
            
        # Pontuação
        score = (
            MAP_CONCORDA_SCORES[respostas['risk_accept']] * 5 +
            MAP_CONCORDA_SCORES[respostas['max_gain']] * 5 +
            MAP_DISCORDA_SCORES[respostas['stable_growth']] * 5 +
            MAP_DISCORDA_SCORES[respostas['avoid_loss']] * 5 +
            MAP_CONHECIMENTO_SCORES[respostas['level']] * 3 +
            MAP_REACTION_SCORES[respostas['reaction']] * 3
        )
        
        nivel = self.determinar_nivel_risco(score)
        return nivel, horizonte, ml_target_days, score

# =============================================================================
# 4. ESTILO E VISUALIZAÇÃO (Cópia do Elite)
# =============================================================================

def obter_template_grafico():
    return {
        'plot_bgcolor': '#f8f9fa',
        'paper_bgcolor': 'white',
        'font': {'family': 'Arial, sans-serif', 'size': 12, 'color': '#343a40'},
        'title': {'font': {'family': 'Arial, sans-serif', 'size': 16, 'color': '#212529', 'weight': 'bold'}, 'x': 0.5, 'xanchor': 'center'},
        'xaxis': {'showgrid': True, 'gridcolor': '#e9ecef', 'showline': True, 'linecolor': '#ced4da'},
        'yaxis': {'showgrid': True, 'gridcolor': '#e9ecef', 'showline': True, 'linecolor': '#ced4da'},
        'colorway': ['#212529', '#495057', '#6c757d', '#adb5bd', '#ced4da']
    }

# =============================================================================
# 5. LÓGICA DO PORTFÓLIO (ANALYZER + ML ENSEMBLE)
# =============================================================================

def calculate_technical_indicators(df):
    """Cálculo manual de indicadores técnicos (Igual Portfolio Analyzer)."""
    df = df.copy()
    df['Returns'] = df['Close'].pct_change()
    
    # RSI 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Volatilidade (20d)
    df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
    
    # Momentum (10d)
    df['Momentum'] = df['Close'] / df['Close'].shift(10) - 1
    
    # SMAs
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    return df.dropna()

class PortfolioBuilder:
    def __init__(self, investment_amount):
        self.investment_amount = investment_amount
        self.data_by_asset = {}
        self.fundamental_data = pd.DataFrame()
        self.ml_predictions = {}
        self.scores_combinados = pd.DataFrame()
        self.selected_assets = []
        self.portfolio_allocation = {}
        self.portfolio_metrics = {}
        self.justifications = {}
        self.success_assets = []
        self.metodo_alocacao = ""
        
    def fetch_data(self, assets, progress_bar=None):
        """Coleta Híbrida: yfinance + Fundamentus"""
        # 1. Fundamentus
        if progress_bar: progress_bar.progress(10, text="Coletando dados fundamentalistas (Fundamentus)...")
        df_fund = obter_dados_fundamentus()
        
        # 2. yfinance
        self.success_assets = []
        if progress_bar: progress_bar.progress(20, text="Baixando histórico de preços (yfinance)...")
        
        # Importa yfinance aqui para garantir uso
        import yfinance as yf
        
        # Filtra apenas ativos que estão no Fundamentus e na lista solicitada
        assets_to_fetch = [a for a in assets if a in df_fund.index or a in ATIVOS_IBOVESPA]
        
        # Baixa em lote para velocidade
        if not assets_to_fetch:
            return False

        tickers_str = " ".join(assets_to_fetch)
        data = yf.download(tickers_str, period=PERIODO_DADOS, group_by='ticker', progress=False, threads=True)
        
        total = len(assets_to_fetch)
        processed = 0
        
        for ticker in assets_to_fetch:
            try:
                if len(assets_to_fetch) == 1:
                    df = data.copy()
                else:
                    df = data[ticker].copy()
                
                if df.empty: continue
                
                # Limpeza básica
                df = df.dropna(how='all')
                if len(df) < MIN_DIAS_HISTORICO: continue
                
                # Indicadores Técnicos
                df = calculate_technical_indicators(df)
                if df.empty: continue
                
                self.data_by_asset[ticker] = df
                self.success_assets.append(ticker)
                
            except Exception:
                continue
            
            processed += 1
            if progress_bar:
                progress = 20 + int((processed / total) * 40) # Vai até 60%
                progress_bar.progress(progress, text=f"Processando {ticker}...")

        # Filtra DF Fundamentalista para os que temos preço
        self.fundamental_data = df_fund.loc[df_fund.index.intersection(self.success_assets)].copy()
        
        # Preenche missings na fundamentalista com média
        self.fundamental_data = self.fundamental_data.fillna(self.fundamental_data.mean())
        
        return len(self.success_assets) >= NUM_ATIVOS_PORTFOLIO

    def train_ensemble_ml(self, target_days):
        """Treina Ensemble (RF + XGB) para prever retorno > 0 em 'target_days'."""
        predictions = {}
        
        features = ['RSI', 'MACD', 'Volatility', 'Momentum', 'SMA_50', 'SMA_200']
        
        for ticker in self.success_assets:
            try:
                df = self.data_by_asset[ticker].copy()
                
                # Target: 1 se preço futuro > preço atual
                df['Target'] = (df['Close'].shift(-target_days) > df['Close']).astype(int)
                
                # Drop NaNs criados pelo shift
                df_train = df.dropna(subset=features + ['Target'])
                if len(df_train) < 100: 
                    predictions[ticker] = {'proba': 0.5, 'auc': 0.5}
                    continue
                
                X = df_train[features]
                y = df_train['Target']
                
                # Split temporal
                split = int(len(X) * 0.8)
                X_train, X_test = X.iloc[:split], X.iloc[split:]
                y_train, y_test = y.iloc[:split], y.iloc[split:]
                
                # Se só tem uma classe no treino, pula
                if len(y_train.unique()) < 2:
                    predictions[ticker] = {'proba': 0.5, 'auc': 0.5}
                    continue

                # Modelos
                rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                xgb = XGBClassifier(n_estimators=100, max_depth=5, eval_metric='logloss', use_label_encoder=False, random_state=42)
                
                rf.fit(X_train, y_train)
                xgb.fit(X_train, y_train)
                
                # Dados atuais para predição
                current_X = df[features].iloc[[-1]]
                
                prob_rf = rf.predict_proba(current_X)[0][1]
                prob_xgb = xgb.predict_proba(current_X)[0][1]
                
                # Ensemble (Média Simples)
                final_prob = (prob_rf + prob_xgb) / 2
                
                # Score simples (Acurácia no teste como proxy de confiança)
                acc_rf = rf.score(X_test, y_test)
                acc_xgb = xgb.score(X_test, y_test)
                avg_acc = (acc_rf + acc_xgb) / 2
                
                predictions[ticker] = {'proba': final_prob, 'auc': avg_acc}
                
            except Exception:
                predictions[ticker] = {'proba': 0.5, 'auc': 0.5}
        
        self.ml_predictions = predictions

    def rank_assets(self, horizon_type):
        """Sistema de Pontuação Multi-Fator (Igual Analyzer, adaptado)."""
        
        # Define pesos baseados no horizonte
        if horizon_type == "CURTO PRAZO":
            w_perf, w_fund, w_tech, w_ml = 0.3, 0.1, 0.3, 0.3
        elif horizon_type == "LONGO PRAZO":
            w_perf, w_fund, w_tech, w_ml = 0.3, 0.4, 0.1, 0.2
        else: # MÉDIO
            w_perf, w_fund, w_tech, w_ml = 0.3, 0.3, 0.2, 0.2
            
        scores = pd.DataFrame(index=self.success_assets)
        
        # 1. Performance (Sharpe)
        sharpes = {}
        for t in self.success_assets:
            rets = self.data_by_asset[t]['Returns']
            vol = rets.std() * np.sqrt(252)
            ret_anual = rets.mean() * 252
            sharpe = (ret_anual - TAXA_LIVRE_RISCO) / vol if vol > 0 else 0
            sharpes[t] = sharpe
        
        scores['Sharpe'] = pd.Series(sharpes)
        scores['Score_Perf'] = zscore(scores['Sharpe'].fillna(0))
        
        # 2. Fundamentos
        # Normaliza: Menor P/L melhor, Maior ROE melhor
        fund = self.fundamental_data.loc[self.success_assets]
        scores['PE'] = fund['PE_Ratio']
        scores['ROE'] = fund['ROE']
        
        # Inverte P/L (menor é melhor) -> zscore negativo
        scores['Score_Fund'] = (zscore(scores['ROE'].fillna(0)) - zscore(scores['PE'].fillna(0))) / 2
        
        # 3. Técnico (RSI + Momentum)
        tech_vals = {}
        for t in self.success_assets:
            last = self.data_by_asset[t].iloc[-1]
            # RSI: perto de 50 é neutro, mas para compra queremos momentum ou oversold.
            # Simplificação: Momentum alto é bom.
            tech_vals[t] = last['Momentum']
        
        scores['Momentum'] = pd.Series(tech_vals)
        scores['Score_Tech'] = zscore(scores['Momentum'].fillna(0))
        
        # 4. ML
        ml_probs = {k: v['proba'] for k, v in self.ml_predictions.items()}
        scores['ML_Prob'] = pd.Series(ml_probs)
        scores['Score_ML'] = zscore(scores['ML_Prob'].fillna(0.5))
        
        # Normaliza tudo entre 0 e 100 para somar
        def norm(s):
            return (s - s.min()) / (s.max() - s.min()) * 100
            
        scores['Total_Score'] = (
            norm(scores['Score_Perf']) * w_perf +
            norm(scores['Score_Fund']) * w_fund +
            norm(scores['Score_Tech']) * w_tech +
            norm(scores['Score_ML']) * w_ml
        )
        
        self.scores_combinados = scores.sort_values('Total_Score', ascending=False)
        
    def cluster_and_select(self):
        """PCA + KMeans para selecionar 5 ativos distintos."""
        df_cluster = self.scores_combinados[['Sharpe', 'PE', 'ROE', 'Momentum', 'ML_Prob']].fillna(0)
        
        # PCA
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df_cluster)
        pca = PCA(n_components=2)
        components = pca.fit_transform(scaled)
        
        # KMeans
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(components)
        
        self.scores_combinados['Cluster'] = clusters
        
        # Seleciona o melhor de cada cluster
        final_list = []
        used_clusters = set()
        
        # Ordena por score total
        candidates = self.scores_combinados.sort_values('Total_Score', ascending=False)
        
        # Pega o melhor de cada cluster
        for ticker, row in candidates.iterrows():
            c = row['Cluster']
            if c not in used_clusters and len(final_list) < NUM_ATIVOS_PORTFOLIO:
                final_list.append(ticker)
                used_clusters.add(c)
        
        # Se faltar (clusters < 5), completa com os melhores restantes
        if len(final_list) < NUM_ATIVOS_PORTFOLIO:
            remaining = [t for t in candidates.index if t not in final_list]
            final_list.extend(remaining[:NUM_ATIVOS_PORTFOLIO - len(final_list)])
            
        self.selected_assets = final_list

    def optimize_markowitz(self, risk_profile):
        """Otimização Mean-Variance."""
        assets = self.selected_assets
        if not assets: return
        
        # DF de Retornos
        prices = pd.DataFrame({t: self.data_by_asset[t]['Close'] for t in assets}).dropna()
        returns = prices.pct_change().dropna()
        
        mu = returns.mean() * 252
        sigma = returns.cov() * 252
        num_assets = len(assets)
        
        def port_vol(w):
            return np.sqrt(np.dot(w.T, np.dot(sigma, w)))
            
        def neg_sharpe(w):
            p_ret = np.dot(w, mu)
            p_vol = port_vol(w)
            return -(p_ret - TAXA_LIVRE_RISCO) / p_vol
            
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((MIN_WEIGHT, MAX_WEIGHT) for _ in range(num_assets))
        init_guess = [1/num_assets] * num_assets
        
        # Define Estratégia
        if risk_profile in ["CONSERVADOR", "INTERMEDIÁRIO"]:
            self.metodo_alocacao = "Minimização de Volatilidade"
            res = minimize(port_vol, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        else:
            self.metodo_alocacao = "Maximização de Sharpe"
            res = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
            
        weights = res.x if res.success else init_guess
        
        # Alocação Final
        self.portfolio_allocation = {
            t: {'weight': w, 'amount': w * self.investment_amount}
            for t, w in zip(assets, weights)
        }
        
        # Métricas
        p_ret = np.dot(weights, mu)
        p_vol = port_vol(weights)
        p_sharpe = (p_ret - TAXA_LIVRE_RISCO) / p_vol
        
        self.portfolio_metrics = {
            'annual_return': p_ret,
            'annual_volatility': p_vol,
            'sharpe_ratio': p_sharpe
        }
        
        # Justificativas
        for t in assets:
            row = self.scores_combinados.loc[t]
            ml = self.ml_predictions[t]
            self.justifications[t] = (
                f"Score Total: {row['Total_Score']:.1f} | "
                f"ML Prob: {ml['proba']:.0%} (Conf: {ml['auc']:.2f}) | "
                f"Fundamentos (P/L: {row['PE']:.1f}, ROE: {row['ROE']:.1%})"
            )

    def run_pipeline(self, assets, profile, progress_bar):
        if not self.fetch_data(assets, progress_bar):
            return False
            
        if progress_bar: progress_bar.progress(65, text="Treinando Ensemble ML (RF + XGBoost)...")
        self.train_ensemble_ml(profile['ml_target'])
        
        if progress_bar: progress_bar.progress(80, text="Ranqueando e Clusterizando Ativos...")
        self.rank_assets(profile['horizon'])
        self.cluster_and_select()
        
        if progress_bar: progress_bar.progress(90, text="Otimizando Alocação (Markowitz)...")
        self.optimize_markowitz(profile['risk'])
        
        if progress_bar: progress_bar.progress(100, text="Concluído!")
        return True

# =============================================================================
# 6. INTERFACE STREAMLIT (Design Elite)
# =============================================================================

def configurar_pagina():
    st.set_page_config(page_title="Sistema de Portfólios Adaptativos", page_icon="📈", layout="wide")
    
    # CSS NEUTRO/PRETO (Elite Theme)
    st.markdown("""
        <style>
        :root { --primary-color: #000000; --secondary-color: #6c757d; --background-light: #ffffff; --background-dark: #f8f9fa; --text-color: #212529; --text-color-light: #ffffff; --border-color: #dee2e6; }
        body { background-color: var(--background-light); color: var(--text-color); font-family: 'Arial', sans-serif; }
        .main-header { font-family: 'Arial', sans-serif; color: var(--primary-color); text-align: center; border-bottom: 2px solid var(--border-color); padding-bottom: 10px; font-size: 2.2rem !important; margin-bottom: 20px; font-weight: 600; }
        .stButton button, .stDownloadButton button { border: 1px solid var(--primary-color) !important; color: var(--primary-color) !important; border-radius: 6px; padding: 8px 16px; background-color: transparent !important; font-family: 'Arial', sans-serif !important; }
        .stButton button:hover { background-color: var(--primary-color) !important; color: var(--text-color-light) !important; }
        .stButton button[kind="primary"], .stFormSubmitButton button { background-color: var(--primary-color) !important; color: var(--text-color-light) !important; border: none !important; }
        .stTabs [data-baseweb="tab-list"] { justify-content: center; width: 100%; border-bottom: 2px solid var(--border-color); }
        .stTabs [data-baseweb="tab"] { color: var(--secondary-color); flex-grow: 0 !important; font-family: 'Arial', sans-serif !important; }
        .stTabs [aria-selected="true"] { border-bottom: 2px solid var(--primary-color); color: var(--primary-color); font-weight: 700; }
        .info-box { background-color: var(--background-dark); border-left: 4px solid var(--primary-color); padding: 15px; margin: 10px 0; border-radius: 6px; }
        .stMetric { background-color: var(--background-dark); padding: 10px; border-radius: 6px; }
        .reference-block { background-color: #fdfdfd; border: 1px solid var(--border-color); padding: 12px; margin-bottom: 12px; border-radius: 6px; }
        .reference-block .explanation { font-style: italic; color: var(--secondary-color); font-size: 0.95em; border-top: 1px dashed #e0e0e0; padding-top: 8px; margin-top: 8px; }
        </style>
    """, unsafe_allow_html=True)

def aba_introducao():
    st.markdown("## 📚 Metodologia Quantitativa e Arquitetura do Sistema")
    st.markdown("""
    <div class="info-box">
    <h3>🎯 Visão Geral do Sistema Híbrido</h3>
    <p>Este sistema integra dados de preços em tempo real via <b>yfinance</b> com indicadores fundamentalistas extraídos via WebScraping do <b>Fundamentus</b>. 
    Utiliza um Ensemble de Machine Learning (<b>Random Forest + XGBoost</b>) para prever probabilidades de alta em 3 horizontes temporais.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('### 1. Motor de Análise')
        with st.expander("Etapa 1.1: Coleta de Dados (Híbrida)"):
            st.markdown("- **Preços:** Histórico de 5 anos via API yfinance.\n- **Fundamentos:** P/L, ROE, Margens e Dívida raspados diretamente do Fundamentus.\n- **Técnica:** Cálculo manual de RSI, MACD, Volatilidade e Momentum.")
        with st.expander("Etapa 1.2: Machine Learning Ensemble"):
            st.markdown("- **Modelos:** Random Forest Classifier + XGBoost Classifier.\n- **Target:** Retorno Positivo no horizonte definido (4, 8 ou 12 meses).\n- **Output:** Média das probabilidades dos dois modelos.")
            
    with col2:
        st.markdown('### 2. Construção de Portfólio')
        with st.expander("Etapa 2.1: Perfil & Scoring"):
            st.markdown("O perfil do investidor define o horizonte (Target ML) e a tolerância ao risco (Método de Otimização). Os ativos recebem um Score Multi-Fator (Performance, Fundamentos, Técnica, ML).")
        with st.expander("Etapa 2.2: Otimização"):
            st.markdown("- **Clusterização:** PCA + KMeans selecionam 5 ativos de perfis distintos.\n- **Alocação:** Markowitz (Mean-Variance) otimiza os pesos para Sharpe Máximo ou Volatilidade Mínima.")

def aba_selecao_ativos():
    st.markdown("## 🎯 Definição do Universo de Análise")
    
    modo = st.radio("**Modo de Seleção:**", ["📊 Índice de Referência (Ibovespa)", "🏢 Seleção Setorial", "✍️ Seleção Individual"])
    ativos = []
    
    if "Índice" in modo:
        ativos = TODOS_ATIVOS
        st.success(f"✓ {len(ativos)} ativos do Ibovespa selecionados.")
        
    elif "Setorial" in modo:
        setores = st.multiselect("Escolha Setores:", list(ATIVOS_POR_SETOR.keys()))
        for s in setores: ativos.extend(ATIVOS_POR_SETOR[s])
        ativos = list(set(ativos))
        st.info(f"{len(ativos)} ativos selecionados.")
        
    elif "Individual" in modo:
        ativos = st.multiselect("Selecione Tickers:", TODOS_ATIVOS)
        
    if ativos:
        st.session_state.ativos_analise = ativos
        st.success("Universo definido. Vá para 'Construtor de Portfólio'.")
    else:
        st.warning("Nenhum ativo selecionado.")

def aba_construtor_portfolio():
    if 'ativos_analise' not in st.session_state:
        st.warning("Defina os ativos na aba anterior.")
        return

    if 'builder_complete' not in st.session_state:
        st.markdown('## 📋 Calibração do Perfil')
        with st.form("perfil_form"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Risco")
                p1 = st.radio("1. Tolerância à Volatilidade:", OPTIONS_CONCORDA, index=2)
                p2 = st.radio("2. Foco em Retorno Máximo:", OPTIONS_CONCORDA, index=2)
                p3 = st.radio("3. Prioridade de Estabilidade:", OPTIONS_DISCORDA, index=2)
                p4 = st.radio("4. Aversão à Perda:", OPTIONS_DISCORDA, index=2)
                p5 = st.radio("5. Reação a Queda:", OPTIONS_REACTION, index=1)
                p6 = st.radio("6. Conhecimento:", OPTIONS_CONHECIMENTO, index=1)
            with col2:
                st.markdown("#### Horizonte")
                t1 = st.radio("7. Prazo:", OPTIONS_TIME, index=1)
                l1 = st.radio("8. Liquidez:", OPTIONS_LIQ, index=1)
                inv = st.number_input("Investimento (R$)", 1000, 10000000, 10000)
            
            if st.form_submit_button("🚀 Gerar Alocação", type="primary"):
                respostas = {'risk_accept': p1, 'max_gain': p2, 'stable_growth': p3, 'avoid_loss': p4, 'reaction': p5, 'level': p6, 'time_purpose': t1, 'liquidity': l1}
                analyser = AnalisadorPerfilInvestidor()
                nivel, horiz, ml_days, score = analyser.calcular_perfil(respostas)
                
                st.session_state.profile_data = {'risk': nivel, 'horizon': horiz, 'ml_target': ml_days, 'score': score}
                st.session_state.builder = PortfolioBuilder(inv)
                
                prog = st.progress(0, text="Iniciando Pipeline...")
                ok = st.session_state.builder.run_pipeline(st.session_state.ativos_analise, st.session_state.profile_data, prog)
                prog.empty()
                
                if ok:
                    st.session_state.builder_complete = True
                    st.rerun()
                else:
                    st.error("Falha ao coletar dados suficientes.")
                    
    else:
        b = st.session_state.builder
        p = st.session_state.profile_data
        st.markdown('## ✅ Relatório de Alocação')
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Perfil", p['risk'])
        c2.metric("Horizonte", p['horizon'])
        c3.metric("Sharpe", f"{b.portfolio_metrics['sharpe_ratio']:.2f}")
        c4.metric("Estratégia", b.metodo_alocacao)
        
        if st.button("🔄 Recomeçar"):
            st.session_state.builder_complete = False
            st.rerun()
            
        t1, t2, t3 = st.tabs(["📊 Alocação", "📈 Performance", "❓ Justificativas"])
        
        with t1:
            df_alloc = pd.DataFrame([
                {'Ativo': k, 'Peso': v['weight']*100, 'Valor': v['amount']} 
                for k, v in b.portfolio_allocation.items()
            ])
            c_pie, c_tab = st.columns([1, 2])
            with c_pie:
                fig = px.pie(df_alloc, values='Peso', names='Ativo', hole=0.4)
                fig.update_layout(**obter_template_grafico())
                st.plotly_chart(fig, use_container_width=True)
            with c_tab:
                st.dataframe(df_alloc.style.format({'Peso': '{:.2f}%', 'Valor': 'R$ {:.2f}'}), use_container_width=True)
                
        with t2:
            # Simulação de curva acumulada dos ativos selecionados
            df_prices = pd.DataFrame({t: b.data_by_asset[t]['Close'] for t in b.selected_assets}).dropna()
            if not df_prices.empty:
                ret_cum = (1 + df_prices.pct_change()).cumprod()
                fig = go.Figure()
                for col in ret_cum.columns:
                    fig.add_trace(go.Scatter(x=ret_cum.index, y=ret_cum[col], name=col))
                fig.update_layout(title="Performance Histórica Acumulada", **obter_template_grafico())
                st.plotly_chart(fig, use_container_width=True)
                
        with t3:
            for t, just in b.justifications.items():
                st.markdown(f"""
                <div class="info-box">
                <h4>{t} ({b.portfolio_allocation[t]['weight']:.1%})</h4>
                <p>{just}</p>
                </div>
                """, unsafe_allow_html=True)

def aba_analise_individual():
    st.markdown("## 🔍 Análise de Fatores")
    
    if 'ativos_analise' in st.session_state:
        sel = st.selectbox("Ativo:", st.session_state.ativos_analise)
        if st.button("Analisar", type="primary"):
            with st.spinner("Coletando dados..."):
                # Builder temporário para fetch único
                tmp_b = PortfolioBuilder(0)
                # Tenta pegar dados de Fundamentus e Yahoo só pra esse
                # Nota: Fundamentus pega tudo de uma vez, é rápido
                df_fund_all = obter_dados_fundamentus()
                
                import yfinance as yf
                hist = yf.Ticker(sel).history(period='2y')
                
                if hist.empty or sel not in df_fund_all.index:
                    st.error("Dados indisponíveis.")
                    return
                    
                # Indicadores
                hist = calculate_technical_indicators(hist)
                fund = df_fund_all.loc[sel]
                
                # Mostra Dados
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Preço", f"R$ {hist['Close'].iloc[-1]:.2f}")
                c2.metric("P/L", f"{fund['PE_Ratio']:.2f}")
                c3.metric("ROE", f"{fund['ROE']:.1%}")
                c4.metric("RSI", f"{hist['RSI'].iloc[-1]:.1f}")
                
                # Gráfico
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='Preço'), row=1, col=1)
                fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name='Volume', marker_color='#6c757d'), row=2, col=1)
                fig.update_layout(title=f"Gráfico Técnico - {sel}", **obter_template_grafico())
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### Fundamentos (Fundamentus)")
                st.dataframe(fund.to_frame().T, use_container_width=True)

def aba_referencias():
    st.markdown("## 📚 Referências e Bibliografia")
    st.markdown("---")
    st.markdown("### Machine Learning & Finanças")
    st.markdown("""
    <div class="reference-block"><p><strong>1. Géron, A. Mãos à Obra: Aprendizado de Máquina com Scikit-Learn, Keras e TensorFlow.</strong></p></div>
    <div class="reference-block"><p><strong>2. HILPISCH, Y. J. Python for finance: analyze big financial dat. O'Reilly Media.</strong></p></div>
    <div class="reference-block"><p><strong>3. Documentação Oficial: XGBoost, Scikit-Learn e Fundamentus.</strong></p></div>
    """, unsafe_allow_html=True)

# =============================================================================
# 7. MAIN
# =============================================================================

def main():
    configurar_pagina()
    st.markdown('<h1 class="main-header">Sistema de Portfólios Adaptativos (v9.0 Integrated)</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📚 Metodologia", "🎯 Seleção de Ativos", "🏗️ Construtor de Portfólio", "🔍 Análise Individual", "📖 Referências"
    ])
    
    with tab1: aba_introducao()
    with tab2: aba_selecao_ativos()
    with tab3: aba_construtor_portfolio()
    with tab4: aba_analise_individual()
    with tab5: aba_referencias()

if __name__ == "__main__":
    main()
