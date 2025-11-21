# -*- coding: utf-8 -*-
"""
=============================================================================
SISTEMA DE PORTFÓLIOS ADAPTATIVOS - INTEGRATED ELITE EDITION
=============================================================================

Fusão da lógica analítica do "Portfolio Analyzer" com a interface e 
metodologia do "AutoML Portfolio Elite".

Funcionalidades:
- Coleta de Dados: yfinance (Preços) + Fundamentus (WebScraping Fundamentalista).
- Engenharia de Features: Indicadores técnicos manuais (RSI, MACD, Volatilidade).
- Machine Learning: Ensemble (Random Forest + XGBoost) para 3 horizontes (4, 8, 12 meses).
- Clusterização: PCA + KMeans para diversificação inteligente.
- Otimização: Fronteira Eficiente de Markowitz.

Versão: 1.0.0 (Integration Build)
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

# Garante XGBoost (pode não estar instalado padrão)
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
# 1. CONFIGURAÇÕES E CONSTANTES GLOBAIS
# =============================================================================

PERIODO_DADOS = '5y'  # Histórico longo para treino ML
MIN_DIAS_HISTORICO = 252
NUM_ATIVOS_PORTFOLIO = 5
TAXA_LIVRE_RISCO = 0.1075 # Selic Aproximada
SCORE_PERCENTILE_THRESHOLD = 0.70 # Flexibilizado para garantir ativos suficientes

# Pesos de Alocação (Markowitz Constraints)
PESO_MIN = 0.10
PESO_MAX = 0.30

# Configuração dos Horizontes de ML (Meses -> Dias Úteis aprox.)
HORIZONTES_ML = {
    'Curto Prazo (4m)': 84,
    'Médio Prazo (8m)': 168,
    'Longo Prazo (12m)': 252
}

# Lista de Ativos (Ibovespa Base)
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

# Mapeamento Setorial Simplificado
ATIVOS_POR_SETOR = {
    'Financeiro': ['BBAS3.SA', 'BBDC4.SA', 'ITUB4.SA', 'B3SA3.SA', 'BPAC11.SA', 'SANB11.SA', 'BBSE3.SA'],
    'Materiais Básicos': ['VALE3.SA', 'GGBR4.SA', 'CSNA3.SA', 'SUZB3.SA', 'KLBN11.SA', 'BRAP4.SA'],
    'Petróleo e Gás': ['PETR4.SA', 'PETR3.SA', 'PRIO3.SA', 'UGPA3.SA', 'RAIZ4.SA', 'VBBR3.SA'],
    'Utilidade Pública': ['ELET3.SA', 'ELET6.SA', 'EQTL3.SA', 'CMIG4.SA', 'CPLE6.SA', 'SBSP3.SA', 'TAEE11.SA'],
    'Consumo e Varejo': ['MGLU3.SA', 'LREN3.SA', 'RENT3.SA', 'ASAI3.SA', 'CRFB3.SA', 'RDOR3.SA', 'HAPV3.SA'],
    'Industrial e Bens': ['WEGE3.SA', 'EMBR3.SA', 'AZZA3.SA', 'RAIL3.SA']
}
TODOS_ATIVOS = sorted(list(set(ATIVOS_IBOVESPA)))

# =============================================================================
# 2. CLASSES UTILITÁRIAS DE COLETA E CÁLCULO
# =============================================================================

class FundamentusScraper:
    """
    Realiza WebScraping do site Fundamentus para obter indicadores fundamentalistas em tempo real.
    """
    @staticmethod
    def obter_dados_consolidados():
        """Baixa a tabela principal do Fundamentus com todos os ativos."""
        url = 'https://www.fundamentus.com.br/resultado.php'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        
        try:
            r = requests.get(url, headers=headers)
            r.raise_for_status()
            
            # Tratamento para ler a tabela HTML corretamente (pt-br format)
            df = pd.read_html(StringIO(r.text), decimal=',', thousands='.')[0]
            
            # Renomear colunas para facilitar
            mapper = {
                'Papel': 'ticker',
                'Cotação': 'price',
                'P/L': 'pe_ratio',
                'P/VP': 'pb_ratio',
                'PSR': 'psr',
                'Div.Yield': 'div_yield',
                'P/Ativo': 'price_to_assets',
                'P/Cap.Giro': 'price_to_working_capital',
                'P/EBIT': 'price_to_ebit',
                'P/Ativ Circ Liq': 'price_to_nca',
                'EV/EBIT': 'ev_to_ebit',
                'EV/EBITDA': 'ev_to_ebitda',
                'Mrg Ebit': 'ebit_margin',
                'Mrg. Líq.': 'net_margin',
                'Liq. Corr.': 'current_ratio',
                'ROIC': 'roic',
                'ROE': 'roe',
                'Liq.2meses': 'liquidity_2m',
                'Patrim. Líq': 'equity',
                'Dív.Brut/ Patrim.': 'debt_to_equity',
                'Cresc. Rec.5a': 'revenue_growth_5y'
            }
            df.rename(columns=mapper, inplace=True)
            
            # Limpeza e Conversão
            for col in df.columns:
                if df[col].dtype == object and col != 'ticker':
                    df[col] = df[col].str.replace('%', '').str.replace('.', '').str.replace(',', '.').astype(float)
                    if col in ['div_yield', 'ebit_margin', 'net_margin', 'roic', 'roe', 'revenue_growth_5y']:
                        df[col] = df[col] / 100.0 # Converter percentual para decimal
            
            return df.set_index('ticker')
            
        except Exception as e:
            st.error(f"Erro ao conectar com Fundamentus: {e}")
            return pd.DataFrame()

class EngenheiroFeatures:
    """Calcula indicadores técnicos manuais (estilo portfolio_analyzer.py)."""
    
    @staticmethod
    def calcular_indicadores_tecnicos(df):
        if df.empty: return df
        
        df = df.copy()
        # Retornos
        df['returns'] = df['Close'].pct_change()
        
        # RSI (14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD (12, 26, 9)
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # Volatilidade (20 dias anualizada)
        df['volatility_20d'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Médias Móveis
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        df['sma_200'] = df['Close'].rolling(window=200).mean()
        
        # Momentum (ROC 60 dias)
        df['momentum_60'] = df['Close'].pct_change(periods=60)
        
        return df

    @staticmethod
    def normalizar_score(serie, maior_melhor=True):
        if serie.isnull().all(): return pd.Series(0.5, index=serie.index)
        
        # Winsorize (Clipping de outliers)
        q_low = serie.quantile(0.05)
        q_high = serie.quantile(0.95)
        serie_clipped = serie.clip(q_low, q_high)
        
        min_val = serie_clipped.min()
        max_val = serie_clipped.max()
        
        if max_val == min_val: return pd.Series(0.5, index=serie.index)
        
        norm = (serie_clipped - min_val) / (max_val - min_val)
        return norm if maior_melhor else 1 - norm

class EngineML:
    """
    Ensemble de Machine Learning (Random Forest + XGBoost).
    Prevê para 3 horizontes temporais: Curto, Médio e Longo.
    """
    
    def __init__(self):
        self.resultados = {} # Armazena predições por ticker e horizonte
    
    def treinar_ensemble(self, df_historico, ticker):
        """
        Treina modelos para CP (4m), MP (8m), LP (12m) usando dados históricos.
        Target: Retorno > 0 após N dias.
        """
        if len(df_historico) < MIN_DIAS_HISTORICO: return None
        
        features_col = ['rsi_14', 'macd', 'volatility_20d', 'momentum_60', 'returns']
        df_ml = df_historico[features_col].dropna().copy()
        
        resultados_ticker = {}
        
        for horizonte_nome, dias_target in HORIZONTES_ML.items():
            # Cria Target (1 se retorno futuro > 0, else 0)
            df_ml['target'] = (df_historico['Close'].shift(-dias_target) > df_historico['Close']).astype(int)
            
            # Drop NaNs criados pelo shift
            data_train = df_ml.dropna()
            
            if len(data_train) < 100: # Dados insuficientes para treino seguro
                resultados_ticker[horizonte_nome] = {'proba': 0.5, 'auc': 0.5}
                continue
                
            X = data_train[features_col]
            y = data_train['target']
            
            # Divisão Treino/Teste Temporal
            split = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]
            
            # Verifica se há apenas uma classe
            if len(y_train.unique()) < 2:
                resultados_ticker[horizonte_nome] = {'proba': 0.5, 'auc': 0.5}
                continue
                
            # Modelo 1: Random Forest
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            rf.fit(X_train, y_train)
            
            # Modelo 2: XGBoost
            xgb_model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, eval_metric='logloss', random_state=42)
            xgb_model.fit(X_train, y_train)
            
            # Avaliação (AUC no teste)
            try:
                pred_rf = rf.predict_proba(X_test)[:, 1]
                pred_xgb = xgb_model.predict_proba(X_test)[:, 1]
                ensemble_pred = (pred_rf + pred_xgb) / 2
                auc = roc_auc_score(y_test, ensemble_pred)
            except:
                auc = 0.5
            
            # Predição Final (Usando os dados mais recentes)
            last_features = df_ml[features_col].iloc[[-1]]
            prob_rf = rf.predict_proba(last_features)[0][1]
            prob_xgb = xgb_model.predict_proba(last_features)[0][1]
            final_proba = (prob_rf + prob_xgb) / 2
            
            resultados_ticker[horizonte_nome] = {
                'proba': final_proba,
                'auc': auc
            }
            
        return resultados_ticker

# =============================================================================
# 3. CLASSE PRINCIPAL: CONSTRUTOR DE PORTFÓLIO
# =============================================================================

class ConstrutorPortfolio:
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
        self.engine_ml = EngineML()
        
        import yfinance as yf # Import local
        self.yf = yf

    def coletar_dados(self, tickers_selecionados, progress_bar=None):
        """Orquestra a coleta do yfinance e do fundamentus."""
        
        # 1. Coleta Fundamentalista (Scraping em Batch - Rápido)
        if progress_bar: progress_bar.progress(10, "Acessando Fundamentus (WebScraping)...")
        df_fund = FundamentusScraper.obter_dados_consolidados()
        
        # 2. Coleta de Mercado (yfinance)
        valid_tickers = []
        lista_performance = []
        
        total = len(tickers_selecionados)
        for i, ticker in enumerate(tickers_selecionados):
            if progress_bar: 
                prog = 20 + int((i/total) * 40)
                progress_bar.progress(prog, f"Analisando {ticker} (yfinance + ML)...")
            
            try:
                # Baixa histórico
                hist = self.yf.download(ticker, period=PERIODO_DADOS, progress=False)
                
                if len(hist) < MIN_DIAS_HISTORICO: continue
                
                # Flatten MultiIndex columns se necessário (yfinance update recente)
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.get_level_values(0)
                
                # Calcula Indicadores Técnicos
                hist = EngenheiroFeatures.calcular_indicadores_tecnicos(hist)
                self.dados_mercado[ticker] = hist
                valid_tickers.append(ticker)
                
                # Performance Histórica Simples
                retorno_anual = hist['returns'].mean() * 252
                vol_anual = hist['returns'].std() * np.sqrt(252)
                sharpe = (retorno_anual - TAXA_LIVRE_RISCO) / vol_anual if vol_anual > 0 else 0
                
                # Drawdown
                cum_ret = (1 + hist['returns']).cumprod()
                peak = cum_ret.expanding(min_periods=1).max()
                dd = (cum_ret/peak) - 1
                max_dd = dd.min()
                
                lista_performance.append({
                    'ticker': ticker,
                    'annual_return': retorno_anual,
                    'annual_volatility': vol_anual,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_dd
                })
                
                # 3. Treinamento ML em Tempo Real
                ml_res = self.engine_ml.treinar_ensemble(hist, ticker)
                if ml_res:
                    self.ml_results[ticker] = ml_res
                
            except Exception as e:
                print(f"Erro em {ticker}: {e}")
                continue
        
        if not valid_tickers: return False
        
        # Consolidação
        self.metricas_performance = pd.DataFrame(lista_performance).set_index('ticker')
        
        # Cruzamento com Fundamentalista
        tickers_sem_sa = [t.replace('.SA', '') for t in valid_tickers]
        # Tenta encontrar os dados fundamentais correspondentes
        fund_subset = df_fund[df_fund.index.isin(tickers_sem_sa)].copy()
        # Reindexa para usar os tickers com .SA
        fund_subset.index = [t + '.SA' for t in fund_subset.index]
        self.dados_fundamentos = fund_subset
        
        return True

    def calcular_scores_e_selecionar(self, perfil_risco, horizonte_temporal):
        """Lógica de Pontuação Multi-Fatorial + Clusterização."""
        
        # Define pesos baseados no horizonte
        if 'Curto' in horizonte_temporal:
            w_perf, w_fund, w_tech, w_ml = 0.3, 0.1, 0.3, 0.3
            target_ml = 'Curto Prazo (4m)'
        elif 'Médio' in horizonte_temporal:
            w_perf, w_fund, w_tech, w_ml = 0.3, 0.3, 0.2, 0.2
            target_ml = 'Médio Prazo (8m)'
        else: # Longo
            w_perf, w_fund, w_tech, w_ml = 0.2, 0.5, 0.1, 0.2
            target_ml = 'Longo Prazo (12m)'
            
        # Dataframe Consolidado
        df_master = self.metricas_performance.join(self.dados_fundamentos, how='left')
        
        # Adiciona Tech Current Values
        for t in df_master.index:
            if t in self.dados_mercado:
                df_master.loc[t, 'rsi_curr'] = self.dados_mercado[t]['rsi_14'].iloc[-1]
                df_master.loc[t, 'macd_curr'] = self.dados_mercado[t]['macd_diff'].iloc[-1]
        
        # Adiciona ML Values
        for t in df_master.index:
            res = self.ml_results.get(t, {}).get(target_ml, {'proba': 0.5, 'auc': 0.5})
            df_master.loc[t, 'ml_proba'] = res['proba']
            df_master.loc[t, 'ml_conf'] = res['auc']
            
        # Normalização e Scoring
        scores = pd.DataFrame(index=df_master.index)
        
        # Performance
        scores['s_perf'] = EngenheiroFeatures.normalizar_score(df_master['sharpe_ratio'], True) * w_perf
        
        # Fundamentalista (P/L baixo é bom, ROE alto é bom)
        s_pl = EngenheiroFeatures.normalizar_score(df_master['pe_ratio'], False)
        s_roe = EngenheiroFeatures.normalizar_score(df_master['roe'], True)
        scores['s_fund'] = ((s_pl + s_roe) / 2) * w_fund
        
        # Técnico (RSI não sobrecomprado, MACD forte)
        s_rsi = EngenheiroFeatures.normalizar_score(df_master['rsi_curr'], False) 
        s_macd = EngenheiroFeatures.normalizar_score(df_master['macd_curr'], True)
        scores['s_tech'] = ((s_rsi + s_macd) / 2) * w_tech
        
        # ML (Probabilidade ponderada pela Confiança/AUC)
        s_ml_raw = EngenheiroFeatures.normalizar_score(df_master['ml_proba'], True)
        scores['s_ml'] = (s_ml_raw * df_master['ml_conf']) * w_ml
        
        scores['total_score'] = scores.sum(axis=1)
        self.scores = scores.join(df_master).sort_values('total_score', ascending=False)
        
        # --- Clusterização (PCA + KMeans) para Diversificação ---
        features_cluster = ['sharpe_ratio', 'annual_volatility', 'pe_ratio', 'roe', 'ml_proba']
        data_cluster = self.scores[features_cluster].fillna(0)
        
        # Normaliza para PCA
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_cluster)
        
        # PCA
        pca = PCA(n_components=min(3, len(scaled_data)))
        pca_data = pca.fit_transform(scaled_data)
        
        # KMeans (K ótimo simplificado para 5 ou sqrt(N)/2)
        k = min(NUM_ATIVOS_PORTFOLIO, len(scaled_data))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        self.scores['Cluster'] = kmeans.fit_predict(pca_data)
        
        # --- Seleção Final ---
        # Pega o melhor de cada cluster
        selecionados = []
        clusters_uniques = self.scores['Cluster'].unique()
        
        for c in clusters_uniques:
            best_in_cluster = self.scores[self.scores['Cluster'] == c].index[0]
            selecionados.append(best_in_cluster)
            
        # Se faltar ativos, completa com os melhores scores gerais que sobraram
        remaining = [x for x in self.scores.index if x not in selecionados]
        while len(selecionados) < NUM_ATIVOS_PORTFOLIO and remaining:
            selecionados.append(remaining.pop(0))
            
        self.ativos_selecionados = selecionados[:NUM_ATIVOS_PORTFOLIO]
        return True

    def otimizar_markowitz(self, perfil_risco):
        """Otimização de Pesos (Min Vol ou Max Sharpe)."""
        ativos = self.ativos_selecionados
        if not ativos: return
        
        # DataFrame de Retornos Conjuntos
        retornos_df = pd.DataFrame({t: self.dados_mercado[t]['returns'] for t in ativos}).dropna()
        
        mu = retornos_df.mean() * 252
        cov = retornos_df.cov() * 252
        n = len(ativos)
        
        def get_portfolio_stats(weights):
            weights = np.array(weights)
            ret = np.sum(retornos_df.mean() * weights) * 252
            vol = np.sqrt(np.dot(weights.T, np.dot(retornos_df.cov() * 252, weights)))
            return ret, vol
        
        def neg_sharpe(weights):
            ret, vol = get_portfolio_stats(weights)
            return -(ret - TAXA_LIVRE_RISCO) / vol
            
        def min_vol(weights):
            return get_portfolio_stats(weights)[1]
            
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((PESO_MIN, PESO_MAX) for _ in range(n))
        init_guess = [1/n] * n
        
        if 'Conservador' in perfil_risco or 'Intermediário' in perfil_risco:
            res = minimize(min_vol, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
            metodo = "Mínima Volatilidade"
        else:
            res = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
            metodo = "Máximo Sharpe"
            
        pesos = res.x
        
        # Monta resultado
        for i, ticker in enumerate(ativos):
            self.alocacao[ticker] = {
                'peso': pesos[i],
                'valor': self.investimento * pesos[i]
            }
            
        # Métricas Finais do Portfólio
        ret_p, vol_p = get_portfolio_stats(pesos)
        self.metricas_finais = {
            'retorno_anual': ret_p,
            'volatilidade_anual': vol_p,
            'sharpe': (ret_p - TAXA_LIVRE_RISCO) / vol_p,
            'metodo': metodo
        }
        
        # Gera Justificativas
        for t in ativos:
            row = self.scores.loc[t]
            target_ml = 'Curto' # Default name placeholder
            ml_info = f"Prob. Alta: {row['ml_proba']*100:.1f}%"
            fund_info = f"P/L: {row['pe_ratio']:.1f}, ROE: {row['roe']*100:.1f}%"
            tech_info = f"RSI: {row['rsi_curr']:.1f}"
            self.justificativas[t] = f"{ml_info} | {fund_info} | {tech_info}"

# =============================================================================
# 4. INTERFACE E FLUXO (STREAMLIT)
# =============================================================================

def configurar_pagina():
    st.set_page_config(
        page_title="Sistema de Portfólios Adaptativos",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS Elite Style (Preto/Branco/Cinza Profissional)
    st.markdown("""
        <style>
        :root { --primary-color: #000000; --secondary-color: #6c757d; --bg-light: #ffffff; --bg-dark: #f8f9fa; }
        body { background-color: var(--bg-light); color: #212529; font-family: 'Arial', sans-serif; }
        .main-header { font-family: 'Arial', sans-serif; color: #000000; text-align: center; border-bottom: 2px solid #dee2e6; padding-bottom: 10px; font-size: 2.2rem; margin-bottom: 20px; font-weight: 600; }
        .stButton button { border: 1px solid #000000 !important; color: #000000 !important; background-color: transparent !important; border-radius: 6px; transition: all 0.3s ease; }
        .stButton button:hover { background-color: #000000 !important; color: #ffffff !important; }
        .stButton button[kind="primary"] { background-color: #000000 !important; color: #ffffff !important; border: none !important; }
        .stTabs [data-baseweb="tab-list"] { justify-content: center; border-bottom: 2px solid #dee2e6; }
        .stTabs [data-baseweb="tab"] { color: #6c757d; }
        .stTabs [aria-selected="true"] { border-bottom: 2px solid #000000; color: #000000; font-weight: 700; }
        .info-box { background-color: #f8f9fa; border-left: 4px solid #000000; padding: 15px; margin: 10px 0; border-radius: 6px; }
        .stMetric { background-color: #f8f9fa; padding: 10px; border-radius: 6px; border: 1px solid #dee2e6; }
        </style>
    """, unsafe_allow_html=True)

def obter_template_grafico():
    return {
        'plot_bgcolor': '#f8f9fa',
        'paper_bgcolor': 'white',
        'font': {'family': 'Arial', 'color': '#343a40'},
        'colorway': ['#000000', '#495057', '#adb5bd', '#ced4da']
    }

# --- ABAS DO SISTEMA ---

def aba_introducao():
    st.markdown("## 📚 Metodologia Híbrida e Arquitetura")
    st.markdown("""
    <div class="info-box">
    <h3>🎯 Visão Geral</h3>
    <p>Este sistema integra o poder analítico do WebScraping em tempo real (Fundamentus) com modelos avançados de Machine Learning (Ensemble RF+XGBoost) para construir portfólios resilientes.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 1. Coleta de Dados (Ao Vivo)")
        st.write("- **Mercado:** `yfinance` captura histórico de preços e volumes.")
        st.write("- **Fundamentos:** `Scraper` proprietário extrai dados do Fundamentus.com.br.")
        st.write("- **Engenharia:** Cálculo on-the-fly de RSI, MACD, Volatilidade.")
        
    with col2:
        st.markdown("### 2. Inteligência Artificial")
        st.write("- **Ensemble:** Combinação de Random Forest e XGBoost.")
        st.write("- **Multi-Horizonte:** Previsões simultâneas para 4, 8 e 12 meses.")
        st.write("- **Clusterização:** PCA + KMeans para garantir diversificação estrutural.")

def aba_selecao_ativos():
    st.markdown("## 🎯 Universo de Análise (Ibovespa)")
    
    modo = st.radio("Modo de Seleção:", ["Índice Completo", "Seleção Setorial", "Manual"], horizontal=True)
    
    selecionados = []
    if modo == "Índice Completo":
        selecionados = ATIVOS_IBOVESPA
        st.success(f"{len(selecionados)} ativos selecionados.")
        
    elif modo == "Seleção Setorial":
        setores = st.multiselect("Setores:", list(ATIVOS_POR_SETOR.keys()))
        for s in setores:
            selecionados.extend(ATIVOS_POR_SETOR[s])
        selecionados = list(set(selecionados))
        
    else:
        selecionados = st.multiselect("Ativos:", TODOS_ATIVOS, default=TODOS_ATIVOS[:5])
        
    if selecionados:
        st.session_state.ativos_analise = selecionados
        st.dataframe(pd.DataFrame({'Tickers': selecionados}).T, hide_index=True)
    else:
        st.warning("Selecione pelo menos um ativo.")

def aba_construtor_portfolio():
    if 'ativos_analise' not in st.session_state:
        st.warning("Vá para a aba 'Seleção de Ativos' primeiro.")
        return

    # Questionário Simplificado
    st.markdown("### 📋 Perfil do Investidor")
    col1, col2 = st.columns(2)
    with col1:
        risco = st.select_slider("Tolerância ao Risco", ["Conservador", "Intermediário", "Moderado", "Arrojado", "Avançado"], value="Moderado")
        capital = st.number_input("Capital (R$)", 1000.0, 10000000.0, 10000.0)
    with col2:
        horizonte = st.select_slider("Horizonte Temporal", ["Curto Prazo (4m)", "Médio Prazo (8m)", "Longo Prazo (12m)"], value="Médio Prazo (8m)")
    
    if st.button("🚀 Construir Portfólio", type="primary"):
        builder = ConstrutorPortfolio(capital)
        st.session_state.builder = builder
        
        progresso = st.progress(0, text="Iniciando motores...")
        
        # 1. Coleta e ML
        sucesso = builder.coletar_dados(st.session_state.ativos_analise, progresso)
        if not sucesso:
            st.error("Falha na coleta de dados.")
            return
            
        # 2. Scoring e Seleção
        progresso.progress(80, "Realizando Clusterização e Seleção...")
        builder.calcular_scores_e_selecionar(risco, horizonte)
        
        # 3. Otimização
        progresso.progress(95, "Otimizando Markowitz...")
        builder.otimizar_markowitz(risco)
        
        progresso.progress(100, "Pronto!")
        st.session_state.analise_completa = True
        st.rerun()

    # Exibição dos Resultados
    if st.session_state.get('analise_completa'):
        b = st.session_state.builder
        
        st.markdown("---")
        st.markdown("### ✅ Portfólio Otimizado")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retorno Esp. (a.a.)", f"{b.metricas_finais['retorno_anual']*100:.2f}%")
        c2.metric("Volatilidade (a.a.)", f"{b.metricas_finais['volatilidade_anual']*100:.2f}%")
        c3.metric("Sharpe Ratio", f"{b.metricas_finais['sharpe']:.2f}")
        c4.metric("Estratégia", b.metricas_finais['metodo'])
        
        # Gráfico Pizza
        df_alloc = pd.DataFrame([
            {'Ativo': k, 'Peso': v['peso']*100, 'Valor': v['valor']} 
            for k, v in b.alocacao.items()
        ])
        
        fig = px.pie(df_alloc, values='Peso', names='Ativo', title='Alocação de Capital', hole=0.4)
        fig.update_layout(obter_template_grafico())
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabela Detalhada
        st.markdown("#### Detalhamento Tático")
        # Adiciona justificativas na tabela
        df_alloc['Justificativa (ML | Fund | Tech)'] = df_alloc['Ativo'].map(b.justificativas)
        st.dataframe(df_alloc.style.format({'Peso': '{:.2f}%', 'Valor': 'R$ {:,.2f}'}), use_container_width=True)

def aba_analise_individual():
    st.markdown("## 🔍 Raio-X do Ativo")
    
    if 'builder' not in st.session_state or not st.session_state.builder.dados_mercado:
        st.warning("Execute o Construtor primeiro para carregar os dados.")
        return
        
    b = st.session_state.builder
    ativos_disp = list(b.dados_mercado.keys())
    
    ativo = st.selectbox("Selecione o Ativo:", ativos_disp)
    
    if st.button("Analisar", type="primary"):
        df_hist = b.dados_mercado[ativo]
        df_fund = b.dados_fundamentos[b.dados_fundamentos.index == ativo]
        ml_res = b.ml_results.get(ativo, {})
        
        # Tabulações Internas
        t1, t2, t3 = st.tabs(["Gráfico & Tech", "Fundamentos", "Inteligência Artificial"])
        
        with t1:
            # Gráfico Candle + Médias
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df_hist.index, open=df_hist['Open'], high=df_hist['High'], low=df_hist['Low'], close=df_hist['Close'], name='Preço'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['sma_50'], name='SMA 50', line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['sma_200'], name='SMA 200', line=dict(color='blue')), row=1, col=1)
            
            # RSI
            fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['rsi_14'], name='RSI', line=dict(color='black')), row=2, col=1)
            fig.add_hline(y=70, row=2, col=1, line_dash="dot", line_color="red")
            fig.add_hline(y=30, row=2, col=1, line_dash="dot", line_color="green")
            
            fig.update_layout(title=f"Análise Técnica: {ativo}", height=600, **obter_template_grafico())
            st.plotly_chart(fig, use_container_width=True)
            
        with t2:
            if not df_fund.empty:
                row = df_fund.iloc[0]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("P/L", f"{row['pe_ratio']:.2f}")
                c2.metric("ROE", f"{row['roe']*100:.2f}%")
                c3.metric("Div. Yield", f"{row['div_yield']*100:.2f}%")
                c4.metric("Margem Líq.", f"{row['net_margin']*100:.2f}%")
                
                st.table(df_fund.T)
            else:
                st.warning("Dados fundamentalistas não encontrados para este ativo (Fundamentus).")
                
        with t3:
            st.markdown("### Previsões do Ensemble (Random Forest + XGBoost)")
            
            res_data = []
            for horiz, res in ml_res.items():
                res_data.append({
                    'Horizonte': horiz,
                    'Probabilidade Alta': f"{res['proba']*100:.2f}%",
                    'Confiança (AUC)': f"{res['auc']:.2f}",
                    'Sinal': "🟢 Compra" if res['proba'] > 0.55 else ("🔴 Venda" if res['proba'] < 0.45 else "⚪ Neutro")
                })
            
            st.dataframe(pd.DataFrame(res_data), use_container_width=True)
            
            st.info("O modelo Ensemble combina árvores de decisão (RF) com Gradient Boosting (XGB) para reduzir variância e viés.")

def aba_referencias():
    st.markdown("## 📚 Referências Bibliográficas e Fontes")
    st.markdown("""
    <div class="info-box">
    <p><strong>1. Markowitz, H. (1952).</strong> Portfolio Selection. The Journal of Finance.</p>
    <p><em>Base para a otimização de Média-Variância utilizada no módulo de alocação.</em></p>
    </div>
    
    <div class="info-box">
    <p><strong>2. Chen, T., & Guestrin, C. (2016).</strong> XGBoost: A Scalable Tree Boosting System.</p>
    <p><em>Algoritmo principal utilizado no Ensemble de Machine Learning para previsão de retornos.</em></p>
    </div>
    
    <div class="info-box">
    <p><strong>3. Fundamentus & Yahoo Finance</strong></p>
    <p><em>Fontes de dados primárias para análise fundamentalista (WebScraping) e técnica (API).</em></p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# 5. MAIN APP LOOP
# =============================================================================

def main():
    if 'builder' not in st.session_state:
        st.session_state.builder = None
        st.session_state.analise_completa = False
        st.session_state.ativos_analise = []
        
    configurar_pagina()
    
    st.markdown('<h1 class="main-header">Sistema de Portfólios Adaptativos (Integrated Elite)</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📚 Metodologia",
        "🎯 Seleção de Ativos",
        "🏗️ Construtor de Portfólio",
        "🔍 Análise Individual",
        "📖 Referências"
    ])
    
    with tab1: aba_introducao()
    with tab2: aba_selecao_ativos()
    with tab3: aba_construtor_portfolio()
    with tab4: aba_analise_individual()
    with tab5: aba_referencias()

if __name__ == "__main__":
    main()
