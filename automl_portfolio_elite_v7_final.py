# -*- coding: utf-8 -*-
"""
=============================================================================
SISTEMA DE PORTFÓLIOS ADAPTATIVOS - ELITE V8 (REAL-TIME ENGINE)
=============================================================================

Fusão da Interface Profissional (V7) com Motor Simplificado Real-Time (Analyzer).

Mudanças Estruturais:
- ETL: Substituído GCS estático por Coleta Real-Time (YFinance + Fundamentus).
- ML: Treinamento local instantâneo (RandomForest + XGBoost) para 3 horizontes.
- Otimização: Markowitz (Média-Variância) + Clusterização (PCA/KMeans).
- Design: Estritamente mantido o tema "Clean/Black" do V7.

=============================================================================
"""

# --- 1. CORE LIBRARIES ---
import warnings
import numpy as np
import pandas as pd
import sys
import time
import requests
import json
from io import StringIO
from datetime import datetime, timedelta
import traceback

# --- 2. SCIENTIFIC ---
from scipy.optimize import minimize
from scipy.stats import zscore

# --- 3. STREAMLIT & PLOTTING ---
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- 4. DATA ---
import yfinance as yf

# --- 5. ML ---
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier # Fallback

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURAÇÕES GLOBAIS
# =============================================================================

PERIODO_DADOS = '2y' # Período suficiente para cálculo técnico e ML rápido
MIN_DIAS_HISTORICO = 120
NUM_ATIVOS_PORTFOLIO = 5
TAXA_LIVRE_RISCO = 0.1175
SCORE_PERCENTILE_THRESHOLD = 0.70

WEIGHT_PERFORMANCE = 0.40
WEIGHT_FUNDAMENTAL = 0.30
WEIGHT_TECHNICAL = 0.30

PESO_MIN = 0.10
PESO_MAX = 0.30

# Lista base IBOVESPA
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
TODOS_ATIVOS = sorted(list(set(ATIVOS_IBOVESPA)))

# =============================================================================
# 2. MAPEAMENTOS DE QUESTIONÁRIO (IDÊNTICO AO V7)
# =============================================================================

OPTIONS_CONCORDA = [
    "CT: (Concordo Totalmente) - Estou confortável com altas flutuações, pois entendo que são o preço para retornos potencialmente maiores.",
    "C: (Concordo) - Aceito alguma volatilidade, mas espero que os ganhos compensem o risco assumido de forma clara.",
    "N: (Neutro) - Tenho dificuldade em opinar; minha decisão dependeria do momento e do ativo específico.",
    "D: (Discordo) - Prefiro estratégias mais cautelosas, mesmo que isso signifique um potencial de retorno menor.",
    "DT: (Discordo Totalmente) - Não estou disposto a ver meu patrimônio flutuar significativamente; prefiro segurança absoluta."
]
MAP_CONCORDA = {OPTIONS_CONCORDA[0]: 'CT', OPTIONS_CONCORDA[1]: 'C', OPTIONS_CONCORDA[2]: 'N', OPTIONS_CONCORDA[3]: 'D', OPTIONS_CONCORDA[4]: 'DT'}

OPTIONS_DISCORDA = [
    "CT: (Concordo Totalmente) - A preservação do capital é minha prioridade máxima, acima de qualquer ganho potencial.",
    "C: (Concordo) - É muito importante para mim evitar perdas, mesmo que isso limite o crescimento do meu portfólio.",
    "N: (Neutro) - Busco um equilíbrio; não quero perdas excessivas, mas sei que algum risco é necessário para crescer.",
    "D: (Discordo) - Estou focado no crescimento de longo prazo e entendo que perdas de curto prazo fazem parte do processo.",
    "DT: (Discordo Totalmente) - Meu foco é maximizar o retorno; perdas de curto prazo são irrelevantes se a tese de longo prazo for válida."
]
MAP_DISCORDA = {OPTIONS_DISCORDA[0]: 'CT', OPTIONS_DISCORDA[1]: 'C', OPTIONS_DISCORDA[2]: 'N', OPTIONS_DISCORDA[3]: 'D', OPTIONS_DISCORDA[4]: 'DT'}

OPTIONS_REACTION_DETALHADA = [
    "A: (Vender Imediatamente) - Venderia a posição para evitar perdas maiores; prefiro realizar o prejuízo e reavaliar.",
    "B: (Manter e Reavaliar) - Manteria a calma, reavaliaria os fundamentos do ativo e o cenário macro para tomar uma decisão.",
    "C: (Comprar Mais) - Encararia como uma oportunidade de compra, aumentando a posição a um preço menor, se os fundamentos estiverem intactos."
]
MAP_REACTION = {OPTIONS_REACTION_DETALHADA[0]: 'A', OPTIONS_REACTION_DETALHADA[1]: 'B', OPTIONS_REACTION_DETALHADA[2]: 'C'}

OPTIONS_CONHECIMENTO_DETALHADA = [
    "A: (Avançado) - Sinto-me confortável analisando balanços (fundamentalista), gráficos (técnica) e cenários macroeconômicos.",
    "B: (Intermediário) - Entendo os conceitos básicos (Renda Fixa vs. Variável, risco vs. retorno) e acompanho o mercado.",
    "C: (Iniciante) - Tenho pouca ou nenhuma experiência prática em investimentos além da poupança ou produtos bancários simples."
]
MAP_CONHECIMENTO = {OPTIONS_CONHECIMENTO_DETALHADA[0]: 'A', OPTIONS_CONHECIMENTO_DETALHADA[1]: 'B', OPTIONS_CONHECIMENTO_DETALHADA[2]: 'C'}

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

SCORE_MAP_ORIGINAL = {'CT': 5, 'C': 4, 'N': 3, 'D': 2, 'DT': 1}
SCORE_MAP_INV_ORIGINAL = {'CT': 1, 'C': 2, 'N': 3, 'D': 4, 'DT': 5}
SCORE_MAP_REACTION_ORIGINAL = {'A': 1, 'B': 3, 'C': 5} # A=Vender(1), B=Manter(3), C=Comprar(5)
SCORE_MAP_CONHECIMENTO_ORIGINAL = {'A': 5, 'B': 3, 'C': 1}

# =============================================================================
# 3. CLASSES DE PERFIL E ESTILO
# =============================================================================

class AnalisadorPerfilInvestidor:
    """Mesma lógica calibrada do V7."""
    
    def determinar_nivel_risco(self, pontuacao: int) -> str:
        if pontuacao <= 46: return "CONSERVADOR"
        elif pontuacao <= 67: return "INTERMEDIÁRIO"
        elif pontuacao <= 88: return "MODERADO"
        elif pontuacao <= 109: return "MODERADO-ARROJADO"
        else: return "AVANÇADO"
    
    def determinar_horizonte_ml(self, liquidez_key: str, objetivo_key: str) -> tuple[str, int]:
        # A=Curto, B=Médio, C=Longo
        mapa = {'A': 1, 'B': 2, 'C': 3}
        val = max(mapa.get(liquidez_key, 1), mapa.get(objetivo_key, 1))
        
        if val == 3: return "LONGO PRAZO", 252 # 12 meses
        elif val == 2: return "MÉDIO PRAZO", 168 # 8 meses
        else: return "CURTO PRAZO", 84 # 4 meses
    
    def calcular_perfil(self, respostas: dict) -> tuple[str, str, int, int]:
        # Mapeia chaves (ex: 'CT') para valores numéricos
        s1 = SCORE_MAP_ORIGINAL.get(respostas['risk_accept'], 3)
        s2 = SCORE_MAP_ORIGINAL.get(respostas['max_gain'], 3)
        s3 = SCORE_MAP_INV_ORIGINAL.get(respostas['stable_growth'], 3)
        s4 = SCORE_MAP_INV_ORIGINAL.get(respostas['avoid_loss'], 3)
        s5 = SCORE_MAP_CONHECIMENTO_ORIGINAL.get(respostas['level'], 3)
        s6 = SCORE_MAP_REACTION_ORIGINAL.get(respostas['reaction'], 3)

        pontuacao = (s1 * 5 + s2 * 5 + s3 * 5 + s4 * 5 + s5 * 3 + s6 * 3)
        nivel = self.determinar_nivel_risco(pontuacao)
        
        # Extrai só a letra (A, B, C)
        liq = respostas['liquidity'][0]
        obj = respostas['time_purpose'][0]
        
        horiz, dias_ml = self.determinar_horizonte_ml(liq, obj)
        return nivel, horiz, dias_ml, pontuacao

def obter_template_grafico() -> dict:
    """Template Limpo/Neutro do V7."""
    return {
        'plot_bgcolor': '#f8f9fa',
        'paper_bgcolor': 'white',
        'font': {'family': 'Arial, sans-serif', 'size': 12, 'color': '#343a40'},
        'title': {'font': {'family': 'Arial, sans-serif', 'size': 16, 'color': '#212529', 'weight': 'bold'}, 'x': 0.5, 'xanchor': 'center'},
        'xaxis': {'showgrid': True, 'gridcolor': '#e9ecef'},
        'yaxis': {'showgrid': True, 'gridcolor': '#e9ecef'},
        'colorway': ['#212529', '#495057', '#6c757d', '#adb5bd', '#ced4da']
    }

# =============================================================================
# 4. ENGENHARIA DE DADOS (SIMPLIFICADA: YFINANCE + FUNDAMENTUS)
# =============================================================================

class EngenheiroFeatures:
    """Cálculo local de indicadores (estilo Analyzer)."""
    @staticmethod
    def normalizar(serie: pd.Series, maior_melhor: bool = True) -> pd.Series:
        if serie.isnull().all(): return pd.Series(0.5, index=serie.index)
        q_low, q_high = serie.quantile(0.02), serie.quantile(0.98)
        clipped = serie.clip(q_low, q_high)
        min_v, max_v = clipped.min(), clipped.max()
        if max_v == min_v: return pd.Series(0.5, index=serie.index)
        norm = (clipped - min_v) / (max_v - min_v)
        return norm if maior_melhor else (1 - norm)

    @staticmethod
    def calcular_tecnicos(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < 60: return df
        df = df.copy()
        df['Returns'] = df['Close'].pct_change()
        
        # RSI 14
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi_14'] = 100 - (100 / (1 + (gain/loss)))
        
        # MACD
        exp12 = df['Close'].ewm(span=12).mean()
        exp26 = df['Close'].ewm(span=26).mean()
        df['macd'] = exp12 - exp26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # Volatilidade e Momentum
        df['volatility_annual'] = df['Returns'].rolling(20).std() * np.sqrt(252)
        df['momentum_60'] = df['Close'] / df['Close'].shift(60) - 1
        
        return df.dropna()

class ColetorDados:
    """Motor Híbrido: WebScraping (Fundamentus) + API (YFinance)."""
    
    def buscar_fundamentos(self) -> pd.DataFrame:
        url = 'https://www.fundamentus.com.br/resultado.php'
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            r = requests.get(url, headers=headers)
            df = pd.read_html(StringIO(r.text), decimal=',', thousands='.')[0]
            
            # Renomeia para padronizar com sistema V7
            rename_map = {
                'Papel': 'Ticker', 'P/L': 'pe_ratio', 'P/VP': 'pb_ratio',
                'Div.Yield': 'div_yield', 'ROE': 'roe', 'Mrg. Líq.': 'net_margin',
                'Liq. Corr.': 'current_ratio', 'Dív.Brut/ Patrim.': 'debt_to_equity'
            }
            df = df.rename(columns=rename_map)
            df['Ticker'] = df['Ticker'].astype(str) + '.SA'
            
            # Limpeza percentual
            for col in ['div_yield', 'roe', 'net_margin']:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace('%','').str.replace('.','').str.replace(',','.').astype(float)/100
            
            # Adiciona setor fictício se não tiver (Fundamentus tabela geral não tem setor explícito fácil)
            # Para manter design V7 que mostra setor, faremos um merge simples com lista estática se necessário,
            # ou deixamos "Desconhecido"
            return df.set_index('Ticker')
        except:
            return pd.DataFrame()

    def coletar_historico(self, tickers: list) -> dict:
        dados = {}
        if not tickers: return dados
        
        # Download em lote para performance
        tickers_str = " ".join(tickers)
        try:
            data = yf.download(tickers_str, period=PERIODO_DADOS, group_by='ticker', threads=True, progress=False)
            for t in tickers:
                if len(tickers) > 1:
                    df = data[t].copy()
                else:
                    df = data.copy()
                
                df = df.dropna(how='all')
                if len(df) > MIN_DIAS_HISTORICO:
                    df = EngenheiroFeatures.calcular_tecnicos(df)
                    dados[t] = df
        except:
            pass
        return dados

# =============================================================================
# 5. MACHINE LEARNING & OTIMIZAÇÃO (LÓGICA SIMPLIFICADA)
# =============================================================================

class MotorAnalise:
    """Processa ML e Portfólio."""
    
    def __init__(self):
        self.features_ml = ['rsi_14', 'macd', 'macd_diff', 'volatility_annual', 'momentum_60']
        
    def treinar_ensemble(self, dados_ativos: dict, dias_target: int) -> pd.DataFrame:
        res = []
        for t, df in dados_ativos.items():
            if len(df) < 150: continue
            
            # Target: Retorno futuro > 0
            df_ml = df.copy()
            df_ml['target'] = (df_ml['Close'].shift(-dias_target) > df_ml['Close']).astype(int)
            df_ml = df_ml.dropna()
            
            if df_ml.empty: continue
            
            X = df_ml[self.features_ml]
            y = df_ml['target']
            
            # Treino simples (sem grid search demorado)
            rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
            xgb = XGBClassifier(n_estimators=50, max_depth=3, eval_metric='logloss', random_state=42)
            
            rf.fit(X, y)
            xgb.fit(X, y)
            
            # Predição atual
            last = df[self.features_ml].iloc[[-1]]
            p_rf = rf.predict_proba(last)[0][1]
            p_xgb = xgb.predict_proba(last)[0][1]
            prob_final = (p_rf + p_xgb) / 2
            
            # Confiança (Acurácia OOB aproximada ou score de treino penalizado)
            conf = rf.score(X, y) * 0.9 # Penalização leve
            
            res.append({'Ticker': t, 'ML_Proba': prob_final, 'ML_Confidence': conf})
            
        return pd.DataFrame(res).set_index('Ticker')

    def otimizar_markowitz(self, retornos: pd.DataFrame, risco: str) -> dict:
        if retornos.empty: return {}
        mu = retornos.mean() * 252
        cov = retornos.cov() * 252
        n = len(retornos.columns)
        
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bnds = tuple((PESO_MIN, PESO_MAX) for _ in range(n))
        init = np.array([1/n]*n)
        
        def stats(w):
            return np.dot(w, mu), np.sqrt(np.dot(w.T, np.dot(cov, w)))
        
        if 'CONSERVADOR' in risco:
            # Min Vol
            fun = lambda w: stats(w)[1]
        else:
            # Max Sharpe
            fun = lambda w: -(stats(w)[0] - TAXA_LIVRE_RISCO) / stats(w)[1]
            
        try:
            res = minimize(fun, init, method='SLSQP', bounds=bnds, constraints=cons)
            return {c: w for c, w in zip(retornos.columns, res.x)}
        except:
            return {c: 1/n for c in retornos.columns}

# =============================================================================
# 6. CONTROLLER (CONSTRUTOR)
# =============================================================================

class ConstrutorPortfolio:
    def __init__(self, investimento):
        self.investimento = investimento
        self.dados_ativos = {}
        self.dados_fund = pd.DataFrame()
        self.predicoes_ml = pd.DataFrame()
        self.scores = pd.DataFrame()
        self.selecionados = []
        self.alocacao = {}
        self.metricas = {}
        self.pesos_fatores = {}
        self.justificativas = {}
        self.metodo_alocacao = ""

    def executar(self, tickers, perfil, pbar):
        motor = MotorAnalise()
        coletor = ColetorDados()
        
        pbar.progress(10, text="Conectando Fundamentus (Scraping)...")
        self.dados_fund = coletor.buscar_fundamentos()
        validos = [t for t in tickers if t in self.dados_fund.index]
        
        pbar.progress(30, text="Baixando Histórico YFinance...")
        self.dados_ativos = coletor.coletar_historico(validos)
        tickers_finais = list(self.dados_ativos.keys())
        
        if len(tickers_finais) < NUM_ATIVOS_PORTFOLIO: return False
        
        # ML
        dias_ml = perfil.get('ml_lookback_days', 168)
        pbar.progress(50, text=f"Treinando Ensemble para {dias_ml} dias...")
        self.predicoes_ml = motor.treinar_ensemble(self.dados_ativos, dias_ml)
        
        # Scoring
        pbar.progress(70, text="Calculando Scores Multi-Fatoriais...")
        self._calcular_scores(tickers_finais, perfil['time_horizon'])
        
        # Clusterização
        pbar.progress(80, text="Clusterização (PCA + KMeans)...")
        self._selecionar_cluster()
        
        # Alocação
        pbar.progress(90, text="Otimizando Markowitz...")
        retornos = pd.DataFrame({t: self.dados_ativos[t]['Returns'] for t in self.selecionados}).dropna()
        pesos = motor.otimizar_markowitz(retornos, perfil['risk_level'])
        
        self.alocacao = {t: {'weight': w, 'amount': w*self.investimento} for t, w in pesos.items()}
        if 'CONSERVADOR' in perfil['risk_level']: self.metodo_alocacao = "Mínima Volatilidade"
        else: self.metodo_alocacao = "Máximo Sharpe"
        
        # Métricas Finais
        self._calc_metricas_finais(retornos, list(pesos.values()))
        self._gerar_justificativas()
        
        pbar.progress(100, text="Concluído!")
        return True

    def _calcular_scores(self, tickers, horizonte):
        # Pesos Dinâmicos
        if "CURTO" in horizonte: wp, wf, wt, wm = 0.3, 0.1, 0.4, 0.2
        elif "LONGO" in horizonte: wp, wf, wt, wm = 0.3, 0.5, 0.1, 0.1
        else: wp, wf, wt, wm = 0.3, 0.3, 0.2, 0.2
        self.pesos_fatores = {'Perf': wp, 'Fund': wf, 'Tec': wt, 'ML': wm}
        
        scores = pd.DataFrame(index=tickers)
        norm = EngenheiroFeatures.normalizar
        
        for t in tickers:
            df = self.dados_ativos[t]
            fund = self.dados_fund.loc[t]
            ml = self.predicoes_ml.loc[t] if t in self.predicoes_ml.index else pd.Series()
            
            # Raw Values
            scores.loc[t, 'sharpe'] = (df['Returns'].mean()*252)/(df['Returns'].std()*np.sqrt(252))
            scores.loc[t, 'pe_ratio'] = fund.get('pe_ratio', 20)
            scores.loc[t, 'roe'] = fund.get('roe', 0.1)
            scores.loc[t, 'rsi'] = df['rsi_14'].iloc[-1]
            scores.loc[t, 'ml_proba'] = ml.get('ML_Proba', 0.5)
            scores.loc[t, 'ml_conf'] = ml.get('ML_Confidence', 0.5)

        # Cálculo Ponderado
        s_perf = norm(scores['sharpe']) * wp
        s_fund = (norm(scores['pe_ratio'], False)*0.5 + norm(scores['roe'])*0.5) * wf
        s_tec = norm(scores['rsi'], False) * wt # RSI baixo = compra (simplificação)
        s_ml = (norm(scores['ml_proba'])*0.7 + norm(scores['ml_conf'])*0.3) * wm
        
        scores['total_score'] = s_perf + s_fund + s_tec + s_ml
        self.scores = scores.sort_values('total_score', ascending=False)

    def _selecionar_cluster(self):
        # PCA + KMeans
        df_feat = self.scores[['sharpe', 'pe_ratio', 'roe', 'rsi', 'ml_proba']].dropna()
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df_feat)
        
        pca = PCA(n_components=min(3, len(scaled)))
        comps = pca.fit_transform(scaled)
        
        km = KMeans(n_clusters=NUM_ATIVOS_PORTFOLIO, random_state=42)
        clusters = km.fit_predict(comps)
        
        self.scores['Cluster'] = pd.Series(clusters, index=df_feat.index)
        
        # Seleciona melhor de cada cluster
        final = []
        for c in range(NUM_ATIVOS_PORTFOLIO):
            cand = self.scores[self.scores['Cluster'] == c].sort_values('total_score', ascending=False)
            if not cand.empty: final.append(cand.index[0])
            
        # Completa se faltar
        if len(final) < NUM_ATIVOS_PORTFOLIO:
            rest = [x for x in self.scores.index if x not in final]
            final.extend(rest[:NUM_ATIVOS_PORTFOLIO - len(final)])
            
        self.selecionados = final[:NUM_ATIVOS_PORTFOLIO]

    def _calc_metricas_finais(self, retornos, pesos):
        w = np.array(pesos)
        mu = retornos.mean() * 252
        cov = retornos.cov() * 252
        
        ret = np.dot(w, mu)
        vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        shp = (ret - TAXA_LIVRE_RISCO) / vol
        
        self.metricas = {'annual_return': ret, 'annual_volatility': vol, 'sharpe_ratio': shp}

    def _gerar_justificativas(self):
        for t in self.selecionados:
            s = self.scores.loc[t]
            txt = f"Score: {s['total_score']:.2f} | Sharpe: {s['sharpe']:.2f} | P/L: {s['pe_ratio']:.1f} | ML Prob: {s['ml_proba']:.1%}"
            self.justificativas[t] = txt

# =============================================================================
# 7. INTERFACE STREAMLIT (DESIGN IDÊNTICO AO V7)
# =============================================================================

def configurar_pagina():
    st.set_page_config(page_title="Sistema de Portfólios Adaptativos", page_icon="📈", layout="wide")
    # CSS COPIADO DO V7 PARA MANTER IDENTIDADE VISUAL
    st.markdown("""
        <style>
        :root { --primary-color: #000000; --secondary-color: #6c757d; --background-light: #ffffff; --background-dark: #f8f9fa; --text-color: #212529; --text-color-light: #ffffff; --border-color: #dee2e6; }
        body { background-color: var(--background-light); color: var(--text-color); }
        .main-header { font-family: 'Arial', sans-serif; color: var(--primary-color); text-align: center; border-bottom: 2px solid var(--border-color); padding-bottom: 10px; font-size: 2.2rem !important; margin-bottom: 20px; font-weight: 600; }
        .stButton button, .stDownloadButton button { border: 1px solid var(--primary-color) !important; color: var(--primary-color) !important; border-radius: 6px; padding: 8px 16px; background-color: transparent !important; }
        .stButton button:hover, .stDownloadButton button:hover { background-color: var(--primary-color) !important; color: var(--text-color-light) !important; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); }
        .stButton button[kind="primary"], .stFormSubmitButton button { background-color: var(--primary-color) !important; color: var(--text-color-light) !important; border: none !important; }
        .stTabs [data-baseweb="tab-list"] { gap: 24px; border-bottom: 2px solid var(--border-color); display: flex; justify-content: center; width: 100%; }
        .stTabs [data-baseweb="tab"] { height: 40px; background-color: transparent; color: var(--secondary-color); font-weight: 500; flex-grow: 0 !important; }
        .stTabs [aria-selected="true"] { border-bottom: 2px solid var(--primary-color); color: var(--primary-color); font-weight: 700; }
        .info-box { background-color: var(--background-dark); border-left: 4px solid var(--primary-color); padding: 15px; margin: 10px 0; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
        .stMetric { padding: 10px 15px; background-color: var(--background-dark); border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 10px; }
        </style>
    """, unsafe_allow_html=True)

def aba_introducao():
    st.markdown("## 📚 Metodologia Quantitativa e Arquitetura do Sistema")
    st.markdown("""
    <div class="info-box">
    <h3>🎯 Visão Geral (Engine V8 Real-Time)</h3>
    <p>Este sistema foi atualizado para operar em <b>tempo real</b>. Em vez de processar dados offline, ele agora conecta-se diretamente
    às fontes de mercado e executa a modelagem preditiva instantaneamente.</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('### 1. Motor de Dados (Live)')
        with st.expander("Etapa 1.1: Coleta Híbrida"):
            st.markdown("- **Fundamentus (WebScraping):** Extrai indicadores como P/L, ROE e Margens diretamente do site.\n- **YFinance (API):** Coleta cotações e volume para cálculo de indicadores técnicos.")
        with st.expander("Etapa 1.2: Machine Learning On-The-Fly"):
            st.markdown("Treina um **Ensemble (Random Forest + XGBoost)** no momento da execução para prever a probabilidade de alta nos horizontes definidos (Curto, Médio ou Longo Prazo).")

    with c2:
        st.markdown('### 2. O Painel de Otimização')
        with st.expander("Etapa 2.1: Ranqueamento Multi-Fatorial"):
            st.markdown("Calcula Score Total combinando Performance, Fundamentos, Técnicos e a Probabilidade ML recém-calculada.")
        with st.expander("Etapa 2.2: Clusterização e Alocação"):
            st.markdown("Utiliza **PCA + KMeans** para agrupar ativos similares e seleciona os líderes de cada cluster. A alocação final segue a Fronteira Eficiente de Markowitz.")

def aba_selecao_ativos():
    st.markdown("## 🎯 Definição do Universo de Análise")
    mode = st.radio("**Modo de Seleção:**", ["📊 Índice de Referência (Todos do Ibovespa)", "✍️ Seleção Individual"])
    
    if "Índice" in mode:
        st.session_state.ativos_analise = ATIVOS_IBOVESPA
        st.success(f"✓ **{len(ATIVOS_IBOVESPA)} ativos** definidos.")
        with st.expander("📋 Visualizar Tickers"):
            st.write(", ".join([t.replace('.SA','') for t in ATIVOS_IBOVESPA]))
    else:
        sel = st.multiselect("Pesquise e selecione:", TODOS_ATIVOS, default=ATIVOS_IBOVESPA[:5])
        st.session_state.ativos_analise = sel

def aba_construtor_portfolio():
    if not st.session_state.get('ativos_analise'):
        st.warning("⚠️ Defina os ativos na aba 'Seleção de Ativos'.")
        return

    # FASE 1: QUESTIONÁRIO (ESTRUTURA IDÊNTICA AO V7)
    if not st.session_state.get('builder_complete'):
        st.markdown('## 📋 Calibração do Perfil de Risco')
        with st.form("form_perfil"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Tolerância ao Risco")
                q1 = st.radio("**1. Tolerância à Volatilidade:**", OPTIONS_CONCORDA, index=2)
                q2 = st.radio("**2. Foco em Retorno Máximo:**", OPTIONS_CONCORDA, index=2)
                q3 = st.radio("**3. Prioridade de Estabilidade:**", OPTIONS_DISCORDA, index=2)
                q4 = st.radio("**4. Aversão à Perda:**", OPTIONS_DISCORDA, index=2)
                q5 = st.radio("**5. Reação a Queda de 10%:**", OPTIONS_REACTION_DETALHADA, index=1)
                q6 = st.radio("**6. Nível de Conhecimento:**", OPTIONS_CONHECIMENTO_DETALHADA, index=1)
            with c2:
                st.markdown("#### Horizonte e Capital")
                q7 = st.radio("**7. Horizonte de Investimento:**", OPTIONS_TIME_HORIZON_DETALHADA, index=2)
                q8 = st.radio("**8. Necessidade de Liquidez:**", OPTIONS_LIQUIDEZ_DETALHADA, index=2)
                st.markdown("---")
                inv = st.number_input("Capital Total (R$)", value=10000.0, step=1000.0)
            
            sub = st.form_submit_button("🚀 Gerar Alocação Otimizada", type="primary")
            
        if sub:
            resp = {'risk_accept': MAP_CONCORDA[q1], 'max_gain': MAP_CONCORDA[q2], 
                    'stable_growth': MAP_DISCORDA[q3], 'avoid_loss': MAP_DISCORDA[q4], 
                    'reaction': MAP_REACTION[q5], 'level': MAP_CONHECIMENTO[q6], 
                    'time_purpose': [q7[0]], 'liquidity': [q8[0]]} # Adaptação formato lista
            
            analiser = AnalisadorPerfilInvestidor()
            niv, hor, dias, sco = analiser.calcular_perfil(resp)
            
            perf = {'risk_level': niv, 'time_horizon': hor, 'ml_lookback_days': dias, 'risk_score': sco}
            st.session_state.perfil = perf
            
            builder = ConstrutorPortfolio(inv)
            prog = st.empty()
            pbar = prog.progress(0, text="Iniciando Engine V8...")
            
            suc = builder.executar(st.session_state.ativos_analise, perf, pbar)
            prog.empty()
            
            if suc:
                st.session_state.builder = builder
                st.session_state.builder_complete = True
                st.rerun()
    
    # FASE 2: RESULTADOS (ESTRUTURA IDÊNTICA AO V7)
    else:
        b = st.session_state.builder
        p = st.session_state.perfil
        
        st.markdown('## ✅ Relatório de Alocação Otimizada')
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Perfil Identificado", p['risk_level'], f"Score: {p['risk_score']}")
        c2.metric("Horizonte", p['time_horizon'])
        c3.metric("Sharpe Estimado", f"{b.metricas['sharpe_ratio']:.2f}")
        c4.metric("Estratégia", b.metodo_alocacao)
        
        if st.button("🔄 Recalibrar Perfil"):
            st.session_state.builder_complete = False
            st.rerun()
            
        t1, t2, t3, t4, t5 = st.tabs(["📊 Alocação", "📈 Performance", "🤖 Fator ML", "📉 Volatilidade", "❓ Justificativas"])
        
        with t1:
            ca, cb = st.columns([1, 2])
            df_a = pd.DataFrame([{'Ativo': k.replace('.SA',''), 'Peso': v['weight']*100, 'Valor': v['amount']} for k,v in b.alocacao.items()])
            with ca:
                fig = px.pie(df_a, values='Peso', names='Ativo', hole=0.4, title="Distribuição")
                fig.update_layout(obter_template_grafico())
                st.plotly_chart(fig, use_container_width=True)
            with cb:
                st.dataframe(df_a.style.format({'Peso': '{:.2f}%', 'Valor': 'R$ {:.2f}'}), use_container_width=True)
                
        with t2:
            st.info("Performance projetada com base nos dados históricos recentes (Markowitz).")
            c1, c2 = st.columns(2)
            c1.metric("Retorno Anual Esperado", f"{b.metricas['annual_return']*100:.1f}%")
            c2.metric("Volatilidade Esperada", f"{b.metricas['annual_volatility']*100:.1f}%")

        with t3:
            st.markdown("#### Predições do Ensemble (RandomForest + XGBoost)")
            ml_data = b.scores[['ml_proba', 'ml_conf']].loc[b.selecionados]
            st.dataframe(ml_data.style.format("{:.1%}"), use_container_width=True)
            
        with t4:
            st.markdown("#### Volatilidade Anualizada (Histórica)")
            vols = b.scores['volatility_annual'].loc[b.selecionados] * 100
            st.bar_chart(vols)
            
        with t5:
            st.markdown("#### Racional da Seleção")
            for t, j in b.justificativas.items():
                st.markdown(f"""<div class="info-box"><h4>{t.replace('.SA','')}</h4><p>{j}</p></div>""", unsafe_allow_html=True)

def aba_analise_individual():
    st.markdown("## 🔍 Análise de Fatores por Ticker")
    if 'ativos_analise' in st.session_state:
        sel = st.selectbox("Selecione um ticker:", st.session_state.ativos_analise)
        if st.button("🔄 Executar Análise", type="primary"):
            with st.spinner("Processando Real-Time..."):
                col = ColetorDados()
                hist = col.coletar_historico([sel])
                if sel in hist:
                    df = hist[sel]
                    last = df.iloc[-1]
                    
                    t1, t2, t3 = st.tabs(["📊 Visão Geral", "💼 Fundamentos", "🔧 Técnicos"])
                    with t1:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Preço", f"R$ {last['Close']:.2f}")
                        c2.metric("RSI", f"{last['rsi_14']:.1f}")
                        c3.metric("MACD", f"{last['macd']:.3f}")
                        
                        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
                        fig.update_layout(obter_template_grafico())
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with t2:
                        try:
                            fund = col.buscar_fundamentos().loc[sel]
                            st.dataframe(fund.to_frame().T, use_container_width=True)
                        except: st.warning("Fundamentos indisponíveis.")
                        
                    with t3:
                        st.line_chart(df[['rsi_14', 'volatility_annual']])

def aba_referencias():
    st.markdown("## 📚 Referências e Bibliografia")
    st.markdown("---")
    # Conteúdo V7
    st.markdown("""
    <div class="info-box">
    <p><strong>Markowitz, H. (1952)</strong> - Portfolio Selection.</p>
    </div>
    <div class="info-box">
    <p><strong>López de Prado, M. (2018)</strong> - Advances in Financial Machine Learning.</p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# 8. MAIN
# =============================================================================

def main():
    if 'builder' not in st.session_state: st.session_state.builder = None
    configurar_pagina()
    st.markdown('<h1 class="main-header">Sistema de Portfólios Adaptativos (v8.7)</h1>', unsafe_allow_html=True)
    
    t1, t2, t3, t4, t5 = st.tabs(["📚 Metodologia", "🎯 Seleção de Ativos", "🏗️ Construtor de Portfólio", "🔍 Análise Individual", "📖 Referências"])
    
    with t1: aba_introducao()
    with t2: aba_selecao_ativos()
    with t3: aba_construtor_portfolio()
    with t4: aba_analise_individual()
    with t5: aba_referencias()

if __name__ == "__main__":
    main()
