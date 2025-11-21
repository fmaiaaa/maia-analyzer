# -*- coding: utf-8 -*-
"""
=============================================================================
SISTEMA DE PORTFÓLIOS ADAPTATIVOS - HÍBRIDO (v8.8.0)
=============================================================================

Fusão da interface do AutoML Elite com a lógica Backend do Portfolio Analyzer.
- Coleta de Preços: yfinance
- Coleta Fundamentalista: pynvest (Fundamentus)
- Engenharia de Features: Simplificada (RSI, MACD, Volatilidade, Momentum, SMAs)
- Machine Learning: Random Forest (Treino local em tempo real)
- Horizontes ML: 4 meses (84 dias), 8 meses (168 dias), 12 meses (252 dias)

=============================================================================
"""

# --- 1. CORE LIBRARIES & UTILITIES ---
import warnings
import numpy as np
import pandas as pd
import sys
import time
import traceback
from datetime import datetime, timedelta

# --- 2. SCIENTIFIC / STATISTICAL TOOLS ---
from scipy.optimize import minimize
from scipy.stats import zscore

# --- 3. STREAMLIT & VISUALIZATION ---
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- 4. FINANCIAL DATA & ML ---
import yfinance as yf
from pynvest.scrappers.fundamentus import Fundamentus
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# --- CONFIGURATION ---
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)

# =============================================================================
# 1. CONFIGURAÇÕES E CONSTANTES GLOBAIS
# =============================================================================

PERIODO_DADOS = '5y'
MIN_DIAS_HISTORICO = 252  # 1 ano útil
NUM_ATIVOS_PORTFOLIO = 5
TAXA_LIVRE_RISCO = 0.1075

# Pesos de Otimização (Baseado em Portfolio Analyzer)
PESO_MIN = 0.10
PESO_MAX = 0.30

# Mapeamento de Horizontes ML (Dias Úteis aproximados)
HORIZONTES_ML = {
    'Curto (4 Meses)': 84,
    'Médio (8 Meses)': 168,
    'Longo (12 Meses)': 252
}

# =============================================================================
# 2. LISTAS DE ATIVOS (IBOVESPA)
# =============================================================================

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

# Setores simplificados para fins de UI
ATIVOS_POR_SETOR = {
    'Geral': ATIVOS_IBOVESPA
}
TODOS_ATIVOS = sorted(list(set(ATIVOS_IBOVESPA)))

# =============================================================================
# 3. UTILS E ENGENHARIA DE FEATURES (Portado de Portfolio Analyzer)
# =============================================================================

class EngenheiroFeatures:
    """Implementa cálculo de indicadores técnicos idêntico ao Portfolio Analyzer."""

    @staticmethod
    def calcular_indicadores_tecnicos(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula RSI, MACD, Volatilidade, Momentum e SMAs.
        """
        if df.empty: return df
        
        df = df.copy()
        
        # Garante coluna Returns
        if 'Returns' not in df.columns:
            df['Returns'] = df['Close'].pct_change()

        # 1. RSI (14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))

        # 2. MACD (12, 26, 9)
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # 3. Volatilidade (20d anualizada)
        df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)

        # 4. Momentum (10d)
        df['Momentum'] = df['Close'] / df['Close'].shift(10) - 1

        # 5. Médias Móveis
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()

        return df

    @staticmethod
    def normalizar_zscore(serie: pd.Series, maior_melhor: bool = True) -> pd.Series:
        """Normalização via Z-Score (0-100) usada no Portfolio Analyzer."""
        serie_clean = serie.replace([np.inf, -np.inf], np.nan).dropna()
        if serie_clean.empty or serie_clean.std() == 0:
            return pd.Series(50, index=serie.index)
        
        z = zscore(serie_clean, nan_policy='omit')
        
        # Z-score mapeado para 0-100 (clamp em +/- 3 desvios)
        normalized = 50 + (np.clip(z, -3, 3) / 3) * 50
        normalized_series = pd.Series(normalized, index=serie_clean.index)
        
        if not maior_melhor:
            normalized_series = 100 - normalized_series
            
        return normalized_series.reindex(serie.index, fill_value=50)

# =============================================================================
# 4. COLETA DE DADOS (YFINANCE + PYNVEST)
# =============================================================================

class ColetorDadosHibrido:
    """Coleta preços via yfinance e fundamentos via pynvest."""
    
    def __init__(self):
        self.pynvest_scrapper = Fundamentus()
        self.dados_por_ativo = {}
        self.dados_fundamentalistas = pd.DataFrame()
        self.metricas_performance = pd.DataFrame()
        self.ativos_sucesso = []

    def _mapear_indicadores_pynvest(self, df_pynvest: pd.DataFrame) -> dict:
        """Mapeia as colunas do Pynvest para o padrão do Portfolio Analyzer."""
        # Se o pynvest retornar Vazio ou None
        if df_pynvest is None or df_pynvest.empty:
            return {}
        
        # Pega a primeira linha (último balanço/cotação)
        # O DataFrame do Pynvest geralmente tem 1 linha por ativo se consultado individualmente
        row = df_pynvest.iloc[0]
        
        # Extração segura com conversão
        def get_val(col):
            val = row.get(col, np.nan)
            return float(val) if val is not None else np.nan

        return {
            'PE_Ratio': get_val('vlr_ind_p_sobre_l'),
            'PB_Ratio': get_val('vlr_ind_p_sobre_vp'),
            'ROE': get_val('vlr_ind_roe') * 100, # Converter para %
            'Div_Yield': get_val('vlr_ind_div_yield') * 100, # Converter para %
            'EV_EBIT': get_val('vlr_ind_ev_sobre_ebit'),
            'Margem_Liq': get_val('vlr_ind_margem_liq') * 100,
            'Liquidez_Corr': get_val('vlr_liquidez_corr')
        }

    def coletar_dados_lote(self, simbolos: list, progress_bar=None):
        """Itera sobre os símbolos coletando histórico e fundamentos."""
        
        lista_fundamentos = []
        metricas_list = []
        total = len(simbolos)
        
        for i, simbolo in enumerate(simbolos):
            if progress_bar:
                progress_bar.progress((i / total) * 0.5, text=f"Coletando dados: {simbolo}...")
            
            try:
                # 1. Coleta Histórica (Yfinance)
                # Tenta .SA, se falhar tenta sem
                tickers_to_try = [simbolo]
                if not simbolo.endswith('.SA'): tickers_to_try.append(simbolo + '.SA')
                
                hist = pd.DataFrame()
                ticker_obj = None
                
                for t_try in tickers_to_try:
                    ticker_obj = yf.Ticker(t_try)
                    hist = ticker_obj.history(period=PERIODO_DADOS)
                    if not hist.empty:
                        break
                
                if hist.empty or len(hist) < MIN_DIAS_HISTORICO:
                    continue

                # Engenharia de Features Imediata
                df_tecnico = EngenheiroFeatures.calcular_indicadores_tecnicos(hist)
                self.dados_por_ativo[simbolo] = df_tecnico.dropna()

                # 2. Coleta Fundamentalista (Pynvest)
                # Pynvest usa ticker sem .SA normalmente, ou precisa tratar
                ticker_clean = simbolo.replace('.SA', '').lower()
                
                try:
                    # Tenta pegar indicadores via pynvest
                    # Nota: O método do pynvest pode demorar um pouco
                    df_fund_raw = self.pynvest_scrapper.coleta_indicadores_de_ativo(ticker_clean)
                    fund_data = self._mapear_indicadores_pynvest(df_fund_raw)
                except Exception:
                    # Fallback para dados vazios se falhar ou ticker não existir no fundamentus
                    fund_data = {
                        'PE_Ratio': np.nan, 'PB_Ratio': np.nan, 'ROE': np.nan, 
                        'Div_Yield': np.nan, 'EV_EBIT': np.nan, 'Margem_Liq': np.nan
                    }

                fund_data['Ticker'] = simbolo
                
                # Info adicional do yfinance para setor se pynvest falhar
                if 'sector' in ticker_obj.info:
                    fund_data['Sector'] = ticker_obj.info['sector']
                else:
                    # Tenta pegar do pynvest
                    fund_data['Sector'] = df_fund_raw['nome_setor'].iloc[0] if (df_fund_raw is not None and not df_fund_raw.empty) else 'Unknown'

                lista_fundamentos.append(fund_data)
                self.ativos_sucesso.append(simbolo)
                
                # 3. Métricas de Performance Simples
                returns = df_tecnico['Returns'].dropna()
                ann_ret = returns.mean() * 252
                ann_vol = returns.std() * np.sqrt(252)
                sharpe = (ann_ret - TAXA_LIVRE_RISCO) / ann_vol if ann_vol > 0 else 0
                
                metricas_list.append({
                    'Ticker': simbolo,
                    'sharpe_ratio': sharpe,
                    'annual_return': ann_ret,
                    'annual_volatility': ann_vol
                })

            except Exception as e:
                # print(f"Erro ao coletar {simbolo}: {e}")
                continue
        
        if not self.ativos_sucesso:
            return False

        self.dados_fundamentalistas = pd.DataFrame(lista_fundamentos).set_index('Ticker')
        self.metricas_performance = pd.DataFrame(metricas_list).set_index('Ticker')
        
        # Tratamento de NaNs nos fundamentos (Preencher com média ou 0)
        self.dados_fundamentalistas = self.dados_fundamentalistas.fillna(self.dados_fundamentalistas.mean(numeric_only=True)).fillna(0)
        
        return True

    def coletar_ativo_unico(self, simbolo: str):
        """Método auxiliar para análise individual."""
        self.coletar_dados_lote([simbolo])
        if simbolo in self.dados_por_ativo:
            return self.dados_por_ativo[simbolo], self.dados_fundamentalistas.loc[simbolo]
        return None, None

# =============================================================================
# 5. MACHINE LEARNING (PREDIÇÃO LOCAL)
# =============================================================================

class MachineLearningEngine:
    """Motor ML baseado no Portfolio Analyzer, expandido para múltiplos horizontes."""
    
    def __init__(self, dados_por_ativo, dados_fundamentalistas):
        self.dados_por_ativo = dados_por_ativo
        self.dados_fundamentalistas = dados_fundamentalistas
        self.predicoes = {} # {ticker: {horizonte: {proba, auc}}}

    def treinar_e_prever(self, horizons_map, progress_bar=None):
        """
        Treina Random Forest para cada ativo e cada horizonte.
        horizons_map: dict {'Nome': dias} ex: {'4 Meses': 84}
        """
        features_base = ['RSI', 'MACD', 'Volatility', 'Momentum', 'SMA_50', 'SMA_200']
        fund_cols = ['PE_Ratio', 'PB_Ratio', 'Div_Yield', 'ROE']
        
        total_steps = len(self.dados_por_ativo)
        
        for i, (ticker, df) in enumerate(self.dados_por_ativo.items()):
            if progress_bar:
                # Atualiza progresso (do 50% até 80%)
                progress = 0.5 + (i / total_steps) * 0.3
                progress_bar.progress(progress, text=f"Treinando ML: {ticker}...")

            self.predicoes[ticker] = {}
            
            # Adiciona fundamentos ao DF temporal para treino
            df_ml = df.copy()
            if ticker in self.dados_fundamentalistas.index:
                fund_row = self.dados_fundamentalistas.loc[ticker]
                for col in fund_cols:
                    if col in fund_row:
                        df_ml[col] = fund_row[col]
            else:
                for col in fund_cols: df_ml[col] = 0

            cols_treino = [c for c in features_base + fund_cols if c in df_ml.columns]

            for nome_hz, dias in horizons_map.items():
                try:
                    # Cria Target: Preço subiu após N dias?
                    df_ml['Target'] = np.where(df_ml['Close'].shift(-dias) > df_ml['Close'], 1, 0)
                    
                    # Remove NaNs criados pelo shift
                    df_train = df_ml.dropna(subset=cols_treino + ['Target']).copy()
                    
                    if len(df_train) < 100: # Mínimo de dados
                        self.predicoes[ticker][nome_hz] = {'proba': 0.5, 'auc': 0.5}
                        continue

                    X = df_train[cols_treino].iloc[:-dias] # Evita lookahead no treino
                    y = df_train['Target'].iloc[:-dias]
                    
                    # Checa se tem 2 classes
                    if len(np.unique(y)) < 2:
                        self.predicoes[ticker][nome_hz] = {'proba': 0.5, 'auc': 0.5}
                        continue

                    # Random Forest (Configuração leve do Portfolio Analyzer)
                    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
                    
                    # Validação Cruzada para AUC
                    tscv = TimeSeriesSplit(n_splits=3)
                    scores = cross_val_score(model, X, y, cv=tscv, scoring='roc_auc')
                    auc = scores.mean()

                    # Treino Final e Predição
                    model.fit(X, y)
                    
                    # Prevê usando os dados mais recentes disponíveis
                    last_row = df_ml[cols_treino].iloc[[-1]]
                    proba = model.predict_proba(last_row)[0][1]

                    self.predicoes[ticker][nome_hz] = {
                        'proba': proba,
                        'auc': auc if not np.isnan(auc) else 0.5
                    }

                except Exception:
                    self.predicoes[ticker][nome_hz] = {'proba': 0.5, 'auc': 0.5}

# =============================================================================
# 6. LÓGICA DE CONSTRUÇÃO DE PORTFÓLIO (BACKEND)
# =============================================================================

class ConstrutorPortfolioHibrido:
    
    def __init__(self, valor_investimento):
        self.valor_investimento = valor_investimento
        self.coletor = ColetorDadosHibrido()
        self.ml_engine = None
        self.scores = pd.DataFrame()
        self.ativos_selecionados = []
        self.alocacao = {}
        self.metricas_finais = {}
        self.justificativas = {}
        self.pesos_fatores = {}

    def executar_pipeline(self, simbolos, perfil, progress_bar):
        # 1. Coleta
        if not self.coletor.coletar_dados_lote(simbolos, progress_bar):
            return False
        
        # 2. Machine Learning
        self.ml_engine = MachineLearningEngine(self.coletor.dados_por_ativo, self.coletor.dados_fundamentalistas)
        self.ml_engine.treinar_e_prever(HORIZONTES_ML, progress_bar)
        
        # 3. Pontuação e Seleção (Lógica Portfolio Analyzer)
        progress_bar.progress(0.85, text="Calculando Scores e Ranqueamento...")
        self._calcular_scores(perfil)
        self._selecionar_ativos()
        
        # 4. Otimização (Markowitz Simples)
        progress_bar.progress(0.95, text="Otimizando Alocação (Markowitz)...")
        self._otimizar_alocacao(perfil['risk_level'])
        
        progress_bar.progress(1.0, text="Finalizado!")
        return True

    def _calcular_scores(self, perfil):
        """
        Implementa a lógica de scoring estrita do portfolio_analyzer.py
        (Weighted Sum de Z-Scores).
        """
        horizonte = perfil.get('time_horizon', 'MÉDIO PRAZO')
        
        # Pesos Adaptativos
        if horizonte == "CURTO PRAZO":
            W_PERF, W_FUND, W_TECH = 0.40, 0.10, 0.50
        elif horizonte == "LONGO PRAZO":
            W_PERF, W_FUND, W_TECH = 0.40, 0.50, 0.10
        else: # Médio
            W_PERF, W_FUND, W_TECH = 0.40, 0.30, 0.30
        
        self.pesos_fatores = {'Performance': W_PERF, 'Fundamentos': W_FUND, 'Técnicos': W_TECH, 'ML': 0.30}

        # Dataframe Combinado
        combined = self.coletor.metricas_performance.join(self.coletor.dados_fundamentalistas, how='inner')
        
        # Adiciona indicadores técnicos atuais
        for ticker in combined.index:
            if ticker in self.coletor.dados_por_ativo:
                df = self.coletor.dados_por_ativo[ticker]
                combined.loc[ticker, 'RSI_current'] = df['RSI'].iloc[-1]
                combined.loc[ticker, 'MACD_current'] = df['MACD'].iloc[-1]
        
        scores = pd.DataFrame(index=combined.index)
        
        # --- 1. PERFORMANCE (Sharpe) ---
        scores['perf_score'] = EngenheiroFeatures.normalizar_zscore(combined['sharpe_ratio'], True) * W_PERF
        
        # --- 2. FUNDAMENTOS (P/L, ROE) ---
        # Portfolio analyzer usa P/L (menor melhor) e ROE (maior melhor)
        fund_score = EngenheiroFeatures.normalizar_zscore(combined['PE_Ratio'], False) * 0.5
        fund_score += EngenheiroFeatures.normalizar_zscore(combined['ROE'], True) * 0.5
        scores['fund_score'] = fund_score * W_FUND
        
        # --- 3. TÉCNICOS (RSI, MACD) ---
        tech_score = 0
        if W_TECH > 0:
            # RSI próximo de 50 é neutro? Portfolio Analyzer: 100 - abs(rsi - 50) -> proximidade de 50?
            # Vamos manter a lógica do analyzer original:
            # "rsi_proximity_score = 100 - abs(combined['RSI_current'] - 50)"
            rsi_prox = 100 - abs(combined.get('RSI_current', 50) - 50)
            tech_score += EngenheiroFeatures.normalizar_zscore(rsi_prox, True) * 0.5
            tech_score += EngenheiroFeatures.normalizar_zscore(combined.get('MACD_current', 0), True) * 0.5
        scores['tech_score'] = tech_score * W_TECH
        
        # --- 4. MACHINE LEARNING (Ponderado por AUC) ---
        # Seleciona o horizonte relevante baseado no perfil
        map_hz = {'CURTO PRAZO': 'Curto (4 Meses)', 'MÉDIO PRAZO': 'Médio (8 Meses)', 'LONGO PRAZO': 'Longo (12 Meses)'}
        target_hz = map_hz.get(horizonte, 'Médio (8 Meses)')
        
        ml_probs = []
        ml_aucs = []
        for ticker in combined.index:
            preds = self.ml_engine.predicoes.get(ticker, {}).get(target_hz, {'proba': 0.5, 'auc': 0.5})
            ml_probs.append(preds['proba'])
            ml_aucs.append(preds['auc'])
            
        combined['ML_Proba'] = ml_probs
        combined['ML_Confidence'] = ml_aucs
        
        # Score ML Base
        ml_base = (
            EngenheiroFeatures.normalizar_zscore(pd.Series(ml_probs, index=combined.index), True) * 0.5 +
            EngenheiroFeatures.normalizar_zscore(pd.Series(ml_aucs, index=combined.index), True) * 0.5
        ) * 0.3 # Peso fixo do ML na composição final
        
        scores['ml_score'] = ml_base * combined['ML_Confidence'] # Pondera pela confiança real
        
        # SCORE TOTAL
        soma_pesos = W_PERF + W_FUND + W_TECH
        scores['base_score'] = scores[['perf_score', 'fund_score', 'tech_score']].sum(axis=1) / soma_pesos
        
        # Mix Final: 70% Base + Score ML
        scores['total_score'] = scores['base_score'] * 0.7 + scores['ml_score']
        
        self.scores = scores.join(combined).sort_values('total_score', ascending=False)

    def _selecionar_ativos(self):
        """Seleciona Top N ativos garantindo mínima diversificação se possível."""
        # Simplificação: Top N direto do Score Total
        self.ativos_selecionados = self.scores.head(NUM_ATIVOS_PORTFOLIO).index.tolist()
        
    def _otimizar_alocacao(self, nivel_risco):
        """Otimização Markowitz (MinVol ou MaxSharpe)."""
        if not self.ativos_selecionados: return
        
        # Extrai retornos dos selecionados
        retornos_df = pd.DataFrame({
            t: self.coletor.dados_por_ativo[t]['Returns'] 
            for t in self.ativos_selecionados
        }).dropna()
        
        if retornos_df.empty:
            self.alocacao = {t: {'weight': 1/len(self.ativos_selecionados), 'amount': 0} for t in self.ativos_selecionados}
            return

        mu = retornos_df.mean() * 252
        sigma = retornos_df.cov() * 252
        num_assets = len(mu)
        
        # Constraints e Bounds
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((PESO_MIN, PESO_MAX) for _ in range(num_assets))
        init_guess = np.array([1/num_assets] * num_assets)
        
        def portfolio_vol(w):
            return np.sqrt(np.dot(w.T, np.dot(sigma, w)))
            
        def neg_sharpe(w):
            ret = np.dot(w, mu)
            vol = np.sqrt(np.dot(w.T, np.dot(sigma, w)))
            return -(ret - TAXA_LIVRE_RISCO) / vol if vol > 0 else 0
            
        # Define Objetivo
        if 'CONSERVADOR' in nivel_risco or 'INTERMEDIÁRIO' in nivel_risco:
            fun = portfolio_vol
            metodo = "Miníma Volatilidade"
        else:
            fun = neg_sharpe
            metodo = "Máximo Sharpe"
            
        try:
            res = minimize(fun, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
            pesos = res.x if res.success else init_guess
        except:
            pesos = init_guess
            metodo += " (Fallback)"
            
        # Formatar Alocação
        self.alocacao = {}
        for i, ticker in enumerate(self.ativos_selecionados):
            self.alocacao[ticker] = {
                'weight': pesos[i],
                'amount': pesos[i] * self.valor_investimento
            }
            
        # Calcula Métricas Finais do Portfólio
        ret_port = np.dot(pesos, mu)
        vol_port = np.sqrt(np.dot(pesos.T, np.dot(sigma, pesos)))
        sharpe_port = (ret_port - TAXA_LIVRE_RISCO) / vol_port if vol_port > 0 else 0
        
        self.metricas_finais = {
            'annual_return': ret_port,
            'annual_volatility': vol_port,
            'sharpe_ratio': sharpe_port,
            'metodo': metodo
        }
        
        # Gera Justificativas
        for ticker in self.ativos_selecionados:
            row = self.scores.loc[ticker]
            txt = (f"Score: {row['total_score']:.2f} | Sharpe: {row['sharpe_ratio']:.2f} | "
                   f"P/L: {row['PE_Ratio']:.2f} | ML Prob: {row['ML_Proba']:.1%}")
            self.justificativas[ticker] = txt

# =============================================================================
# 7. ANALISADOR DE PERFIL
# =============================================================================

class AnalisadorPerfilInvestidor:
    """Mesma lógica do original, apenas mantendo a estrutura."""
    def calcular_perfil(self, respostas):
        # Lógica simplificada para manter compatibilidade
        # Score Map
        score_map = {'CT: Concordo Totalmente': 5, 'C: Concordo': 4, 'N: Neutro': 3, 'D: Discordo': 2, 'DT: Discordo Totalmente': 1}
        # Pontuação
        pts = sum([score_map.get(respostas.get(k, 'N: Neutro'), 3) for k in ['risk_accept', 'max_gain']])
        
        if pts <= 6: risk = "CONSERVADOR"
        elif pts <= 8: risk = "MODERADO"
        else: risk = "ARROJADO"
        
        # Horizonte
        hz_map = {'A': 'CURTO PRAZO', 'B': 'MÉDIO PRAZO', 'C': 'LONGO PRAZO'}
        hz = hz_map.get(respostas.get('time_purpose', 'B')[0], 'MÉDIO PRAZO') # Pega a primeira letra da resposta
        
        return risk, hz, 0, pts * 10

# =============================================================================
# 8. INTERFACE STREAMLIT (ESTILO AUTOML ELITE)
# =============================================================================

def configurar_pagina():
    st.set_page_config(page_title="AutoML Portfolio Elite v8", page_icon="📈", layout="wide")
    st.markdown("""
        <style>
        .main-header { font-family: 'Arial'; color: #000000; text-align: center; font-size: 2.2rem; font-weight: 600; margin-bottom: 20px; }
        .stButton button { border: 1px solid #000; color: #000; background-color: transparent; }
        .stButton button:hover { background-color: #000; color: #fff; }
        .stTabs [data-baseweb="tab-list"] { justify-content: center; border-bottom: 2px solid #dee2e6; }
        .stTabs [data-baseweb="tab"] { flex-grow: 0; }
        .stMetric { background-color: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 4px solid #000; }
        </style>
    """, unsafe_allow_html=True)

def aba_selecao_ativos():
    st.markdown("## 🎯 Seleção do Universo de Investimento")
    st.info("O sistema utiliza dados do Yahoo Finance (Preços) e Fundamentus (Indicadores) em tempo real.")
    
    modo = st.radio("Modo de Seleção:", ["Índice Ibovespa (Completo)", "Seleção Manual"], horizontal=True)
    
    if modo == "Índice Ibovespa (Completo)":
        ativos = ATIVOS_IBOVESPA
        st.success(f"{len(ativos)} ativos selecionados.")
    else:
        ativos = st.multiselect("Selecione os ativos:", TODOS_ATIVOS, default=TODOS_ATIVOS[:5])
    
    st.session_state.ativos_selecionados_user = ativos

def aba_construtor_portfolio():
    if 'ativos_selecionados_user' not in st.session_state:
        st.warning("Selecione os ativos na aba anterior.")
        return

    st.markdown("## 🏗️ Construtor de Portfólio")
    
    with st.form("form_perfil"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Perfil de Risco")
            r1 = st.selectbox("1. Aceito volatilidade?", ["CT: Concordo Totalmente", "C: Concordo", "N: Neutro", "D: Discordo"], index=1, key='risk_accept')
            r2 = st.selectbox("2. Busco retorno máximo?", ["CT: Concordo Totalmente", "C: Concordo", "N: Neutro", "D: Discordo"], index=1, key='max_gain')
        with col2:
            st.markdown("### Objetivos")
            time_hz = st.selectbox("Horizonte de Tempo:", 
                                 ["A: Curto (< 1 ano)", "B: Médio (1-5 anos)", "C: Longo (> 5 anos)"], index=1, key='time_purpose')
            capital = st.number_input("Capital (R$):", value=10000.0, step=1000.0)
            
        submit = st.form_submit_button("🚀 Iniciar Otimização AI")
    
    if submit:
        analisador = AnalisadorPerfilInvestidor()
        risk, hz, _, _ = analisador.calcular_perfil({'risk_accept': r1, 'max_gain': r2, 'time_purpose': time_hz})
        
        perfil = {'risk_level': risk, 'time_horizon': hz}
        
        st.info(f"Perfil Detectado: **{risk}** | Horizonte: **{hz}**")
        
        construtor = ConstrutorPortfolioHibrido(capital)
        st.session_state.construtor = construtor
        
        prog_bar = st.progress(0, text="Iniciando coleta de dados (yfinance + pynvest)...")
        
        try:
            sucesso = construtor.executar_pipeline(st.session_state.ativos_selecionados_user, perfil, prog_bar)
            if sucesso:
                st.session_state.pipeline_concluido = True
                st.rerun()
            else:
                st.error("Falha na coleta de dados. Verifique sua conexão.")
        except Exception as e:
            st.error(f"Erro Crítico: {e}")
            st.code(traceback.format_exc())

    # Exibição de Resultados
    if st.session_state.get('pipeline_concluido'):
        c = st.session_state.construtor
        st.divider()
        st.markdown("### ✅ Alocação Otimizada")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Retorno Esp. (a.a.)", f"{c.metricas_finais['annual_return']:.1%}")
        col2.metric("Volatilidade (a.a.)", f"{c.metricas_finais['annual_volatility']:.1%}")
        col3.metric("Sharpe Ratio", f"{c.metricas_finais['sharpe_ratio']:.2f}")
        
        cols = st.columns(2)
        with cols[0]:
            # Gráfico Pizza
            df_alloc = pd.DataFrame.from_dict(c.alocacao, orient='index')
            fig = px.pie(df_alloc, values='amount', names=df_alloc.index, title="Distribuição de Capital")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
        with cols[1]:
            # Tabela
            df_display = pd.DataFrame([
                {
                    'Ativo': k, 
                    'Peso (%)': f"{v['weight']:.1%}", 
                    'Valor (R$)': f"R$ {v['amount']:,.2f}",
                    'Justificativa': c.justificativas[k]
                }
                for k, v in c.alocacao.items()
            ])
            st.dataframe(df_display, hide_index=True, use_container_width=True)

def aba_analise_individual():
    st.markdown("## 🔍 Análise Individual (Machine Learning)")
    
    ativo = st.selectbox("Selecione o Ativo:", ATIVOS_IBOVESPA)
    
    if st.button("Analisar Ativo"):
        with st.spinner(f"Processando {ativo} via yfinance/pynvest/ML..."):
            coletor = ColetorDadosHibrido()
            df_tec, s_fund = coletor.coletar_ativo_unico(ativo)
            
            if df_tec is None:
                st.error("Dados não encontrados.")
                return
            
            # Treino ML Local para os 3 horizontes
            ml = MachineLearningEngine({ativo: df_tec}, pd.DataFrame([s_fund], index=[ativo]))
            ml.treinar_e_prever(HORIZONTES_ML)
            preds = ml.predicoes.get(ativo, {})
            
            # --- DASHBOARD ---
            st.markdown(f"### 📊 {ativo} - Dashboard Integrado")
            
            # 1. Indicadores
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Preço Atual", f"R$ {df_tec['Close'].iloc[-1]:.2f}")
            c2.metric("RSI (14)", f"{df_tec['RSI'].iloc[-1]:.1f}")
            c3.metric("P/L", f"{s_fund.get('PE_Ratio', 0):.2f}")
            c4.metric("ROE", f"{s_fund.get('ROE', 0):.1f}%")
            
            # 2. ML Predictions
            st.markdown("#### 🤖 Predições de IA (Random Forest)")
            c_ml1, c_ml2, c_ml3 = st.columns(3)
            
            def show_ml_card(col, titulo, key):
                d = preds.get(key, {'proba': 0.5, 'auc': 0})
                prob = d['proba']
                color = "green" if prob > 0.6 else "red" if prob < 0.4 else "gray"
                col.markdown(f"**{titulo}**")
                col.markdown(f"<h2 style='color:{color}'>{prob:.1%}</h2>", unsafe_allow_html=True)
                col.caption(f"Probabilidade de Alta (AUC: {d['auc']:.2f})")
            
            show_ml_card(c_ml1, "Curto (4 Meses)", 'Curto (4 Meses)')
            show_ml_card(c_ml2, "Médio (8 Meses)", 'Médio (8 Meses)')
            show_ml_card(c_ml3, "Longo (12 Meses)", 'Longo (12 Meses)')
            
            # 3. Gráfico
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df_tec.index, open=df_tec['Open'], high=df_tec['High'], low=df_tec['Low'], close=df_tec['Close'], name='Preço'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_tec.index, y=df_tec['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
            fig.add_hline(y=70, line_dash='dot', row=2, col=1)
            fig.add_hline(y=30, line_dash='dot', row=2, col=1)
            fig.update_layout(height=600, title_text="Análise Técnica", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

def main():
    if 'pipeline_concluido' not in st.session_state:
        st.session_state.pipeline_concluido = False
        
    configurar_pagina()
    st.markdown('<h1 class="main-header">Sistema de Portfólios Adaptativos v8.8 (Híbrido)</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📚 Introdução", "🎯 Seleção", "🏗️ Construtor", "🔍 Análise ML"])
    
    with tab1:
        st.markdown("### Metodologia Híbrida")
        st.write("Este sistema combina a interface do AutoML Elite com o motor lógico simplificado do Portfolio Analyzer.")
        st.info("- **Preços:** yfinance\n- **Fundamentos:** pynvest (Fundamentus)\n- **ML:** Random Forest (Treino Local em Tempo Real)")
        
    with tab2: aba_selecao_ativos()
    with tab3: aba_construtor_portfolio()
    with tab4: aba_analise_individual()

if __name__ == "__main__":
    main()
