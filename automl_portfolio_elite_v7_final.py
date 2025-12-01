# -*- coding: utf-8 -*-
"""
=============================================================================
SISTEMA DE OTIMIZAÇÃO QUANTITATIVA
=============================================================================

Modelo de Alocação de Ativos com Métodos Adaptativos.
- Preços: Estratégia Linear com Fail-Fast (YFinance -> TvDatafeed -> Estático Global). 
- Fundamentos: Coleta Exaustiva Pynvest (50+ indicadores).
- Lógica de Construção (V9.4): Pesos Dinâmicos + Seleção por Clusterização.
- Modelagem (V9.35): Seleção Dinâmica de Modelos ML (Simples/Complexo) e Tratamento Robusto de Fallback.

Versão: 9.32.37 (Final Build: Professional UI, Dynamic ML/GARCH, Robust Fallback, Custom Allocation)
=============================================================================
"""

# --- 1. CORE LIBRARIES & UTILITIES ---
import warnings
# Movemos o filtro para o topo para suprimir SyntaxWarnings de bibliotecas importadas
warnings.filterwarnings('ignore') 

import numpy as np
import pandas as pd
import subprocess
import sys
import os
import time
from datetime import datetime, timedelta
import json
import traceback
import math

# --- 2. SCIENTIFIC / STATISTICAL TOOLS ---
from scipy.optimize import minimize
from scipy.stats import zscore, norm
# NOVAS IMPORTAÇÕES ML
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

# Tenta importar XGBoost e trata a ausência
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Aviso: XGBoost não instalado. O modelo complexo usará apenas RF.")


# --- 3. STREAMLIT, DATA ACQUISITION, & PLOTTING ---
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests

# --- IMPORTAÇÕES PARA COLETA LIVE (HÍBRIDO) ---
# 1. TvDatafeed (Primário)
try:
    from tvDatafeed import TvDatafeed, Interval
    TV_DATAFEED_AVAILABLE = True
except ImportError:
    TV_DATAFEED_AVAILABLE = False
    print("Aviso: tvDatafeed não instalado. Usando fallback para yfinance.")

# 2. Yfinance (Secundário)
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Aviso: yfinance não instalado. Usando fallback estático.")

# 3. Pynvest (Fundamentos)
try:
    from pynvest.scrappers.fundamentus import Fundamentus
    FUNDAMENTUS_AVAILABLE = True
except ImportError:
    FUNDAMENTUS_AVAILABLE = False
    print("Aviso: pynvest não instalado. Fundamentos serão desabilitados.")


# --- CONFIGURAÇÕES GLOBAIS ---
# Mapeamento para nomes de classes CSS do Streamlit
PRIMARY_BUTTON_STYLE = "primary"
SECONDARY_BUTTON_STYLE = "secondary"

# Constantes para a simulação Markowitz/Otimização
RISK_FREE_RATE = 0.05 # Taxa livre de risco anual (Exemplo: 5%)
N_SIMULATIONS = 10000 # Número de simulações de Monte Carlo
DEFAULT_TIMEFRAME = '3y' # Padrão de 3 anos de dados
DEFAULT_INTERVAL = '1wk' # Padrão de intervalo semanal

# --- UTILITIES ---

# Função de log de debug
def log_debug(message):
    if 'debug_logs' not in st.session_state:
        st.session_state.debug_logs = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.debug_logs.append(f"[{timestamp}] {message}")
    # print(f"DEBUG: {message}") # Descomente para logar no console do Streamlit

# --- CLASSE CORE: BUILDER (Otimização e Análise) ---

class PortfolioBuilder:
    def __init__(self, ativos_analise, horizonte=DEFAULT_TIMEFRAME, intervalo=DEFAULT_INTERVAL):
        log_debug(f"Inicializando PortfolioBuilder com ativos: {ativos_analise}")
        self.ativos = ativos_analise
        self.horizonte = horizonte
        self.intervalo = intervalo
        self.dados_historicos = pd.DataFrame()
        self.dados_fundamentais = {}
        self.dados_tecnicos = {}
        self.predicoes_ml = {}
        self.markowitz_results = None
        self.modelos_treinados = {}
        self.precos_atuais = {} # Dicionário para armazenar o preço atual de cada ativo

    # ... [Omitindo métodos de coleta de dados e tratamento de erros por brevidade] ...

    # =========================================================================
    # ML/GARCH: Modelos Preditivos e de Risco (NOVA IMPLEMENTAÇÃO)
    # =========================================================================
    
    def _criar_features_ml(self, df_ativo, lags=[1, 2, 3, 5, 10]):
        """Cria features de lag e retorno para os modelos ML."""
        df = df_ativo[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # 1. Target (vai subir ou descer no próximo período)
        df['Retorno'] = df['Close'].pct_change().shift(-1)
        df['Target'] = (df['Retorno'] > 0).astype(int)
        
        # 2. Lags
        for col in ['Close', 'Open', 'High', 'Low', 'Volume']:
            for lag in lags:
                df[f'{col}_Lag_{lag}'] = df[col].shift(lag)
                
        # 3. Retornos Logarítmicos
        df['Log_Ret_1d'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Log_Ret_5d'] = np.log(df['Close'] / df['Close'].shift(5))
        
        # Remove NaNs criados pelos lags e pelo target shift
        df.dropna(inplace=True)
        
        return df

    def _treinar_modelo_simples(self, ativo, dados_historicos):
        """Modelo Simples: Elastic Net (Seleção) + Logistic Regression (Modelagem)."""
        log_debug(f"Iniciando treinamento ML Simples para {ativo}")
        
        # Features: apenas Close, Open, High, Low, Volume (lags)
        df = self._criar_features_ml(dados_historicos, lags=[1, 2, 3, 5, 10])
        
        if len(df) < 50:
            log_debug(f"Dados insuficientes para ML Simples em {ativo}. Len: {len(df)}")
            return False

        # Prepara features e target
        X = df.drop(columns=['Retorno', 'Target', 'Volume']) # Remove Target, Retorno e Volume (Volume puro não é feature de lag)
        y = df['Target']
        
        if X.empty or y.empty: return False

        # Split Temporal (80/20)
        split_point = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

        # Escalonamento
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 1. Elastic Net para Seleção de Features (Regressão)
        # Usamos como proxy para encontrar features importantes para a classificação
        enet = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=1000)
        enet.fit(X_train_scaled, y_train)
        
        # Seleciona features com coeficiente absoluto > um limiar
        coeficientes = pd.Series(enet.coef_, index=X.columns)
        features_selecionadas = coeficientes[abs(coeficientes) > 1e-4].index.tolist()
        
        if not features_selecionadas:
            log_debug(f"Elastic Net não selecionou features relevantes para {ativo}.")
            return False
            
        # 2. Logistic Regression (Apenas com as features selecionadas)
        X_train_sel = X_train[features_selecionadas]
        X_test_sel = X_test[features_selecionadas]
        
        scaler_final = StandardScaler() # Re-escalona com as features selecionadas
        X_train_final = scaler_final.fit_transform(X_train_sel)
        X_test_final = scaler_final.transform(X_test_sel)

        model = LogisticRegression(C=1.0, solver='liblinear', max_iter=1000, random_state=42)
        model.fit(X_train_final, y_train)

        # 3. Avaliação e Predição
        y_pred_proba = model.predict_proba(X_test_final)[:, 1]
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        
        # Predição Próximo Período
        X_futuro_raw = X.iloc[[-1]][features_selecionadas] # Usa a última linha do DF original para prever o futuro
        X_futuro = scaler_final.transform(X_futuro_raw)
        proba_futura = model.predict_proba(X_futuro)[0, 1]

        self.predicoes_ml[ativo] = {
            'modelo': 'Simples (Enet + LogReg)',
            'auc_roc_score': auc_roc,
            'proba_futura_alta': proba_futura,
            'features_selecionadas': features_selecionadas,
            'features_importantes_logreg': {f: model.coef_[0][i] for i, f in enumerate(features_selecionadas)},
            'status': 'Treinado com Sucesso'
        }
        self.modelos_treinados[f'{ativo}_simples'] = model
        log_debug(f"ML Simples para {ativo} treinado com AUC: {auc_roc:.4f}")
        return True
        
    def _treinar_modelo_complexo(self, ativo, dados_historicos):
        """Modelo Complexo: Elastic Net (Seleção) + Ensemble RF/XGBoost (Modelagem)."""
        log_debug(f"Iniciando treinamento ML Complexo para {ativo}")
        
        # Features: lags mais extensos (simulando a complexidade)
        df = self._criar_features_ml(dados_historicos, lags=[1, 2, 3, 5, 10, 20, 40])
        
        if len(df) < 50: return False

        X = df.drop(columns=['Retorno', 'Target', 'Volume'])
        y = df['Target']
        
        if X.empty or y.empty: return False

        split_point = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

        # Escalonamento (Necessário para Elastic Net)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 1. Elastic Net para Seleção de Features
        enet = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=1000)
        enet.fit(X_train_scaled, y_train)
        coeficientes = pd.Series(enet.coef_, index=X.columns)
        features_selecionadas = coeficientes[abs(coeficientes) > 1e-4].index.tolist()
        
        if not features_selecionadas:
            log_debug(f"Elastic Net não selecionou features relevantes para o ML Complexo em {ativo}.")
            return False
            
        X_train_final = X_train[features_selecionadas]
        X_test_final = X_test[features_selecionadas]
        
        # Re-escalona (opcional, mas bom para consistência)
        scaler_final = StandardScaler()
        X_train_final_scaled = scaler_final.fit_transform(X_train_final)
        X_test_final_scaled = scaler_final.transform(X_test_final)

        # 2. Treinamento Ensemble (RF + XGBoost - Tuning Mock)
        
        # Modelo RF (hiperparâmetros mockados/iniciais)
        rf_model = RandomForestClassifier(n_estimators=300, max_depth=7, random_state=42)
        rf_model.fit(X_train_final_scaled, y_train)
        rf_proba = rf_model.predict_proba(X_test_final_scaled)[:, 1]

        # Modelo XGBoost (se disponível, tuning mockado/inicial)
        if XGB_AVAILABLE:
            xgb_model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss', random_state=42)
            xgb_model.fit(X_train_final_scaled, y_train)
            xgb_proba = xgb_model.predict_proba(X_test_final_scaled)[:, 1]
            
            # Ensemble: Média das probabilidades
            ensemble_proba_test = (rf_proba + xgb_proba) / 2
            
            # Predição Próximo Período
            X_futuro_raw = X.iloc[[-1]][features_selecionadas]
            X_futuro_scaled = scaler_final.transform(X_futuro_raw)
            rf_proba_futura = rf_model.predict_proba(X_futuro_scaled)[0, 1]
            xgb_proba_futura = xgb_model.predict_proba(X_futuro_scaled)[0, 1]
            proba_futura = (rf_proba_futura + xgb_proba_futura) / 2
            
        else: # Fallback: Apenas RF
            ensemble_proba_test = rf_proba
            X_futuro_raw = X.iloc[[-1]][features_selecionadas]
            X_futuro_scaled = scaler_final.transform(X_futuro_raw)
            proba_futura = rf_model.predict_proba(X_futuro_scaled)[0, 1]

        # 3. Avaliação
        auc_roc = roc_auc_score(y_test, ensemble_proba_test)

        self.predicoes_ml[ativo] = {
            'modelo': 'Complexo (Enet + Ensemble)',
            'auc_roc_score': auc_roc,
            'proba_futura_alta': proba_futura,
            'features_selecionadas': features_selecionadas,
            'status': 'Treinado com Sucesso',
            'base_models': ['RF', 'XGB' if XGB_AVAILABLE else 'RF_Solo']
        }
        self.modelos_treinados[f'{ativo}_complexo'] = (rf_model, xgb_model if XGB_AVAILABLE else None)
        log_debug(f"ML Complexo para {ativo} treinado com AUC: {auc_roc:.4f}")
        return True

    def treinar_ml_e_garch(self, modelos_selecionados=['Simples', 'Complexo', 'GARCH']):
        """Treina modelos ML (Simples/Complexo) e de Risco (GARCH) para todos os ativos."""
        self.predicoes_ml = {}
        for ativo in self.ativos:
            if ativo in self.dados_historicos:
                if 'Simples' in modelos_selecionados:
                    self._treinar_modelo_simples(ativo, self.dados_historicos[ativo])
                if 'Complexo' in modelos_selecionados:
                    self._treinar_modelo_complexo(ativo, self.dados_historicos[ativo])
                if 'GARCH' in modelos_selecionados:
                    self._treinar_modelo_garch(ativo, self.dados_historicos[ativo]) # GARCH permanece o mesmo
        log_debug("Treinamento de ML e GARCH concluído para todos os ativos.")


    # Omiti _treinar_modelo_garch e _objetivo_funcao/otimizar_markowitz, pois não foram alterados
    # ...


# --- FUNÇÕES STREAMLIT UI ---

def configurar_pagina():
    """Configura o título da página e injeta CSS customizado."""
    st.set_page_config(
        page_title="Sistema de Otimização Quantitativa",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # INJEÇÃO DE CSS CUSTOMIZADO
    st.markdown("""
    <style>
    /* Estilo para a fonte principal */
    .main-header {
        color: #0E768E;
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* Estilo para os cards de métricas */
    [data-testid="stMetric"] {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Estilo para o botão de seleção persistente e layout full-width */
    /* 1. Ocupa a largura total da coluna e centraliza */
    div.st-emotion-cache-1r7r3f5 { /* Seletor comum de coluna Streamlit */
        display: flex;
        justify-content: center; 
        align-items: center;
    }

    /* 2. Estilo base do botão */
    .stButton>button {
        border-radius: 0.5rem;
        padding: 0.6rem 1rem; /* Aumenta um pouco o padding vertical */
        margin: 0.25rem;
        transition: all 0.2s ease-in-out;
        width: 100%; /* Ocupa a largura total do seu container/coluna */
        border: 1px solid #0E768E;
        color: #0E768E;
        background-color: white;
    }
    
    /* 3. Estilo para o botão Padrão (secundário) no hover */
    .stButton>button:hover:not([data-testid="stClickableButton-primary"]) {
        background-color: #f0f2f6 !important; 
        color: #0E768E !important;
        border-color: #0E768E !important;
    }

    /* 4. Estilo para o botão PRIMÁRIO (que será o "selecionado" via lógica Python) */
    .stButton button[data-testid="stClickableButton-primary"] {
        background-color: #0E768E; /* Azul escuro principal */
        color: white;
        border-color: #0E768E;
        box-shadow: 0 0 10px rgba(14, 118, 142, 0.5); /* Sombra para destacar */
    }

    /* 5. Mantém o estilo do primário no hover */
    .stButton button[data-testid="stClickableButton-primary"]:hover {
        background-color: #0A5F73; /* Um pouco mais escuro no hover do selecionado */
        color: white;
        border-color: #0A5F73;
    }
    
    /* NOVO: CSS para centralizar texto em colunas (usado para Setor/Indústria) */
    .centered-text p {
        text-align: center !important;
    }
    
    /* NOVO: Ajuste para o radio button (Ponto 1) */
    [data-testid="stRadio"] label {
        margin-right: 15px;
    }
    
    </style>
    """, unsafe_allow_html=True)


# Função para mostrar o resumo de mercado (o card superior)
def mostrar_resumo_ativo(builder, ativo, dados_historicos, predicoes_ml):
    """Exibe o resumo de mercado, incluindo a seção ML, com novo layout de 5 colunas (Ponto 4)."""
    if ativo not in dados_historicos or dados_historicos[ativo].empty:
        st.error(f"Não há dados históricos para {ativo}.")
        return

    df = dados_historicos[ativo]
    dados_ml = predicoes_ml.get(ativo, {})

    # Valores de Mercado
    preco_atual = builder.precos_atuais.get(ativo, df['Close'].iloc[-1] if not df.empty else 0.0)
    preco_anterior = df['Close'].iloc[-2] if len(df) > 1 else preco_atual
    variacao_percentual = (preco_atual - preco_anterior) / preco_anterior * 100 if preco_anterior else 0.0
    volume_medio = df['Volume'].mean() if not df.empty else 0
    volatilidade_anualizada = df['Close'].pct_change().std() * np.sqrt(252) * 100 if len(df) > 2 else 0.0

    # Busca por dados fundamentais
    dados_fund = builder.dados_fundamentais.get(ativo, {})
    setor = dados_fund.get('Setor', 'Não Informado')
    industria = dados_fund.get('Subsetor', 'Não Informado')

    st.markdown(f"**{ativo} - Resumo de Mercado**")
    
    # NOVO: Layout de 5 colunas para Preço, Volume, Volatilidade, Setor, Indústria (Ponto 4)
    col_p, col_v, col_vol, col_setor, col_ind = st.columns(5)
    
    col_p.metric(
        "Preço",
        f"R$ {preco_atual:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
        f"{variacao_percentual:+.2f}%"
    )
    col_v.metric("Volume Médio", f"{int(volume_medio):,}".replace(",", "."))
    col_vol.metric("Vol. Anualizada (Hist)", f"{volatilidade_anualizada:,.2f}%".replace(",", "X").replace(".", ",").replace("X", "."))

    # Convertendo Setor e Indústria para Custom Metrics para manter o alinhamento
    def display_custom_metric(col, label, value):
        col.markdown(f"""
        <div class="centered-text" data-testid="stMetric" style="padding: 15px; border-radius: 10px; background-color: #f0f2f6; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
            <p style="font-size: 11px; color: #555; margin: 0; padding: 0;">{label}</p>
            <p style="font-size: 16px; font-weight: bold; color: #0E768E; margin: 0; padding: 0;">{value}</p>
        </div>
        """, unsafe_allow_html=True)
        
    display_custom_metric(col_setor, "Setor", setor)
    display_custom_metric(col_ind, "Indústria", industria)

    # --- CORREÇÃO E MELHORIA DA SEÇÃO ML ---
    # Verifica se pelo menos um dos modelos foi treinado com sucesso para este ativo
    is_ml_trained = len(dados_ml) > 0 and dados_ml.get('status') == 'Treinado com Sucesso'
    
    # Exibição da Mensagem ML (Corrigida)
    if is_ml_trained:
        modelo_nome = dados_ml.get('modelo', 'ML Desconhecido')
        if 'auc_roc_score' in dados_ml and dados_ml['auc_roc_score'] > 0.5:
             # Modelo de classificação como LogReg ou Ensemble
             score = dados_ml['auc_roc_score']
             status_msg = f"✅ Modelo ML Treinado: **{modelo_nome}** (AUC-ROC: {score:.4f})."
        elif 'pred_vol_condicional' in dados_ml:
            # Modelo de risco como GARCH
            vol_cond = dados_ml['pred_vol_condicional']
            status_msg = f"✅ Modelo ML/Risco Treinado: **{modelo_nome}** (Vol. Condicional: {vol_cond:.2f}%)."
        else:
            status_msg = "⚠️ Modelo ML Treinado, mas sem métricas conhecidas."

        st.info(status_msg)
    else:
        st.warning("ℹ️ Modelo ML Não Treinado: A pipeline de Machine Learning falhou ou está desativada para este ativo. A classificação se baseia puramente nos fatores Fundamentais e Técnicos (Modo Geral com ML Desativado).")

# Função que gera a aba de análise individual
def aba_analise_individual():
    """Permite a análise individual de um ativo."""
    st.title("🔍 Análise Individual de Ativo")
    
    builder = st.session_state.builder
    if not builder:
        st.warning("Por favor, selecione os ativos e execute a coleta de dados na aba 'Seleção de Ativos' primeiro.")
        return

    ativos_disponiveis = builder.ativos
    if not ativos_disponiveis:
        st.warning("Nenhum ativo disponível para análise.")
        return

    # Seletor de Ativo
    ativo_selecionado = st.selectbox(
        "Selecione o Ativo para Análise Detalhada",
        options=ativos_disponiveis,
        key='ativo_analise_individual'
    )

    if ativo_selecionado:
        # 1. Resumo de Mercado (inclui o status ML)
        st.markdown("---")
        mostrar_resumo_ativo(
            builder, 
            ativo_selecionado, 
            builder.dados_historicos, 
            builder.predicoes_ml
        )
        st.markdown("---")

        # 2. Dados Históricos (Gráfico de Preços)
        st.subheader(f"Gráfico Histórico de Preços de {ativo_selecionado}")
        df_hist = builder.dados_historicos.get(ativo_selecionado)
        if df_hist is not None and not df_hist.empty:
            fig = px.line(df_hist, y='Close', title=f"Preço de Fechamento de {ativo_selecionado}")
            st.plotly_chart(fig, use_container_width=True)

        # 3. Dados Fundamentais
        st.subheader("Fatores Fundamentais (Último Balanço)")
        dados_fund = builder.dados_fundamentais.get(ativo_selecionado)
        if dados_fund:
            df_fund = pd.Series(dados_fund).to_frame(name='Valor')
            st.table(df_fund)
        else:
            st.info("Dados fundamentais não disponíveis ou não carregados.")
        
        # 4. Dados de Machine Learning/Risco (Ponto 2: Gráfico de Previsão)
        st.subheader("Predições e Métricas de Machine Learning/Risco")
        dados_ml = builder.predicoes_ml.get(ativo_selecionado, {})
        
        is_logreg_trained = dados_ml.get('modelo', '').startswith('Simples') and dados_ml.get('status') == 'Treinado com Sucesso'
        is_complex_trained = dados_ml.get('modelo', '').startswith('Complexo') and dados_ml.get('status') == 'Treinado com Sucesso'
        is_garch_trained = dados_ml.get('modelo') == 'GARCH(1,1)' and dados_ml.get('status') == 'Treinado com Sucesso'
        
        if is_logreg_trained or is_complex_trained:
            modelo_nome = dados_ml.get('modelo', 'Modelo de Classificação')
            proba = dados_ml.get('proba_futura_alta', 0.5)
            auc = dados_ml.get('auc_roc_score', 0.5)
            
            st.markdown(f"##### Resultados do {modelo_nome}")
            col1, col2 = st.columns(2)
            col1.metric("AUC-ROC Score (Teste)", f"{auc:.4f}")
            col2.metric("Probabilidade de Alta (Próx. Período)", f"{proba * 100:.2f}%")
            
            # NOVO: Gráfico de Previsão de Probabilidade (Ponto 2)
            st.markdown("##### Previsão de Probabilidade de Alta")
            
            fig_proba = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = proba * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Probabilidade de Fechamento em Alta (Próx. Período)"},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#0E768E"},
                    'bar': {'color': "#0E768E"},
                    'bgcolor': "white",
                    'steps': [
                        {'range': [0, 50], 'color': "lightcoral"},
                        {'range': [50, 100], 'color': "lightgreen"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50}}))

            fig_proba.update_layout(autosize=False, width=400, height=300)
            st.plotly_chart(fig_proba, use_container_width=False)
            
            st.markdown("###### Features Selecionadas e Importância")
            # Exibe os coeficientes do modelo logreg ou features selecionadas
            if is_logreg_trained:
                features_data = dados_ml.get('features_importantes_logreg', {})
                df_features = pd.DataFrame(features_data.items(), columns=['Feature', 'Coeficiente'])
                st.dataframe(df_features.sort_values(by='Coeficiente', ascending=False), hide_index=True)
            elif is_complex_trained:
                st.info(f"Modelo Ensemble (Base: {', '.join(dados_ml.get('base_models', []))}). Features selecionadas por Elastic Net: {', '.join(dados_ml.get('features_selecionadas', []))}")

        elif is_garch_trained:
            st.markdown("##### Resultados do Modelo GARCH(1,1) (Risco)")
            col1, col2 = st.columns(2)
            vol_cond = dados_ml.get('pred_vol_condicional', 0.0)
            var_99 = dados_ml.get('VaR_99_percentual', 0.0)
            col1.metric("Volatilidade Condicional Predita", f"{vol_cond:.4f}%")
            col2.metric("VaR 99% (Valor em Risco)", f"{var_99:.4f}%")
            st.info("O GARCH modela a volatilidade dos retornos. O VaR 99% indica a perda máxima esperada em 1% dos casos.")

        else:
            st.info("Nenhum modelo de Machine Learning/Risco foi treinado com sucesso para este ativo.")

        # NOVO: Análise Sintética (Ponto 8)
        st.markdown("---")
        st.subheader("📊 Análise Sintética e Clusterização")
        st.markdown("A pontuação final é usada pelo algoritmo de Clusterização para a seleção final de ativos, combinando fatores Fundamentais, Técnicos e ML.")
        
        # Mocking the scores based on current data for demonstration
        df_change = df_hist['Close'].pct_change().iloc[-1] * 100 if len(df_hist) > 1 else 0.0
        pl_ratio = builder.dados_fundamentais.get(ativo_selecionado, {}).get('P/L', 15.0)
        
        # Placeholder Score Fundamental: Baseado em P/L (quanto menor, melhor)
        fundamental_score = round(10 - math.log10(max(1, pl_ratio * 0.5)), 2) # Mock: P/L baixo -> Score alto
        
        # Placeholder Score Técnico: Baseado em performance recente (momentum)
        technical_score = round(5 + min(5, max(-5, df_change * 0.5)), 2) # Mock: Retorno alto -> Score alto
        
        # Score ML: Probabilidade de Alta * AUC
        ml_score = round(dados_ml.get('proba_futura_alta', 0.5) * dados_ml.get('auc_roc_score', 0.5) * 10, 2)
        
        col_fund, col_tec, col_ml, col_cluster = st.columns(4)
        
        col_fund.metric("Score Fundamental", f"{max(1, fundamental_score):.2f}/10")
        col_tec.metric("Score Técnico", f"{max(1, technical_score):.2f}/10")
        col_ml.metric("Score ML (Prob. Alta * AUC)", f"{max(1, ml_score):.2f}/10")
        col_cluster.metric("Cluster de Seleção", "Alto Crescimento")
        
        st.info("A Clusterização final agrupa ativos com perfis de Risco/Retorno/ML semelhantes para garantir diversificação coerente.")


# Função que gera a aba de seleção de ativos
def aba_selecao_ativos():
    """Permite a seleção do universo de ativos e configuração de parâmetros."""
    st.title("🎯 Seleção de Ativos e Coleta de Dados")
    
    # Campo de Ativos (Mantido o placeholder da versão anterior)
    st.markdown("#### Universo de Ativos")
    ativos_input = st.text_area(
        "Insira os Tickers de Ativos (separados por vírgula)",
        value="ABEV3, PETR4, VALE3, ITUB4, BBDC4",
        key='ativos_input'
    )
    
    # Parâmetros de Tempo
    st.markdown("#### Parâmetros Temporais")
    col_h, col_i = st.columns(2)
    horizonte = col_h.selectbox(
        "Horizonte de Dados Históricos",
        options=['1y', '3y', '5y'],
        index=1,
        key='horizonte_select'
    )
    intervalo = col_i.selectbox(
        "Intervalo de Coleta",
        options=['1d', '1wk', '1mo'],
        index=1,
        key='intervalo_select'
    )
    
    # Implementação de Prazos de Predição (Ponto 1)
    st.markdown("#### Configurações de Predição e Risco")
    
    if 'prediction_period' not in st.session_state:
        st.session_state.prediction_period = 5
        
    st.session_state.prediction_period = st.radio(
        "Prazos de Predição (Dias Úteis Futuros)",
        options=[1, 5, 21, 63],
        index=[1, 5, 21, 63].index(st.session_state.prediction_period),
        horizontal=True,
        key='prediction_period_radio',
        help="Selecione o horizonte de previsão para os modelos ML (e.g., 5 dias úteis = 1 semana)."
    )
    
    # MOCK: Placeholder para a execução
    if st.button("▶️ Coletar e Processar Dados"):
        st.session_state.ativos_para_analise = [a.strip().upper() for a in ativos_input.split(',') if a.strip()]
        if st.session_state.ativos_para_analise:
            st.session_state.builder = PortfolioBuilder(
                st.session_state.ativos_para_analise,
                horizonte=horizonte,
                intervalo=intervalo
            )
            
            # MOCK: Simula a coleta e o treinamento de ML/Markowitz
            with st.spinner("Coletando dados, treinando modelos ML e otimizando Markowitz..."):
                time.sleep(2) # Simula o tempo de processamento
                
                # Mock de Dados Históricos e Preços Atuais
                mock_dates = pd.to_datetime(pd.date_range(end=datetime.now(), periods=100, freq='W'))
                for ativo in st.session_state.builder.ativos:
                    mock_data = pd.DataFrame({
                        'Close': np.random.rand(100) * 10 + 10,
                        'Open': np.random.rand(100) * 10 + 10,
                        'High': np.random.rand(100) * 10 + 10,
                        'Low': np.random.rand(100) * 10 + 10,
                        'Volume': np.random.randint(100000, 5000000, 100)
                    }, index=mock_dates)
                    st.session_state.builder.dados_historicos[ativo] = mock_data
                    st.session_state.builder.precos_atuais[ativo] = mock_data['Close'].iloc[-1]
                
                # Mock de Dados Fundamentais
                for ativo in st.session_state.builder.ativos:
                    st.session_state.builder.dados_fundamentais[ativo] = {
                        'Setor': 'Mock Setor', 
                        'Subsetor': 'Mock Indústria',
                        'P/L': np.random.uniform(5, 30),
                        'ROE': np.random.uniform(0.1, 0.3)
                    }

                # Executa os novos treinamentos de ML (Simples, Complexo, GARCH)
                st.session_state.builder.treinar_ml_e_garch(modelos_selecionados=['Simples', 'Complexo'])
                
                # Otimização Markowitz (Mockada)
                st.session_state.builder.markowitz_results = {
                    'Sharpe Ratio Máximo': 1.5 + np.random.rand(),
                    'Retorno Anualizado (%)': 15 + np.random.rand() * 10,
                    'Volatilidade Anualizada (%)': 10 + np.random.rand() * 5,
                    'Pesos Otimizados': {
                        st.session_state.builder.ativos[0]: 0.4,
                        st.session_state.builder.ativos[1]: 0.3,
                        st.session_state.builder.ativos[2]: 0.2,
                        st.session_state.builder.ativos[3]: 0.1
                    }
                }
            
            st.session_state.builder_complete = True
            st.success("Processamento concluído! Vá para as abas de Portfólio e Análise.")
        else:
            st.error("Por favor, insira pelo menos um ativo para análise.")


# Função que gera a aba de construtor de portfólio
def aba_construtor_portfolio():
    """Aba principal para otimização Markowitz e detalhamento da alocação."""
    st.title("🏗️ Construtor de Portfólio Otimizado")
    
    builder = st.session_state.builder
    if not builder or not st.session_state.get('builder_complete', False):
        st.warning("Por favor, selecione e processe os ativos na aba 'Seleção de Ativos' primeiro.")
        return

    # --- CORREÇÃO DO IndexError (Ponto 5 e 6) ---
    assets = builder.ativos
    if assets: 
        # Checagem robusta: se algum ativo tem AUC > 0.5 (classificação) OU Vol Condicional (risco)
        is_ml_actually_trained = any(
            builder.predicoes_ml.get(ativo, {}).get('auc_roc_score', 0.0) > 0.5 or 
            'pred_vol_condicional' in builder.predicoes_ml.get(ativo, {})
            for ativo in assets
        )
        if not is_ml_actually_trained:
             st.warning("ℹ️ Modelo ML Não Treinado: A pipeline de Machine Learning falhou para todos os ativos no universo de análise.")

    # --- LAYOUT DE BOTÕES FULL WIDTH (EXEMPLO) ---
    st.markdown("#### Botões de Ação")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    if 'selected_button' not in st.session_state:
        st.session_state.selected_button = 'otimizar'

    # Botões (Mantido o estilo de seleção persistente)
    col1.button(
        "Otimizar", 
        key='otimizar', 
        type=PRIMARY_BUTTON_STYLE if st.session_state.selected_button == 'otimizar' else SECONDARY_BUTTON_STYLE,
        on_click=lambda: st.session_state.update(selected_button='otimizar')
    )
    col2.button(
        "Monte Carlo", 
        key='monte_carlo', 
        type=PRIMARY_BUTTON_STYLE if st.session_state.selected_button == 'monte_carlo' else SECONDARY_BUTTON_STYLE,
        on_click=lambda: st.session_state.update(selected_button='monte_carlo')
    )
    col3.button(
        "Limites de Risco", 
        key='risco', 
        type=PRIMARY_BUTTON_STYLE if st.session_state.selected_button == 'risco' else SECONDARY_BUTTON_STYLE,
        on_click=lambda: st.session_state.update(selected_button='risco')
    )
    col4.button(
        "Rebalancear", 
        key='rebalancear', 
        type=PRIMARY_BUTTON_STYLE if st.session_state.selected_button == 'rebalancear' else SECONDARY_BUTTON_STYLE,
        on_click=lambda: st.session_state.update(selected_button='rebalancear')
    )
    col5.button(
        "Exportar", 
        key='exportar', 
        type=PRIMARY_BUTTON_STYLE if st.session_state.selected_button == 'exportar' else SECONDARY_BUTTON_STYLE,
        on_click=lambda: st.session_state.update(selected_button='exportar')
    )
    
    st.markdown("---")
    
    # --- RESULTADOS MARKOWITZ ---
    if st.session_state.selected_button == 'otimizar' and builder.markowitz_results:
        st.subheader("Resultados da Otimização Markowitz (Max Sharpe)")
        
        sharpe = builder.markowitz_results['Sharpe Ratio Máximo']
        retorno = builder.markowitz_results['Retorno Anualizado (%)']
        volatilidade = builder.markowitz_results['Volatilidade Anualizada (%)']
        
        col_s, col_r, col_v = st.columns(3)
        col_s.metric("Sharpe Ratio Máximo", f"{sharpe:,.4f}".replace(",", "X").replace(".", ",").replace("X", "."))
        col_r.metric("Retorno Anualizado Esperado", f"{retorno:,.2f}%".replace(",", "X").replace(".", ",").replace("X", "."))
        col_v.metric("Volatilidade Anualizada", f"{volatilidade:,.2f}%".replace(",", "X").replace(".", ",").replace("X", "."))

        st.markdown("##### Pesos Otimizados por Ativo")
        pesos_df = pd.DataFrame(
            {'Peso (%)': [v * 100 for v in builder.markowitz_results['Pesos Otimizados'].values()]},
            index=builder.markowitz_results['Pesos Otimizados'].keys()
        )
        st.dataframe(pesos_df.sort_values(by='Peso (%)', ascending=False).style.format({'Peso (%)': "{:.2f}%"}))
        
        # --- CÁLCULO DE AÇÕES INTEIRAS E SALDO RESTANTE (Lógica Confirmada) ---
        st.markdown("---")
        st.subheader("💰 Alocação Financeira com Ações Inteiras")
        st.markdown("Insira o valor total da sua carteira para calcular a quantidade exata de ações a serem compradas e o saldo restante (cash-back).")

        valor_total_carteira = st.number_input(
            "Valor Total Disponível para Investimento (R$)",
            min_value=100.0,
            value=10000.00,
            step=100.0,
            format="%.2f",
            key='valor_carteira_final_input'
        )

        if valor_total_carteira > 0:
            pesos_otimizados = builder.markowitz_results['Pesos Otimizados']
            ativos_na_carteira = list(pesos_otimizados.keys())
            
            precos_atuais = {
                ativo: builder.precos_atuais.get(ativo) 
                for ativo in ativos_na_carteira if builder.precos_atuais.get(ativo) is not None
            }

            if precos_atuais and len(precos_atuais) == len(pesos_otimizados):
                dados_alocacao = []
                dinheiro_gasto_total = 0.0
                
                for ativo, peso in pesos_otimizados.items():
                    if ativo in precos_atuais and peso > 1e-4:
                        preco = precos_atuais[ativo]
                        
                        valor_alvo = valor_total_carteira * peso
                        
                        # Cálculo de ações inteiras (math.floor)
                        num_acoes = math.floor(valor_alvo / preco)
                        
                        # Dinheiro realmente gasto
                        dinheiro_gasto = num_acoes * preco
                        
                        dinheiro_gasto_total += dinheiro_gasto
                        
                        formato_moeda = lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                        
                        dados_alocacao.append({
                            'Ativo': ativo,
                            'Peso Alvo (%)': f"{peso * 100:.2f}%",
                            'Preço Unitário': formato_moeda(preco),
                            'Valor Alvo': formato_moeda(valor_alvo),
                            'Ações a Comprar': num_acoes,
                            'Valor Gasto': formato_moeda(dinheiro_gasto),
                            'Diferença (Cash-In)': formato_moeda(valor_alvo - dinheiro_gasto)
                        })
                
                # Saldo restante (calculado como o total - o gasto com ações inteiras)
                saldo_restante = valor_total_carteira - dinheiro_gasto_total

                if dados_alocacao:
                    df_alocacao = pd.DataFrame(dados_alocacao)
                    st.table(df_alocacao)
                    
                    col_gasto, col_saldo = st.columns(2)
                    col_gasto.metric(
                        "Dinheiro Total Gasto (Ações)", 
                        formato_moeda(dinheiro_gasto_total)
                    )
                    col_saldo.metric(
                        "Saldo Restante (Cash-Back)", 
                        formato_moeda(saldo_restante), 
                        delta=f"{saldo_restante / valor_total_carteira * 100:.2f}% do total" if valor_total_carteira > 0 else None
                    )
                    st.info("O saldo restante representa o valor que não pôde ser investido devido à restrição de compra de ações inteiras (não fracionárias).")
                else:
                    st.warning("Não foi possível calcular a alocação de ações inteiras. Verifique se os preços e pesos estão disponíveis e se há ativos com peso relevante.")
            else:
                st.warning("Os preços atuais de todos os ativos otimizados não foram carregados ou encontrados. Por favor, refaça a coleta de dados.")
        else:
            st.info("Insira um valor maior que R$ 100,00 para o cálculo da alocação de ações inteiras.")
            
    elif builder.markowitz_results is None:
        st.error("Nenhuma alocação significativa para exibir. Otimização não retornou pesos.")


# --- OUTRAS ABAS (Placeholder) ---

def aba_metodologia():
    st.title("📚 Metodologia")
    st.info("O código aqui não foi modificado. Esta aba explica a metodologia (Markowitz, ML, etc.).")

def aba_referencias():
    st.title("📖 Referências")
    st.info("O código aqui não foi modificado.")

# --- MAIN EXECUTION ---

def main():
    if 'builder' not in st.session_state:
        st.session_state.builder = None
        st.session_state.builder_complete = False
        st.session_state.profile = {}
        st.session_state.ativos_para_analise = []
        st.session_state.analisar_ativo_triggered = False
        
    configurar_pagina()
    # Garante que st.session_state.debug_logs está inicializado para log_debug()
    if 'debug_logs' not in st.session_state:
        st.session_state.debug_logs = []
    
    st.markdown('<h1 class="main-header">Sistema de Otimização Quantitativa</h1>', unsafe_allow_html=True)
    
    # Esta linha foi simplificada no código de produção para uso das abas
    tabs_list = ["📚 Metodologia", "🎯 Seleção de Ativos", "🏗️ Construtor de Portfólio", "🔍 Análise Individual", "📖 Referências"]
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(tabs_list)
    
    with tab1: aba_metodologia()
    with tab2: aba_selecao_ativos()
    with tab3: aba_construtor_portfolio() # Contém a nova lógica de alocação e correção de bug
    with tab4: aba_analise_individual() # Contém a exibição da seção ML e Análise Sintética
    with tab5: aba_referencias()
    
if __name__ == '__main__':
    # MOCK: Funções de ML/GARCH ausentes no snippet original, mas necessárias para o Builder.
    # Elas foram substituídas pelas novas funções _treinar_modelo_simples e _treinar_modelo_complexo
    # e GARCH (mantido o original).
    # Caso este código seja executado fora do Streamlit e o Markowitz esteja comentado,
    # ele irá rodar, mas a simulação não.
    try:
        # A chamada principal é main()
        pass
    except NameError:
        # Se as classes ou funções não existirem (no contexto de um snippet), apenas printa.
        print("A execução real depende de todas as classes e funções do PortfolioBuilder estarem definidas.")
