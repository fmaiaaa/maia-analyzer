# -*- coding: utf-8 -*-
"""
================================================================================
GERADOR DE DADOS FINANCEIROS V25.3 - VERSÃO CLOUD RUN JOB
================================================================================
ALTERAÇÕES DESTA VERSÃO:
1. [Cloud Run] Removido '!pip install' (vai para requirements.txt).
2. [Cloud Run] Removida autenticação manual ('google.colab auth'). A autenticação
   agora é automática via Conta de Serviço.
3. [Cloud Run] Removido o loop principal sobre 'ATIVOS_PARA_COLETA'.
4. [Cloud Run] O script agora lê uma variável de ambiente 'TICKER_ID' para
   saber qual ativo processar.
5. [Cloud Run] O 'GCS_DISPONIVEL' foi removido; o script assume que o GCS está
   disponível e que a Conta de Serviço tem permissão.
6. [Cloud Run] A barra de progresso 'tqdm' foi removida do loop principal.
"""

# ----------------------------------------------------------------------------
# 📦 INSTALAÇÃO E IMPORTAÇÕES DE BIBLIOTECAS
# ----------------------------------------------------------------------------
# A instalação !pip foi movida para requirements.txt

import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm # Mantido para loops internos, mas removido do loop principal
from datetime import datetime
import time
import warnings
import os
import traceback
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Novas importações V22/V23/V24
import requests
from bs4 import BeautifulSoup
import shap
import re

# Importações de ML e Estatística
from arch import arch_model
import ta
from ta.utils import dropna
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import optuna

# Importações Google Cloud
# [REMOVIDO] from google.colab import auth
from google.cloud import storage # Importação direta
GCS_DISPONIVEL = True # Assumimos que está disponível

# --- Configurações Globais ---
# ... (mesmas configurações de warnings, optuna, pandas) ...
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.DEBUG)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# ----------------------------------------------------------------------------
# 🏦 CONSTANTES GLOBAIS E CONFIGURAÇÕES DO SCRIPT
# ----------------------------------------------------------------------------

# [REMOVIDO] A lista 'ATIVOS_PARA_COLETA_RAW' não é mais necessária aqui.
# O ticker virá da variável de ambiente.

# --- Configurações de Período ---
DATA_INICIO_HISTORICO = '2015-01-01'
DATA_FIM_HISTORICO = datetime.now().strftime('%Y-%m-%d')
DATA_INICIO_MACRO = '2014-01-01'

# --- [V25] Configurações de Features e Targets (V21) ---
LOOKAHEAD_DAYS = {
    'curto_prazo': 84,
    'medio_prazo': 168,
    'longo_prazo': 252
}
JANELAS_MOMENTUM = [5, 10, 21, 63, 126, 252]
JANELAS_VOLATILIDADE = [21, 63, 252]
JANELA_GARCH = 100
N_PCA_COMPONENTS = 15

# --- Configurações de ML (WFCV e HPO) ---
N_SPLITS_WFCV = 5
N_TRIALS_HPO = 30
ML_SCORING_METRIC = 'roc_auc'
WFCV_TEST_SIZE_RATIO = 0.15
WFCV_MIN_TRAIN_SIZE = 504
MI_FEATURE_PERCENTILE = 0.8
MI_FEATURE_MIN_K = 30

# --- Configurações Google Cloud Storage (GCS) ---
GCS_BUCKET_NAME = 'meu-portfolio-dados-gratuitos'
GCS_BASE_PATH = 'dados_financeiros_v25_2_debug'

# ----------------------------------------------------------------------------
# ... (Todas as funções auxiliares: obter_dados_macro_avancados, _limpar_valor_fundamentus, ...) ...
# ... (obter_dados_fundamentalistas_fundamentus, coletar_dados_base, ...) ...
# ... (adicionar_indicadores_tecnicos, adicionar_volatilidade_garch, ...) ...
# ... (adicionar_padroes_candlestick, preparar_features_e_target_raw, ...) ...
# ... (WalkForwardCV, rodar_wfc_com_hpo, todas as funções hpo_*, ...) ...
# ... (gerar_explicabilidade_shap, salvar_no_gcs, salvar_dataframe_gcs, ...) ...
# ... (Abaixo estão as funções inalteradas. O conteúdo está omitido aqui
#      para focar nas mudanças, mas elas devem estar no arquivo final.) ...
#
# <COPIAR TODAS AS FUNÇÕES DO ARQUIVO V25.3 AQUI, DE 'obter_dados_macro_avancados'
# ATÉ 'salvar_dados_separados'>
#
# (Exemplo de uma função inalterada)
def obter_dados_macro_avancados(start_date, end_date=DATA_FIM_HISTORICO):
    logging.debug("[MACRO] Iniciando coleta de dados macro expandidos...")
    tickers_macro = {
        'USDBRL=X': 'retorno_usdbrl',
        '^GSPC': 'retorno_gspc', # S&P 500
        '^VIX': 'retorno_vix',   # Volatilidade
        'CL=F': 'retorno_oil',   # Petróleo WTI
        '^BVSP': 'retorno_ibov' # Ibovespa
    }
    try:
        df_macro_combinado = pd.DataFrame()
        for ticker, nome_coluna in tickers_macro.items():
            logging.debug(f"[MACRO] Baixando dados para {ticker}...")
            df_temp = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not df_temp.empty:
                df_macro_combinado[nome_coluna] = df_temp['Close'].pct_change()
            else:
                logging.warning(f"[MACRO] Aviso: Não foi possível baixar dados de {ticker}.")
        if df_macro_combinado.empty:
            logging.error("[MACRO] Erro: Nenhuma série macro foi baixada.")
            return pd.DataFrame()
        df_macro_combinado = df_macro_combinado.ffill()
        df_macro_combinado = df_macro_combinado.fillna(0)
        df_macro_combinado.index = pd.to_datetime(df_macro_combinado.index).tz_localize(None)
        logging.info(f"[MACRO] Dados Macro Expandidos coletados. {len(df_macro_combinado)} registros.")
        logging.debug(f"[MACRO] Dados combinados: {df_macro_combinado.shape[0]} linhas, colunas: {list(df_macro_combinado.columns)}")
        return df_macro_combinado
    except Exception as e:
        logging.error(f"[MACRO] Erro ao obter dados macro expandidos: {e}", exc_info=True)
        return pd.DataFrame()

def _limpar_valor_fundamentus(valor_str):
    if valor_str is None: return np.nan
    try:
        valor_str = str(valor_str).strip()
        if valor_str.endswith('%'):
            return float(valor_str.replace('%', '').replace('.', '').replace(',', '.')) / 100.0
        valor_limpo = valor_str.replace('.', '').replace(',', '.')
        valor_limpo = re.sub(r"[^0-9.-]", "", valor_limpo)
        if valor_limpo == "-": return np.nan
        return float(valor_limpo)
    except Exception: return np.nan

def obter_dados_fundamentalistas_fundamentus(ticker):
    logging.debug(f"[Fundamentus] Iniciando scraping estático para {ticker}...")
    url = f"https://www.fundamentus.com.br/detalhes.php?papel={ticker}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers)
        logging.debug(f"[Fundamentus] Página de {ticker} acessada com status {response.status_code}")
        if response.status_code != 200:
            logging.error(f"[Fundamentus] Falha ao acessar {url}. Status: {response.status_code}")
            return {}
        soup = BeautifulSoup(response.text, 'lxml')
        tables = soup.find_all('table')
        if len(tables) < 5:
            logging.warning(f"[Fundamentus] Estrutura da página inesperada para {ticker}. Menos de 5 tabelas encontradas.")
            return {}
        dados_fundamentus = {}
        for i, table in enumerate(tables[2:5]):
            logging.debug(f"[Fundamentus] Processando Tabela {i+2}...")
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) == 6:
                    for i in [0, 2, 4]:
                        chave = cols[i].get_text(strip=True).replace('?', '')
                        valor = cols[i+1].get_text(strip=True)
                        if chave: dados_fundamentus[chave] = _limpar_valor_fundamentus(valor)
                elif len(cols) == 4:
                    for i in [0, 2]:
                        chave = cols[i].get_text(strip=True).replace('?', '')
                        valor = cols[i+1].get_text(strip=True)
                        if chave: dados_fundamentus[chave] = _limpar_valor_fundamentus(valor)
                elif len(cols) == 2 and 'Nro. Ações' in cols[0].get_text():
                     chave = cols[0].get_text(strip=True).replace('?', '')
                     valor = cols[1].get_text(strip=True)
                     if chave: dados_fundamentus[chave] = _limpar_valor_fundamentus(valor)
        logging.debug(f"[Fundamentus] Dados brutos extraídos (JSON): {json.dumps(dados_fundamentus, indent=2)}")
        dados_renomeados = {
            'P/L': dados_fundamentus.get('P/L'), 'P/VP': dados_fundamentus.get('P/VP'), 'P/EBIT': dados_fundamentus.get('P/EBIT'),
            'PSR': dados_fundamentus.get('PSR'), 'Div.Yield': dados_fundamentus.get('Div. Yield'), 'EV/EBIT': dados_fundamentus.get('EV/EBIT'),
            'EV/EBITDA': dados_fundamentus.get('EV/EBITDA'), 'Mrg.Ebit': dados_fundamentus.get('Mrg. Ebit'), 'Mrg.Liq.': dados_fundamentus.get('Mrg. Líq.'),
            'ROE': dados_fundamentus.get('ROE'), 'ROIC': dados_fundamentus.get('ROIC'), 'Liq.Corr.': dados_fundamentus.get('Liq. Corr.'),
            'Cresc.Rec.5a': dados_fundamentus.get('Cresc. Rec.5a'), 'LPA': dados_fundamentus.get('LPA'), 'Liq.Seca': dados_fundamentus.get('Liq. Seca'),
            'D.B./Patri.': dados_fundamentus.get('Dív. Bruta / Patrim.'), 'ValorMercado': dados_fundamentus.get('Valor de mercado'),
            'ValorFirma': dados_fundamentus.get('Valor da firma'), 'PatrimLiq': dados_fundamentus.get('Patrim. Líq'),
            'AtivoTotal': dados_fundamentus.get('Ativo'), 'Rec.Liq.12m': dados_fundamentus.get('Receita Líquida'),
            'EBITDA.12m': dados_fundamentus.get('EBITDA'), 'LucroBruto.12m': dados_fundamentus.get('Lucro Bruto'),
            'LucroLiq.12m': dados_fundamentus.get('Lucro Líquido'), 'NumAcoes': dados_fundamentus.get('Nro. Ações'),
        }
        dados_finais = {k: v for k, v in dados_renomeados.items() if pd.notna(v)}
        logging.info(f"[Fundamentus] Scraping estático para {ticker} concluído. {len(dados_finais)} indicadores coletados.")
        logging.debug(f"[Fundamentus] Indicadores FINAIS (Renomeados e Limpos): {list(dados_finais.keys())}")
        return dados_finais
    except Exception as e:
        logging.error(f"[Fundamentus] Erro ao fazer scraping para {ticker}: {e}", exc_info=True)
        traceback.print_exc()
        return {}

def coletar_dados_base(ticker, start_date, end_date):
    logging.info(f"Coletando dados base para {ticker}...")
    ticker_yf = ticker + ".SA"
    ticker_obj = yf.Ticker(ticker_yf)
    df_historico = None
    dict_fundamentos_estaticos = None
    try:
        logging.debug(f"[YFinance] Baixando dados históricos para {ticker_yf}...")
        df_historico = ticker_obj.history(start=start_date, end=end_date, auto_adjust=False)
        if df_historico.empty:
            logging.warning(f"[YFinance] Nenhum dado histórico encontrado para {ticker_yf}.")
            return None, None
        df_historico.index = pd.to_datetime(df_historico.index).tz_localize(None)
        df_historico.columns = [col.lower().replace(' ', '_') for col in df_historico.columns]
        df_historico['ticker'] = ticker
        logging.info(f"[YFinance] Dados históricos de {ticker_yf} coletados ({len(df_historico)} registros).")
        logging.debug(f"[YFinance] Intervalo de datas: {df_historico.index.min().date()} a {df_historico.index.max().date()}.")
    except Exception as e:
        logging.error(f"[YFinance] Erro ao coletar dados históricos para {ticker_yf}: {e}", exc_info=True)
        traceback.print_exc()
        return None, None
    try:
        logging.debug(f"[Fundamentus] Coletando dados fundamentalistas estáticos (Scraping) para {ticker}...")
        dict_fundamentos_estaticos = obter_dados_fundamentalistas_fundamentus(ticker)
        if not dict_fundamentos_estaticos:
            logging.warning(f"[Fundamentus] ⚠️ Falha ao coletar dados do Fundamentus para {ticker}. O ativo será processado sem dados fundamentalistas.")
            dict_fundamentos_estaticos = {}
        else:
            logging.info(f"[Fundamentus] ✅ Dados estáticos de {ticker} coletados com sucesso.")
        return df_historico, dict_fundamentos_estaticos
    except Exception as e:
        logging.error(f"[Fundamentus] ❌ Erro inesperado na coleta de dados estáticos: {e}", exc_info=True)
        traceback.print_exc()
        return df_historico, {}

def adicionar_indicadores_tecnicos(df):
    logging.debug("[TA] Calculando indicadores técnicos (TA-Lib)...")
    df_ta = df.copy()
    colunas_originais = set(df_ta.columns)
    logging.debug("[TA]  Calculando: Momentum (Retornos, MOM)...")
    for lag in JANELAS_MOMENTUM:
        df_ta[f'retorno_{lag}d'] = df_ta['close'].pct_change(lag)
        df_ta[f'mom_{lag}d'] = df_ta['close'] / df_ta['close'].shift(lag) - 1
    logging.debug("[TA]  Calculando: RSI, Stochastic...")
    df_ta['rsi'] = ta.momentum.RSIIndicator(df_ta['close'], window=14).rsi()
    df_ta['stoch_k'] = ta.momentum.StochasticOscillator(df_ta['high'], df_ta['low'], df_ta['close'], window=14, smooth_window=3).stoch()
    logging.debug("[TA]  Calculando: Volatilidade (Rolling Std), ATR...")
    for lag in JANELAS_VOLATILIDADE:
        df_ta[f'vol_{lag}d'] = df_ta['close'].pct_change().rolling(window=lag).std() * np.sqrt(lag)
    df_ta['atr'] = ta.volatility.AverageTrueRange(df_ta['high'], df_ta['low'], df_ta['close'], window=14).average_true_range()
    logging.debug("[TA]  Calculando: Trend (SMA, EMA, MACD, ADX)...")
    df_ta['sma_21'] = ta.trend.SMAIndicator(df_ta['close'], window=21).sma_indicator()
    df_ta['ema_50'] = ta.trend.EMAIndicator(df_ta['close'], window=50).ema_indicator()
    df_ta['ema_200'] = ta.trend.EMAIndicator(df_ta['close'], window=200).ema_indicator()
    macd = ta.trend.MACD(df_ta['close'], window_slow=26, window_fast=12, window_sign=9)
    df_ta['macd'] = macd.macd()
    df_ta['macd_signal'] = macd.macd_signal()
    adx_obj = ta.trend.ADXIndicator(df_ta['high'], df_ta['low'], df_ta['close'], window=14)
    df_ta['adx'] = adx_obj.adx()
    df_ta['adx_pos'] = adx_obj.adx_pos()
    df_ta['adx_neg'] = adx_obj.adx_neg()
    logging.debug("[TA]  Calculando: Volume (OBV, MFI)...")
    df_ta['obv'] = ta.volume.OnBalanceVolumeIndicator(df_ta['close'], df_ta['volume']).on_balance_volume()
    df_ta['mfi'] = ta.volume.MFIIndicator(df_ta['high'], df_ta['low'], df_ta['close'], df_ta['volume'], window=14).money_flow_index()
    logging.debug("[TA]  Calculando: Relação Preço/MAs...")
    df_ta['price_div_sma_21'] = df_ta['close'] / df_ta['sma_21']
    df_ta['price_div_ema_50'] = df_ta['close'] / df_ta['ema_50']
    df_ta['price_div_ema_200'] = df_ta['close'] / df_ta['ema_200']
    df_ta['sma_21_div_ema_50'] = df_ta['sma_21'] / df_ta['ema_50']
    logging.debug("[TA] Cálculo de indicadores técnicos concluído.")
    colunas_novas = set(df_ta.columns) - colunas_originais
    logging.debug(f"[TA] {len(colunas_novas)} colunas técnicas adicionadas (ex: {list(colunas_novas)[:5]}...).")
    return df_ta

def adicionar_volatilidade_garch(df, janela_garch=JANELA_GARCH):
    logging.debug("[GARCH] Calculando volatilidade GARCH(1,1)...")
    df_garch = df.copy()
    df_garch['retorno_diario'] = df_garch['close'].pct_change().fillna(0)
    vol_garch = []
    if len(df_garch) < janela_garch:
        logging.warning(f"[GARCH] Aviso: Dados insuficientes (< {janela_garch} dias). Pulando GARCH.")
        df_garch['vol_garch'] = np.nan
        return df_garch
    total_janelas = len(df_garch) - janela_garch
    logging.debug(f"[GARCH] Iniciando loop GARCH para {total_janelas} janelas...")
    for i in range(total_janelas):
        try:
            retornos_janela = df_garch['retorno_diario'].iloc[i : i + janela_garch] * 100
            if retornos_janela.var() < 1e-6:
                vol_garch.append(np.nan)
                continue
            gm = arch_model(retornos_janela, p=1, q=1, vol='Garch', dist='Normal', rescale=False)
            res = gm.fit(update_freq=0, disp='off')
            forecast = res.forecast(horizon=1)
            pred_vol = np.sqrt(forecast.variance.iloc[-1, 0])
            vol_garch.append(pred_vol / 100.0)
            if (i+1) % 100 == 0:
                 logging.debug(f"[GARCH]  Calculado GARCH: Janela {i+1}/{total_janelas}...")
        except Exception as e:
            logging.debug(f"[GARCH] Erro GARCH no índice {i}: {e}")
            vol_garch.append(np.nan)
    df_garch['vol_garch'] = [np.nan] * (janela_garch) + vol_garch
    logging.debug("[GARCH] Cálculo GARCH concluído.")
    return df_garch

def adicionar_padroes_candlestick(df):
    logging.debug("[Candlestick] Calculando padrões de candlestick manuais...")
    df_cs = df.copy()
    if 'open' not in df_cs.columns or df_cs['open'].isnull().all():
        logging.warning("[Candlestick] Coluna 'open' ausente ou vazia. Pulando padrões.")
        return df_cs
    body = df_cs['close'] - df_cs['open']
    body_abs = body.abs()
    range_total = df_cs['high'] - df_cs['low']
    range_total = range_total.replace(0, np.nan)
    upper_shadow = df_cs['high'] - np.maximum(df_cs['close'], df_cs['open'])
    lower_shadow = np.minimum(df_cs['close'], df_cs['open']) - df_cs['low']
    df_cs['cs_doji'] = (body_abs < (range_total * 0.1)).astype(int)
    df_cs['cs_marubozu_white'] = ((body > 0) & (body_abs > (range_total * 0.95)) & (upper_shadow < (range_total * 0.05)) & (lower_shadow < (range_total * 0.05))).astype(int)
    df_cs['cs_marubozu_black'] = ((body < 0) & (body_abs > (range_total * 0.95)) & (upper_shadow < (range_total * 0.05)) & (lower_shadow < (range_total * 0.05))).astype(int)
    body_prev = body.shift(1)
    df_cs['cs_engulfing_bullish'] = ((body_prev < 0) & (body > 0) & (df_cs['open'] < df_cs['close'].shift(1)) & (df_cs['close'] > df_cs['open'].shift(1))).astype(int)
    df_cs['cs_engulfing_bearish'] = ((body_prev > 0) & (body < 0) & (df_cs['open'] > df_cs['close'].shift(1)) & (df_cs['close'] < df_cs['open'].shift(1))).astype(int)
    df_cs['cs_hammer'] = (((body_abs > (range_total * 0.1)) & (body_abs < (range_total * 0.3))) &
                          (upper_shadow < (range_total * 0.1)) &
                          (lower_shadow > (body_abs * 2.0))).astype(int)
    colunas_novas = ['cs_doji', 'cs_marubozu_white', 'cs_marubozu_black', 'cs_engulfing_bullish', 'cs_engulfing_bearish', 'cs_hammer']
    logging.debug(f"[Candlestick] {len(colunas_novas)} colunas de padrões adicionadas: {colunas_novas}")
    return df_cs

def preparar_features_e_target_raw(df_combinado, lookahead_days_map):
    logging.debug("[ML Prep] Preparando features (X) e retornos futuros (raw)...")
    df_ml = df_combinado.copy()
    dropna_cols = []
    for name, days in lookahead_days_map.items():
        retorno_col = f'retorno_futuro_{name}'
        df_ml[retorno_col] = df_ml['close'].shift(-days) / df_ml['close'] - 1
        dropna_cols.append(retorno_col)
    logging.debug(f"[ML Prep]  Calculados retornos futuros para: {list(lookahead_days_map.keys())}")
    colunas_para_remover = [
        'open', 'high', 'low', 'close', 'adj_close', 'volume', 'dividends', 'stock_splits', 'ticker',
        'retorno_diario',
    ]
    features_colunas = [col for col in df_ml.columns if col not in colunas_para_remover and not col.startswith('retorno_futuro_') and not col.startswith('target_')]
    df_ml = df_ml.dropna(subset=dropna_cols)
    X = df_ml[features_colunas]
    df_ml_metadata = df_ml.drop(columns=features_colunas)
    logging.debug(f"[ML Prep]  Shape (linhas, colunas) após dropna dos targets: {df_ml.shape}")
    logging.debug(f"[ML Prep] Realizando imputação de NaNs. Features originais: {len(X.columns)}")
    for col in X.columns:
        if X[col].isnull().any():
            X[f'{col}_is_nan'] = X[col].isnull().astype(int)
    features_colunas_com_nan = list(X.columns)
    imputer = SimpleImputer(strategy='median')
    X_imputado = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputado, columns=features_colunas_com_nan, index=X.index)
    logging.debug(f"[ML Prep]  Imputação concluída. Features com flags NaN: {len(X.columns)}")
    logging.debug("[ML Prep] Aplicando StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_final = pd.DataFrame(X_scaled, columns=features_colunas_com_nan, index=X.index)
    logging.debug("[PCA] Calculando e adicionando Componentes Principais (PCA)...")
    n_pca = min(N_PCA_COMPONENTS, len(X_final.columns))
    if n_pca > 0:
        pca = PCA(n_components=n_pca, random_state=42)
        X_pca = pca.fit_transform(X_final)
        pca_cols = [f'PCA_{i}' for i in range(n_pca)]
        X_pca_df = pd.DataFrame(X_pca, columns=pca_cols, index=X_final.index)
        X_final_com_pca = pd.concat([X_final, X_pca_df], axis=1)
        logging.debug(f"[PCA] {n_pca} componentes PCA adicionados. Total de features agora: {len(X_final_com_pca.columns)}")
    else:
        logging.warning("[PCA] Pular PCA, número de features ou componentes é 0.")
        X_final_com_pca = X_final
    logging.debug(f"[ML Prep] Shape final de X (features): {X_final_com_pca.shape}")
    logging.debug(f"[ML Prep] Shape final de df_ml_metadata (targets): {df_ml_metadata.shape}")
    logging.debug("[ML Prep] Preparação de X_final (scaled + PCA) e df_ml_metadata (raw returns) concluída.")
    return X_final_com_pca, df_ml_metadata

class WalkForwardCV:
    def __init__(self, n_splits, min_train_size, test_size_ratio):
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.test_size_ratio = test_size_ratio
        logging.debug(f"WFCV inicializado: n_splits={n_splits}, min_train={min_train_size}, test_ratio={test_size_ratio}")
    def split(self, X):
        n_samples = len(X)
        if n_samples < self.min_train_size * (1 + self.test_size_ratio):
            logging.warning(f"Aviso WFCV: Dados insuficientes ({n_samples}) para {self.n_splits} splits com min_train={self.min_train_size}.")
            if n_samples >= self.min_train_size:
                test_size_calc = int(n_samples * self.test_size_ratio)
                if test_size_calc == 0 and n_samples - self.min_train_size > 0:
                   test_size_calc = n_samples - self.min_train_size
                if n_samples - self.min_train_size > 0 and test_size_calc > 0:
                    logging.debug("WFCV: Tentando 1 split.")
                    train_end = n_samples - test_size_calc
                    yield np.arange(0, train_end), np.arange(train_end, n_samples)
                else:
                    logging.error("WFCV: Não é possível gerar nenhum split. Pulando WFCV.")
                    return
            else:
               logging.error("WFCV: Não é possível gerar nenhum split. Pulando WFCV.")
               return
        test_size = int(n_samples * self.test_size_ratio)
        if test_size == 0: test_size = 1
        start_points = np.linspace(self.min_train_size, n_samples - test_size, self.n_splits, dtype=int)
        logging.debug(f"WFCV: Gerando {self.n_splits} splits. Tamanho do teste: {test_size}. Pontos de início de treino: {start_points}")
        for train_end in start_points:
            test_end = train_end + test_size
            if test_end > n_samples:
                test_end = n_samples
                train_end = max(self.min_train_size, n_samples - test_size)
            if train_end >= test_end:
                logging.warning(f"WFCV: Pulando split inválido (train_end={train_end} >= test_end={test_end})")
                continue
            logging.debug(f"WFCV Split: Treino [0:{train_end}], Teste [{train_end}:{test_end}]")
            yield np.arange(0, train_end), np.arange(train_end, test_end)

def rodar_wfc_com_hpo(X, y, ticker, target_name, n_splits_wfc, n_trials_hpo, modelos_config):
    logging.info(f"[{target_name}] Iniciando WFCV + HPO (Stacking) para {ticker} ({n_splits_wfc} splits, {n_trials_hpo} trials/split)...")
    wfc = WalkForwardCV(n_splits=n_splits_wfc, min_train_size=WFCV_MIN_TRAIN_SIZE, test_size_ratio=WFCV_TEST_SIZE_RATIO)
    resultados_folds = []
    best_params_folds = {}
    if not any(wfc.split(X)):
        logging.error(f"[{target_name}] ❌ Falha: Dados insuficientes para {ticker} após WFCV split. Pulando ML.")
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float)
    wfc = WalkForwardCV(n_splits=n_splits_wfc, min_train_size=WFCV_MIN_TRAIN_SIZE, test_size_ratio=WFCV_TEST_SIZE_RATIO)
    y_true_final = pd.Series(dtype=float)
    y_pred_proba_stacking_final = pd.Series(dtype=float)
    for fold, (train_idx, test_idx) in enumerate(wfc.split(X)):
        fold_start_time = time.time()
        logging.info(f"      [{target_name}] Fold {fold+1}/{n_splits_wfc}...")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        logging.debug(f"      Fold {fold+1}: Treino {len(X_train)} amostras ({X_train.index.min()} a {X_train.index.max()})")
        logging.debug(f"      Fold {fold+1}: Teste {len(X_test)} amostras ({X_test.index.min()} a {X_test.index.max()})")
        fold_metadata = {
            'ticker': ticker, 'target_name': target_name, 'fold': fold + 1, 'train_size': len(X_train), 'test_size': len(X_test),
            'train_start_date': X_train.index.min().strftime('%Y-%m-%d'), 'train_end_date': X_train.index.max().strftime('%Y-%m-%d'),
            'test_start_date': X_test.index.min().strftime('%Y-%m-%d'), 'test_end_date': X_test.index.max().strftime('%Y-%m-%d'),
        }
        if len(np.unique(y_train)) < 2:
            logging.warning(f"      Aviso: Fold {fold+1} tem apenas uma classe no treino. Pulando fold.")
            continue
        if len(np.unique(y_test)) < 2:
            logging.warning(f"      Aviso: Fold {fold+1} tem apenas uma classe no teste. Pulando fold (não é possível calcular AUC).")
            continue
        X_meta_train_fold = pd.DataFrame(index=X_train.index)
        X_meta_test_fold = pd.DataFrame(index=X_test.index)
        for nome_modelo, config in modelos_config.items():
            logging.debug(f"      Fold {fold+1}: Iniciando HPO para {nome_modelo}...")
            def objective(trial):
                params = config['hpo_space'](trial)
                if nome_modelo == 'CatBoost': model = config['model'](**params, verbose=False, random_state=42)
                elif nome_modelo == 'LGBM': model = config['model'](**params, random_state=42, n_jobs=1, verbose=-1)
                elif nome_modelo == 'XGB': model = config['model'](**params, random_state=42, n_jobs=1, eval_metric='logloss')
                elif nome_modelo == 'SVC': model = config['model'](**params, random_state=42, probability=True)
                else: model = config['model'](**params, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                try: preds_proba = model.predict_proba(X_test)[:, 1]
                except (AttributeError, Exception): preds_proba = model.predict(X_test)
                return roc_auc_score(y_test, preds_proba)
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials_hpo, n_jobs=-1)
            best_params = study.best_params
            best_params_folds[f"{nome_modelo}_fold_{fold+1}_{target_name}"] = best_params
            logging.debug(f"      Fold {fold+1}: HPO {nome_modelo} concluído. Melhor AUC: {study.best_value:.4f}")
            if nome_modelo == 'CatBoost': model_final = config['model'](**best_params, verbose=False, random_state=42)
            elif nome_modelo == 'LGBM': model_final = config['model'](**best_params, random_state=42, n_jobs=1, verbose=-1)
            elif nome_modelo == 'XGB': model_final = config['model'](**best_params, random_state=42, n_jobs=1, eval_metric='logloss')
            elif nome_modelo == 'SVC': model_final = config['model'](**best_params, random_state=42, probability=True)
            else: model_final = config['model'](**best_params, random_state=42, n_jobs=-1)
            model_final.fit(X_train, y_train)
            preds_binario_base = model_final.predict(X_test)
            preds_proba_base = model_final.predict_proba(X_test)[:, 1]
            metrics_base = {'modelo': nome_modelo, 'roc_auc': roc_auc_score(y_test, preds_proba_base), 'best_hpo_score (roc_auc)': study.best_value}
            resultados_folds.append({**fold_metadata, **metrics_base})
            X_meta_test_fold[nome_modelo] = preds_proba_base
            logging.debug(f"         Gerando meta-features (manual CV loop) para {nome_modelo}...")
            if len(X_train) < 50:
                logging.warning(f"         Aviso: X_train muito pequeno ({len(X_train)} < 50) para cv_predict loop. Usando predições de treino (com risco de overfit).")
                inner_preds_array = model_final.predict_proba(X_train)[:, 1]
                X_meta_train_fold[nome_modelo] = inner_preds_array
            else:
                test_size_inner = int(len(X_train) * 0.2); test_size_inner = max(1, test_size_inner)
                max_splits = min(3, max(1, (len(X_train) // test_size_inner) - 1))
                tscv_inner = TimeSeriesSplit(n_splits=max_splits, test_size=test_size_inner)
                inner_preds = pd.Series(np.nan, index=X_train.index, dtype=float)
                for inner_train_idx, inner_test_idx in tscv_inner.split(X_train):
                    if len(inner_test_idx) == 0 or len(inner_train_idx) == 0: continue
                    X_inner_train, X_inner_test = X_train.iloc[inner_train_idx], X_train.iloc[inner_test_idx]
                    y_inner_train = y_train.iloc[inner_train_idx]
                    if nome_modelo == 'CatBoost': model_inner = config['model'](**best_params, verbose=False, random_state=42)
                    elif nome_modelo == 'LGBM': model_inner = config['model'](**best_params, random_state=42, n_jobs=1, verbose=-1)
                    elif nome_modelo == 'XGB': model_inner = config['model'](**best_params, random_state=42, n_jobs=1, eval_metric='logloss')
                    elif nome_modelo == 'SVC': model_inner = config['model'](**best_params, random_state=42, probability=True)
                    else: model_inner = config['model'](**best_params, random_state=42, n_jobs=-1)
                    try:
                        if len(np.unique(y_inner_train)) < 2: preds_fold = 0.5
                        else:
                            model_inner.fit(X_inner_train, y_inner_train)
                            preds_fold = model_inner.predict_proba(X_inner_test)[:, 1]
                        inner_preds.iloc[inner_test_idx] = preds_fold
                    except Exception as e: inner_preds.iloc[inner_test_idx] = 0.5
                X_meta_train_fold[nome_modelo] = inner_preds
        logging.info(f"      Fold {fold+1}: Treinando Meta-Modelo (Stacking Nível 1) com Regularização CV (L1/L2/ElasticNet)...")
        X_meta_train_fold = X_meta_train_fold.fillna(0.5)
        meta_model = LogisticRegressionCV(
            Cs=10, cv=TimeSeriesSplit(n_splits=3), penalty='elasticnet', solver='saga',
            l1_ratios=[0.1, 0.5, 0.9], random_state=42, max_iter=3000, n_jobs=-1
        )
        meta_model.fit(X_meta_train_fold, y_train)
        logging.debug(f"      Fold {fold+1}: Meta-ModeloCV C escolhido: {meta_model.C_[0]:.4f}, L1_Ratio: {meta_model.l1_ratio_[0]}")
        preds_proba_stack = meta_model.predict_proba(X_meta_test_fold)[:, 1]
        preds_binario_stack = meta_model.predict(X_meta_test_fold)
        metrics_stack = {
            'modelo': 'STACKING_ENSEMBLE', 'accuracy': accuracy_score(y_test, preds_binario_stack),
            'precision': precision_score(y_test, preds_binario_stack, zero_division=0),
            'recall': recall_score(y_test, preds_binario_stack, zero_division=0),
            'f1': f1_score(y_test, preds_binario_stack, zero_division=0),
            'roc_auc': roc_auc_score(y_test, preds_proba_stack), 'best_hpo_score (roc_auc)': np.nan,
        }
        resultados_folds.append({**fold_metadata, **metrics_stack})
        fold_time = time.time() - fold_start_time
        logging.info(f"      Fold {fold+1} (Stacking) concluído em {fold_time:.2f}s. AUC: {metrics_stack['roc_auc']:.4f}")
        if fold == (n_splits_wfc - 1) or (not any(wfc.split(X)) and fold == 0):
             y_true_final = y_test
             y_pred_proba_stacking_final = pd.Series(preds_proba_stack, index=y_test.index)
    df_resultados_ml = pd.DataFrame(resultados_folds)
    df_metadados_ml = pd.DataFrame(best_params_folds.items(), columns=['model_fold_target', 'best_params'])
    df_previsoes_finais = pd.DataFrame({'y_true': y_true_final, 'y_pred_proba_stacking': y_pred_proba_stacking_final}).dropna()
    return df_resultados_ml, df_metadados_ml, df_previsoes_finais

def hpo_lgbm(trial):
    return {
        'objective': 'binary', 'metric': 'binary_logloss', 'n_estimators': trial.suggest_int('n_estimators', 100, 700),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True), 'num_leaves': trial.suggest_int('num_leaves', 20, 60),
        'max_depth': trial.suggest_int('max_depth', 5, 20), 'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9), 'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0),
    }
def hpo_xgb(trial):
    return {
        'objective': 'binary:logistic', 'n_estimators': trial.suggest_int('n_estimators', 100, 700),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True), 'max_depth': trial.suggest_int('max_depth', 5, 20),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9), 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0), 'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0),
    }
def hpo_rf(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 700), 'max_depth': trial.suggest_int('max_depth', 10, 50),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50), 'min_samples_split': trial.suggest_int('min_samples_split', 10, 100),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.7]),
    }
def hpo_catboost(trial):
    return {
        'iterations': trial.suggest_int('iterations', 100, 700), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 5, 12), 'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9), 'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 0.9),
    }
def hpo_svc(trial):
    return {
        'C': trial.suggest_float('C', 0.1, 20.0, log=True), 'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
        'gamma': trial.suggest_float('gamma', 1e-4, 1.0, log=True), 'degree': trial.suggest_int('degree', 2, 4),
    }

def gerar_explicabilidade_shap(X, y, df_metadados_ml, target_name, n_splits_wfc, modelos_config):
    logging.info(f"[{target_name}] [SHAP] Gerando explicabilidade (XAI) para os modelos base...")
    modelos_shap = {k: v for k, v in modelos_config.items() if k != 'SVC'}
    all_shap_summaries = {}
    for nome_modelo, config in modelos_shap.items():
        try:
            logging.debug(f"      [{target_name}] [SHAP] Calculando SHAP para {nome_modelo}...")
            param_key = f"{nome_modelo}_fold_{n_splits_wfc}_{target_name}"
            param_row = df_metadados_ml[df_metadados_ml['model_fold_target'] == param_key]
            if param_row.empty:
                logging.warning(f"      Aviso SHAP: Hiperparâmetros para {param_key} não encontrados. Pulando.")
                continue
            best_params = param_row['best_params'].values[0]
            if isinstance(best_params, str): best_params = eval(best_params)
            if nome_modelo == 'CatBoost': model_final_shap = config['model'](**best_params, verbose=False, random_state=42)
            elif nome_modelo == 'LGBM': model_final_shap = config['model'](**best_params, random_state=42, n_jobs=1, verbose=-1)
            elif nome_modelo == 'XGB': model_final_shap = config['model'](**best_params, random_state=42, n_jobs=1, eval_metric='logloss')
            else: model_final_shap = config['model'](**best_params, random_state=42, n_jobs=-1)
            model_final_shap.fit(X, y)
            logging.debug(f"      [SHAP] Calculando TreeExplainer para {nome_modelo}...")
            background_data = shap.sample(X, 100) if len(X) > 100 else X
            explainer = shap.TreeExplainer(model_final_shap, background_data)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list): shap_values = shap_values[1]
            shap_summary_df = pd.DataFrame(shap_values, columns=X.columns).abs().mean()
            all_shap_summaries[nome_modelo] = shap_summary_df
        except Exception as e:
            logging.error(f"      Erro ao calcular SHAP para {nome_modelo}: {e}", exc_info=True)
            traceback.print_exc()
    if not all_shap_summaries:
        logging.warning(f"   [{target_name}] [SHAP] Nenhum sumário SHAP foi gerado.")
        return pd.DataFrame()
    df_final_shap = pd.DataFrame(all_shap_summaries).fillna(0)
    df_final_shap = df_final_shap.reindex(df_final_shap.mean(axis=1).sort_values(ascending=False).index)
    logging.info(f"   [{target_name}] [SHAP] Sumário de explicabilidade gerado.")
    return df_final_shap

def salvar_no_gcs(bucket, gcs_path, local_path):
    if not GCS_DISPONIVEL: return
    try:
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        logging.debug(f"[GCS] Arquivo {local_path} salvo em gs://{bucket.name}/{gcs_path}")
    except Exception as e:
        logging.error(f"[GCS] Falha no upload para {gcs_path}: {e}", exc_info=True)

def salvar_dataframe_gcs(df, bucket, gcs_path_base, ticker, sufixo, local=True):
    if df is None or df.empty:
        logging.debug(f"DataFrame {ticker}_{sufixo} está vazio. Pulando salvamento.")
        return
    local_filename = f"{ticker}_{sufixo}.csv"
    gcs_path = f"{gcs_path_base}/{ticker}/{local_filename}"
    try:
        df.to_csv(local_filename)
        if GCS_DISPONIVEL and bucket:
            salvar_no_gcs(bucket, gcs_path, local_filename)
        if os.path.exists(local_filename):
            os.remove(local_filename)
    except Exception as e:
        logging.error(f"Erro ao salvar {local_filename}: {e}", exc_info=True)

# ----------------------------------------------------------------------------
# 🚀 FUNÇÃO PRINCIPAL DE ORQUESTRAÇÃO (V25.1 - MULTI-TARGET)
# ----------------------------------------------------------------------------

def processar_ativo_completo(ticker, df_dados_macro, gcs_bucket=None):
    logging.info(f"--- 🚀 Processando Ativo: {ticker} ---")
    start_time_ativo = time.time()
    modelos_config = {
        'LGBM': {'model': LGBMClassifier, 'hpo_space': hpo_lgbm},
        'XGB': {'model': XGBClassifier, 'hpo_space': hpo_xgb},
        'RF': {'model': RandomForestClassifier, 'hpo_space': hpo_rf},
        'CatBoost': {'model': CatBoostClassifier, 'hpo_space': hpo_catboost},
        'SVC': {'model': SVC, 'hpo_space': hpo_svc},
    }
    df_historico, dict_fundamentos_estaticos = coletar_dados_base(
        ticker,
        start_date=DATA_INICIO_HISTORICO,
        end_date=DATA_FIM_HISTORICO
    )
    if df_historico is None or df_historico.empty:
        logging.error(f"[Pipeline] ❌ Falha: Não foi possível obter dados históricos para {ticker}. Pulando ativo.")
        return False
    df_tecnico = adicionar_indicadores_tecnicos(df_historico)
    df_garch = adicionar_volatilidade_garch(df_tecnico)
    df_candlestick = adicionar_padroes_candlestick(df_garch)
    df_final_tecnico = df_candlestick
    logging.debug("Combinando Features (Técnicos + Macro)...")
    df_macro_reindexado = df_dados_macro.reindex(df_final_tecnico.index, method='ffill')
    df_combinado_tecnico_macro = pd.merge(
        df_final_tecnico, df_macro_reindexado,
        left_index=True, right_index=True, how='left'
    )
    logging.debug("Combinando Features (Fundamentos Estáticos)...")
    df_combinado_final = df_combinado_tecnico_macro.copy()
    if dict_fundamentos_estaticos:
        for chave, valor in dict_fundamentos_estaticos.items():
            df_combinado_final[chave] = valor
    df_fundamentos_estatico = pd.DataFrame.from_dict(
        dict_fundamentos_estaticos, orient='index', columns=['Valor']
    )
    X_final_scaled_pca, df_ml_metadata = preparar_features_e_target_raw(
        df_combinado_final,
        LOOKAHEAD_DAYS
    )
    if X_final_scaled_pca.empty or df_ml_metadata.empty:
        logging.warning(f"[Pipeline] ❌ Falha: Não há dados suficientes para ML após preparação (X/y vazios). Pulando ML.")
        salvar_dados_separados(
            ticker, df_final_tecnico, df_fundamentos_estatico,
            None, None, None, None, None,
            gcs_bucket
        )
        return True
    all_results_ml = []
    all_metadata_ml = []
    all_mi_scores = {}
    all_shap_summaries = {}
    all_final_predictions = {}
    for target_name, days in LOOKAHEAD_DAYS.items():
        logging.info(f"--- 🎯 Iniciando ML para Target: {target_name} ({days}d) ---")
        retorno_col = f'retorno_futuro_{target_name}'
        logging.debug(f"[{target_name}] Criando target binário (vs mediana {days}d)...")
        target_threshold_rolling = df_ml_metadata[retorno_col].rolling(window=252).median().shift(days)
        target_threshold_rolling = target_threshold_rolling.fillna(0)
        y_specific = (df_ml_metadata[retorno_col] > target_threshold_rolling).astype(int)
        y_specific = y_specific.loc[X_final_scaled_pca.index].dropna()
        X_aligned = X_final_scaled_pca.loc[y_specific.index]
        logging.debug(f"[{target_name}] Dados alinhados. Shape X: {X_aligned.shape}, Shape y: {y_specific.shape}")
        if len(X_aligned) < WFCV_MIN_TRAIN_SIZE:
             logging.warning(f"[{target_name}] ❌ Pulando: Dados insuficientes ({len(X_aligned)}) para o target.")
             continue
        logging.info(f"[{target_name}] [MI] Calculando Informação Mútua para {len(X_aligned.columns)} features...")
        X_aligned = X_aligned.replace([np.inf, -np.inf], 0)
        mi_scores = mutual_info_classif(X_aligned, y_specific)
        mi_df = pd.DataFrame({'feature': X_aligned.columns, 'mi_score': mi_scores})
        mi_df = mi_df.sort_values('mi_score', ascending=False)
        k_features = max(MI_FEATURE_MIN_K, int(len(X_aligned.columns) * MI_FEATURE_PERCENTILE))
        k_features = min(k_features, len(X_aligned.columns))
        top_k_features = mi_df.head(k_features)['feature'].values
        X_mi_selected = X_aligned[top_k_features]
        all_mi_scores[target_name] = mi_df
        logging.info(f"[{target_name}] [MI] Selecionando {len(top_k_features)} features.")
        logging.debug(f"[{target_name}] [MI] Top 5 features: {list(mi_df.head(5)['feature'])}")
        df_resultados_ml, df_metadados_ml, df_previsoes_finais = rodar_wfc_com_hpo(
            X_mi_selected, y_specific, ticker, target_name,
            n_splits_wfc=N_SPLITS_WFCV,
            n_trials_hpo=N_TRIALS_HPO,
            modelos_config=modelos_config
        )
        df_shap_summary = gerar_explicabilidade_shap(
            X_mi_selected, y_specific, df_metadados_ml, target_name, N_SPLITS_WFCV, modelos_config
        )
        all_results_ml.append(df_resultados_ml)
        all_metadata_ml.append(df_metadados_ml)
        all_shap_summaries[target_name] = df_shap_summary.mean(axis=1)
        all_final_predictions[target_name] = df_previsoes_finais['y_pred_proba_stacking'].iloc[-1] if not df_previsoes_finais.empty else np.nan
    logging.info("Consolidando e salvando todos os artefatos...")
    df_resultados_ml_final = pd.concat(all_results_ml) if all_results_ml else pd.DataFrame()
    df_metadados_ml_final = pd.concat(all_metadata_ml) if all_metadata_ml else pd.DataFrame()
    df_mi_scores_final = pd.DataFrame(all_mi_scores) if all_mi_scores else pd.DataFrame()
    df_shap_summary_final = pd.DataFrame(all_shap_summaries) if all_shap_summaries else pd.DataFrame()
    df_ml_results_final = pd.DataFrame.from_dict(all_final_predictions, orient='index', columns=['last_stacking_proba'])
    df_ml_results_final.index.name = 'target_name'
    salvar_dados_separados(
        ticker,
        df_final_tecnico,
        df_fundamentos_estatico,
        df_ml_results_final,
        df_resultados_ml_final,
        df_metadados_ml_final,
        df_mi_scores_final,
        df_shap_summary_final,
        gcs_bucket
    )
    end_time_ativo = time.time()
    logging.info(f"--- ✅ Concluído: {ticker} em {end_time_ativo - start_time_ativo:.2f}s ---")
    return True

def salvar_dados_separados(
    ticker, df_historico_tecnico, df_fundamentos_estatico,
    df_ml_results_final,
    df_resultados_ml, df_metadados_ml, df_mi_scores, df_shap_summary,
    gcs_bucket
    ):
    gcs_path_base = GCS_BASE_PATH
    logging.debug(f"[Salvar] Salvando 7 artefatos para {ticker} em {gcs_path_base}...")
    try:
        salvar_dataframe_gcs(df_historico_tecnico, gcs_bucket, gcs_path_base, ticker, '1_dados_tecnicos')
        salvar_dataframe_gcs(df_fundamentos_estatico, gcs_bucket, gcs_path_base, ticker, '2_dados_fundamentus_estatico')
        salvar_dataframe_gcs(df_ml_results_final, gcs_bucket, gcs_path_base, ticker, '3_ml_previsoes_finais')
        salvar_dataframe_gcs(df_resultados_ml, gcs_bucket, gcs_path_base, ticker, '4_ml_resultados_wfcv')
        salvar_dataframe_gcs(df_metadados_ml, gcs_bucket, gcs_path_base, ticker, '5_ml_metadados_hpo')
        salvar_dataframe_gcs(df_mi_scores, gcs_bucket, gcs_path_base, ticker, '6_ml_mi_scores')
        salvar_dataframe_gcs(df_shap_summary, gcs_bucket, gcs_path_base, ticker, '7_ml_shap_summary')
        if GCS_DISPONIVEL and gcs_bucket:
             logging.info(f"[GCS] ✅ Todos os 7 artefatos de {ticker} salvos em gs://{GCS_BUCKET_NAME}/{gcs_path_base}/{ticker}/")
        else:
             logging.info(f"[Local] ✅ Todos os 7 artefatos de {ticker} gerados (salvamento GCS desativado).")
    except Exception as e:
        logging.error(f"[Salvar] ❌ Erro ao salvar dados de {ticker}: {e}", exc_info=True)
        traceback.print_exc()

# ----------------------------------------------------------------------------
# 🏁 BLOCO DE EXECUÇÃO (MAIN) (V25.1)
# ----------------------------------------------------------------------------

if __name__ == "__main__":

    # [V25.3] Configura o logging básico para usar STDOUT, evitando conflito com TQDM
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    logging.info("=" * 70)
    logging.info(" GERADOR DE DADOS FINANCEIROS V25.3 - DEBUG CORRIGIDO (STDOUT)")
    logging.info("=" * 70)
    logging.warning("⚠️ AVISO: Este script (V25) usa dados fundamentalistas ESTÁTICOS (Fundamentus).")
    logging.warning("⚠️ ISSO CAUSA LOOK-AHEAD BIAS E INVALIDA O BACKTEST.")
    logging.info("-" * 70)
    
    # [ALTERADO] Bloco principal modificado para Cloud Run Job
    
    # 1. Inicializa Clientes (Autenticação Automática)
    logging.info("Inicializando clientes GCP (autenticação via Conta de Serviço)...")
    storage_client = storage.Client()
    gcs_bucket = storage_client.get_bucket(GCS_BUCKET_NAME)
    logging.info(f"✅ Conectado ao GCS Bucket: {GCS_BUCKET_NAME}")

    # 2. Coleta Dados Macro (executado em todos os jobs, mas o YFinance tem cache)
    logging.info("🌐 Coletando Dados Macro Expandidos (S&P 500, VIX, Óleo...)...")
    df_dados_macro = obter_dados_macro_avancados(start_date=DATA_INICIO_MACRO)
    if df_dados_macro.empty:
        logging.critical("❌ Erro crítico: Não foi possível obter dados macro. Saindo.")
        sys.exit(1) # Falha o job

    # 3. Pega o Ticker da Variável de Ambiente
    ticker = os.environ.get('TICKER_ID')
    
    if not ticker:
        logging.critical("❌ Erro Crítico: Variável de ambiente 'TICKER_ID' não definida.")
        logging.critical("Este script deve ser executado com uma variável de ambiente (ex: TICKER_ID=PETR4 python gerador_financeiro_cloudrun.py)")
        sys.exit(1) # Falha o job

    logging.info(f"✅ Job iniciado para processar Ticker: {ticker}")

    # 4. Processa o ÚNICO ativo
    try:
        if processar_ativo_completo(ticker, df_dados_macro, gcs_bucket):
            logging.info(f"🏁 PROCESSAMENTO CONCLUÍDO COM SUCESSO para {ticker} 🏁")
            sys.exit(0) # Sucesso
        else:
            logging.error(f"🏁 PROCESSAMENTO FALHOU para {ticker} 🏁")
            sys.exit(1) # Falha
    except Exception as e:
        logging.critical(f"\n--- ❌ ERRO INESPERADO (Crítico) em {ticker} ---")
        logging.critical(f"Erro: {e}", exc_info=True)
        traceback.print_exc()
        logging.critical(f"--------------------------------------------------")
        sys.exit(1) # Falha