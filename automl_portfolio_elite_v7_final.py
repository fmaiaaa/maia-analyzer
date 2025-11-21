# -*- coding: utf-8 -*-
"""
=============================================================================
SISTEMA DE PORTFÓLIOS ADAPTATIVOS - OTIMIZAÇÃO QUANTITATIVA (VERSÃO NATIVA)
=============================================================================

Versão Adaptada: 8.10.0 (Integração Biblioteca Mercados)
- Macroeconomia: Taxa Livre de Risco (CDI/Selic) dinâmica via `mercados.bcb`.
- Inflação: Coleta de IPCA via `mercados.ibge` para cálculo de Retorno Real.
- Dados Técnicos: yfinance (OHLCV em tempo real).
- Dados Fundamentalistas: pynvest (Scraping avançado do Fundamentus).
- Integração: Uso híbrido para maximizar robustez e profundidade de dados.

Dependências Extras Necessárias:
pip install yfinance pynvest plotly scipy scikit-learn pandas numpy mercados
=============================================================================
"""

# --- 1. CORE LIBRARIES & UTILITIES ---
import warnings
import numpy as np
import pandas as pd
import sys
import time
import datetime
from datetime import timedelta
import traceback
import json

# --- 2. SCIENTIFIC / STATISTICAL TOOLS ---
from scipy.optimize import minimize

# --- 3. STREAMLIT & VISUALIZATION ---
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- 4. DATA SOURCES ---
import yfinance as yf

# Pynvest para Fundamentos
try:
    from pynvest.scrappers.fundamentus import Fundamentus
except ImportError:
    warnings.warn("Biblioteca 'pynvest' não encontrada. Instale com 'pip install pynvest'.")

# Mercados para Macroeconomia e Dados Oficiais (BCB, IBGE, B3)
try:
    from mercados.bcb import BancoCentral
    from mercados.ibge import IBGE
    from mercados.b3 import B3
    MERCADOS_AVAILABLE = True
except ImportError:
    MERCADOS_AVAILABLE = False
    warnings.warn("Biblioteca 'mercados' não encontrada. Instale com 'pip install mercados'. Usando valores fallback.")

# --- 5. MACHINE LEARNING (SCIKIT-LEARN) ---
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --- 6. CONFIGURATION ---
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURAÇÕES E CONSTANTES GLOBAIS
# =============================================================================

PERIODO_DADOS = '5y'
MIN_DIAS_HISTORICO = 200 
NUM_ATIVOS_PORTFOLIO = 5
SCORE_PERCENTILE_THRESHOLD = 0.75 

# Taxa Livre de Risco Padrão (Fallback caso a API do BCB falhe)
TAXA_LIVRE_RISCO_FALLBACK = 0.1075 

# =============================================================================
# 2. PONDERAÇÕES E REGRAS DE OTIMIZAÇÃO
# =============================================================================

WEIGHT_PERFORMANCE = 0.35
WEIGHT_FUNDAMENTAL = 0.40
WEIGHT_TECHNICAL = 0.25
WEIGHT_ML = 0.00 

PESO_MIN = 0.05
PESO_MAX = 0.35

# =============================================================================
# 3. LISTAS DE ATIVOS (IBOVESPA)
# =============================================================================

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

# =============================================================================
# 4. CLASSE: COLETOR MACROECONÔMICO (MERCADOS)
# =============================================================================

class ColetorMacroEconomico:
    """
    Responsável por coletar dados oficiais do governo (BCB, IBGE) usando a biblioteca 'mercados'.
    Fornece o contexto macroeconômico (Juros, Inflação) para a otimização do portfólio.
    """
    def __init__(self):
        self.taxa_selic_atual = TAXA_LIVRE_RISCO_FALLBACK
        self.ipca_acumulado_12m = 0.045 # Fallback ~4.5%
        self.data_referencia = datetime.date.today()
        
        if MERCADOS_AVAILABLE:
            try:
                self.bc = BancoCentral()
                self.ibge = IBGE()
            except Exception as e:
                print(f"Erro ao inicializar clientes Mercados: {e}")
                self.bc = None
                self.ibge = None
        else:
            self.bc = None
            self.ibge = None

    def atualizar_dados_macro(self):
        """Coleta Selic/CDI e IPCA atualizados."""
        if not self.bc or not self.ibge:
            return False

        try:
            # 1. Coletar CDI/Selic (Série 432 ou 11 do SGS - Usando atalho da lib se houver, ou busca direta)
            # A lib mercados.bcb tem serie_temporal. Vamos tentar pegar a Selic Meta ou CDI.
            # CDI Diário anualizado costuma ser a referência.
            
            # Vamos pegar o CDI acumulado dos últimos 12 meses ou a meta atual?
            # Para otimização de Sharpe, usa-se a taxa livre de risco ATUAL (anualizada).
            # Vamos tentar pegar a série 432 (Taxa de juros - Meta Selic definida pelo Copom % a.a.)
            
            # Nota: A biblioteca mercados usa bc.serie_temporal(codigo, inicio, fim)
            # 432: Meta Selic
            hoje = self.data_referencia
            inicio = hoje - timedelta(days=30)
            
            serie_selic = self.bc.serie_temporal(432, inicio=inicio, fim=hoje)
            
            if serie_selic:
                # Pega o último valor disponível
                ultimo_dado = list(serie_selic)[-1]
                self.taxa_selic_atual = float(ultimo_dado.valor) / 100.0
            
            # 2. Coletar IPCA (IBGE)
            # O método historico("IPCA") retorna a série.
            historico_ipca = list(self.ibge.historico("IPCA"))
            if historico_ipca:
                # Calcular acumulado 12 meses
                # A lista contém objetos Taxa(data, valor). Valor é o número índice.
                # Variação 12m = (Indice Atual / Indice 12m atrás) - 1
                
                if len(historico_ipca) > 12:
                    indice_atual = historico_ipca[-1].valor
                    indice_12m_atras = historico_ipca[-13].valor
                    self.ipca_acumulado_12m = float((indice_atual / indice_12m_atras) - 1)
                
            return True
            
        except Exception as e:
            st.warning(f"Erro ao atualizar dados macro com 'mercados': {e}. Usando fallback.")
            return False

    def get_taxa_livre_risco(self):
        return self.taxa_selic_atual

    def get_inflacao_real(self):
        return self.ipca_acumulado_12m

# Instância global
coletor_macro = ColetorMacroEconomico()

# =============================================================================
# 5. CLASSE: ANALISADOR DE PERFIL (MANTIDO)
# =============================================================================

SCORE_MAP_ORIGINAL = {'CT: Concordo Totalmente': 5, 'C: Concordo': 4, 'N: Neutro': 3, 'D: Discordo': 2, 'DT: Discordo Totalmente': 1}
SCORE_MAP_INV_ORIGINAL = {'CT: Concordo Totalmente': 1, 'C: Concordo': 2, 'N: Neutro': 3, 'D: Discordo': 4, 'DT: Discordo Totalmente': 5}
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

class AnalisadorPerfilInvestidor:
    def __init__(self):
        self.nivel_risco = ""
        self.horizonte_tempo = ""
    
    def determinar_nivel_risco(self, pontuacao: int) -> str:
        if pontuacao <= 46: return "CONSERVADOR"
        elif pontuacao <= 67: return "INTERMEDIÁRIO"
        elif pontuacao <= 88: return "MODERADO"
        elif pontuacao <= 109: return "MODERADO-ARROJADO"
        else: return "AVANÇADO"
    
    def determinar_horizonte_ml(self, liquidez_key: str, objetivo_key: str) -> tuple[str, int]:
        time_map = { 'A': 5, 'B': 20, 'C': 30 }
        final_lookback = max( time_map.get(liquidez_key, 5), time_map.get(objetivo_key, 5) )
        
        if final_lookback >= 30: return "LONGO PRAZO", 30
        elif final_lookback >= 20: return "MÉDIO PRAZO", 20
        else: return "CURTO PRAZO", 5
    
    def calcular_perfil(self, respostas: dict) -> tuple[str, str, int, int]:
        score_risk = SCORE_MAP_ORIGINAL.get(respostas['risk_accept'], 3)
        s_gain = SCORE_MAP_ORIGINAL.get(respostas['max_gain'], 3)
        s_stab = SCORE_MAP_INV_ORIGINAL.get(respostas['stable_growth'], 3)
        s_loss = SCORE_MAP_INV_ORIGINAL.get(respostas['avoid_loss'], 3)
        s_know = SCORE_MAP_CONHECIMENTO_ORIGINAL.get(respostas['level'], 3)
        s_reac = SCORE_MAP_REACTION_ORIGINAL.get(respostas['reaction'], 3)
        
        pontuacao = (score_risk * 5 + s_gain * 5 + s_stab * 5 + s_loss * 5 + s_know * 3 + s_reac * 3)
        
        liquidez_key = respostas['liquidity'][0] if respostas['liquidity'] else 'C'
        objetivo_key = respostas['time_purpose'][0] if respostas['time_purpose'] else 'C'
        
        nivel = self.determinar_nivel_risco(pontuacao)
        horizonte, lookback = self.determinar_horizonte_ml(liquidez_key, objetivo_key)
        
        return nivel, horizonte, lookback, pontuacao

# =============================================================================
# 6. ENGENHARIA DE FEATURES E CÁLCULOS TÉCNICOS
# =============================================================================

class CalculadoraTecnica:
    """Calcula indicadores técnicos e métricas estatísticas localmente."""
    
    @staticmethod
    def calcular_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calcular_macd(series, slow=26, fast=12, signal=9):
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        diff = macd - signal_line
        return macd, signal_line, diff

    @staticmethod
    def calcular_bollinger(series, window=20, num_std=2):
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        upper = rolling_mean + (rolling_std * num_std)
        lower = rolling_mean - (rolling_std * num_std)
        width = (upper - lower) / rolling_mean
        return upper, lower, width

    @staticmethod
    def normalizar(serie: pd.Series, maior_melhor: bool = True) -> pd.Series:
        if serie.isnull().all(): return pd.Series(0.5, index=serie.index)
        # Clip para remover outliers extremos que distorcem o score
        q_low = serie.quantile(0.05)
        q_high = serie.quantile(0.95)
        serie_clipped = serie.clip(q_low, q_high)
        
        min_val, max_val = serie_clipped.min(), serie_clipped.max()
        if max_val == min_val: return pd.Series(0.5, index=serie.index)
        
        norm = (serie_clipped - min_val) / (max_val - min_val)
        return norm if maior_melhor else 1 - norm

# =============================================================================
# 7. COLETA DE DADOS (YFINANCE + PYNVEST + MERCADOS)
# =============================================================================

class ColetorDadosNativo:
    """Coleta dados via Yahoo Finance, Pynvest e usa Mercados para validação."""
    
    def __init__(self, periodo=PERIODO_DADOS):
        self.periodo = periodo
        self.dados_por_ativo = {}
        self.dados_fundamentalistas = pd.DataFrame()
        self.metricas_performance = pd.DataFrame()
        self.ativos_sucesso = []
        self.pynvest_scrapper = Fundamentus()
        
        # Dados Macro
        self.taxa_livre_risco = coletor_macro.get_taxa_livre_risco()

    def _limpar_valor_pynvest(self, valor):
        """Converte strings formatadas (ex: '10,5%') para float."""
        if isinstance(valor, (int, float)):
            return float(valor)
        if isinstance(valor, str):
            valor = valor.replace('%', '').replace('.', '').replace(',', '.')
            try:
                return float(valor)
            except ValueError:
                return np.nan
        return np.nan

    def coletar_fundamentos_pynvest(self, lista_ativos, progress_callback=None):
        """Itera sobre os ativos coletando dados ricos do Pynvest."""
        dados_consolidados = []
        
        # Mapeamento de colunas Pynvest -> Padrão Interno
        mapa_cols = {
            'vlr_ind_p_sobre_l': 'pe_ratio',
            'vlr_ind_p_sobre_vp': 'pb_ratio',
            'vlr_ind_roe': 'roe',
            'vlr_ind_roic': 'roic',
            'vlr_ind_div_yield': 'div_yield',
            'vlr_ind_ev_sobre_ebitda': 'ev_ebitda',
            'vlr_ind_margem_liq': 'net_margin',
            'vlr_ind_margem_ebit': 'operating_margin',
            'vlr_ind_divida_bruta_sobre_patrim': 'debt_to_equity',
            'vlr_liquidez_corr': 'current_ratio',
            'nome_setor': 'sector',
            'nome_subsetor': 'industry',
            'vlr_mercado': 'market_cap',
            'pct_cresc_rec_liq_ult_5a': 'revenue_growth_5y',
            'vol_med_neg_2m': 'liquidity'
        }

        total = len(lista_ativos)
        for i, ticker in enumerate(lista_ativos):
            ticker_sem_sa = ticker.replace('.SA', '')
            
            if progress_callback and i % 5 == 0: 
                progress_callback(int((i / total) * 40) + 10, f"Coletando fundamentos: {ticker}...")

            try:
                df_ativo = self.pynvest_scrapper.coleta_indicadores_de_ativo(ticker_sem_sa)
                
                if df_ativo is not None and not df_ativo.empty:
                    row = df_ativo.iloc[0].to_dict()
                    dados_limpos = {'Ticker': ticker}
                    
                    for col_pynvest, col_interna in mapa_cols.items():
                        val = row.get(col_pynvest)
                        
                        if col_interna in ['sector', 'industry']:
                            dados_limpos[col_interna] = str(val) if val else 'Desconhecido'
                        else:
                            val_float = self._limpar_valor_pynvest(val)
                            
                            if col_interna in ['roe', 'roic', 'div_yield', 'net_margin', 'operating_margin', 'revenue_growth_5y']:
                                dados_limpos[col_interna] = val_float / 100.0 if pd.notnull(val_float) else 0.0
                            else:
                                dados_limpos[col_interna] = val_float if pd.notnull(val_float) else 0.0
                    
                    dados_limpos['price_sales'] = self._limpar_valor_pynvest(row.get('vlr_ind_psr'))
                    dados_limpos['asset_turnover'] = self._limpar_valor_pynvest(row.get('vlr_ind_giro_ativos'))
                    
                    dados_consolidados.append(dados_limpos)
                    
            except Exception as e:
                continue
        
        if not dados_consolidados:
            return pd.DataFrame()
            
        df_final = pd.DataFrame(dados_consolidados).set_index('Ticker')
        return df_final

    def processar_ativo(self, ticker, df_prices, df_fund_row):
        """Calcula indicadores técnicos e une com fundamentos."""
        if len(df_prices) < MIN_DIAS_HISTORICO: return None

        df = df_prices.copy()
        df['returns'] = df['Close'].pct_change()
        
        # Cálculos Técnicos Locais
        df['rsi_14'] = CalculadoraTecnica.calcular_rsi(df['Close'], 14)
        df['macd'], df['macd_signal'], df['macd_diff'] = CalculadoraTecnica.calcular_macd(df['Close'])
        df['bb_upper'], df['bb_lower'], df['bb_width'] = CalculadoraTecnica.calcular_bollinger(df['Close'])
        df['volatility_30d'] = df['returns'].rolling(21).std() * np.sqrt(252)
        
        # Performance Histórica
        total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
        ann_return = df['returns'].mean() * 252
        ann_vol = df['returns'].std() * np.sqrt(252)
        
        # Sharpe (usando a taxa dinâmica do ColetorMacro)
        if ann_vol > 0.0001:
            sharpe = (ann_return - self.taxa_livre_risco) / ann_vol
        else:
            sharpe = 0

        # Drawdown
        cum_ret = (1 + df['returns']).cumprod()
        peak = cum_ret.expanding().max()
        dd = (cum_ret - peak) / peak
        max_dd = dd.min()

        if not df_fund_row.empty:
            for col, val in df_fund_row.items():
                df.loc[df.index[-1], col] = val

        metrics = {
            'sharpe': sharpe,
            'retorno_anual': ann_return,
            'volatilidade_anual': ann_vol,
            'max_drawdown': max_dd,
            'garch_volatility': ann_vol 
        }
        
        df['ML_Proba'] = 0.5
        df['ML_Confidence'] = 0.0

        return df, metrics

    def coletar_tudo(self, simbolos, progress_bar=None):
        """Fluxo principal de coleta."""
        
        # 0. Atualiza Macroeconomia (via Mercados)
        if progress_bar: progress_bar.progress(5, "Atualizando dados macroeconômicos (BCB/IBGE)...")
        coletor_macro.atualizar_dados_macro()
        self.taxa_livre_risco = coletor_macro.get_taxa_livre_risco()

        # 1. Fundamentos (Scraping individual via Pynvest)
        df_fund_all = self.coletar_fundamentos_pynvest(simbolos, progress_bar.progress if progress_bar else None)
        
        # 2. Preços (Yfinance Batch é rápido)
        if progress_bar: progress_bar.progress(60, "Baixando cotações históricas (YFinance)...")
        try:
            data_yf = yf.download(simbolos, period=self.periodo, group_by='ticker', auto_adjust=True, progress=False, threads=True)
        except Exception as e:
            st.error(f"Erro no download yfinance: {e}")
            return False

        lista_fundamentalistas = []
        metricas_simples_list = []

        if progress_bar: progress_bar.progress(80, "Processando indicadores técnicos...")
        
        for simbolo in simbolos:
            try:
                if len(simbolos) == 1:
                    df_ativo = data_yf
                else:
                    if simbolo in data_yf:
                        df_ativo = data_yf[simbolo]
                    else:
                        continue
                
                df_ativo = df_ativo.dropna(subset=['Close'])
                if df_ativo.empty: continue
                
                fund_row = df_fund_all.loc[simbolo] if simbolo in df_fund_all.index else pd.Series()
                
                resultado = self.processar_ativo(simbolo, df_ativo, fund_row)
                if not resultado: continue
                
                df_final, metrics = resultado
                
                self.dados_por_ativo[simbolo] = df_final
                self.ativos_sucesso.append(simbolo)
                
                fund_dict = fund_row.to_dict()
                fund_dict['Ticker'] = simbolo
                fund_dict.update(metrics) 
                lista_fundamentalistas.append(fund_dict)
                
                metricas_simples_list.append({
                    'Ticker': simbolo,
                    **metrics
                })
                
            except Exception as e:
                continue

        if not lista_fundamentalistas: return False
        
        self.dados_fundamentalistas = pd.DataFrame(lista_fundamentalistas).set_index('Ticker')
        self.metricas_performance = pd.DataFrame(metricas_simples_list).set_index('Ticker')
        
        return True
        
    def coletar_ativo_unico(self, ativo):
        df_fund_single = self.coletar_fundamentos_pynvest([ativo])
        fund_row = df_fund_single.iloc[0] if not df_fund_single.empty else pd.Series()
        
        df_yf = yf.download(ativo, period=self.periodo, auto_adjust=True, progress=False)
        
        if df_yf.empty: return None, None
        
        res = self.processar_ativo(ativo, df_yf, fund_row)
        if res:
            return res[0], res[1] 
        return None, None

# =============================================================================
# 8. OTIMIZADOR DE PORTFÓLIO (MPT)
# =============================================================================

class OtimizadorPortfolioAvancado:
    def __init__(self, returns_df, taxa_livre_risco):
        self.returns = returns_df
        self.mean_returns = returns_df.mean() * 252
        self.cov_matrix = returns_df.cov() * 252
        self.num_ativos = len(returns_df.columns)
        self.taxa_livre_risco = taxa_livre_risco

    def otimizar(self, estrategia='MaxSharpe'):
        if self.num_ativos == 0: return {}
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((PESO_MIN, PESO_MAX) for _ in range(self.num_ativos))
        initial_guess = np.array([1/self.num_ativos] * self.num_ativos)
        
        def portfolio_stats(weights):
            ret = np.dot(weights, self.mean_returns)
            vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            return ret, vol

        if estrategia == 'MinVolatility':
            fun = lambda w: portfolio_stats(w)[1]
        else: # MaxSharpe
            fun = lambda w: -((portfolio_stats(w)[0] - self.taxa_livre_risco) / portfolio_stats(w)[1])

        res = minimize(fun, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if res.success:
            return {k: v for k, v in zip(self.returns.columns, res.x)}
        return {k: 1.0/self.num_ativos for k in self.returns.columns}

# =============================================================================
# 9. CONSTRUTOR DE PORTFÓLIO
# =============================================================================

class ConstrutorPortfolioAutoML:
    def __init__(self, valor_investimento):
        self.valor_investimento = valor_investimento
        self.coletor = ColetorDadosNativo()
        self.dados_por_ativo = {}
        self.dados_fundamentalistas = pd.DataFrame()
        self.metricas_performance = pd.DataFrame()
        self.ativos_selecionados = []
        self.alocacao_portfolio = {}
        self.metricas_portfolio = {}
        self.pesos_atuais = {}
        self.scores_combinados = pd.DataFrame()
        self.metodo_alocacao_atual = ""
        self.justificativas_selecao = {}

    def executar_pipeline(self, simbolos, perfil_inputs, progress_bar=None):
        sucesso = self.coletor.coletar_tudo(simbolos, progress_bar)
        if not sucesso: return False
        
        self.dados_por_ativo = self.coletor.dados_por_ativo
        self.dados_fundamentalistas = self.coletor.dados_fundamentalistas
        self.metricas_performance = self.coletor.metricas_performance
        
        if progress_bar: progress_bar.progress(90, "Calculando Scores Avançados (ROIC, EV/EBITDA)...")
        self.pontuar_ativos(perfil_inputs.get('time_horizon', 'MÉDIO PRAZO'))
        
        if progress_bar: progress_bar.progress(95, "Selecionando Melhores Ativos...")
        self.selecionar_ativos()
        
        if progress_bar: progress_bar.progress(98, "Otimizando Pesos (Markowitz com Macro)...")
        self.otimizar_alocacao(perfil_inputs.get('risk_level', 'MODERADO'))
        
        self.calcular_metricas_finais()
        self.gerar_justificativas()
        if progress_bar: progress_bar.progress(100, "Concluído!")
        
        return True

    def pontuar_ativos(self, horizonte):
        w_perf, w_fund, w_tech = WEIGHT_PERFORMANCE, WEIGHT_FUNDAMENTAL, WEIGHT_TECHNICAL
        if horizonte == 'CURTO PRAZO': w_perf, w_fund, w_tech = 0.4, 0.2, 0.4
        elif horizonte == 'LONGO PRAZO': w_perf, w_fund, w_tech = 0.2, 0.7, 0.1
        
        self.pesos_atuais = {'Performance': w_perf, 'Fundamentos': w_fund, 'Técnicos': w_tech, 'ML': 0}

        df = self.dados_fundamentalistas.copy()
        
        s_perf = CalculadoraTecnica.normalizar(df['sharpe'], True) * w_perf
        
        s_pe = CalculadoraTecnica.normalizar(df['pe_ratio'].replace(0, np.nan), False) 
        s_evebitda = CalculadoraTecnica.normalizar(df.get('ev_ebitda', pd.Series(0)).replace(0, np.nan), False)
        
        s_roe = CalculadoraTecnica.normalizar(df['roe'], True)
        s_roic = CalculadoraTecnica.normalizar(df.get('roic', pd.Series(0)), True)
        
        s_fund = (s_pe * 0.2 + s_evebitda * 0.3 + s_roe * 0.2 + s_roic * 0.3) * w_fund
        
        tech_vals = {t: {} for t in df.index}
        for t, d in self.dados_por_ativo.items():
            tech_vals[t]['rsi'] = d['rsi_14'].iloc[-1] if 'rsi_14' in d else 50
            tech_vals[t]['macd_diff'] = d['macd_diff'].iloc[-1] if 'macd_diff' in d else 0
        
        df_tech = pd.DataFrame(tech_vals).T
        s_rsi = CalculadoraTecnica.normalizar(df_tech['rsi'], False) 
        s_macd = CalculadoraTecnica.normalizar(df_tech['macd_diff'], True)
        s_tech = (s_rsi * 0.5 + s_macd * 0.5) * w_tech
        
        df['total_score'] = s_perf + s_fund + s_tech
        
        self.scores_combinados = df.join(df_tech)
        self.scores_combinados = self.scores_combinados.sort_values('total_score', ascending=False)

    def selecionar_ativos(self):
        top_n = self.scores_combinados.head(NUM_ATIVOS_PORTFOLIO * 3)
        
        try:
            features_cols = ['sharpe', 'pe_ratio', 'roe', 'ev_ebitda', 'net_margin', 'debt_to_equity']
            cols_to_use = [c for c in features_cols if c in top_n.columns]
            
            features = top_n[cols_to_use].fillna(0)
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            kmeans = KMeans(n_clusters=NUM_ATIVOS_PORTFOLIO, n_init='auto', random_state=42).fit(features_scaled)
            top_n['cluster'] = kmeans.labels_
            
            selecionados = []
            for c in range(NUM_ATIVOS_PORTFOLIO):
                cluster_assets = top_n[top_n['cluster'] == c]
                if not cluster_assets.empty:
                    selecionados.append(cluster_assets['total_score'].idxmax())
            
            while len(selecionados) < NUM_ATIVOS_PORTFOLIO:
                for cand in top_n.index:
                    if cand not in selecionados:
                        selecionados.append(cand)
                        break
            
            self.ativos_selecionados = selecionados[:NUM_ATIVOS_PORTFOLIO]
            
        except Exception as e:
            self.ativos_selecionados = self.scores_combinados.head(NUM_ATIVOS_PORTFOLIO).index.tolist()

    def otimizar_alocacao(self, risco):
        returns = pd.DataFrame({t: self.dados_por_ativo[t]['returns'] for t in self.ativos_selecionados}).dropna()
        # Passa a taxa livre de risco dinâmica do coletor macro
        opt = OtimizadorPortfolioAvancado(returns, self.coletor.taxa_livre_risco)
        
        method = 'MinVolatility' if risco in ['CONSERVADOR', 'INTERMEDIÁRIO'] else 'MaxSharpe'
        pesos = opt.otimizar(method)
        
        self.metodo_alocacao_atual = f"Otimização {method} (MPT)"
        self.alocacao_portfolio = {
            t: {'weight': w, 'amount': w * self.valor_investimento}
            for t, w in pesos.items()
        }

    def calcular_metricas_finais(self):
        if not self.alocacao_portfolio: return
        
        w_dict = {k: v['weight'] for k, v in self.alocacao_portfolio.items()}
        returns = pd.DataFrame({t: self.dados_por_ativo[t]['returns'] for t in w_dict.keys()}).dropna()
        
        weights = np.array([w_dict[c] for c in returns.columns])
        port_ret = returns.dot(weights)
        
        cum_ret = (1 + port_ret).cumprod()
        dd = (cum_ret - cum_ret.expanding().max()) / cum_ret.expanding().max()
        
        ann_ret = port_ret.mean() * 252
        ann_vol = port_ret.std() * np.sqrt(252)
        
        # Calcula Retorno Real (Descontado o IPCA)
        ipca = coletor_macro.get_inflacao_real()
        real_return = ((1 + ann_ret) / (1 + ipca)) - 1
        
        self.metricas_portfolio = {
            'annual_return': ann_ret,
            'real_return': real_return,
            'annual_volatility': ann_vol,
            'sharpe_ratio': (ann_ret - self.coletor.taxa_livre_risco) / ann_vol if ann_vol > 0 else 0,
            'max_drawdown': dd.min(),
            'total_investment': self.valor_investimento
        }

    def gerar_justificativas(self):
        for t in self.ativos_selecionados:
            row = self.scores_combinados.loc[t]
            
            def safe_fmt(val, is_pct=False):
                if pd.isna(val): return "N/A"
                return f"{val*100:.1f}%" if is_pct else f"{val:.1f}"

            self.justificativas_selecao[t] = (
                f"Score: {row['total_score']:.2f} | "
                f"EV/EBITDA: {safe_fmt(row.get('ev_ebitda'))} | "
                f"ROIC: {safe_fmt(row.get('roic'), True)} | "
                f"P/L: {safe_fmt(row.get('pe_ratio'))}"
            )

# =============================================================================
# 10. INTERFACE STREAMLIT
# =============================================================================

def configurar_pagina():
    st.set_page_config(page_title="Portfolio Quant Mercados", layout="wide", page_icon="📈")
    st.markdown("""
    <style>
        .main-header {font-size: 2rem; color: #111; text-align: center; border-bottom: 2px solid #ddd; padding: 10px;}
        .stMetric {background-color: #f9f9f9; padding: 10px; border-radius: 5px; border: 1px solid #eee;}
    </style>
    """, unsafe_allow_html=True)

def aba_analise_individual():
    st.markdown("## 🔍 Análise Individual (Pynvest Data)")
    
    ativo = st.text_input("Digite o Ticker (ex: VALE3.SA)", value="VALE3.SA").upper()
    if not ativo.endswith(".SA"): ativo += ".SA"
    
    if st.button("Analisar Ativo"):
        coletor = ColetorDadosNativo()
        with st.spinner(f"Coletando dados de {ativo} via Pynvest & YFinance..."):
            df, fund = coletor.coletar_ativo_unico(ativo)
        
        if df is None:
            st.error("Ativo não encontrado ou sem dados suficientes.")
            return
            
        st.subheader("Indicadores de Valuation e Rentabilidade")
        col1, col2, col3, col4 = st.columns(4)
        
        def fmt(val, pct=False):
            if val is None or pd.isna(val): return "N/A"
            return f"{val*100:.1f}%" if pct else f"{val:.2f}"
        
        col1.metric("P/L", fmt(fund.get('pe_ratio')))
        col2.metric("EV/EBITDA", fmt(fund.get('ev_ebitda')))
        col3.metric("ROIC", fmt(fund.get('roic'), True))
        col4.metric("ROE", fmt(fund.get('roe'), True))
        
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Div. Yield", fmt(fund.get('div_yield'), True))
        col6.metric("Margem Líq.", fmt(fund.get('net_margin'), True))
        col7.metric("Dívida/PL", fmt(fund.get('debt_to_equity')))
        col8.metric("Setor", fund.get('sector', 'N/A'))

        st.markdown("---")
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Preço'), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'), row=2, col=1)
        fig.update_layout(height=600, title=f"{ativo} - Dados Históricos")
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Ver Tabela Completa de Indicadores"):
            st.json(fund)

def main():
    if 'builder' not in st.session_state:
        st.session_state.builder = None
        st.session_state.ativos_para_analise = ATIVOS_IBOVESPA

    configurar_pagina()
    st.markdown('<h1 class="main-header">Sistema de Portfólios (Engine: Pynvest + Mercados)</h1>', unsafe_allow_html=True)
    
    # Sidebar com Status Macro
    st.sidebar.header("📊 Macroeconomia (Mercados)")
    if MERCADOS_AVAILABLE:
        coletor_macro.atualizar_dados_macro()
        st.sidebar.success("Conectado: BCB / IBGE")
        st.sidebar.metric("Taxa Selic/CDI (a.a.)", f"{coletor_macro.get_taxa_livre_risco()*100:.2f}%")
        st.sidebar.metric("IPCA 12m (IBGE)", f"{coletor_macro.get_inflacao_real()*100:.2f}%")
    else:
        st.sidebar.error("Biblioteca 'mercados' ausente.")
        st.sidebar.metric("Taxa Fallback", f"{TAXA_LIVRE_RISCO_FALLBACK*100:.2f}%")
    
    tab1, tab2, tab3 = st.tabs(["Configuração & Perfil", "Construção de Portfólio", "Análise Individual"])
    
    with tab1:
        st.info("Esta versão integra a biblioteca `mercados` para dados macroeconômicos oficiais e `pynvest` para fundamentos.")
        with st.form("perfil_form"):
            st.subheader("Calibração")
            risco = st.select_slider("Tolerância ao Risco", options=["CONSERVADOR", "MODERADO", "ARROJADO"], value="MODERADO")
            horizonte = st.radio("Horizonte", ["CURTO PRAZO", "MÉDIO PRAZO", "LONGO PRAZO"], index=1)
            capital = st.number_input("Capital (R$)", value=10000.0)
            if st.form_submit_button("Salvar Perfil"):
                st.session_state.profile = {'risk_level': risco, 'time_horizon': horizonte, 'investment': capital}
                st.success("Perfil Salvo!")

    with tab2:
        if 'profile' in st.session_state:
            st.write(f"Universo de Análise: {len(st.session_state.ativos_para_analise)} ativos.")
            if st.button("🚀 Gerar Portfólio"):
                builder = ConstrutorPortfolioAutoML(st.session_state.profile['investment'])
                progress = st.progress(0, "Iniciando Coleta Macro & Micro...")
                
                sucesso = builder.executar_pipeline(st.session_state.ativos_para_analise, st.session_state.profile, progress)
                
                if sucesso:
                    st.session_state.builder = builder
                    st.success("Portfólio Gerado com Sucesso!")
                else:
                    st.error("Falha na geração.")
            
            if st.session_state.builder:
                b = st.session_state.builder
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Retorno Esp. (a.a.)", f"{b.metricas_portfolio['annual_return']*100:.1f}%")
                c2.metric("Retorno REAL (a.a.)", f"{b.metricas_portfolio['real_return']*100:.1f}%", help="Descontado IPCA")
                c3.metric("Volatilidade (a.a.)", f"{b.metricas_portfolio['annual_volatility']*100:.1f}%")
                c4.metric("Sharpe", f"{b.metricas_portfolio['sharpe_ratio']:.2f}")
                
                st.subheader("Alocação Sugerida")
                df_alloc = pd.DataFrame([
                    {'Ativo': k, 'Peso (%)': v['weight']*100, 'Valor (R$)': v['amount']} 
                    for k, v in b.alocacao_portfolio.items()
                ])
                st.dataframe(df_alloc, use_container_width=True)
                
                fig = px.pie(df_alloc, values='Valor (R$)', names='Ativo', title="Distribuição do Portfólio")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Justificativas (Fundamentalistas)")
                for t, just in b.justificativas_selecao.items():
                    st.text(f"{t}: {just}")
                
        else:
            st.warning("Defina o perfil na Aba 1 primeiro.")

    with tab3:
        aba_analise_individual()

if __name__ == "__main__":
    main()
