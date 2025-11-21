# =============================================================================
# SISTEMA DE ANÁLISE E OTIMIZAÇÃO DE PORTFÓLIO DE INVESTIMENTOS
# =============================================================================
# Descrição: Sistema completo para análise de perfil de investidor, otimização
#            de portfólio usando teoria de Markowitz e Machine Learning
# Versão: 6.0.9 (CORREÇÃO DE AUC N/A: ML Lookback Máximo reduzido para 30 dias)
# =============================================================================

# =============================================================================
# IMPORTAÇÕES E CONFIGURAÇÕES INICIAIS
# =============================================================================

import warnings
import numpy as np
import pandas as pd
import subprocess
import sys
import streamlit as st
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.stats import zscore 

# Desativa avisos desnecessários para melhor experiência do usuário
warnings.filterwarnings("ignore")

# =============================================================================
# INSTALAÇÃO AUTOMÁTICA DE DEPENDÊNCIAS
# =============================================================================

# Dicionário com os pacotes necessários para o sistema funcionar
REQUIRED_PACKAGES = {
    'yfinance': 'yfinance',         # Coleta de dados financeiros
    'plotly': 'plotly',             # Visualizações interativas
    'streamlit': 'streamlit',       # Interface web
    'sklearn': 'scikit-learn',       # Machine Learning
    'scipy': 'scipy'                # Otimização matemática
}

def ensure_package(module_name, package_name):
    """
    Verifica se um pacote está instalado e instala automaticamente se necessário.
    
    Args:
        module_name: Nome do módulo para importação
        package_name: Nome do pacote no pip
    """
    try:
        __import__(module_name)
    except ImportError:
        print(f"Instalando {package_name}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', package_name])
        
        if 'streamlit' in sys.modules:
            st.warning(f"{package_name} foi instalado. Por favor, **reexecute** o servidor Streamlit.")
            st.stop()
        else:
            print(f"{package_name} instalado. Por favor, reexecute o script.")

# Tenta importar todas as bibliotecas necessárias
try:
    # Garante que todos os pacotes estão instalados
    for module, package in REQUIRED_PACKAGES.items():
        ensure_package(module.split('.')[0], package)
    
    # Importações de bibliotecas externas
    import yfinance as yf
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier 
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    
except Exception as e:
    if 'streamlit' in sys.modules:
        st.error(f"Erro ao carregar ou instalar bibliotecas: {e}")
    else:
        print(f"Erro ao carregar ou instalar bibliotecas: {e}")
    sys.exit(1)

# =============================================================================
# CONSTANTES GLOBAIS DO SISTEMA
# =============================================================================

# Taxa livre de risco (Selic atual - 14.75% ao ano)
RISK_FREE_RATE = 0.1475

# Peso mínimo que um único ativo deve ter no portfólio (10%)
MIN_WEIGHT = 0.10

# Peso máximo que um único ativo pode ter no portfólio (30%)
MAX_WEIGHT = 0.30

# Número de ativos no portfólio final
NUM_ASSETS_IN_PORTFOLIO = 5

# Período fixo de análise histórica (2 anos)
FIXED_ANALYSIS_PERIOD = '2y'

MIN_HISTORY_DAYS = 60

# =============================================================================
# MAPEAMENTOS DE PONTUAÇÃO DO QUESTIONÁRIO DE PERFIL
# =============================================================================

# Pontuação para respostas de concordância (1-5)
SCORE_MAP = {
    'CT: Concordo Totalmente': 5,
    'C: Concordo': 4,
    'N: Neutro': 3,
    'D: Discordo': 2,
    'DT: Discordo Totalmente': 1,
    'A': 5,
    'B': 3,
    'C': 1
}

# Pontuação invertida (para perguntas onde discordar = mais risco)
SCORE_MAP_INV = {
    'CT: Concordo Totalmente': 1,
    'C: Concordo': 2,
    'N: Neutro': 3,
    'D: Discordo': 4,
    'DT: Discordo Totalmente': 5
}

# Pontuação para nível de conhecimento
SCORE_MAP_CONHECIMENTO = {
    'A: Avançado': 5,
    'B: Intermediário': 3,
    'C: Iniciante': 1
}

# Pontuação para reação a perdas
SCORE_MAP_REACTION = {
    'A: Venderia': 1,
    'B: Manteria': 3,
    'C: Compraria mais': 5
}

# =============================================================================
# LISTA DE ATIVOS DISPONÍVEIS PARA ANÁLISE
# =============================================================================

ALL_ASSETS = [
    'ALOS3.SA', 'ABEV3.SA', 'ASAI3.SA', 'AURE3.SA', 'AZZA3.SA',
    'B3SA3.SA', 'BBSE3.SA', 'BBDC3.SA', 'BBDC4.SA', 'BRAP4.SA',
    'BBAS3.SA', 'BRKM5.SA', 'BRAV3.SA', 'BPAC11.SA', 'CXSE3.SA',
    'CEAB3.SA', 'CMIG4.SA', 'COGN3.SA', 'CPLE6.SA', 'CSAN3.SA',
    'CPFE3.SA', 'CMIN3.SA', 'CURY3.SA', 'CVCB3.SA', 'CYRE3.SA',
    'DIRR3.SA', 'ELET3.SA', 'ELET6.SA', 'EMBR3.SA', 'ENGI11.SA',
    'ENEV3.SA', 'EGIE3.SA', 'EQTL3.SA', 'FLRY3.SA', 'GGBR4.SA',
    'GOAU4.SA', 'HAPV3.SA', 'HYPE3.SA', 'IGTI11.SA', 'IRBR3.SA',
    'ISAE4.SA', 'ITSA4.SA', 'ITUB4.SA', 'KLBN11.SA', 'RENT3.SA',
    'LREN3.SA', 'MGLU3.SA', 'POMO4.SA', 'MBRF3.SA', 'BEEF3.SA',
    'MOTV3.SA', 'MRVE3.SA', 'MULT3.SA', 'NATU3.SA', 'PCAR3.SA',
    'PETR3.SA', 'PETR4.SA', 'RECV3.SA', 'PRIO3.SA', 'PSSA3.SA',
    'RADL3.SA', 'RAIZ4.SA', 'RDOR3.SA', 'RAIL3.SA', 'SBSP3.SA',
    'SANB11.SA', 'CSNA3.SA', 'SLCE3.SA', 'SMFT3.SA', 'SUZB3.SA',
    'TAEE11.SA', 'VIVT3.SA', 'TIMS3.SA', 'TOTS3.SA', 'UGPA3.SA',
    'USIM5.SA', 'VALE3.SA', 'VAMO3.SA', 'VBBR3.SA', 'VIVA3.SA',
    'WEGE3.SA', 'YDUQ3.SA'
]

# Lista padrão usada na inicialização
DEFAULT_ASSETS = ALL_ASSETS

# =============================================================================
# CLASSE: ANALISADOR DE PERFIL DO INVESTIDOR
# =============================================================================

class InvestorProfileAnalyzer:
    """
    Classe responsável por analisar o perfil de risco do investidor
    baseado em respostas de questionário e determinar horizonte temporal.
    """
    
    def __init__(self):
        """Inicializa o analisador com valores padrão."""
        self.risk_level = ""        # Perfil de risco (CONSERVADOR, MODERADO, etc.)
        self.time_horizon = ""      # Horizonte temporal (CURTO, MÉDIO, LONGO PRAZO)
        self.ml_lookback_days = 5    # Janela de previsão do modelo ML

    def determine_risk_level(self, score):
        """
        Traduz a pontuação numérica em perfil de risco qualitativo.
        
        Args:
            score: Pontuação total do questionário (máximo ~70)
            
        Returns:
            String com o perfil de risco
        """
        if score <= 18:
            return "CONSERVADOR"
        elif score <= 30:
            return "INTERMEDIÁRIO"
        elif score <= 45:
            return "MODERADO"
        elif score <= 60:
            return "MODERADO-ARROJADO"
        else:
            return "AVANÇADO"

    def determine_ml_lookback(self, liquidity_key, purpose_key):
        """
        Define a janela de previsão do modelo ML e o horizonte temporal
        baseado nas necessidades de liquidez e objetivo do investidor.
        
        Args:
            liquidity_key: Resposta sobre necessidade de liquidez (A/B/C)
            purpose_key: Resposta sobre prazo de investimento (A/B/C)
            
        Returns:
            Tupla (horizonte_temporal, dias_previsao)
        """
        # Mapeamento de respostas para dias de previsão
        time_map = {
            'A': 5,    # Curto prazo: 5 dias
            'B': 20,   # Médio prazo: 20 dias
            'C': 30    # ALTERADO: Longo prazo agora usa 30 dias para melhor AUC/Robustez
        }
        
        # Usa o maior período entre liquidez e objetivo
        final_lookback = max(
            time_map.get(liquidity_key, 5),
            time_map.get(purpose_key, 5)
        )
        
        # Define horizonte temporal baseado no lookback
        if final_lookback >= 30:
            self.time_horizon = "LONGO PRAZO"
            self.ml_lookback_days = 30 # Usando 30 dias
        elif final_lookback >= 20:
            self.time_horizon = "MÉDIO PRAZO"
            self.ml_lookback_days = 20
        else:
            self.time_horizon = "CURTO PRAZO"
            self.ml_lookback_days = 5

        return self.time_horizon, self.ml_lookback_days
        
    def calculate_profile(self, risk_answers):
        """
        Calcula o perfil completo do investidor a partir das respostas.
        
        Args:
            risk_answers: Dicionário com todas as respostas do questionário
            
        Returns:
            Tupla (perfil_risco, horizonte_temporal, dias_ml, pontuacao)
        """
        # Calcula pontuação ponderada de todas as respostas
        score = (
            SCORE_MAP[risk_answers['risk_accept']] * 5 +       # Aceitação de risco
            SCORE_MAP[risk_answers['max_gain']] * 5 +          # Busca por ganho máximo
            SCORE_MAP_INV[risk_answers['stable_growth']] * 5 + # Preferência por estabilidade
            SCORE_MAP_INV[risk_answers['avoid_loss']] * 5 +    # Aversão a perdas
            SCORE_MAP_CONHECIMENTO[risk_answers['level']] * 3 + # Nível de conhecimento
            SCORE_MAP_REACTION[risk_answers['reaction']] * 3   # Reação a quedas
        )
        
        # Determina perfil de risco baseado na pontuação
        risk_level = self.determine_risk_level(score)
        
        # Determina horizonte temporal e janela ML
        time_horizon, ml_lookback = self.determine_ml_lookback(
            risk_answers['liquidity'],
            risk_answers['time_purpose']
        )
        
        return risk_level, time_horizon, ml_lookback, score

# =============================================================================
# FUNÇÕES DE ESTILO E VISUALIZAÇÃO
# =============================================================================

def get_chart_template():
    """
    Retorna template de layout para gráficos Plotly com estilo
    minimalista e fonte Times New Roman.
    
    Returns:
        Dicionário com configurações de layout
    """
    return {
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'font': {
            'family': 'Times New Roman',
            'size': 12,
            'color': 'black'
        },
        'title': {
            'font': {
                'family': 'Times New Roman',
                'size': 16,
                'color': '#2c3e50',
                'weight': 'bold'
            },
            'x': 0.5,
            'xanchor': 'center',
            'text': '' 
        },
        'xaxis': {
            'showgrid': True,
            'gridcolor': 'lightgray',
            'showline': True,
            'linecolor': 'black',
            'linewidth': 1,
            'mirror': False,
            'tickfont': {'family': 'Times New Roman', 'color': 'black'},
            'title': {'font': {'family': 'Times New Roman', 'color': 'black'}},
            'zeroline': False
        },
        'yaxis': {
            'showgrid': True,
            'gridcolor': 'lightgray',
            'showline': True,
            'linecolor': 'black',
            'linewidth': 1,
            'mirror': False,
            'tickfont': {'family': 'Times New Roman', 'color': 'black'},
            'title': {'font': {'family': 'Times New Roman', 'color': 'black'}},
            'zeroline': False
        },
        'legend': {
            'font': {'family': 'Times New Roman', 'color': 'black'},
            'bgcolor': 'rgba(255, 255, 255, 0.8)',
            'bordercolor': 'lightgray',
            'borderwidth': 1,
            'orientation': "h", 
            'y': 1.02, 
            'x': 0.01 
        },
        'colorway': ['#2c3e50', '#7f8c8d', '#3498db', '#e74c3c', '#27ae60']
    }

# =============================================================================
# FUNÇÕES DE ANÁLISE TÉCNICA E FUNDAMENTALISTA
# =============================================================================

def calculate_technical_indicators(df):
    """
    Calcula indicadores técnicos para análise de ativos.
    
    Indicadores calculados:
    - RSI (Relative Strength Index): Força relativa do ativo
    - MACD (Moving Average Convergence Divergence): Convergência de médias móveis
    - Volatilidade: Desvio padrão anualizado dos retornos
    - Momentum: Taxa de mudança de preço
    - SMA 50 e 200: Médias móveis simples
    
    Args:
        df: DataFrame com dados históricos (deve conter coluna 'Close')
        
    Returns:
        DataFrame com indicadores técnicos adicionados
    """
    # Calcula retornos percentuais diários
    df['Returns'] = df['Close'].pct_change()
    
    # RSI (14 períodos)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # MACD (12, 26, 9)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Volatilidade anualizada (janela de 20 dias)
    df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
    
    # Momentum (10 períodos)
    df['Momentum'] = df['Close'] / df['Close'].shift(10) - 1
    
    # Médias móveis simples
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    # Remove linhas com valores NaN resultantes dos cálculos
    return df.dropna()

def calculate_risk_metrics(returns):
    """
    Calcula métricas de risco e retorno para uma série de retornos.
    
    Métricas calculadas:
    - Retorno anual: Média dos retornos anualizada
    - Volatilidade anual: Desvio padrão anualizado
    - Sharpe Ratio: Retorno ajustado ao risco
    - Max Drawdown: Maior queda do pico ao vale
    
    Args:
        returns: Série pandas com retornos diários
        
    Returns:
        Dicionário com as métricas calculadas
    """
    # Anualiza retorno e volatilidade (252 dias úteis por ano)
    annual_return = returns.mean() * 252
    annual_volatility = returns.std() * np.sqrt(252)
    
    # Sharpe Ratio: (Retorno - Taxa libre de risco) / Volatilidade
    sharpe_ratio = (annual_return - RISK_FREE_RATE) / annual_volatility if annual_volatility > 0 else 0
    
    # Calcula Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }

# =============================================================================
# CLASSE: OTIMIZADOR DE PORTFÓLIO (MARKOWITZ)
# =============================================================================

class PortfolioOptimizer:
    """
    Implementa otimização de portfólio usando teoria moderna de Markowitz.
    
    Estratégias disponíveis:
    - MinVolatility: Minimiza a volatilidade do portfólio
    - MaxSharpe: Maximiza o Sharpe Ratio (retorno ajustado ao risco)
    """

    def __init__(self, returns_df):
        """
        Inicializa o otimizador com dados de retornos.
        
        Args:
            returns_df: DataFrame com retornos diários dos ativos
        """
        self.returns = returns_df
        self.mean_returns = returns_df.mean() * 252  # Retornos anualizados
        self.cov_matrix = returns_df.cov() * 252     # Matriz de covariância anualizada
        self.num_assets = len(returns_df.columns)

    def portfolio_stats(self, weights):
        """
        Calcula estatísticas do portfólio para um dado conjunto de pesos.
        
        Args:
            weights: Array numpy com pesos dos ativos
            
        Returns:
            Tupla (retorno_esperado, volatilidade)
        """
        p_return = np.dot(weights, self.mean_returns)
        p_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return p_return, p_std

    def negative_sharpe(self, weights):
        """
        Calcula o Sharpe Ratio negativo (para minimização).
        
        Args:
            weights: Array numpy com pesos dos ativos
            
        Returns:
            Sharpe Ratio negativo
        """
        p_return, p_std = self.portfolio_stats(weights)
        return -(p_return - RISK_FREE_RATE) / p_std if p_std != 0 else -100

    def minimize_volatility(self, weights):
        """
        Retorna a volatilidade do portfólio (para minimização).
        
        Args:
            weights: Array numpy com pesos dos ativos
            
        Returns:
            Volatilidade do portfólio
        """
        return self.portfolio_stats(weights)[1]

    def optimize(self, strategy):
        """
        Executa a otimização do portfólio.
        
        Args:
            strategy: 'MinVolatility' ou 'MaxSharpe'
            
        Returns:
            Dicionário com pesos otimizados por ativo, ou None se falhar
        """
        # Restrição: soma dos pesos = 1 (100%)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Limites: cada ativo entre o peso mínimo (10%) e o peso máximo (30%)
        bounds = tuple((MIN_WEIGHT, MAX_WEIGHT) for _ in range(self.num_assets))
        
        # Chute inicial: pesos iguais
        initial_guess = np.array([1/self.num_assets] * self.num_assets)

        # Define função objetivo baseada na estratégia
        if strategy == 'MinVolatility':
            objective = self.minimize_volatility
        elif strategy == 'MaxSharpe':
            objective = self.negative_sharpe
        else:
            return None

        try:
            # Executa otimização usando SLSQP (Sequential Least Squares Programming)
            result = minimize(
                objective,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                # Retorna dicionário com pesos por ativo
                return {asset: weight for asset, weight in zip(self.returns.columns, result.x)}
            else:
                return None
                
        except Exception:
            return None

# =============================================================================
# CLASSE PRINCIPAL: CONSTRUTOR DE PORTFÓLIO
# =============================================================================

class PortfolioBuilder:
    """
    Classe principal que orquestra todo o processo de construção de portfólio:
    1. Coleta de dados de mercado
    2. Cálculo de indicadores técnicos e fundamentalistas
    3. Aplicação de Machine Learning para previsões
    4. Pontuação e ranqueamento de ativos
    5. Seleção final e otimização de alocação
    """
    
    def __init__(self, investment_amount, period=FIXED_ANALYSIS_PERIOD):
        """
        Inicializa o construtor de portfólio.
        
        Args:
            investment_amount: Valor total a ser investido
            period: Período de análise histórica (padrão: 2 anos)
        """
        self.investment_amount = investment_amount
        self.period = period
        
        # Armazenamento de dados
        self.data_by_asset = {}           # Dados históricos por ativo
        self.performance_data = pd.DataFrame() # Métricas de performance
        self.fundamental_data = pd.DataFrame() # Dados fundamentalistas
        self.ml_predictions = {}              # Previsões do modelo ML
        self.successful_assets = []           # Ativos com dados válidos
        self.combined_scores = pd.DataFrame() # Scores combinados
        
        # Resultados finais
        self.portfolio_metrics = {}           # Métricas do portfólio final
        self.portfolio_allocation = {}        # Alocação de capital
        self.selected_assets = []             # Ativos selecionados
        self.current_allocation_method = "Não Aplicado"  # Método de otimização usado
        self.selection_justifications = {}    # Justificativas de seleção
        self.dashboard_profile = {}           # Perfil do investidor
        self.current_weights = {}             # Pesos usados na pontuação

    def get_asset_sector(self, symbol):
        """
        Retorna o setor do ativo.
        
        Args:
            symbol: Ticker do ativo
            
        Returns:
            String com o setor ou 'Unknown'
        """
        if symbol in self.fundamental_data.index:
            return self.fundamental_data.loc[symbol, 'Sector']
        return 'Unknown'

    def collect_market_data(self, symbols):
        """
        Coleta dados de mercado para uma lista de símbolos.
        
        Args:
            symbols: Lista de tickers para coletar
            
        Returns:
            True se coletou dados suficientes, False caso contrário
        """
        self.successful_assets = []
        self.data_by_asset = {}
        fundamentals_list = []
        
        def get_ticker_variations(symbol):
            """
            Gera variações do ticker para aumentar chance de sucesso.
            """
            variations = [symbol]
            if symbol.endswith('.SA'):
                base = symbol[:-3]
                variations.append(base)
                variations.append(base + '.SAO')
            else:
                if len(symbol) <= 6 and not symbol.endswith(('-USD', '.L', '.PA', '.NY')):
                    variations.append(symbol + '.SA')
                    variations.append(symbol + '.SAO')
            if len(symbol) >= 5 and symbol[-2:].isdigit():
                base_no_digits = symbol[:-2]
                variations.append(base_no_digits)
                if not base_no_digits.endswith('.SA'):
                    variations.append(base_no_digits + '.SA')
            
            return variations

        print(f"\n{'='*60}")
        print(f"INICIANDO COLETA DE DADOS - {len(symbols)} ativos")
        print(f"Período: {self.period} | Mínimo de dias: {MIN_HISTORY_DAYS}")
        print(f"{'='*60}\n")
        
        failed_assets = []
        
        for symbol in tqdm(symbols, desc=f"[Coleta {FIXED_ANALYSIS_PERIOD}] Baixando dados"):
            found_valid_data = False
            last_error = None
            
            for attempt_ticker in get_ticker_variations(symbol):
                try:
                    ticker = yf.Ticker(attempt_ticker)
                    hist_attempt = ticker.history(period=self.period)
                    info_attempt = ticker.info

                    if not hist_attempt.empty and len(hist_attempt) >= MIN_HISTORY_DAYS:
                        hist = hist_attempt
                        info = info_attempt
                        found_valid_data = True
                        break
                        
                except Exception as e:
                    last_error = str(e)
                    continue
            
            if not found_valid_data:
                failed_assets.append({
                    'symbol': symbol,
                    'error': last_error or 'Dados insuficientes',
                    'variations_tried': get_ticker_variations(symbol)
                })
                continue

            try:
                # Calcula indicadores técnicos
                df = calculate_technical_indicators(hist.copy())
                
                if df.empty or len(df) < MIN_HISTORY_DAYS:
                    failed_assets.append({
                        'symbol': symbol,
                        'error': f'Dados insuficientes após indicadores: {len(df)} dias',
                        'variations_tried': get_ticker_variations(symbol)
                    })
                    continue
                
                # Armazena dados do ativo
                self.data_by_asset[symbol] = df
                self.successful_assets.append(symbol)
                
                # Coleta dados fundamentalistas
                sector = info.get('sector', 'Unknown')
                
                # Heurística para identificar fundos e ETFs
                if not sector or sector == 'None':
                    if '11' in symbol or 'HODL' in symbol or 'FIAGRO' in symbol.upper() or symbol in ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'VTI', 'VOO']:
                        sector = 'Fundo/ETF'
                    else:
                        sector = 'Unknown'
                
                fundamentals_list.append({
                    'Ticker': symbol,
                    'Sector': sector,
                    'PE_Ratio': info.get('trailingPE', np.nan),
                    'PB_Ratio': info.get('priceToBook', np.nan),
                    'Div_Yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else np.nan,
                    'ROE': info.get('returnOnEquity', np.nan) * 100 if info.get('returnOnEquity') else np.nan
                })
                
            except Exception as e:
                failed_assets.append({
                    'symbol': symbol,
                    'error': f'Erro no processamento: {str(e)}',
                    'variations_tried': get_ticker_variations(symbol)
                })

        print("\n" + '=' * 60)
        print("RESULTADO DA COLETA")
        print('=' * 60)
        print(f"✓ Ativos coletados com sucesso: {len(self.successful_assets)}")
        print(f"✗ Ativos que falharam: {len(failed_assets)}")
        print('=' * 60 + "\n")
        
        if len(self.successful_assets) < NUM_ASSETS_IN_PORTFOLIO:
            # CORREÇÃO: Usando format() e chaves duplas {{}} para evitar o erro do f-string com caracteres especiais/emojis
            print("\n⚠️ ERRO: Apenas {} ativos coletados.".format(len(self.successful_assets)))
            print("    Necessário: {} ativos mínimos".format(NUM_ASSETS_IN_PORTFOLIO))
            print("    Sugestão: Verifique sua conexão com a internet ou tente novamente mais tarde.\n")
            return False

        # Prepara DataFrame de dados fundamentalistas
        self.fundamental_data = pd.DataFrame(fundamentals_list).set_index('Ticker').loc[self.successful_assets]
        self.fundamental_data = self.fundamental_data.replace([np.inf, -np.inf], np.nan)
        self.fundamental_data.fillna(self.fundamental_data.median(numeric_only=True), inplace=True)
        self.fundamental_data.fillna(0, inplace=True)

        # Calcula métricas de performance para cada ativo
        metrics = {
            s: calculate_risk_metrics(self.data_by_asset[s]['Returns'])
            for s in self.successful_assets
            if 'Returns' in self.data_by_asset[s]
        }
        self.performance_data = pd.DataFrame(metrics).T
        
        return True

    def calculate_cross_sectional_features(self):
        """
        Calcula features relativas ao setor (análise cross-sectional).
        """
        df_fund = self.fundamental_data.copy()
        
        # Calcula médias por setor
        sector_means = df_fund.groupby('Sector')[['PE_Ratio', 'PB_Ratio']].transform('mean')
        
        # Calcula razões relativas ao setor
        df_fund['pe_rel_sector'] = df_fund['PE_Ratio'] / sector_means['PE_Ratio']
        df_fund['pb_rel_sector'] = df_fund['PB_Ratio'] / sector_means['PB_Ratio']
        
        # Limpa valores infinitos e NaN
        df_fund = df_fund.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        
        self.fundamental_data = df_fund
        return self.fundamental_data

    def apply_clustering_and_ml(self, ml_lookback_days):
        """
        Aplica técnicas de Machine Learning:
        1. Clustering (PCA + K-Means) para agrupar ativos similares
        2. Random Forest para prever direção futura do preço
        """
        
        # ===== PARTE 1: CLUSTERING =====
        clustering_df = self.fundamental_data[['PE_Ratio', 'PB_Ratio', 'Div_Yield', 'ROE']].join(
            self.performance_data[['sharpe_ratio', 'annual_volatility']],
            how='inner'
        ).fillna(0)

        if len(clustering_df) >= 5:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(clustering_df)
            n_comp = min(data_scaled.shape[1], 3)
            pca = PCA(n_components=n_comp)
            data_pca = pca.fit_transform(data_scaled)
            n_clusters_fit = min(len(data_pca), 5)
            if n_clusters_fit < 2:
                n_clusters_fit = 1

            kmeans = KMeans(n_clusters=n_clusters_fit, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(data_pca)
            
            cluster_series = pd.Series(clusters, index=clustering_df.index)
            self.fundamental_data['Cluster'] = self.fundamental_data.index.map(cluster_series).fillna(-1).astype(int)
        else:
            self.fundamental_data['Cluster'] = 0

        # ===== PARTE 2: RANDOM FOREST CLASSIFIER (MODELO MELHORADO) =====
        features_ml = [
            'RSI', 'MACD', 'Volatility', 'Momentum', 'SMA_50', 'SMA_200',
            'PE_Ratio', 'PB_Ratio', 'Div_Yield', 'ROE',
            'pe_rel_sector', 'pb_rel_sector', 'Cluster'
        ]
        
        for symbol in tqdm(self.successful_assets, desc="[ML] Treinando Random Forest"):
            df = self.data_by_asset[symbol].copy()
            
            # Cria target: 1 se preço subiu após N dias, 0 caso contrário
            df['Future_Direction'] = np.where(
                df['Close'].pct_change(ml_lookback_days).shift(-ml_lookback_days) > 0,
                1,
                0
            )
            
            if symbol in self.fundamental_data.index:
                fund_data = self.fundamental_data.loc[symbol].to_dict()
                for col in [f for f in features_ml if f not in df.columns]:
                    if col in fund_data:
                        df[col] = fund_data[col]
            
            current_features = [f for f in features_ml if f in df.columns]
            df.dropna(subset=current_features + ['Future_Direction'], inplace=True)
            
            if len(df) < MIN_HISTORY_DAYS:
                continue
            
            X = df[current_features].iloc[:-ml_lookback_days].copy()
            y = df['Future_Direction'].iloc[:-ml_lookback_days]

            if 'Cluster' in X.columns:
                X['Cluster'] = X['Cluster'].astype(str)
            
            numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = ['Cluster'] if 'Cluster' in X.columns else []

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), [f for f in numeric_cols if 'rel_sector' not in f]),
                    ('rel', 'passthrough', [f for f in numeric_cols if 'rel_sector' in f]),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
                ],
                remainder='passthrough'
            )
            
            # --- Usando RandomForestClassifier ---
            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced'))
            ])
            
            try:
                # O Random Forest é mais robusto, mas ainda precisa de 2 classes
                if len(np.unique(y)) < 2:
                    self.ml_predictions[symbol] = {
                        'predicted_proba_up': 0.5,
                        'auc_roc_score': 0.5, 
                        'model_name': 'Inconclusivo (Classe Única)'
                    }
                    print(f"✗ Erro ML em {symbol}: Classe única. Predição neutra (0.5).")
                    continue
                
                # Treina o modelo
                model.fit(X, y)
                
                scores = cross_val_score(
                    model, X, y,
                    cv=TimeSeriesSplit(n_splits=5),
                    scoring='roc_auc'
                )
                auc_roc_score = scores.mean()
                
                last_features = df[current_features].iloc[[-ml_lookback_days]].copy()
                if 'Cluster' in last_features.columns:
                    last_features['Cluster'] = last_features['Cluster'].astype(str)
                
                proba = model.predict_proba(last_features)[0][1]
                
                # Armazena previsão
                self.ml_predictions[symbol] = {
                    'predicted_proba_up': proba,
                    'auc_roc_score': auc_roc_score,
                    'model_name': 'RandomForestClassifier'
                }
                
            except Exception as e:
                print(f"✗ Erro ML em {symbol}: {str(e)}")
                self.ml_predictions[symbol] = {
                    'predicted_proba_up': 0.5,
                    'auc_roc_score': np.nan, 
                    'model_name': 'Erro no Treino'
                }
                pass

    def _normalize_score(self, series, higher_better=True):
        """
        Normaliza uma série de scores entre 0-100 usando Z-score robusto.
        """
        series_clean = series.replace([np.inf, -np.inf], np.nan).dropna()
        
        if series_clean.empty or series_clean.std() == 0:
            return pd.Series(50, index=series.index)
        
        z = zscore(series_clean, nan_policy='omit')
        
        normalized_series = pd.Series(
            50 + (z.clip(-3, 3) / 3) * 50,
            index=series_clean.index
        )
        
        if not higher_better:
            normalized_series = 100 - normalized_series
        
        return normalized_series.reindex(series.index, fill_value=50)

    def score_and_rank_assets(self, time_horizon):
        """
        Pontua e ranqueia ativos usando sistema de scoring multi-fator.
        
        A pontuação ML é PONDERADA pela sua confiança (AUC-ROC).
        """
        
        if time_horizon == "CURTO PRAZO":
            WEIGHT_PERF, WEIGHT_FUND, WEIGHT_TECH = 0.40, 0.10, 0.50
        elif time_horizon == "LONGO PRAZO":
            WEIGHT_PERF, WEIGHT_FUND, WEIGHT_TECH = 0.40, 0.50, 0.10
        else:
            WEIGHT_PERF, WEIGHT_FUND, WEIGHT_TECH = 0.40, 0.30, 0.30

        self.current_weights = {
            'Performance': WEIGHT_PERF,
            'Fundamentos': WEIGHT_FUND,
            'Técnicos': WEIGHT_TECH
        }
        
        combined = self.performance_data.join(self.fundamental_data, how='inner').copy()
        
        for symbol in combined.index:
            if symbol in self.data_by_asset and 'RSI' in self.data_by_asset[symbol].columns:
                df = self.data_by_asset[symbol]
                combined.loc[symbol, 'RSI_current'] = df['RSI'].iloc[-1]
                combined.loc[symbol, 'MACD_current'] = df['MACD'].iloc[-1]

        scores = pd.DataFrame(index=combined.index)
        
        # ===== SCORE DE PERFORMANCE =====
        scores['performance_score'] = self._normalize_score(
            combined['sharpe_ratio'],
            higher_better=True
        ) * WEIGHT_PERF
        
        # ===== SCORE FUNDAMENTALISTA =====
        fund_score = self._normalize_score(
            combined.get('PE_Ratio', 50),
            higher_better=False
        ) * 0.5
        fund_score += self._normalize_score(
            combined.get('ROE', 50),
            higher_better=True
        ) * 0.5
        scores['fundamental_score'] = fund_score * WEIGHT_FUND
        
        # ===== SCORE TÉCNICO =====
        tech_score = 0
        if WEIGHT_TECH > 0:
            tech_weights = {'RSI_current': 0.5, 'MACD_current': 0.5}
            
            rsi_proximity_score = 100 - abs(combined['RSI_current'] - 50)
            tech_score += self._normalize_score(
                rsi_proximity_score.clip(0, 100),
                higher_better=True
            ) * tech_weights['RSI_current']
            
            tech_score += self._normalize_score(
                combined.get('MACD_current', 50),
                higher_better=True
            ) * tech_weights['MACD_current']

        scores['technical_score'] = tech_score * WEIGHT_TECH
        
        # ===== SCORE DE MACHINE LEARNING (PONDERADO PELO AUC) =====
        ml_scores = pd.Series({
            s: self.ml_predictions.get(s, {'predicted_proba_up': 0.5})['predicted_proba_up']
            for s in combined.index
        })
        confidence_scores = pd.Series({
            s: self.ml_predictions.get(s, {'auc_roc_score': np.nan})['auc_roc_score']
            for s in combined.index
        })
        
        combined['ML_Proba'] = ml_scores
        # Substitui NaN por 0.5 (confiança neutra) para o cálculo do score.
        combined['ML_Confidence'] = confidence_scores.fillna(0.5) 
        
        # Score ML base (0-30): Probabilidade Normalizada (15%) + Confiança Normalizada (15%)
        ml_score_base = (
             self._normalize_score(combined['ML_Proba'], higher_better=True) * 0.5 +
             self._normalize_score(combined['ML_Confidence'], higher_better=True) * 0.5
        ) * 0.3
        
        # Pondera o Score ML pela CONFIANÇA REAL (AUC). 
        scores['ml_score_weighted'] = ml_score_base * combined['ML_Confidence'] 
        
        # A soma total do peso base (Performance+Fundamentos+Técnicos) é 70%
        score_sum = (WEIGHT_PERF + WEIGHT_FUND + WEIGHT_TECH)
        scores['base_score'] = scores[[
            'performance_score',
            'fundamental_score',
            'technical_score'
        ]].sum(axis=1) / score_sum if score_sum > 0 else 50
        
        # Score final: 70% da Base Ponderada + Score ML Ponderado pelo AUC
        scores['total_score'] = scores['base_score'] * 0.7 + scores['ml_score_weighted']
        
        self.combined_scores = scores.join(combined).sort_values('total_score', ascending=False)
        return self.combined_scores

    def select_final_portfolio(self):
        """
        Seleciona os 5 ativos finais do portfólio com diversificação setorial.
        """
        ranked_assets = self.combined_scores.index.tolist()
        final_portfolio = []
        selected_sectors = set()

        for asset in ranked_assets:
            sector = self.get_asset_sector(asset)
            
            is_broad_asset = any(key in sector for key in [
                'ETF', 'Fundo', 'Fiagro', 'Fixed Income', 'Commodities', 'Unknown'
            ])
            
            if is_broad_asset:
                current_funds = len([
                    a for a in final_portfolio
                    if any(key in self.get_asset_sector(a) for key in ['ETF', 'Fundo', 'Fiagro'])
                ])
                
                if asset not in final_portfolio and len(final_portfolio) < NUM_ASSETS_IN_PORTFOLIO and current_funds < 2:
                    final_portfolio.append(asset)
            
            elif sector not in selected_sectors:
                final_portfolio.append(asset)
                selected_sectors.add(sector)
            
            if len(final_portfolio) >= NUM_ASSETS_IN_PORTFOLIO:
                break
        
        self.selected_assets = final_portfolio
        return self.selected_assets

    def optimize_allocation(self, risk_level):
        """
        Otimiza a alocação de capital usando teoria de Markowitz.
        """
        if len(self.selected_assets) < NUM_ASSETS_IN_PORTFOLIO:
            self.current_allocation_method = "ERRO: Ativos Insuficientes para Otimização Mínima"
            return {asset: 1/len(self.selected_assets) for asset in self.selected_assets} if self.selected_assets else {}
        
        final_returns_df = pd.concat(
            [self.data_by_asset[s]['Returns'] for s in self.selected_assets if s in self.data_by_asset],
            axis=1,
            keys=self.selected_assets
        ).dropna()
        
        if final_returns_df.shape[0] < 50:
            weights = {asset: 1/len(self.selected_assets) for asset in self.selected_assets}
            self.current_allocation_method = 'PESOS IGUAIS (Dados insuficientes)'
        else:
            optimizer = PortfolioOptimizer(final_returns_df)
            
            if 'CONSERVADOR' in risk_level or 'INTERMEDIÁRIO' in risk_level:
                weights = optimizer.optimize('MinVolatility')
                self.current_allocation_method = 'MINIMIZAÇÃO DE VOLATILIDADE'
            else:
                weights = optimizer.optimize('MaxSharpe')
                self.current_allocation_method = 'MAXIMIZAÇÃO DE SHARPE'

        # Fallback para pesos iguais se otimização falhar
        if weights is None:
            weights = {asset: 1/len(self.selected_assets) for asset in self.selected_assets}
            self.current_allocation_method += " | FALLBACK (Otimização Falhou)"
        
        # Normaliza e garante que os pesos somem 100%
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

        # Calcula valores monetários
        self.portfolio_allocation = {
            s: {
                'weight': w,
                'amount': self.investment_amount * w
            }
            for s, w in weights.items()
        }
        
        return self.portfolio_allocation
        
    def calculate_portfolio_metrics(self):
        """
        Calcula métricas consolidadas do portfólio final.
        """
        if not self.selected_assets or not self.portfolio_allocation:
            return {}

        returns_data = {
            s: self.data_by_asset[s]['Returns']
            for s in self.selected_assets
            if s in self.data_by_asset
        }
        returns_df = pd.DataFrame(returns_data).dropna()

        if returns_df.empty:
            return {}

        weights_dict = {s: self.portfolio_allocation[s]['weight'] for s in self.selected_assets}
        weights = np.array([weights_dict[s] for s in returns_df.columns])
        weights = weights / weights.sum()

        portfolio_returns = (returns_df * weights).sum(axis=1)

        metrics = calculate_risk_metrics(portfolio_returns)
        self.portfolio_metrics = {
            **metrics,
            'total_investment': self.investment_amount
        }
        
        return self.portfolio_metrics
        
    def generate_justifications(self):
        """
        Gera justificativas textuais para a seleção de cada ativo.
        """
        for symbol in self.selected_assets:
            justification = []
            
            # Adiciona informações de performance
            if symbol in self.performance_data.index:
                perf = self.performance_data.loc[symbol]
                justification.append(
                    f"Perf: Sharpe {perf['sharpe_ratio']:.3f}, "
                    f"Retorno Anual {perf['annual_return']*100:.2f}%."
                )
            
            # Adiciona informações fundamentalistas
            if symbol in self.fundamental_data.index:
                fund = self.fundamental_data.loc[symbol]
                fund_list = [
                    f"P/L: {fund['PE_Ratio']:.2f}",
                    f"ROE: {fund['ROE']:.2f}%"
                ]
                fund_list = [item for item in fund_list if not pd.isna(float(item.split(': ')[1].replace('%', '').replace('P/L: ', '').replace('ROE: ', '')))]
                if fund_list:
                    justification.append(f"Fund: {', '.join(fund_list)}.")
                
                if 'pe_rel_sector' in fund and not pd.isna(fund['pe_rel_sector']):
                    justification.append(f"Relativo: P/L Setor {fund['pe_rel_sector']:.2f}x.")

            # Adiciona previsão ML (Ajustado para incluir AUC e tratar NaN)
            if symbol in self.ml_predictions:
                ml = self.ml_predictions[symbol]
                
                proba_up = ml.get('predicted_proba_up', 0.5)
                auc_score = ml.get('auc_roc_score', np.nan)
                
                # CORREÇÃO: Usar 'N/A' se o AUC for NaN ou o modelo falhou
                auc_str = f"{auc_score:.3f}" if not pd.isna(auc_score) else "N/A"
                
                justification.append(
                    f"ML: Prob. Alta {proba_up*100:.1f}% "
                    f"(AUC {auc_str})."
                )
            
            self.selection_justifications[symbol] = " | ".join(justification)
        
        return self.selection_justifications
        
    def run_complete_pipeline(self, custom_symbols=None, profile_inputs=None):
        """
        Executa o pipeline completo de construção de portfólio.
        """
        self.dashboard_profile = profile_inputs
        ml_lookback_days = profile_inputs['ml_lookback_days']
        risk_level = profile_inputs['risk_level']
        time_horizon = profile_inputs['time_horizon']
        
        # Etapa 1: Coleta de dados
        if not self.collect_market_data(custom_symbols):
            return False

        # Etapa 2: Features cross-sectional
        self.calculate_cross_sectional_features()
        self.performance_data = self.performance_data.dropna(subset=['sharpe_ratio'])
        
        # Etapa 3: Machine Learning
        self.apply_clustering_and_ml(ml_lookback_days=ml_lookback_days)
        
        # Etapa 4: Pontuação e ranqueamento
        self.score_and_rank_assets(time_horizon=time_horizon)
        
        # Etapa 5: Seleção final
        self.select_final_portfolio()
        
        # Etapa 6: Otimização de alocação
        self.optimize_allocation(risk_level=risk_level)
        
        # Etapa 7: Métricas do portfólio
        self.calculate_portfolio_metrics()
        
        # Etapa 8: Justificativas
        self.generate_justifications()
        
        return True

# =============================================================================
# INTERFACE STREAMLIT - PÁGINAS
# =============================================================================

def show_portfolio_builder_page():
    """
    Página principal: Construtor de Portfólio Otimizado.
    """
    # Inicializa estado da sessão
    if 'builder_complete' not in st.session_state:
        st.session_state.builder_complete = False

    # ===== FASE 1: QUESTIONÁRIO =====
    if not st.session_state.builder_complete:
        st.markdown('### 1. Defina Seu Perfil de Risco e Horizonte')
        
        with st.form("investor_profile_form"):
            col_risk, col_time = st.columns(2)
            
            # Opções de resposta
            options_score = [
                'CT: Concordo Totalmente',
                'C: Concordo',
                'N: Neutro',
                'D: Discordo',
                'DT: Discordo Totalmente'
            ]
            options_reaction = ['A: Venderia', 'B: Manteria', 'C: Compraria mais']
            options_level_abc = ['A: Avançado', 'B: Intermediário', 'C: Iniciante']

            # Coluna 1: Perguntas de tolerância ao risco
            with col_risk:
                st.markdown("#### Tolerância ao Risco")
                p2_risk = st.radio(
                    "**1. Aceito risco de curto prazo por retorno de longo prazo**",
                    options=options_score,
                    index=2
                )
                p3_gain = st.radio(
                    "**2. Ganhar o máximo é minha prioridade, mesmo com risco**",
                    options=options_score,
                    index=2
                )
                p4_stable = st.radio(
                    "**3. Prefiro crescimento constante, sem volatilidade**",
                    options=options_score,
                    index=2
                )
                p5_loss = st.radio(
                    "**4. Evitar perdas é mais importante que crescimento**",
                    options=options_score,
                    index=2
                )
                p511_reaction = st.radio(
                    "**5. Se meus investimentos caíssem 10%, eu:**",
                    options=options_reaction,
                    index=1
                )
                p_level = st.radio(
                    "**6. Meu nível de conhecimento em investimentos:**",
                    options=options_level_abc,
                    index=1
                )

            # Coluna 2: Horizonte temporal e capital
            with col_time:
                st.markdown("#### Horizonte Temporal e Capital")
                p211_time = st.radio(
                    "**7. Prazo máximo para reavaliação de estratégia:**",
                    options=[
                        'A: Curto (até 1 ano)',
                        'B: Médio (1-5 anos)',
                        'C: Longo (5+ anos)'
                    ],
                    index=2
                )[0]  # Pega apenas a letra
                
                p311_liquid = st.radio(
                    "**8. Necessidade de liquidez (prazo mínimo para resgate):**",
                    options=[
                        'A: Menos de 6 meses',
                        'B: Entre 6 meses e 2 anos',
                        'C: Mais de 2 anos'
                    ],
                    index=2
                )[0]  # Pega apenas a letra

                st.markdown("---")
                investment = st.number_input(
                    "Valor de Investimento (R$)",
                    min_value=1000,
                    max_value=10000000,
                    value=100000,
                    step=10000
                )
                
                st.markdown(f"**Período de Análise:** {FIXED_ANALYSIS_PERIOD}")
                st.markdown(f"**Restrição de Peso:** Mínimo **{MIN_WEIGHT*100:.0f}%** e Máximo **{MAX_WEIGHT*100:.0f}%** por ativo.")


            # Botão de submissão
            submitted = st.form_submit_button("Gerar Portfólio Otimizado", type="primary")

            if submitted:
                # Prepara respostas
                risk_answers = {
                    'risk_accept': p2_risk,
                    'max_gain': p3_gain,
                    'stable_growth': p4_stable,
                    'avoid_loss': p5_loss,
                    'reaction': p511_reaction,
                    'level': p_level,
                    'time_purpose': p211_time,
                    'liquidity': p311_liquid
                }
                
                # Analisa perfil
                analyzer = InvestorProfileAnalyzer()
                risk_level, horizon, lookback, score = analyzer.calculate_profile(risk_answers)

                # Armazena perfil na sessão
                st.session_state.profile = {
                    'risk_level': risk_level,
                    'time_horizon': horizon,
                    'ml_lookback_days': lookback,
                    'risk_score': score
                }
                
                # Cria construtor de portfólio
                builder = PortfolioBuilder(investment)
                st.session_state.builder = builder

                # Executa pipeline completo
                with st.spinner(f'Criando recomendações para **PERFIL {risk_level}** ({horizon}). Isso levará alguns minutos.'):
                    success = builder.run_complete_pipeline(
                        custom_symbols=ALL_ASSETS,
                        profile_inputs=st.session_state.profile
                    )

                    if not success:
                        st.error(
                            "Falha fatal: Não foi possível coletar dados para o número mínimo "
                            "de 5 ativos com 60+ dias de histórico. Tente novamente."
                        )
                        return

                    st.session_state.builder_complete = True
                    st.rerun()
    
    # ===== FASE 2: RESULTADOS =====
    else:
        builder = st.session_state.builder
        profile = st.session_state.profile
        assets = builder.selected_assets
        allocation = builder.portfolio_allocation
        
        st.markdown('### Resultados do Portfólio Otimizado')
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Perfil de Risco", profile['risk_level'], f"Score: {profile.get('risk_score', 'N/A')}")
        col2.metric("Horizonte de Tempo", profile['time_horizon'])
        col3.metric("Sharpe Ratio do Portfólio", f"{builder.portfolio_metrics.get('sharpe_ratio', 0):.3f}")
        col4.metric("Estratégia de Alocação", builder.current_allocation_method)

        # Botão para recomeçar
        if st.button("Recomeçar Análise de Perfil", key='recomecar_analise'):
            st.session_state.builder_complete = False
            st.rerun()

        st.markdown("---")
        
        col_alloc, col_just = st.columns([1, 2])

        # Prepara dados para gráfico de pizza
        alloc_data_chart = pd.DataFrame([
            {'Ativo': a, 'Peso (%)': allocation[a]['weight'] * 100}
            for a in assets if a in allocation
        ])
        
        # Gráfico de alocação
        with col_alloc:
            st.markdown('#### 🥧 Alocação de Capital (Mín 10%, Máx 30%)')
            if not alloc_data_chart.empty:
                fig_alloc = px.pie(
                    alloc_data_chart,
                    values='Peso (%)',
                    names='Ativo',
                    hole=0.3
                )
                fig_layout = get_chart_template()
                fig_layout['title']['text'] = f"Distribuição para {builder.current_allocation_method}"
                fig_alloc.update_layout(**fig_layout)
                st.plotly_chart(fig_alloc, use_container_width=True)
            else:
                st.warning("Sem dados de alocação para mostrar.")

        # Tabela e justificativas
        with col_just:
            st.markdown('#### Tabela e Justificativas')
            
            # Prepara dados da tabela
            alloc_table_data = []
            for asset in assets:
                if asset in allocation:
                    weight = allocation[asset]['weight']
                    amount = allocation[asset]['amount']
                    sector = builder.get_asset_sector(asset)
                    ml_info = builder.ml_predictions.get(asset, {
                        'predicted_proba_up': 0.5,
                        'auc_roc_score': np.nan 
                    })
                    
                    # Tratar AUC nan para exibição na tabela
                    auc_score = ml_info.get('auc_roc_score', np.nan)
                    auc_display = f"{auc_score:.3f}" if not pd.isna(auc_score) else "N/A"

                    alloc_table_data.append({
                        'Ativo': asset,
                        'Setor': sector,
                        'Peso (%)': weight * 100,
                        'Valor (R$)': amount,
                        'ML (Prob. Alta)': f"{ml_info['predicted_proba_up']*100:.1f}%",
                        'ML (AUC)': auc_display 
                    })
            
            # Formata e exibe tabela
            alloc_df_table = pd.DataFrame(alloc_table_data).sort_values('Peso (%)', ascending=False)
            alloc_df_table['Peso (%)'] = alloc_df_table['Peso (%)'].round(2)
            alloc_df_table['Valor (R$)'] = alloc_df_table['Valor (R$)'].apply(
                lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            )
            # Exibe a coluna ML (AUC)
            st.dataframe(alloc_df_table, use_container_width=True, hide_index=True)
            
            # Justificativas detalhadas
            st.markdown("---")
            st.markdown('##### Detalhe da Seleção:')
            for asset in assets:
                if asset in builder.portfolio_allocation:
                     st.markdown(
                        f"**{asset}** ({builder.portfolio_allocation[asset]['weight']*100:.1f}%) — "
                        f"{builder.selection_justifications.get(asset, 'Sem justificativa detalhada.')}"
                    )
                else:
                    st.markdown(f"**{asset}** — Sem dados de alocação final.")


def show_individual_analysis_page():
    """
    Página: Análise Individual de Ativo.
    
    Permite análise detalhada de um único ativo, incluindo preço atual, 
    todos os indicadores técnicos e fundamentalistas usados, previsão ML,
    e gráficos de preços, volume e indicadores.
    """
    st.markdown('### Análise Individual de Ativo')
    
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.text_input("Digite o ticker do ativo:", placeholder="Ex: PETR4.SA, AAPL")
    with col2:
        st.info(f"Análise histórica: {FIXED_ANALYSIS_PERIOD}")
        period = FIXED_ANALYSIS_PERIOD
        
    if st.button("Analisar Ativo", type="primary") and symbol:
        with st.spinner(f'Analisando **{symbol}**...'):
            try:
                symbol_yfin = symbol.upper()
                
                # Tenta primeiro com o ticker original
                ticker = yf.Ticker(symbol_yfin)
                hist = ticker.history(period=period)
                
                # Se falhar e for ticker brasileiro, tenta com .SA
                if (hist.empty or len(hist) < MIN_HISTORY_DAYS) and not symbol_yfin.endswith(('.SA', '-USD')):
                    if len(symbol_yfin) <= 6:
                        symbol_yfin += '.SA'
                        ticker = yf.Ticker(symbol_yfin)
                        hist = ticker.history(period=period)
                        
                info = ticker.info
                
                if hist.empty or len(hist) < MIN_HISTORY_DAYS:
                    st.error(f"Não foi possível obter dados suficientes para {symbol}. Mínimo: {MIN_HISTORY_DAYS} dias.")
                    return
                
                # Calcula indicadores técnicos
                df = calculate_technical_indicators(hist.copy())
                df['Volume'] = hist.loc[df.index, 'Volume'] # Adiciona Volume na mesma data que os indicadores
                df.dropna(inplace=True)
                
                # Último preço
                current_price = df['Close'].iloc[-1]
                
                # Calcula métricas de risco
                returns = df['Returns'].dropna()
                metrics = calculate_risk_metrics(returns)
                
                # ===== PREVISÃO ML (5 DIAS) =====
                lookback = 5
                
                # Adiciona coluna Future_Direction
                df['Future_Direction'] = np.where(
                    df['Close'].pct_change(lookback).shift(-lookback) > 0,
                    1,
                    0
                )
                
                features = ['RSI', 'MACD', 'Volatility', 'Momentum', 'SMA_50', 'SMA_200']
                df_ml = df.dropna(subset=features + ['Future_Direction'])
                
                proba_up = 0.5
                auc_score = 0.0
                
                if len(df_ml) >= MIN_HISTORY_DAYS:
                    X = df_ml[features].iloc[:-lookback]
                    y = df_ml['Future_Direction'].iloc[:-lookback]
                    
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    # --- Usando RandomForestClassifier ---
                    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
                    
                    # Trata o erro de classe única (melhoria integrada)
                    if len(np.unique(y)) < 2:
                        st.warning(f"⚠️ Aviso ML: O ativo **{symbol_yfin}** teve apenas uma direção no período de treino. A previsão ML não foi realizada.")
                    else:
                        model.fit(X_scaled, y)
                        
                        scores_cv = cross_val_score(
                            model, X_scaled, y,
                            cv=TimeSeriesSplit(n_splits=3),
                            scoring='roc_auc'
                        )
                        auc_score = scores_cv.mean()
                        
                        proba_up = model.predict_proba(
                            scaler.transform(df_ml[features].iloc[[-lookback]])
                        )[0][1]

            except Exception as e:
                st.error(f"Erro na análise de {symbol}: {e}")
                return

            st.success(f"Análise de **{symbol_yfin}** concluída!")
            st.markdown("---")

            # =================================================================
            # APRESENTAÇÃO DE DADOS: METRICS E FUNDAMENTOS
            # =================================================================
            st.markdown("#### Resumo e Fundamentos")
            
            col_price, col_sector, col_pl, col_div = st.columns(4)
            col_price.metric("Preço Atual", f"R$ {current_price:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
            sector_name = info.get('sector', 'N/A')
            col_sector.metric("Setor", sector_name)
            col_pl.metric("P/L (Trailing)", f"{info.get('trailingPE', 0):.2f}")
            col_div.metric("Div. Yield", f"{info.get('dividendYield', 0)*100:.2f}%")
            
            st.markdown("#### Métricas de Risco/Retorno (Histórico: 2 Anos)")
            col_ret, col_vol, col_sharp, col_mdd = st.columns(4)
            col_ret.metric("Retorno Anual", f"{metrics['annual_return']*100:.2f}%")
            col_vol.metric("Volatilidade Anual", f"{metrics['annual_volatility']*100:.2f}%")
            col_sharp.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}")
            col_mdd.metric("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%")

            # =================================================================
            # PREVISÃO ML
            # =================================================================
            st.markdown("#### Previsão do Random Forest (Próximos 5 dias)")
            
            # Tratamento de AUC nan na exibição
            auc_display = f"{auc_score:.3f}" if not pd.isna(auc_score) else "N/A"
            
            if len(df_ml) >= MIN_HISTORY_DAYS and len(np.unique(df_ml['Future_Direction'].iloc[:-lookback])) >= 2:
                if proba_up > 0.5:
                    status, emoji = "**BULLISH**", "🚀"
                elif proba_up < 0.5:
                    status, emoji = "**BEARISH**", "🔻"
                else:
                    status, emoji = "**NEUTRO**", "🔘"
                    
                st.info(
                    f"Probabilidade de alta nos próximos 5 dias: **{proba_up*100:.1f}%** ({status} {emoji}). "
                    f"Confiança do Modelo (AUC-ROC): **{auc_display}**."
                )
            else:
                st.warning("A previsão ML não foi realizada devido à falta de variabilidade de classes (apenas alta ou apenas baixa) ou dados insuficientes no período de treino.")

            # =================================================================
            # GRÁFICOS: PREÇO, VOLUME, RSI, MACD
            # =================================================================
            st.markdown("#### Análise Gráfica Detalhada (Preço, Volume, RSI e MACD)")
            
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.5, 0.1, 0.2, 0.2],
            )
            
            # --- ROW 1: CANDLESTICK E MÉDIAS MÓVEIS ---
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Preço'
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='#f39c12', width=1)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['SMA_200'], name='SMA 200', line=dict(color='#2ecc71', width=1)),
                row=1, col=1
            )
            fig.update_yaxes(title_text=f"Preço ({symbol_yfin})", row=1, col=1)

            # --- ROW 2: VOLUME ---
            fig.add_trace(
                go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='#7f8c8d'),
                row=2, col=1
            )
            fig.update_yaxes(title_text="Volume", row=2, col=1)

            # --- ROW 3: RSI ---
            fig.add_trace(
                go.Scatter(x=df.index, y=df['RSI'], name='RSI (14)', line=dict(color='#2c3e50')),
                row=3, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="#e74c3c", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#27ae60", row=3, col=1)
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)

            # --- ROW 4: MACD ---
            fig.add_trace(
                go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='#3498db')),
                row=4, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='#f1c40f')),
                row=4, col=1
            )
            fig.update_yaxes(title_text="MACD", row=4, col=1)

            # --- Configuração Final do Layout (CORRIGIDO) ---
            fig_layout = get_chart_template()
            fig_layout['title']['text'] = f"Preço, Volume e Indicadores de {symbol_yfin} ({period})"
            
            fig.update_layout(
                **fig_layout,
                height=900,
                xaxis_rangeslider_visible=False,
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # =================================================================
            # INDICADORES NÃO-GRÁFICOS: DADOS DE TABELA
            # =================================================================
            st.markdown("#### Indicadores Técnicos Atuais")
            
            last_indicators = df.iloc[-1][['RSI', 'MACD', 'Momentum', 'Volatility', 'SMA_50', 'SMA_200']]
            indicators_df = pd.DataFrame(last_indicators).T
            
            for col in ['RSI', 'MACD', 'Momentum', 'Volatility', 'SMA_50', 'SMA_200']:
                indicators_df[col] = indicators_df[col].apply(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

            indicators_df = indicators_df.rename(columns={
                'RSI': 'RSI (14)',
                'MACD': 'MACD',
                'Momentum': 'Momentum (10d)',
                'Volatility': 'Volatilidade Anual',
                'SMA_50': 'Média Móvel (50d)',
                'SMA_200': 'Média Móvel (200d)'
            })

            st.dataframe(indicators_df, use_container_width=True, hide_index=True)


# =============================================================================
# CONFIGURAÇÃO E FUNÇÃO PRINCIPAL
# =============================================================================

def set_page_config():
    """
    Configura a página Streamlit e injeta CSS customizado.
    """
    st.set_page_config(
        page_title="Portfolio Adaptativo",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # CSS customizado para tema Times New Roman
    st.markdown("""
        <style>
        .main-header {
            font-family: 'Times New Roman', serif;
            color: #2c3e50;
            text-align: center;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 10px;
            font-size: 2.2rem !important;
        }
        html, body, [class*="st-"] {
            font-family: 'Times New Roman', serif;
        }
        .stButton button {
            border: 1px solid #2c3e50;
            color: #2c3e50;
            border-radius: 4px;
        }
        .stButton button:hover {
            background-color: #7f8c8d;
            color: white;
        }
        .stButton button[kind="primary"] {
            background-color: #2c3e50;
            color: white;
            border: none;
        }
        .stRadio div, .stSelectbox {
            border-radius: 4px;
            padding: 5px;
        }
        </style>
        """, unsafe_allow_html=True)

def main():
    """
    Função principal que inicializa e gerencia a aplicação Streamlit.
    """
    # Inicializa estado da sessão
    if 'builder' not in st.session_state:
        st.session_state.builder = PortfolioBuilder(investment_amount=100000)
        st.session_state.builder_complete = False
        st.session_state.profile = {}
    
    # Configura página
    set_page_config()
    
    # Menu lateral
    st.sidebar.markdown(
        '<p style="font-size: 20px; font-weight: bold; color: #2c3e50;">📈 Menu de Navegação</p>',
        unsafe_allow_html=True
    )
    
    # Páginas disponíveis
    PAGES = {
        "Criar Portfólio Otimizado": show_portfolio_builder_page,
        "Análise Individual de Ativo": show_individual_analysis_page
    }
    
    # Seleção de página
    selection = st.sidebar.radio("Selecione a Ferramenta:", list(PAGES.keys()))
    
    # Exibe título da página
    st.markdown(f'<h1 class="main-header">{selection}</h1>', unsafe_allow_html=True)
    
    # Renderiza página selecionada
    page = PAGES[selection]
    page()

# =============================================================================
# PONTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    main()
