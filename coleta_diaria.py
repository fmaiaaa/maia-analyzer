"""
=============================================================================
SCRIPT ETL - COLETA DIÁRIA DE DADOS FINANCEIROS (ADAPTADO PARA GOOGLE SHEETS)
=============================================================================

Este script deve ser executado diariamente (via GitHub Actions) para:
1. Coletar dados de TODOS os ativos do yfinance
2. Processar e engenheirar features
3. Salvar os dados em abas do Google Sheets para consumo pelo Streamlit Cloud.

Uso:
    python coleta_diaria.py

Execução na Nuvem (GitHub Actions):
    A autenticação usa o Secret GOOGLE_CREDENTIALS_JSON.
"""
import gspread
from gspread_dataframe import set_with_dataframe
from oauth2client.service_account import ServiceAccountCredentials
import json
import os # Necessário para ler o Secret do GitHub Actions
import sys
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
import time
import ta # Import adicionado, pois é usado na função calcular_indicadores_tecnicos

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURAÇÕES E CONSTANTES
# =============================================================================

# 📌 O QUE ALTERAR: COLE AQUI O ID DA SUA PLANILHA MESTRA DO GOOGLE SHEETS
PLANILHA_MESTRA_ID = "1g6lnrB-N4kgrbEiZKEYaoll4la9TnfBJh4VvioqbC78" 

# Configurações de coleta
PERIODO_DADOS = 'max'
MIN_DIAS_HISTORICO = 252
TAXA_LIVRE_RISCO = 0.1075
MAX_RETRIES = 3
RETRY_DELAY = 2

# Lista completa de ativos (mantida para referência)
ATIVOS_IBOVESPA = [
    'ALOS3.SA', 'ABEV3.SA', 'ASAI3.SA', 'AURE3.SA', 'AZZA3.SA', 'B3SA3.SA',
    'BBSE3.SA', 'BBDC3.SA', 'BBDC4.SA', 'BRAP4.SA', 'BBAS3.SA', 'BRKM5.SA',
    'BRAV3.SA', 'BPAC11.SA', 'CXSE3.SA', 'CEAB3.SA', 'CMIG4.SA', 'COGN3.SA',
    'CPLE6.SA', 'CSAN3.SA', 'CPFE3.SA', 'CMIN3.SA', 'CURY3.SA', 'CVCB3.SA',
    'CYRE3.SA', 'DIRR3.SA', 'ELET3.SA', 'ELET6.SA', 'EMBR3.SA', 'ENGI11.SA',
    'ENEV3.SA', 'EGIE3.SA', 'EQTL3.SA', 'FLRY3.SA', 'GGBR4.SA', 'GOAU4.SA',
    'HAPV3.SA', 'HYPE3.SA', 'IGTI11.SA', 'IRBR3.SA', 'ISAE4.SA', 'ITSA4.SA',
    'ITUB4.SA', 'KLBN11.SA', 'RENT3.SA', 'LREN3.SA', 'MGLU3.SA', 'POMO4.SA',
    'MBRF3.SA', 'BEEF3.SA', 'MOTV3.SA', 'MRVE3.SA', 'MULT3.SA', 'NATU3.SA',
    'PCAR3.SA', 'PETR3.SA', 'PETR4.SA', 'RECV3.SA', 'PRIO3.SA', 'PSSA3.SA',
    'RADL3.SA', 'RAIZ4.SA', 'RDOR3.SA', 'RAIL3.SA', 'SBSP3.SA', 'SANB11.SA',
    'CSNA3.SA', 'SLCE3.SA', 'SMFT3.SA', 'SUZB3.SA', 'TAEE11.SA', 'VIVT3.SA',
    'TIMS3.SA', 'TOTS3.SA', 'UGPA3.SA', 'USIM5.SA', 'VALE3.SA', 'VAMO3.SA',
    'VBBR3.SA', 'VIVA3.SA', 'WEGE3.SA', 'YDUQ3.SA'
]

ATIVOS_POR_SETOR = {
    'Bens Industriais': ['NATU3.SA', 'AMOB3.SA', 'ISAE4.SA', 'BHIA3.SA', 'ZAMP3.SA', 'AERI3.SA'],
    'Consumo Cíclico': ['AZZA3.SA', 'ALOS3.SA', 'VIIA3.SA', 'RDNI3.SA', 'SLED4.SA', 'RSID3.SA'],
    'Consumo não Cíclico': ['PRVA3.SA', 'SMTO3.SA', 'MDIA3.SA', 'CAML3.SA', 'AGRO3.SA', 'BEEF3.SA'],
    'Financeiro': ['CSUD3.SA', 'INBR31.SA', 'BIDI3.SA', 'BIDI4.SA', 'IGTI11.SA', 'IGTI3.SA'],
    'Materiais Básicos': ['LAND3.SA', 'DEXP4.SA', 'RANI3.SA', 'PMAM3.SA', 'FESA4.SA', 'EUCA3.SA'],
    'Petróleo, Gás e Biocombustíveis': ['SRNA3.SA', 'VBBR3.SA', 'RAIZ4.SA', 'RECV3.SA', 'PRIO3.SA'],
    'Saúde': ['ONCO3.SA', 'VVEO3.SA', 'PARD3.SA', 'BIOM3.SA', 'BALM3.SA', 'PNVL3.SA', 'AALR3.SA'],
    'Tecnologia da Informação': ['CLSA3.SA', 'LVTC3.SA', 'G2DI33.SA', 'IFCM3.SA', 'GOGL35.SA'],
    'Telecomunicações': ['BRIT3.SA', 'FIQE3.SA', 'DESK3.SA', 'TIMS3.SA', 'VIVT3.SA', 'TELB4.SA'],
    'Utilidade Pública': ['BRAV3.SA', 'AURE3.SA', 'MEGA3.SA', 'CEPE6.SA', 'CEED3.SA', 'EEEL4.SA']
}

# Lista completa de todos os ativos
TODOS_ATIVOS = []
for setor, ativos in ATIVOS_POR_SETOR.items():
    TODOS_ATIVOS.extend(ativos)
TODOS_ATIVOS.extend(ATIVOS_IBOVESPA)
TODOS_ATIVOS = sorted(list(set(TODOS_ATIVOS)))

print(f"Total de ativos para coleta: {len(TODOS_ATIVOS)}")

# =============================================================================
# FUNÇÕES DE COLETA, PROCESSAMENTO E ESCRITA
# =============================================================================

def autenticar_e_escrever_sheets(worksheet_name, df):
    """Autentica com o JSON do GitHub Secret e escreve o DataFrame na aba correta."""
    try:
        # 1. Autenticação com Credenciais (lê o JSON da variável de ambiente)
        credentials_json = os.environ.get('GOOGLE_CREDENTIALS_JSON')
        if not credentials_json:
            print("❌ ERRO: Variável de ambiente GOOGLE_CREDENTIALS_JSON não encontrada.")
            print("       (O ETL só funciona com o Secret configurado no GitHub Actions!)")
            return False
        
        # Converte o JSON string (do Secret) em um dicionário Python
        # O JSON deve estar em uma única linha no Secret do GitHub!
        creds_dict = json.loads(credentials_json)

        # Conecta
        creds = ServiceAccountCredentials.from_json_keyfile_dict(
            creds_dict,
            scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        )
        client = gspread.authorize(creds)
        
        # 2. Abre a Planilha Mestra
        spreadsheet = client.open_by_key(PLANILHA_MESTRA_ID)
        
        # 3. Seleciona a aba (worksheet) e escreve
        worksheet = spreadsheet.worksheet(worksheet_name)
        
        # O DataFrame precisa ter o índice resetado para ser escrito como dados
        df_para_escrever = df.reset_index()
        
        # Limpa o conteúdo existente 
        worksheet.clear()
        
        # Escreve o DataFrame. write_index=False porque reset_index já adicionou a coluna 'index'
        set_with_dataframe(worksheet, df_para_escrever, include_index=False) 
        print(f"  ✓ Dados '{worksheet_name}' escritos no Google Sheets.")
        return True
        
    except Exception as e:
        print(f"  ❌ ERRO CRÍTICO ao escrever no Google Sheets ({worksheet_name}): {str(e)}")
        return False


def coletar_dados_macroeconomicos():
    """Coleta dados macroeconômicos"""
    # ... (CORPO DA FUNÇÃO MANTIDO)
    print("\n📊 Coletando dados macroeconômicos...")
    
    dados_macro = {}
    indices = {
        'IBOV': '^BVSP',
        'SP500': '^GSPC',
        'VIX': '^VIX',
        'USD_BRL': 'BRL=X',
        'GOLD': 'GC=F',
        'OIL': 'CL=F'
    }
    
    for nome, simbolo in indices.items():
        for tentativa in range(MAX_RETRIES):
            try:
                ticker = yf.Ticker(simbolo)
                hist = ticker.history(period=PERIODO_DADOS)
                
                if not hist.empty:
                    dados_macro[nome] = hist['Close'].pct_change()
                    print(f"  ✓ {nome}: {len(hist)} dias")
                    break
                else:
                    print(f"  ⚠️ {nome}: Sem dados (tentativa {tentativa + 1}/{MAX_RETRIES})")
                    if tentativa < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
            except Exception as e:
                print(f"  ⚠️ {nome}: Erro - {str(e)[:50]} (tentativa {tentativa + 1}/{MAX_RETRIES})")
                if tentativa < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    dados_macro[nome] = pd.Series()
    
    return dados_macro


def calcular_indicadores_tecnicos(df):
    """Calcula indicadores técnicos usando biblioteca ta"""
    try:
        # Importado no topo: import ta
        
        # Retornos
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Volatilidade
        df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        df['volatility_60'] = df['returns'].rolling(window=60).std() * np.sqrt(252)
        
        # Drawdown
        cumulative = (1 + df['returns']).cumprod()
        running_max = cumulative.expanding().max()
        df['drawdown'] = (cumulative - running_max) / running_max
        
        # Adiciona todos os indicadores da biblioteca ta
        df = ta.add_all_ta_features(
            df, open="Open", high="High", low="Low", close="Close", volume="Volume",
            fillna=True
        )
        
        # Features temporais
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        return df
        
    except Exception as e:
        print(f"  ⚠️ Erro ao calcular indicadores técnicos: {str(e)[:80]}")
        return df

def adicionar_correlacoes_macro(df, simbolo, dados_macro):
    """Adiciona correlações com indicadores macroeconômicos"""
    # ... (CORPO DA FUNÇÃO MANTIDO)
    if not dados_macro or 'returns' not in df.columns:
        return df
    
    try:
        if df['returns'].isnull().all():
            return df

        for nome, serie_macro in dados_macro.items():
            if serie_macro.empty or serie_macro.isnull().all():
                continue
            
            combined_df = pd.DataFrame({
                'asset_returns': df['returns'],
                'macro_returns': serie_macro.reindex(df.index)
            }).dropna()

            if len(combined_df) > 60:
                corr_rolling = combined_df['asset_returns'].rolling(60).corr(combined_df['macro_returns'])
                df[f'corr_{nome.lower()}'] = corr_rolling.reindex(df.index)
            else:
                df[f'corr_{nome.lower()}'] = np.nan
                
    except Exception as e:
        print(f"  ⚠️ {simbolo}: Erro ao calcular correlações macro - {str(e)[:80]}")
    
    return df

def calcular_features_fundamentalistas(info):
    """Extrai features fundamentalistas"""
    # ... (CORPO DA FUNÇÃO MANTIDO)
    features = {
        'pe_ratio': info.get('trailingPE', np.nan),
        'forward_pe': info.get('forwardPE', np.nan),
        'pb_ratio': info.get('priceToBook', np.nan),
        'ps_ratio': info.get('priceToSalesTrailing12Months', np.nan),
        'peg_ratio': info.get('pegRatio', np.nan),
        'ev_ebitda': info.get('enterpriseToEbitda', np.nan),
        'div_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else np.nan,
        'payout_ratio': info.get('payoutRatio', np.nan) * 100 if info.get('payoutRatio') else np.nan,
        'roe': info.get('returnOnEquity', np.nan) * 100 if info.get('returnOnEquity') else np.nan,
        'roa': info.get('returnOnAssets', np.nan) * 100 if info.get('returnOnAssets') else np.nan,
        'roic': info.get('returnOnCapital', np.nan) * 100 if info.get('returnOnCapital') else np.nan,
        'profit_margin': info.get('profitMargins', np.nan) * 100 if info.get('profitMargins') else np.nan,
        'operating_margin': info.get('operatingMargins', np.nan) * 100 if info.get('operatingMargins') else np.nan,
        'gross_margin': info.get('grossMargins', np.nan) * 100 if info.get('grossMargins') else np.nan,
        'debt_to_equity': info.get('debtToEquity', np.nan),
        'current_ratio': info.get('currentRatio', np.nan),
        'quick_ratio': info.get('quickRatio', np.nan),
        'revenue_growth': info.get('revenueGrowth', np.nan) * 100 if info.get('revenueGrowth') else np.nan,
        'earnings_growth': info.get('earningsGrowth', np.nan) * 100 if info.get('earningsGrowth') else np.nan,
        'market_cap': info.get('marketCap', np.nan),
        'enterprise_value': info.get('enterpriseValue', np.nan),
        'beta': info.get('beta', np.nan),
        'sector': info.get('sector', 'Unknown'),
        'industry': info.get('industry', 'Unknown')
    }
    
    return features

def coletar_ativo_com_retry(ticker, periodo=PERIODO_DADOS):
    """Coleta dados de um ativo com retry"""
    # ... (CORPO DA FUNÇÃO MANTIDO)
    for tentativa in range(MAX_RETRIES):
        try:
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period=periodo)
            
            if not hist.empty and len(hist) >= MIN_DIAS_HISTORICO:
                return hist, ticker_obj
            else:
                if tentativa < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    
        except Exception as e:
            if tentativa < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"  ❌ {ticker}: Falha após {MAX_RETRIES} tentativas - {str(e)[:50]}")
                
    return None, None

# =============================================================================
# FUNÇÃO PRINCIPAL DE ETL
# =============================================================================

def executar_etl():
    """Executa o processo completo de ETL"""
    
    print("="*80)
    print(f"INICIANDO ETL - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total de ativos: {len(TODOS_ATIVOS)}")
    print("="*80)
    
    # 1. Coletar dados macroeconômicos
    dados_macro = coletar_dados_macroeconomicos()
    
    # 2. Coletar e processar dados dos ativos
    print(f"\n📥 Coletando dados de {len(TODOS_ATIVOS)} ativos...")
    
    dados_historicos_list = []
    lista_fundamentalistas = []
    metricas_performance = {}
    ativos_sucesso = []
    ativos_falha = []
    
    for ticker in tqdm(TODOS_ATIVOS, desc="Processando ativos"):
        try:
            # Coleta dados históricos
            hist, ticker_obj = coletar_ativo_com_retry(ticker)
            
            if hist is None:
                ativos_falha.append(ticker)
                continue
            
            # Processa dados técnicos
            df = calcular_indicadores_tecnicos(hist)
            df = adicionar_correlacoes_macro(df, ticker, dados_macro)
            df = df.dropna()
            
            # Validação após limpeza
            min_dias_flexivel = max(180, int(MIN_DIAS_HISTORICO * 0.7))
            if len(df) < min_dias_flexivel: // <--- AQUI! O DF ESTÁ VAZIO OU COM POUCOS DIAS
                ativos_falha.append(ticker)
                continue
            
            # Adiciona coluna de ticker para identificação
            df['ticker'] = ticker
            dados_historicos_list.append(df)
            ativos_sucesso.append(ticker) // <--- ESTA LINHA NUNCA FOI ATINGIDA
            
            # Coleta dados fundamentalistas
            try:
                info = ticker_obj.info
                features_fund = calcular_features_fundamentalistas(info)
                features_fund['Ticker'] = ticker
                lista_fundamentalistas.append(features_fund)
            except Exception as fund_e:
                print(f"  ⚠️ {ticker}: Erro fundamentalista - {str(fund_e)[:50]}")
                # Adiciona linha com NaNs
                features_fund = {'Ticker': ticker}
                expected_cols = ['pe_ratio', 'forward_pe', 'pb_ratio', 'ps_ratio', 'peg_ratio', 
                                'ev_ebitda', 'div_yield', 'payout_ratio', 'roe', 'roa', 'roic',
                                'profit_margin', 'operating_margin', 'gross_margin', 'debt_to_equity',
                                'current_ratio', 'quick_ratio', 'revenue_growth', 'earnings_growth',
                                'market_cap', 'enterprise_value', 'beta', 'sector', 'industry']
                for col in expected_cols:
                    features_fund[col] = np.nan
                lista_fundamentalistas.append(features_fund)
            
            # Calcula métricas de performance
            if 'returns' in df.columns and 'drawdown' in df.columns:
                returns = df['returns']
                metricas_performance[ticker] = {
                    'retorno_anual': returns.mean() * 252,
                    'volatilidade_anual': returns.std() * np.sqrt(252),
                    'sharpe': (returns.mean() * 252 - TAXA_LIVRE_RISCO) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
                    'max_drawdown': df['drawdown'].min()
                }
            
        except Exception as e:
            print(f"  ❌ {ticker}: Erro no processamento - {str(e)[:80]}")
            ativos_falha.append(ticker)
            continue
    
    # 3. Consolidar e salvar dados
    print(f"\n💾 Salvando dados processados no Google Sheets...")
    
    # Salvar dados históricos
    if dados_historicos_list:
        df_historico_completo = pd.concat(dados_historicos_list, ignore_index=False)
        autenticar_e_escrever_sheets('Dados_Historicos', df_historico_completo) # <--- NOVO
        print(f"  ✓ Históricos salvos: {len(df_historico_completo)} linhas")
    
    # Salvar dados fundamentalistas
    if lista_fundamentalistas:
        df_fundamentalista = pd.DataFrame(lista_fundamentalistas).set_index('Ticker')
        df_fundamentalista = df_fundamentalista.replace([np.inf, -np.inf], np.nan)
        
        # Normalizar dados numéricos
        scaler = RobustScaler()
        numeric_cols = df_fundamentalista.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df_fundamentalista[col].isnull().any():
                median_val = df_fundamentalista[col].median()
                df_fundamentalista[col] = df_fundamentalista[col].fillna(median_val)
        
        df_fundamentalista[numeric_cols] = scaler.fit_transform(df_fundamentalista[numeric_cols])
        autenticar_e_escrever_sheets('Fundamentalistas', df_fundamentalista) # <--- NOVO
        print(f"  ✓ Fundamentalistas salvos: {len(df_fundamentalista)} ativos")
    
    # Salvar métricas de performance
    if metricas_performance:
        df_metricas = pd.DataFrame(metricas_performance).T
        autenticar_e_escrever_sheets('Metricas', df_metricas) # <--- NOVO
        print(f"  ✓ Métricas salvos: {len(df_metricas)} ativos")
    
    # Salvar dados macro
    if dados_macro:
        df_macro = pd.DataFrame(dados_macro)
        autenticar_e_escrever_sheets('Macro', df_macro) # <--- NOVO
        print(f"  ✓ Dados macro salvos: {len(df_macro)} linhas")
    
    # 4. Relatório final (Metadata e Relatório final mantidos)
    print("\n" + "="*80)
    print("RELATÓRIO FINAL DO ETL")
    print("="*80)
    print(f"✓ Ativos processados com sucesso: {len(ativos_sucesso)}/{len(TODOS_ATIVOS)}")
    print(f"❌ Ativos com falha: {len(ativos_falha)}/{len(TODOS_ATIVOS)}")
    
    if ativos_falha:
        print(f"\nAtivos com falha ({len(ativos_falha)}):")
        for ticker in ativos_falha[:20]:
            print(f"  • {ticker}")
        if len(ativos_falha) > 20:
            print(f"  ... e mais {len(ativos_falha) - 20} ativos")
    
    print(f"\n✓ ETL concluído em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return len(ativos_sucesso) > 0

# =============================================================================
# EXECUÇÃO
# =============================================================================

if __name__ == "__main__":
    try:
        sucesso = executar_etl()
        sys.exit(0 if sucesso else 1)
    except Exception as e:
        print(f"\n❌ ERRO CRÍTICO NO ETL: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
