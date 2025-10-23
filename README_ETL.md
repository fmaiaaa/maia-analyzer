# Sistema ETL - Coleta Diária de Dados Financeiros

## Visão Geral

Este sistema desacopla a coleta de dados do yfinance do dashboard Streamlit, resolvendo problemas de:
- Rate limiting da API do Yahoo Finance
- Lentidão no carregamento do dashboard
- Instabilidade por requisições repetitivas

## Arquitetura

\`\`\`
coleta_diaria.py (ETL)  →  dados_financeiros/*.parquet  →  automl_portfolio_elite_v7_final.py (Streamlit)
\`\`\`

### Arquivos Gerados

- `dados_historicos.parquet`: Dados históricos de preços + indicadores técnicos
- `dados_fundamentalistas.parquet`: Métricas fundamentalistas normalizadas
- `metricas_performance.parquet`: Sharpe, retorno, volatilidade, drawdown
- `dados_macro.parquet`: Índices macroeconômicos (IBOV, SP500, VIX, etc.)
- `metadata.parquet`: Informações sobre a coleta (data, ativos processados, etc.)

## Uso

### 1. Primeira Execução

\`\`\`bash
# Executar coleta inicial
python coleta_diaria.py
\`\`\`

Isso irá:
- Coletar dados de TODOS os ativos configurados
- Processar indicadores técnicos (30+ indicadores via biblioteca `ta`)
- Calcular features fundamentalistas
- Salvar tudo em arquivos Parquet comprimidos

### 2. Executar o Dashboard

\`\`\`bash
streamlit run automl_portfolio_elite_v7_final.py
\`\`\`

O dashboard agora apenas **lê** os dados dos arquivos Parquet, sem fazer requisições ao yfinance (exceto na aba de Análise Individual, que tem cache de 6 horas).

### 3. Agendar Coleta Diária

#### Linux/Mac (crontab)

\`\`\`bash
# Editar crontab
crontab -e

# Adicionar linha (executa às 20:00 de segunda a sexta)
0 20 * * 1-5 cd /caminho/completo/do/projeto && /usr/bin/python3 coleta_diaria.py >> /caminho/logs/etl.log 2>&1
\`\`\`

#### Windows (Task Scheduler)

1. Abrir "Agendador de Tarefas"
2. Criar Tarefa Básica
3. Nome: "ETL Portfolio AutoML"
4. Gatilho: Diariamente às 20:00, dias úteis
5. Ação: Iniciar programa
   - Programa: `python.exe`
   - Argumentos: `coleta_diaria.py`
   - Iniciar em: `C:\caminho\do\projeto`

## Monitoramento

### Verificar Última Coleta

\`\`\`python
import pandas as pd

metadata = pd.read_parquet('./dados_financeiros/metadata.parquet')
print(metadata.iloc[-1])
\`\`\`

### Logs

O script `coleta_diaria.py` imprime logs detalhados:
- Ativos processados com sucesso
- Ativos com falha e motivos
- Tempo de execução
- Estatísticas de coleta

Redirecione para arquivo de log:

\`\`\`bash
python coleta_diaria.py >> logs/etl_$(date +%Y%m%d).log 2>&1
\`\`\`

## Troubleshooting

### Erro: "Arquivos de dados não encontrados"

**Solução**: Execute `python coleta_diaria.py` primeiro.

### Erro: "Apenas X ativos válidos carregados"

**Causas possíveis**:
- Primeira execução do ETL ainda não concluída
- Muitos ativos falharam na coleta (verificar logs)
- Símbolos selecionados não estão na lista de coleta

**Solução**: 
1. Verificar logs do ETL
2. Adicionar mais ativos à lista `TODOS_ATIVOS` em `coleta_diaria.py`
3. Re-executar ETL

### Performance

- **Tempo de coleta**: ~30-60 minutos para 200+ ativos
- **Espaço em disco**: ~50-200 MB (comprimido com gzip)
- **Cache do Streamlit**: Análise individual usa cache de 6 horas

## Manutenção

### Adicionar Novos Ativos

Editar `coleta_diaria.py`:

\`\`\`python
ATIVOS_POR_SETOR = {
    'Novo Setor': ['ATIVO1.SA', 'ATIVO2.SA', ...],
    ...
}
\`\`\`

### Alterar Período de Coleta

\`\`\`python
PERIODO_DADOS = '5y'  # Opções: '1y', '2y', '5y', 'max'
\`\`\`

### Ajustar Retry e Delays

\`\`\`python
MAX_RETRIES = 5  # Aumentar se houver muitas falhas
RETRY_DELAY = 3  # Segundos entre tentativas
\`\`\`

## Benefícios

✅ **Performance**: Dashboard carrega em segundos (vs. minutos antes)  
✅ **Estabilidade**: Sem rate limiting durante uso do dashboard  
✅ **Escalabilidade**: Fácil adicionar centenas de ativos  
✅ **Manutenibilidade**: Coleta e visualização desacopladas  
✅ **Auditoria**: Logs e metadata de cada coleta  

## Próximos Passos

- [ ] Adicionar notificações por email em caso de falha do ETL
- [ ] Implementar coleta incremental (apenas dados novos)
- [ ] Dashboard de monitoramento do ETL
- [ ] Integração com outras fontes de dados (Fundamentus, B3, etc.)
