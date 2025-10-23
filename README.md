# 📈 Sistema AutoML Elite - Otimização Quantitativa de Portfólio

## Versão 6.0.0 - Elite Quantitative AutoML

Sistema completo de otimização de portfólio com Machine Learning avançado, governança de modelo e análise quantitativa profunda.

## 🎯 Recursos Principais

### Machine Learning Elite
- **9 Modelos de Ensemble**: XGBoost, LightGBM, CatBoost, Random Forest, Extra Trees, KNN, SVC, Logistic Regression, Gaussian NB
- **Ponderação por AUC-ROC**: Apenas modelos com AUC > 0.50 são incluídos, ponderados por performance
- **Validação Temporal**: TimeSeriesSplit para evitar data leakage

### Governança de Modelo
- **Monitoramento Contínuo**: Histórico móvel de 20 períodos
- **Alertas Automáticos**: 
  - AUC < 0.65 (mínimo aceitável)
  - Degradação > 5% do máximo histórico
  - Tendência de queda consistente
- **Métricas Completas**: AUC-ROC, Precision, Recall, F1-Score

### Análise Quantitativa
- **30+ Indicadores Técnicos**: RSI, MACD, Bollinger, Stochastic, ADX, ATR, CCI, Williams %R, OBV, MFI, VWAP
- **20+ Métricas Fundamentalistas**: P/L, P/VP, ROE, ROA, ROIC, margens, crescimento
- **Modelagem GARCH/EGARCH**: Volatilidade condicional
- **Modelos Estatísticos**: ARIMA, SARIMA, Prophet, VAR

### Otimização de Portfólio
- **Markowitz com GARCH**: Matriz de covariância com volatilidade condicional
- **Maximização de Sharpe Ratio**
- **Minimização de Volatilidade**
- **CVaR (Conditional Value at Risk)**: Para perfis arrojados

## 🚀 Deploy no Streamlit Cloud

### Passo 1: Preparar Repositório

1. Crie um repositório no GitHub
2. Adicione os arquivos:
   - `novo_enhanced_elite.py`
   - `requirements.txt`
   - `.streamlit/config.toml`
   - `README.md`

### Passo 2: Deploy

1. Acesse https://share.streamlit.io/
2. Conecte sua conta GitHub
3. Selecione o repositório
4. Escolha `novo_enhanced_elite.py` como arquivo principal
5. Clique em "Deploy"

### Passo 3: Configurações Recomendadas

- **RAM**: Mínimo 2GB (recomendado 4GB)
- **CPU**: 2 cores
- **Timeout**: 600 segundos

## 📊 Como Usar

### 1. Seleção de Ativos
- Escolha entre Ibovespa (87 ativos), todos os ativos (300+), setores específicos ou número fixo
- Sistema avalia todos e seleciona os 5 melhores

### 2. Questionário de Perfil
- Responda 8 perguntas baseadas em normas CVM
- Sistema determina perfil de risco e horizonte temporal
- Ponderações adaptativas por perfil

### 3. Construção do Portfólio
- Coleta de dados históricos (máximo disponível)
- Engenharia de features (100+ features)
- Treinamento de 9 modelos ML com ponderação AUC
- Otimização de alocação (Markowitz/CVaR)
- Geração de justificativas

### 4. Análise Individual
- Análise técnica completa
- Análise fundamentalista expandida
- Previsões ML e estatísticas
- Clusterização e similaridade (K-means + PCA)

### 5. Governança de Modelo
- Monitoramento de performance
- Histórico de AUC-ROC
- Alertas de degradação
- Recomendações de ação

## 🎓 Baseado nas Ementas FGV EPGE

### GRDECO222 - Machine Learning
- Ensemble de modelos
- Validação cruzada temporal
- Otimização de hiperparâmetros (Optuna)
- Métricas de classificação

### GRDECO203 - Laboratório de Ciência de Dados em Finanças
- Análise técnica e fundamentalista
- Modelagem de volatilidade (GARCH)
- Otimização de portfólio (Markowitz, CVaR)
- Modelos estatísticos (ARIMA, Prophet)

## 📝 Requisitos

- Python 3.8+
- Streamlit 1.28+
- Bibliotecas listadas em `requirements.txt`

## 🔧 Troubleshooting

### Erro de Memória
- Reduza `NUM_ATIVOS_PORTFOLIO` ou `PERIODO_DADOS`
- Use cache agressivo com `@st.cache_data`

### Timeout
- Aumente timeout nas configurações do Streamlit Cloud
- Desative otimização Optuna para análises rápidas

### Dependências
- Use versões específicas no `requirements.txt`
- Verifique compatibilidade entre bibliotecas

## 📄 Licença

MIT License

## 👨‍💻 Autor

Sistema desenvolvido para aplicação de Machine Learning e Ciência de Dados em Finanças.

---

**Versão**: 6.0.0 Elite Quantitative AutoML  
**Última Atualização**: 2025
