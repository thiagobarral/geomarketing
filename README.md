# Geomarketing | Expansão de lojas para São Paulo

Estudo de **Ciência de Dados aplicada à expansão comercial**: estimar faturamento e classificar o potencial de bairros de São Paulo a partir de dados de lojas e características demográficas do Rio de Janeiro.

**Autor:** [Thiago Barral](https://github.com/thiagobarral) · [LinkedIn](https://www.linkedin.com/in/thiago-barral-data-expert)

## Problema de negócio

Uma empresa alimentícia sediada no RJ deseja abrir filiais em SP, com público-alvo de adultos de 25 a 50 anos das classes A e B. O estudo combina informações de população, faixas etárias, domicílios por renda e renda média para apoiar uma primeira análise de oportunidades.

As saídas são estimativas exploratórias por bairro, não garantias de retorno nem substitutos de um estudo de ponto comercial.

## Dados disponíveis

| Conjunto | Bairros | Uso |
| --- | ---: | --- |
| Rio de Janeiro | 160 | Treinamento e avaliação: faturamento e potencial observados |
| São Paulo | 296 | Estimativas: faturamento e potencial sem valores observados |

A planilha `DadosDesafioCientista.xlsx` contém 456 linhas e 24 colunas. O código utiliza 17 preditores: oito faixas etárias, oito faixas de renda domiciliar e renda média. Identificadores, nome, cidade, estado e os dois alvos não entram como preditores. A população total é omitida por já estar representada pelas faixas etárias.

Valores de renda não numéricos são convertidos em ausentes. A mediana é aprendida somente nos dados de treino de cada divisão, dentro de um `Pipeline`. Linhas não são excluídas por renda ausente e bairros não são interpolados pela ordem da planilha. Registros de SP com população zero ou renda ausente recebem sinalizadores na saída.

**Origem:** base do desafio descrito neste repositório. A data de referência, regras de rotulagem de potencial, período e unidade monetária do faturamento e condições de redistribuição precisam de documentação adicional. Não se assume que a base representa o mercado atual.

## Metodologia revisada

1. Separação de 70% dos bairros do RJ para treino e 30% para teste, estratificada pelo potencial, com semente 7. Ambos os problemas usam os mesmos bairros reservados.
2. Validação cruzada de quatro partes apenas no treino para escolher modelos e hiperparâmetros. Imputação e padronização são ajustadas dentro de cada parte.
3. Regressão: baseline de média, regressão linear, SVR e árvore de decisão; seleção pelo menor RMSE médio de validação.
4. Classificação: baseline da classe majoritária, regressão logística, KNN, SVM e Random Forest; seleção pelo maior F1 macro médio de validação.
5. Avaliação dos modelos selecionados no teste reservado, sem usá-lo para escolher candidatos.
6. Reajuste dos modelos selecionados em todos os 160 bairros do RJ para estimar os bairros de SP.

A busca da árvore avalia nove configurações, em lugar da busca excessiva do notebook histórico. Todas as fontes de aleatoriedade dos modelos usados possuem semente definida. Os escores completos de validação, parâmetros e matriz de confusão estão em [reports/metrics.json](reports/metrics.json). Na regressão, `cv_score` é o **negativo do RMSE**, convenção do scikit-learn.

## Resultados da execução revisada

Execução em 26/10/2026, Python 3.12 e dependências de `requirements.txt`. Treino: 112 bairros; teste: 48 bairros do RJ.

| Tarefa | Modelo escolhido por validação cruzada | Métrica no teste |
| --- | --- | ---: |
| Faturamento | Árvore de decisão | RMSE: **132.993,36** |
| Faturamento | Árvore de decisão | R²: **0,895** |
| Potencial | Random Forest | Acurácia: **79,17%** |
| Potencial | Random Forest | F1 macro: **0,795** |

O RMSE está na mesma unidade numérica do faturamento da planilha; seu período e unidade monetária não estão confirmados. Os resultados acima foram recalculados pelo fluxo revisado. O baseline é incluído na comparação por validação cruzada.

**Não há resultados observados de SP para medir a precisão das estimativas nessa cidade.** R² e métricas de classificação acima são exclusivamente do RJ. As porcentagens e conclusões comerciais das exportações históricas não são resultados desta revisão.

## Como executar

Requer **Python 3.12**. Execute os comandos a partir da raiz do repositório:

```bash
python -m venv .venv
```

Ative o ambiente:

```bash
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Instale as dependências e execute:

```bash
python -m pip install -r requirements.txt
python scripts/train.py
python -m unittest discover -s tests -v
```

Os arquivos são gerados em `outputs/`, diretório ignorado pelo Git:

- `metrics.json`: protocolo, modelos, escores de validação e métricas de teste.
- `previsoes_sp.csv`: 296 bairros com código, nome, cidade, estado, faturamento estimado, potencial estimado e sinalizadores de qualidade. Estimativas negativas de faturamento são limitadas a zero somente na saída de SP; o RMSE de teste usa previsões sem esse ajuste.

Para outros caminhos:

```bash
python scripts/train.py --input DadosDesafioCientista.xlsx --output-dir outputs
```

Para explorar pelo notebook:

```bash
python -m pip install -r requirements-notebook.txt
python -m jupyter lab
```

Abra [geomarketing.ipynb](geomarketing.ipynb) e execute as células em ordem. Não é necessário Google Drive, Sweetviz ou Verstack. O notebook usa o mesmo fluxo do script e grava novas saídas em `outputs/`.

## Estrutura

| Caminho | Conteúdo |
| --- | --- |
| `geomarketing.ipynb` | Exploração e execução guiada do fluxo revisado |
| `scripts/train.py` | Preparação, seleção, avaliação e exportação |
| `tests/test_training.py` | Verificações de preditores, imputação, seleção e base |
| `requirements.txt` | Versões usadas na execução verificada |
| `requirements-notebook.txt` | Dependências opcionais de Jupyter |
| `reports/metrics.json` | Registro da execução revisada |
| `DadosDesafioCientista.xlsx` | Base de entrada preservada |
| `Previsaodefaturamentoepotencial (2).xlsx` | Exportação histórica, não regenerada |
| `Análise do faturamento comparativo (2).html` | Relatório exploratório histórico, não regenerado |

O notebook anterior está preservado no [histórico do repositório](https://github.com/thiagobarral/geomarketing/commits/main/geomarketing.ipynb). O Power BI citado na descrição original não está disponível como arquivo `.pbix` neste repositório.

## Limitações e próximos passos

- **Transferência geográfica:** desempenho em bairros do RJ não comprova generalização para SP; avaliar com resultados reais de novas lojas e divisões espaciais.
- **Amostra pequena:** 48 observações de teste geram estimativas sensíveis à divisão; não foram calculados intervalos de confiança.
- **Dependência espacial:** bairros próximos podem compartilhar características, tornando a divisão aleatória otimista.
- **Dados incompletos:** revisar bairros sem população informada e confirmar a adequação da imputação antes de decisões.
- **Público-alvo:** faixas agregadas de 25–34 e 35–49 anos não isolam exatamente o intervalo de 25–50; não foi criada uma nova segmentação de aderência ao público-alvo nesta revisão.
- **Contexto comercial:** faltam concorrência, aluguel, fluxo de pessoas, acessibilidade e desempenho real de SP. As previsões não mensuram impacto financeiro de uma implantação.
- **Governança:** documentar proveniência, período, unidade dos alvos e direitos de uso dos dados e escolher uma licença de código. Nenhuma licença é presumida nesta revisão.
