# XploreDS

Biblioteca Python e cookbook para experimentos em Data Science

## Estrutura

```
src/xploreds/   → biblioteca (publicavel no PyPI)
cookbook/       → scripts e receitas locais
tests/          → testes automatizados
```

## Módulos da biblioteca (`src/xploreds/`)

Pacotes por domínio; cada módulo segue o prefixo `xds_` (Xplore Data Science).

| Pacote | Arquivo | Responsabilidade |
|--------|---------|------------------|
| `io` | `xds_file.py` | Pastas, JSON e persistência de `pandas` (Parquet, Excel, etc.) |
| `log` | `xds_log.py` | Logging estruturado, runs em `runs/` e perfil de ambiente |
| `utils` | `xds_environment.py` | Localizar a raiz do projeto (`pyproject.toml`) |
| `utils` | `xds_memory.py` | Conversão de tamanhos de memória (KB, MB, GB, …) |
| `datetime` | `xds_date_time.py` | Parsing, timezone e features temporais |
| `data_handle` | `xds_dataframe_pandas.py` | Criação, validação e manipulação de DataFrames |
| `data_analysis` | `xds_eda.py` | Exploração inicial e perfil do dataset |
| `data_analysis` | `xds_stats.py` | Testes, distribuições e resumos estatísticos |
| `data_analysis` | `xds_quality.py` | Schema, completude, consistência e integridade |
| `data_analysis` | `xds_outliers.py` | Detecção e tratamento de outliers |
| `data_analysis` | `xds_drift.py` | Drift de features e alvo entre bases ou no tempo |
| `data_transformation` | `xds_categorical_encoding.py` | Encoding categórico (one-hot, ordinal, etc.) |
| `data_transformation` | `xds_scaling.py` | Normalização e scaling de features numéricas |
| `data_transformation` | `xds_missing_inputation.py` | Detecção e imputação de missing values |
| `data_transformation` | `xds_datetime_encoding.py` | Extração e encoding de features de data/hora |
| `data_visualization` | `xds_plotly.py` | Gráficos interativos com Plotly |
| `models` | `xds_linear_regression.py` | Treino, predição e avaliação — regressão linear |
| `models` | `xds_logistic_regression.py` | Treino, predição e avaliação — regressão logística |
| `models` | `xds_evaluate_model.py` | Métricas, matriz de confusão e diagnósticos |
| `models` | `xds_model.py` | Contratos genéricos de modelo (em construção) |
| `features` | `xds_features.py` | Feature engineering (em construção) |

## Configuração do ambiente

Este projeto usa **[uv](https://docs.astral.sh/uv/)** para ambientes e dependencias.
Nao use `requirements.txt` nem `pip install` — tudo fica em `pyproject.toml` + `uv.lock`.

### 1. Instalar uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Inicializar ambiente do projeto

Na raiz do repositorio (onde esta o `pyproject.toml`):

```bash
uv sync --all-groups
```

O `uv sync` cria `.venv/`, instala as dependencias fixadas em `uv.lock` e registra `xploreds` em modo editavel apontando para `src/xploreds/`.

Para **criar um projeto Python novo** com uv (fora deste repo):

```bash
uv init --python 3.12
uv add pandas          # exemplo: adiciona dependencia e atualiza pyproject.toml + uv.lock
uv sync
```

### 3. Notebooks sem output no Git (nbstripout)

Este repo **nao usa** o pacote [`pre-commit`](https://pre-commit.com/) para notebooks. A limpeza e feita pelo **filtro Git** do [`nbstripout`](https://github.com/kynan/nbstripout) (ja dependencia em `pyproject.toml`).

**Uma vez por clone** (depois de `uv sync`), na raiz do repo:

```bash
uv run nbstripout --install --attributes .gitattributes
```

Isso registra o filtro em `.git/config` e versiona regras em `.gitattributes` (`*.ipynb filter=nbstripout`). No `git add`, outputs sao removidos do que entra no indice; no working tree voce pode continuar executando o notebook normalmente.

Conferir se esta ativo:

```bash
uv run nbstripout --is-installed
uv run nbstripout --status
```

Se recriar o `.venv` ou mudar o caminho do Python, reinstale o filtro:

```bash
uv run nbstripout --install --attributes .gitattributes --force
```

Limpar notebooks ja versionados (one-shot), se necessario:

```bash
git ls-files -z '*.ipynb' | xargs -0 uv run nbstripout
```

Validar CI/local sem alterar arquivos:

```bash
git ls-files -z '*.ipynb' | xargs -0 uv run nbstripout --verify
```

### 4. Execução

Validação do setup:

```bash
uv run xploredev
```

Testes:

```bash
uv run pytest
```

Exemplo de cookbook:

```bash
uv run python cookbook/sorting/test_sorting.py
```

## Versionamento semantico (Commitizen)

Commits seguem **[Conventional Commits](https://www.conventionalcommits.org/)** via [Commitizen](https://commitizen-tools.github.io/commitizen/). A versão do pacote vive em `pyproject.toml` (`[project].version`); o `cz bump` atualiza essa versão, o `CHANGELOG.md` e cria a tag Git `vX.Y.Z`.

### Commits

```bash
git add .
uv run cz commit
```

| Tipo | Quando usar | Bump |
|------|-------------|------|
| `feat` | Nova funcionalidade | minor |
| `fix` | Correcao de bug | patch |
| `docs` | Documentacao | — |
| `refactor` | Refatoracao | patch |
| `test` | Testes | — |
| `chore` | Manutencao | — |

### Release e publicacao

```bash
uv run cz bump          # bump de versao + CHANGELOG + tag
uv build                # gera dist/*.whl
uv run --with xploreds --no-project -- python -c "import xploreds"  # testar artefato
uv run --env-file .env --no-project -- uv publish              # publica no PyPI (quando configurado)
git push && git push --tags
```


