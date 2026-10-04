# XploreDS

Library and cookbook codes for Data Science Projects.

## Estrutura

```
src/xploreds/   → biblioteca (publicavel no PyPI)
cookbook/       → scripts e receitas locais
tests/          → testes automatizados
```

## Environment Setup

Este projeto usa **[uv](https://docs.astral.sh/uv/)** para ambientes e dependencias.
Nao use `requirements.txt` nem `pip install` — tudo fica em `pyproject.toml` + `uv.lock`.

### 1. Instalar uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Instalar dependencias (modo editavel)

```bash
uv sync --all-groups
```

Isso cria `.venv/`, instala libs externas e registra `xploreds` em modo editavel apontando para `src/xploreds/`.

### 3. Hooks de commit (Commitizen)

```bash
uv run pre-commit install --hook-type commit-msg
```

### 4. Validar setup

```bash
uv run pytest
uv run python static/00_templates/script.py
```

## Versionamento semantico (Commitizen)

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
uv publish              # publica no PyPI (quando configurado)
```

Na primeira release (sem tags Git), use `--yes`:

```bash
uv run cz bump --yes
```
