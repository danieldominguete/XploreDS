# XploreDS

Library and cookbook codes for Data Science Projects

## Environment Setup

1. Install UV

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv run cz commit    # em vez de git commit -m
```

2. Install dependencies

```bash
uv sync --group dev
```

3. Install commit hooks (validates Conventional Commits)

```bash
uv run pre-commit install --hook-type commit-msg
```

## Versionamento semântico (Commitizen)

Este projeto usa [Conventional Commits](https://www.conventionalcommits.org/) e [Commitizen](https://commitizen-tools.github.io/commitizen/) para versionamento semântico.

### Commits

Use o assistente interativo em vez de `git commit -m`:

```bash
uv run cz commit
```

Tipos comuns:

| Tipo | Quando usar | Bump |
|------|-------------|------|
| `feat` | Nova funcionalidade | minor |
| `fix` | Correção de bug | patch |
| `docs` | Documentação | — |
| `refactor` | Refatoração | patch |
| `test` | Testes | — |
| `chore` | Manutenção | — |

Breaking change: adicione `!` após o escopo (`feat(api)!: ...`) ou `BREAKING CHANGE:` no corpo.

### Release

Gera nova versão, atualiza `CHANGELOG.md`, `pyproject.toml`, `uv.lock` e cria tag Git:

```bash
uv run cz bump
```

Pré-visualizar próxima versão:

```bash
uv run cz bump --dry-run
```

Na primeira release (sem tags Git ainda), confirme com `--yes`:

```bash
uv run cz bump --yes
```

### Versão atual

A versão canônica fica em `pyproject.toml` (`project.version`) e é espelhada em `xploreds/__init__.py`.
