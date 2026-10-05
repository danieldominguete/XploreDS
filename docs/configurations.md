# Configurações do Projeto

## Estrutura

```
XploreDS/
├── src/xploreds/     # biblioteca publicavel (PyPI)
├── cookbook/         # scripts locais de exemplo
├── tests/            # pytest
├── data/             # datasets locais
├── static/           # templates
└── docs/
```

## IDE (Cursor / VS Code)

Arquivos em `.vscode/`:

| Arquivo | Uso |
|---------|-----|
| `launch.json` | Debug (F5) no **Terminal integrado** |
| `settings.json` | Interpretador `.venv`, `.env` automatico |
| `tasks.json` | Rodar script no terminal via **Cmd+Shift+B** |

Para executar scripts Python com logs no terminal, use **Cmd+Shift+B** ou:

```bash
uv run python static/00_templates/reciple_script.py
```

Os logs tambem sao gravados em `runs/<script>_<timestamp>/`.

## Notebooks (`.ipynb`)

O hook **nbstripout** (pre-commit) remove outputs de notebooks antes de cada commit.

```bash
uv run pre-commit install
uv run pre-commit run nbstripout --all-files
```

## uv — desenvolvimento local

O `uv sync` instala dependencias externas **e** o pacote `xploreds` em modo **editavel**.
Edicoes em `src/xploreds/*.py` refletem imediatamente, sem reinstall.

```bash
uv sync --all-groups
uv run python -c "import xploreds; print(xploreds.__file__)"
uv run pytest
```

Saida esperada do import:

```text
.../XploreDS/src/xploreds/__init__.py
```

## Comandos uv

| Acao | Comando |
|------|---------|
| Instalar deps + lib editavel | `uv sync --all-groups` |
| Adicionar dependencia runtime | `uv add <pacote>` |
| Adicionar dependencia de dev | `uv add --dev <pacote>` |
| Executar script do cookbook | `uv run python cookbook/.../script.py` |
| Rodar testes | `uv run pytest` |
| Build para PyPI | `uv build` |
| Publicar PyPI | `uv publish` |

## Onde ficam as dependencias

- **Runtime** → `[project].dependencies` em `pyproject.toml`
- **Dev** → `[dependency-groups].dev`
- **Versoes fixas** → `uv.lock`
- **Codigo da lib** → `src/xploreds/`

A versao canonica fica em `pyproject.toml` (`project.version`) e e espelhada em `src/xploreds/__init__.py`.
