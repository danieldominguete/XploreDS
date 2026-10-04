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
| `tasks.json` | **Run Python File in Terminal** via `Cmd+Shift+B` |

### Onde a saida aparece

| Acao na IDE | Onde ver a saida |
|-------------|------------------|
| ▶ **Run Python File** (play simples) | Painel **Output → Python** (nao no Terminal) |
| ▶ **Run Python File in Terminal** (dropdown do play) | **Terminal** integrado |
| **F5** Debug | **Terminal** integrado (`launch.json`) |
| **Cmd+F5** Run Without Debugging | **Terminal** integrado (`launch.json`) |
| **Cmd+Shift+B** | **Terminal** integrado (`tasks.json`) |

### Como rodar e ver logs no Terminal

1. **Recomendado:** abra `static/00_templates/script.py` e pressione **Cmd+F5** (Run Without Debugging).
2. Alternativa: **Cmd+Shift+B** (task padrao "Run Python File in Terminal").
3. Alternativa: clique na **seta do botao play** → **Run Python File in Terminal**.
4. Pelo terminal manual: `uv run python static/00_templates/script.py`.

Se usar o play simples (sem dropdown), abra **View → Output** e selecione **Python** no dropdown — a saida esta la, nao no Terminal.

### Checklist se o Terminal continuar vazio

- Interpretador: **Python: Select Interpreter** → `.venv/bin/python`
- Arquivo `.env` na raiz (copie de `.env.example`)
- Aba **Terminal** visivel (nao confundir com **Output** ou **Debug Console**)

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
