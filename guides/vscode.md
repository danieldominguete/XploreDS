# Visual Studio Code

This is my IDE customization recommendations for Visual Studio code.

## Extensions

- Python
- Black Formatter
- Flake 8
- Prettier
- Codeium
- GitLens
- Markdown All in One
- Data Wrangler
- Parquet-viewer
- Jupyter
- Raiwbow csv
- SQLTools


## Settings

`
{
    "workbench.colorTheme": "Default Dark Modern",
    "files.autoSave": "onFocusChange",
    "files.associations": {
        "*.rmd": "markdown"
    },
    "black-formatter.args": ["--line-length", "88"],
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
            },
      },
      "flake8.args": [
    "--max-line-length", "88",
    "--extend-ignore", "E203"
   ],
}
`
