# Catalogar o pacote no pypi

1) Gerar pacote localmente (tar.gz)
  
    `python setup.py sdist`

2) Upload do pacote para o site de testes

    `twine upload --repository-url https://test.pypi.org/legacy/ dist/*`

3) Upload do pacote para o site oficial

    `twine upload dist/*`

Opcional: Gerar pacote whl

`python3 setup.py bdist_wheel`