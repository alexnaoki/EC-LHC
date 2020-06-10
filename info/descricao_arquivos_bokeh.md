# Organização dos arquivos Bokeh

## Arquivos Principais .py para EC-LHC
Apesar de haver diversos arquivos .py e .ipynb dentro do [Github](https://github.com/alexnaoki/EC-LHC). Abaixo será listado os mais relevantes.

- [view_files_bokeh_serve01.py](https://github.com/alexnaoki/EC-LHC/blob/master/bokeh/view_files_bokeh_serve01.py)
- [view_footprint_k15_bokeh_map01.py](https://github.com/alexnaoki/EC-LHC/blob/master/bokeh/view_footprint_k15_bokeh_map01.py)
- [view_footprint_k15_bokeh_map02.py](https://github.com/alexnaoki/EC-LHC/blob/master/bokeh/view_footprint_k15_bokeh_map02.py)

## Arquivos Secundários .py para EC-LHC
Esses arquivos secundários foram criados para auxiliar ou aprimorar o conhecimento da ferramenta, ou o entendimento de uma determinada metodologia. Alguns desses arquivos são utilizados no Jupyter Notebook.

- [view_footprint_k15_bokeh01.py](https://github.com/alexnaoki/EC-LHC/blob/master/bokeh/view_footprint_k15_bokeh01.py)
- [view_footprint_k15_bokeh02.py](https://github.com/alexnaoki/EC-LHC/blob/master/bokeh/view_footprint_k15_bokeh02.py)
- [[Notebook] view_files_bokeh_wind_jupyter.py](https://github.com/alexnaoki/EC-LHC/blob/master/bokeh/view_files_bokeh_wind_jupyter.py)
- [[Notebook] view_files_bokeh_et_jupyter.py](https://github.com/alexnaoki/EC-LHC/blob/master/bokeh/view_files_bokeh_et_jupyter.py)
- [[Notebook] view_files_bokeh_jupyter.py](https://github.com/alexnaoki/EC-LHC/blob/master/bokeh/view_files_bokeh_jupyter.py)

## Como utilizar os programas que utilizam Bokeh (Python)
Existem dois tipos de arquivos que utilizam o Bokeh:

- Arquivo Python (.py)
- Arquivo Notebook (.ipynb)

Apesar de ambos utilizarem Python, seus propósitos diferem.

### Arquivos .py (Arquivos Python)
A utilização dos arquivos **.py** requer a utilização do "Anaconda Command Prompt" ou "Command Prompt" (caso o "variable path" esteja configurado corretamente).

É recomendado a utilização de um ambiente virtual para a utilização (ao invés de usar "(base)").

Primeiramente, entre no ambiente virtual escolhido. E digite o seguinte comando no "Command Prompt"

```
bokeh serve --show [PATH DO ARQUIVO .py]
```

Para a utilização de alguns arquivos .py, deve-se atentar para evitar o conflito de diretórios (pacotes). Em suma, evita mexer arquivos individualmente, se for mover um arquivo mova a pasta.


### Arquivos .ipynb (Arquivos Python Notebook)
Primeiramente é necessário abrir o arquivos **.ipynb** através do Jupyter Notebook ou JupyterLab.

Após aberto o Jupyter é necessário verificar se está com "Kernel" correto. Verifica-se o "Kernel" através do canto superior direito. Ainda, recomenda-se utilizar um Ambiente Virtual para utilização (assim como os arquivos .py).

Com o arquivo .ipynb aberto e com o "Kernel" correto basta verificar se a importação se encontra e se encontra corretamente. Em seguida basta iniciar o programa.
