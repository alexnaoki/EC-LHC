# Descrição dos arquivos de Gapfilling

## Arquivos principais

- [gapfilling_iab3.py](https://github.com/alexnaoki/EC-LHC/blob/master/gapfilling/gapfilling_iab3.py)
> É o arquivo que contêm todos os processos e métricas para utilização dos processos de preenchimento de falhas

### Métodos de preenchimento de falhas

- Baseline
- Variação diurna média (mdv)
- Regressão linear multivariada (lr)
- Random Forest Regressor (rfr)
- Dense Neural Network (dnn)
- [DEPRECIADO] Long-Short Term Memory com ET (lstm_u)
- [DEPRECIADO] Long-Short Term Memory com Conv1d com ET (lstm_conv1d_u)
- Long-Short Term Memory sem ET (lstm_m_v2)
- Penman-Monteith Invertido (pm)


## Recomendação de *workflow*
Para a utilização do arquivo de preenchimento de falhas é recomendado a utilização de um arquivo Jupyter (*.ipynb*) para fácil visualização e modificação.

### Dados de entrada
- Diretório dos dados processados pelo EddyPro
- Diretório dos dados de baixa frequência do IAB3
- Diretório dos dados do IAB1
- Diretório dos dados do IAB2
- Arquivo de Footprint gerado pelo [programa](https://github.com/alexnaoki/EC-LHC/blob/master/bokeh/view_footprint_k15_bokeh_map02.py)

### Utilização no arquivo Jupyter (*.ipynb)

A primeira etapa é a importação do programa [gapfilling_iab3.py](https://github.com/alexnaoki/EC-LHC/blob/master/gapfilling/gapfilling_iab3.py).

```
import sys
sys.path.append(r'[DIRETÓRIO DO gapfilling_iab3.py]')
from gapfilling_iab3 import gapfilling_iab3
```

Após a importação do programa, deve-se adicionar os dados de entrada no programa.

```
a = gapfilling_iab3(
    ep_path=r'[DIRETÓRIO DOS DADOS PROCESSSADOS PELO EDDYPRO]',
    lf_path=r'[DIRETÓRIO DOS DADOS DE BAIXA FREQUÊNCIA DO IAB3]',
    iab1_path=r'[DIRETÓRIO DOS DADOS DO IAB1]',
    iab2_path=r'[DIRETÓRIO DOS DADOS DO IAB2]',
    footprint_file=r'[ARQUIVO DE FOOTPRINT]',
    )
```

Exemplo da importação dos dados:

```
a = gapfilling_iab3(
    ep_path=r'G:\Meu Drive\USP-SHS\Resultados_processados\EddyPro_Fase01020304',
    lf_path=r'G:\Meu Drive\USP-SHS\Mestrado\Dados_Brutos\IAB3',
    iab1_path=r'G:\Meu Drive\USP-SHS\Mestrado\Dados_Brutos\IAB1\IAB1',
    iab2_path=r'G:\Meu Drive\USP-SHS\Mestrado\Dados_Brutos\IAB2\IAB2',
    footprint_file=r'G:\Meu Drive\USP-SHS\Resultados_processados\Footprint\classification_pixel_2018-10-05-00-30to2021-01-08-00-00_pf_90_2.csv'
)
```

A mensagem da leitura dos dados deve ser semelhante a seguinte mensagem:

```
Reading IAB3_EP files...
# IAB3_EP: 5	Inicio: 2018-10-05 00:30:00	Fim: 2021-01-08 00:00:00
Reading IAB3_LF files...
# IAB3_LF: 41	Inicio:2018-09-07 19:30:00	Fim: 2021-01-08 09:30:00
Reading IAB2 files...
# IAB2: 38	Inicio: 2017-02-03 09:20:00	Fim: 2021-01-22 11:30:00
Reading IAB1 files...
# IAB1: 12	Inicio: 2015-07-17 15:00:00	Fim: 2021-01-22 10:50:00
Reading Footprint file...
Inicio: 2018-10-05 00:30:00	Fim: 2021-01-08 00:00:00
Duplicatas:  0
Verificacao de Duplicatas:  0
Duplicatas:  1
Verificacao de Duplicatas:  0
Duplicatas:  840479
Verificacao de Duplicatas:  0
Duplicatas:  507541
Verificacao de Duplicatas:  0
```

Após a correta importação dos dados para o programa, você poderá utilizar o programa para o preenchimento de falhas e verificação de diversas métricas.

Cada método de preenchimento de falhas possui um código que é correspondente, a qual é encontrado listado entre parênteses em cima.
Para o preenchimento de falhas, o seguinte comando deve ser utilizado:

```
# Método de Variação diurna média
a.fill_ET(listOfmethods=['mdv'])

# Ou aplicar diversos métodos de uma vez
a.fill_ET(listOfmethods=['mdv','lr','rfr','dnn','lstm_m_v2'])

# Juntar dados do ET (ótimo sem preenchimento de falhas) + ET (preenchidos)
a.join_ET()
```

Após processados os dados, os comandos mais relevantes são:

```
# Visualização das falhas dos métodos de preenchimento de falhas
a.stats_ET(stats=['gaps'])

# Análise de correlação dos dados de ET real x ET estimado
a.stats_ET(stats=['corr'])

# Dados acumulados dos métodos de ET
a.stats_ET(stats=['sum'])

# Dados do ET (mm/day) médio anual e por estação do ano
a.stats_ET(stats=['daily'])

# Aplicação das métricas de erro (MAE, MBE e RMSE)
a.stats_ET(stats=['error'])

# Análise da normalidade dos dados de ET estimados
a.stats_ET(stats=['normality'])

# Visualização dos heatmaps totais de ET com dados preenchidos
a.stats_ET(stats=['heatmap'])
```

Ainda, existem diversos outras funções que podem ser utilizadas para a análise dos dados.

Por fim, caso seja necessário a extração dos dados preenchidos para outras análises é possível a criação de arquivos *.csv* com o seguinte código.

```
# Lista de métodos para gerar o arquivo .csv
methods = ['ET','ET_mdv_[3, 5, 7]', 'ET_lr', 'ET_rfr','ET_pm', 'ET_dnn', 'ET_lstm_m_v2']

# Loop para os métodos
for method in methods:
    df = a.iab3_ET_timestamp.copy()
    print((df.set_index('TIMESTAMP').resample('1d')[method].sum(min_count=48)/2))

    df_resample = (df.set_index('TIMESTAMP').resample('1d')[method].sum(min_count=48)/2)

# São gerados dados com pelo menos 30 valores no dia, sendo que o máximo é 48.
(df.set_index('TIMESTAMP').resample('1d')[methods].sum(min_count=30)/2).to_csv('daily_ET_all.csv')
```
