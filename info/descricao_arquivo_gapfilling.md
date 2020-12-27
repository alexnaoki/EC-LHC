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
- Long-Short Term Memory com ET (lstm_u)
- Long-Short Term Memory com Conv1d com ET (lstm_conv1d_u)
- Long-Short Term Memory sem ET (lstm_m)
- Penman-Monteith Invertido (pm)


## Recomendação de *workflow*
Para a utilização do arquivo de preenchimento de falhas é recomendado a utilização de um arquivo Jupyter (*.ipynb*) para fácil visualização e modificação.
