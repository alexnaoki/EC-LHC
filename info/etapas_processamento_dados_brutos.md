# Etapas para o processamento de dados brutos
O contínuo processamento de dados brutos para dados processados é de vital importância para a análise de dados e verificação de possíveis erros.

O estado inicial dos dados é o dado ainda no cartão de memória CFM e o estado final que estamos procurando obter é antes de seu processamento. Ou seja, necessita-se realizar uma série de operações para adequá-lo para o processamento.

Essa série de processos foi desenvolvida especificamente para os dados da torre de monitoramento IAB 3, localizado no município de Itirapina/SP.

## Pré-requisitos
As etapas a serem utilizadas requerem a instalação de alguns programas fora do repositório.

- Clone este repositório;
- Python (3.x)
- 'toa_to_tob1' (LoggerNet - CampbellScientific)

## Tutorial de utilização
O seguinte programa identifica arquivos tipo '.dat' onde possui em seu nome 'TOA5_11341.ts_data' a qual é a nomenclatura para dados tipo 'TOA5' e alta frequência (20 Hz). O programa é capaz de ler "família de pastas", ou seja, deve-se fornecer a "pasta mãe" dos arquivos que se deseja ajeitar. Logo, o programa é capaz de unir 2 arquivos 'TOA5' de um mesmo dia.

Porém, para o processamento de dados no *EddyPro* e *EasyFlux* é necessário a utilização de arquivos do tipo **'TOB1'**, sendo assim, necessário a conversão dos arquivos unidos.

Para realizar a união de arquivos TOA5 e seguinte conversão para TOB1 é necessário a utilização do programa [Merger_v3.py](https://github.com/alexnaoki/EC-LHC/blob/master/raw_data_management/Merger_v3.py). Sendo necessário ter o diretório das pastas com arquivos 'TOA5' e 'TOB1' e o caminho para o programa externo 'toa_to_tob1' da 'Campbellsci'. Por exemplo:

```
- DadosDeCampo
-- 05/09/2019
--- TOA5_11341.ts_data12.dat
--- ...
--- TOB1_11341.ts_data12.dat
--- ...
-- 20/09/2019
--- ...
-- 08/10/2019

- CópiaDadosDeCampo
-- 05/09/2019
--- ...
-- ...
```

O programa **Merger_v3.py** possui 5 funções:
- Identificação e união de arquivos de mesmo dia ('TOA5');
- Cópia de arquivos 'TOB1';
- Conversão de arquivos unidos de 'TOA5' para 'TOB1';
- Junção de arquivos unidos e arquivos copiados para uma mesma pasta;
- Renomeação de arquivos para processamento no *EddyPro*.

Para iniciar o programa é necessário atender os pré-requisitos e abrir com o IDLE (e.g. Spyder, Atom, outros) o programa e adicionar a variável para iniciar a classe.
```python
if __name__ == '__main__':
    # Main folder
    # path_teste = r'E:\Teste_merger_fase01'
    # path_teste = r'E:\Teste_merger_fase02'
    path_teste = r'E:\Teste_merger_func_fase03'

    # Start program (Input: Main folder containing other folder containing TOA5 and TOB1 data)
    a = Merger_TOA5(path=path_teste)
```
Utilizar a função para identificar e unir.
```python
    # Identify and Merge data from same day
    a.identifyandmerge_sameday()
```
Copiar arquivos TOB1 em uma nova pasta.
```python
    # Copy Tob1 files to newfolder
    a.copy_tob_files()
```
Converter arquivos TOA5 para TOB1 (NECESSÁRIO PROGRAMA 'toa_to_tob1').
```python
    # Convert TOA5 files to tob1 using merge folder
    a.convert_toa_to_tob1(path_toa_to_tob1=r"C:\Program Files (x86)\Campbellsci\LoggerNet\toa_to_tob1")
```
Copiar arquivos TOB1 que foram unidos para a pasta onde estão os arquivos TOB1 completos.
```python
    # Copy converted TOB1 files to tob1_complete folder
    a.join_mergedFiles_fullFiles()
```
Criar uma nova pasta com os arquivos TOB1 prontos para a utilização no *EddyPro*.
```python    
    # Create new folder and rename TOB1 files to adjust for processing in EddyPro
    a.copying_renaming_tobFiles()
```
O programa é capaz de verificar se um arquivo existe, ou seja, ele não copiará e converterá arquivos cujo já possui na pasta *'/merge_{NOME DA PASTA}/'*, *'/tob1_complete_{NOME DA PASTA}/'* e *'/tob1_date/{NOME DA PASTA}'*.

O *workflow* recomendado para a utilização do programa é de ter uma pasta copiada e separadas dos arquivos armazenados do campo => E sempre adicionar novas subpastas toda vez que ir ao campo => Rodar o programa => Verificar resultado.



Ainda, o programa permite criar uma pasta a qual os arquivos encontram-se renomeados, prontos para utilização em programas como *EddyPro*.
