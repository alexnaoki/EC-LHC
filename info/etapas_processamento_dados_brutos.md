# Etapas para o processamento de dados brutos
O contínuo processamento de dados brutos para dados processados é de vital importância para a análise de dados e verificação de possíveis erros.

O estado inicial dos dados é o dado ainda no cartão de memória CFM e o estado final que estamos procurando obter é antes de seu processamento. Ou seja, necessita-se realizar uma série de operações para adequá-lo para o processamento.

Essa série de processos foi desenvolvida especificamente para os dados da torre de monitoramento IAB 3, localizado no município de Itirapina/SP.

## Pré-requisitos
As etapas a serem utilizadas requerem a instalação de alguns programas fora do repositório.

- Clone este repositório;
- 'toa_to_tob1' (LoggerNet - CampbellScientific)
- (!) EasyFlux
- JupyterLab ou JupyterNotebook

## Tutorial de utilização
O seguinte programa identifica arquivos tipo '.dat' onde possui em seu nome 'TOA5_11341.ts_data' a qual é a nomenclatura para dados tipo 'TOA5' e alta frequência (20 Hz). O programa é capaz de ler "família de pastas", ou seja, deve-se fornecer a "pasta mãe" dos arquivos que se deseja ajeitar. Logo, o programa é capaz de unir 2 arquivos 'TOA5' de um mesmo dia.

Porém, para o processamento de dados no *EddyPro* e *EasyFlux* é necessário a utilização de arquivos do tipo **'TOB1'**, sendo assim, necessário a conversão dos arquivos unidos.

Para realizar a união de arquivos TOA5 e seguinte conversão para TOB1 é necessário a utilização do programa [Merger_v3.py](git\EC-LHC\raw_data_management\Merger_v3.py)
