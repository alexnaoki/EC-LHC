# EC-LHC

Repositório de códigos/programas referente ao **Mestrado em Engenharia Hidráulica e de Saneamento** do **Alex Naoki Asato Kobayashi** pelo Departamento de Engenharia Hidráulica e Saneamento da Escola de Engenharia de São Carlos na Universidade de São Paulo.

Período do Mestrado: **Fevereiro/2019** até **...**.

Qualificação aprovado: **03/03/2020** (Banca: Edson Wendland, Rodrigo Porto, Maria Mercedes)

## Fases do projeto do Mestrado
O projeto de Mestrado apresentado na Qualificação consiste basicamente em 3 (três) partes principais:
1. Definição do programa de processamento para "Eddy Covariance" com dados de baixa frequência da torre de monitoramento (IAB3).
2. Determinação da área de contribuição ("Footprint") da torre de monitoramento (IAB3).
3. Preenchimento de falhas para geração da série temporal da torre de monitoramento (Evapotranspiração/Balanço de Energia).

Para alcançar os objetivos listados acima é necessário a utilização de ferramentas computacionais para análise de dados.

### Análise dos dados processados pelo EddyPro
A análise de dados processados é a etapa final após realizado a conversão e união de dados binários a qual diversas permutações realizados foram medidas 2 principais métricas (inclinação e correlação). Com a utilização de diversas ferramentas computacionais foi possível a variação de diversas condições de dia/hora e filtros para aprimorar o entendimento dos dados. Sendo assim, possibilitando a classificação das permutações e obtendo a permutação que gera o melhor fechamento energético.

Essa análise de dados processados, por utilizar filtro de análise de contribuição, está muito ligado ao tópico abaixo.

### Análise da área de contribuição ("Footprint")
A análise de contribuição têm 3 partes essenciais: (i) dados processados, (ii) método de Kljun et al. (2015) e (iii) mapa de classificação para aceitação. Os dados processados podem ser gerados com dados de Rotação de coordenadas 2D ou Regressão planar, sendo o segundo o mais adequado. O método de Kjlun et al. (2015) foi considerado o mais adequado por ser um método capaz de considerar a heterogeneidade do campo e um método a qual o código para sua utilização foi disponibilizado pela autora. E o mapa para classificação da vegetação foi utilizando o *MapBiomas* a qual é amplamente utilizado.

### Análise do preenchimento de falhas
Descrição da análise do preenchimento de falhas

## Descrição dos códigos

### Processamento dados brutos
Este tópico tem o objetivo de mostrar as etapas necessárias para obtenção de uma série de dados brutos para o processamento.

O tutorial para utilização e recomendação em relação ao *workflow* para o processamento de dados se encontra [aqui](https://github.com/alexnaoki/EC-LHC/blob/master/info/etapas_processamento_dados_brutos.md).

### Bokeh
Para utilização dos arquivos com biblioteca *Bokeh* é necessário seguir algumas etapas descritas [aqui](https://github.com/alexnaoki/EC-LHC/blob/master/info/descricao_arquivos_bokeh.md).

Os programas desenvolvidos utilizando *Bokeh* têm como função principal a interatividade e análise dos dados. Dentre as análises estão: análise de área de contribuição com variação de períodos, visualização e análise de dados para o fechamento do balanço energético com a utilização de filtros de *Mauder e Foken (2004)*, filtro de chuva, filtro de força de sinal e filtro de *Footprint*.

### bqplot
A pasta com bqplot não é mais atualizada
> A biblioteca bqplot é interessante, porém, ela possui um performance ruim e serviu para o aprendizado de algumas ferramentas de interação.
