# Aprend_Fed_SBPO2026

## Visão Geral

Este projeto implementa uma solução de otimização multi-objetivo para Aprendizado Federado (AF), inspirada pelo trabalho documentado em `SBPO_2026_Nilo.pdf`.

A proposta integra seleção de clientes e alocação de recursos na borda, buscando equilibrar os trade-offs entre:
- consumo energético,
- tempo de convergência,
- desempenho do modelo global.

O problema é modelado como uma formulação derivada de programação não linear e inteiro mista, resolvida com algoritmos evolutivos baseados em NSGA-II.

## Estrutura do Projeto

- `FLPOPT/`: pacote principal do projeto, com o problema, o solver e utilitários de otimização.
- `use_cases/optimization/`: scripts de execução para otimização multi-objetivo e para rodar uma instância específica.
- `use_cases/evolution/`: script para simular a evolução iterativa e salvar resultados em `evolucao_teorica/`.
- `use_cases/analysis/`: scripts para análise e visualização dos resultados armazenados.
- `instancia/parse.py`: carrega configurações JSON e converte parâmetros em vetores NumPy.
- `instancia/random_instance.py`: gera arquivos de configuração aleatórios para testes de instância.
- `configs/`: arquivos JSON de configuração usados pelos fluxos de execução.
- `evolucao_teorica/`: artefatos gerados pelas simulações (CSV, logs e entradas de rodada).
- `Figuras/`: gráficos de resultados gerados pelos scripts.

## Dependências

As dependências principais incluem:
- Python 3.x
- NumPy
- pandas
- pymoo

Recomenda-se usar um ambiente virtual e instalar dependências antes de executar os scripts.

## Como Executar

Executar a otimização multi-objetivo:
```bash
python use_cases/optimization/multi_objetivo.py --config configs/config_afea_n11.json
```

Executar uma instância específica do AFEA:
```bash
python use_cases/optimization/run_afea_instance.py --config configs/config_afea_n11.json
```

Gerar uma instância aleatória de configuração:
```bash
python instancia/random_instance.py --N 20 --output configs/config_random.json
```

Rodar a evolução iterativa e salvar resultados em CSV e PNG:
```bash
python use_cases/evolution/evolucao.py
```

Analisar os resultados gerados:
```bash
python use_cases/analysis/analise.py --config configs/config_afea_n11.json
```

## Contexto e Referência

O arquivo `SBPO_2026_Nilo.pdf` contextualiza o problema central do projeto: a alocação integrada de recursos e seleção de clientes em redes de borda para Aprendizado Federado. O estudo mostra como uma abordagem multi-objetivo pode reduzir consumo de energia enquanto preserva a acurácia do modelo global.

## Notas

- As configurações de entrada (`config_n22.json`, `config_random.json`) definem parâmetros como número de clientes, capacidades de CPU, limites de frequência e condições iniciais.
- O módulo `FLPOPT` abstrai a criação do problema e a execução do solver evolutivo.


