# Aprend_Fed_SBPO2026

## Visão Geral

Este projeto implementa uma solução de otimização multi-objetivo para Aprendizado Federado (AF), inspirada pelo trabalho documentado em `SBPO_2026_Nilo.pdf`.

A proposta integra seleção de clientes e alocação de recursos na borda, buscando equilibrar os trade-offs entre:
- consumo energético,
- tempo de convergência,
- desempenho do modelo global.

O problema é modelado como uma formulação derivada de programação não linear e inteiro mista, resolvida com algoritmos evolutivos baseados em NSGA-II.

## Estrutura do Projeto

- `multi_objetivo.py`: executa a otimização multi-objetivo usando configurações JSON e apresenta resultados de Pareto.
- `evolucao.py`: realiza uma evolução iterativa das configurações, salvando soluções e gráficos de fronteira de Pareto em `Figuras/`.
- `instancia/parse.py`: carrega configurações JSON e converte parâmetros em vetores NumPy.
- `instancia/random_instance.py`: gera arquivos de configuração aleatórios para testes de instância.
- `FLPOPT/`: contém a modelagem do problema, o solver e utilitários de visualização.
- `Dados/`: dados de métricas e instâncias de teste.
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
python multi_objetivo.py --config config_n22.json
```

Gerar uma instância aleatória de configuração:
```bash
python instancia/random_instance.py --N 20 --output config_random.json
```

Rodar a evolução iterativa e salvar resultados em CSV e PNG:
```bash
python evolucao.py --config config_n22.json
```

## Contexto e Referência

O arquivo `SBPO_2026_Nilo.pdf` contextualiza o problema central do projeto: a alocação integrada de recursos e seleção de clientes em redes de borda para Aprendizado Federado. O estudo mostra como uma abordagem multi-objetivo pode reduzir consumo de energia enquanto preserva a acurácia do modelo global.

## Notas

- As configurações de entrada (`config_n22.json`, `config_random.json`) definem parâmetros como número de clientes, capacidades de CPU, limites de frequência e condições iniciais.
- O módulo `FLPOPT` abstrai a criação do problema e a execução do solver evolutivo.


