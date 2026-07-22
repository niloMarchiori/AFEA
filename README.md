# Aprend_Fed_SBPO2026

## Visão Geral e Definição do Problema

Este projeto implementa uma solução de otimização multi-objetivo para Aprendizado Federado (AF), focando na alocação integrada de recursos e seleção de clientes em ambientes de computação de borda (*Edge Computing*).

O processo de treinamento do Aprendizado Federado em cenários reais enfrenta restrições severas de hardware e tempo. O problema central (modelado como Programação Não-Linear Inteiro-Mista) consiste em determinar, a cada rodada de treinamento:
1. **Seleção de Dispositivos ($\beta_n$):** Quais clientes participarão ativamente (Variável Binária).
2. **Iterações Locais ($\psi_n$):** A quantidade de épocas de treinamento executadas no dispositivo local (Variável Inteira).
3. **Frequência de CPU ($f_n$):** A velocidade de processamento dedicada por cada cliente (Variável Contínua).
4. **Acurácia Alvo Local ($\theta_n$):** O limiar de desempenho local a ser atingido (Variável Contínua).
5. **Tempo Global da Rodada ($T$):** O tempo estrito imposto para a rodada, limitado pelos atrasos de computação (Variável Contínua).

O problema é matematicamente estruturado para minimizar simultaneamente três objetivos estritamente conflitantes:

1. **$F_1$ (Energia Global):** Minimizar o consumo energético dinâmico total na borda.
$$ \min F_1 = \sum_{n=1}^{N} \beta_n \cdot \psi_n \cdot G(\theta_n) \cdot \frac{\alpha_n}{2} \cdot c_n \cdot S_n \cdot f_n^2 $$

2. **$F_2$ (Justiça na Seleção):** Maximizar a participação de dispositivos excluídos (formulado via minimização negativa), ponderada pela quantidade de dados locais ($S_n$) e pelo histórico de rodadas sem participação ($U_n$).
$$ \min F_2 = -\sum_{n=1}^{N} \beta_n \left( 1 + \frac{S_n}{\sum S_i} \cdot U_n \right) $$

3. **$F_3$ (Tempo Mínimo de Convergência):** Minimizar o gargalo temporal global ponderado para a precisão desejada, balizado pelo atraso computacional do cliente selecionado.
$$ \min F_3 = \max_{n} (\beta_n G(\theta_n)) \cdot T $$

### Restrições Matemáticas ($g \leq 0$)

Para garantir a viabilidade física e teórica do Aprendizado Federado, as soluções devem satisfazer estritamente o seguinte conjunto de restrições:

- **$g_1$ (Limite de Atraso Temporal):** O tempo processado por cada dispositivo selecionado não pode exceder o tempo global alocado $T$.
  $$ \beta_n \cdot \left(\frac{\psi_n \cdot c_n \cdot S_n}{f_n}\right) - T \leq 0, \quad \forall n $$

- **$g_2$ (Factibilidade do Desempenho Local):** Garante que os clientes selecionados treinem épocas locais suficientes ($\psi_n \ge \Psi(\theta_n)$) para atingir a acurácia, enquanto clientes inativos respeitem a punição matemática de não-divergência.
  $$ \beta_n (\Psi(\theta_n) - \psi_n) + (1 - \beta_n)(\psi_n - \Psi(\theta_n)) \leq 0, \quad \forall n $$
  
- **$g_3$ (Monotonicidade Progressiva):** Impede a degradação agressiva do limiar de aprendizagem $\theta_n$ de um dispositivo com relação ao seu histórico da rodada anterior $t-1$.
  $$ 0.99 \cdot \theta_n^{t-1} - \theta_n \leq 0, \quad \forall n $$

- **$g_4$ (Seleção Mínima de Agentes):** Garante que a rodada federada não ocorra vazia.
  $$ 1 - \sum_{n=1}^{N} \beta_n \leq 0 $$

*(Onde $G(\theta_n) = \frac{-\log_2(1-\epsilon_0)}{\theta_n}$ e $\Psi(\theta_n) = -\log_2(1-\theta_n)$ são os parâmetros heurísticos derivados do erro algorítmico do SGD local).*

## Abordagem Heurística: NSGA-II Customizado

Devido à elevada dimensionalidade contínuo-discreta do espaço de busca e à não-linearidade das restrições (ex: divisões por $f_n$ e logaritmos limitadores), algoritmos evolutivos multivariáveis puros tendem a apresentar uma vasta dispersão na geração de soluções factíveis. Para transpor esse gargalo, implementou-se operadores heurísticos analíticos plugados ao framework **NSGA-II** de otimização evolutiva:

- **Amostragem Heurística Ancorada (`CustomFLSampling`):** 
  Durante a inicialização dos indivíduos, as iterações locais ($\psi_n$) não são geradas aleatoriamente, mas sim balizadas no limite rigoroso da precisão esperada $\Psi(\theta_n) = -\log_2(1-\theta_n)$. Perturbações estocásticas induzem variações no entorno desse limite dependendo do status de seleção ($\beta_n$). Mais importante, o tempo limitante da rodada ($T$) de todos os ancestrais é instanciado rigorosamente no seu mínimo matemático factível, blindando o algoritmo de explorar tempos irracionais (ex: bilhões de segundos).

  **Pseudocódigo de Inicialização:**
  ```text
  Para cada indivíduo da população inicial:
      Para cada dispositivo n de 1 a N:
          Psi_alvo = -log2(1 - theta_n)
          
          Se beta_n == 1 (Selecionado):
              perturbação = numero_aleatorio(0, 5 * Psi_alvo)
              psi_n = arredonda_para_cima(Psi_alvo + perturbação)
          Senão (Não Selecionado):
              perturbação = numero_aleatorio(0, Psi_alvo - eps)
              psi_n = maximo(1, arredonda_para_baixo(Psi_alvo - perturbação))
              
      tempo_minimo = Valor_Maximo_Global(beta_n * C_n * S_n * psi_n / f_n)
      Tempo_Rodada (T) = tempo_minimo + numero_aleatorio(0, 0.5 * tempo_minimo)
  ```
  
- **Filtro de Reparo Pós-Mutação (`CustomFLMating`):**
  A prole evolutiva é combinada geneticamente utilizando *Simulated Binary Crossover* (SBX) e perturbações não-lineares via Mutação Polinomial (PM). Para blindar a evolução de mutações catastróficas nas bordas dos domínios, introduziu-se um operador que intercepta probabilisticamente (com chance de $30\%$) os descendentes gerados: as iterações $\psi_n$ são forçadas para as bordas exatas de factibilidade e a variável de restrição temporal $T$ é reajustada para acompanhar cirurgicamente a nova anatomia genética do cromossomo mutado. Isso garante uma busca agressiva e eficiente na real Fronteira de Pareto, superando os mínimos locais inerentes à otimização multiobjetivo pura.

  **Pseudocódigo de Reparo Genético:**
  ```text
  Gere Prole usando Crossover e Mutação Padrão do NSGA-II
  
  Para cada descendente recém-gerado na Prole:
      Se chance_aleatoria() < 0.30:  # 30% de chance de aplicar o filtro
      
          Para cada dispositivo n de 1 a N:
              Psi_alvo = -log2(1 - theta_n)
              
              Se beta_n == 1 (Selecionado):
                  psi_n = arredonda_para_cima(Psi_alvo)
              Senão (Não Selecionado):
                  psi_n = maximo(1, arredonda_para_baixo(Psi_alvo))
                  
          # Recalcula o tempo T para casar perfeitamente com os genes mutados
          Tempo_Rodada (T) = Valor_Maximo_Global(beta_n * C_n * S_n * psi_n / f_n)
  ```

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


