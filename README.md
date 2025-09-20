# dlg-fb — Ataques DLG/iDLG com Feedback (FB)

Experimentos para avaliar vazamento de privacidade via reconstrução de dados (Data Leakage from Gradients) usando os métodos DLG e iDLG, com inicializadores padrão (random) e por realimentação (FB). O script principal gera visualizações da evolução das imagens reconstruídas e um CSV com métricas por execução, para datasets MNIST e CIFAR-100.

Este repositório é voltado a testes controlados e coleta de métricas. Há também notebooks para análise agregada posterior.


## Sumário

- O que o projeto faz
- Requisitos e instalação
- Como executar (rápido e completo)
- Datasets suportados e estrutura
- Parâmetros principais e como ajustar
- Métodos, inicializadores e defesas
- Saídas geradas (imagens e CSVs)
- Análise dos resultados (notebooks)
- Dicas de uso e solução de problemas


## O que o projeto faz

O script `dlg-fb_tests.py`:

- Baixa/usa datasets (MNIST ou CIFAR-100) via `torchvision`.
- Define uma rede simples tipo LeNet e calcula gradientes verdadeiros (com rótulos reais) para 1 amostra por experimento.
- Inicializa um dado “dummy” e (opcionalmente) rótulos dummy e otimiza-os com LBFGS para minimizar a diferença de gradientes entre dummy e verdadeiro.
- Suporta dois métodos de ataque: DLG e iDLG.
- Compara inicializações: `random` e `FB` (feedback a partir de resultados previamente convergidos com `random`).
- Pode aplicar defesas na forma de ruído (Gaussian/Laplacian), atualmente desativadas por padrão.
- Salva uma imagem de evolução por experimento (várias iterações por figura) e um CSV com métricas.


## Requisitos e instalação

Pré-requisitos:

- Python 3.9+ recomendado
- pip

Instalação em ambiente virtual (opcional, mas recomendado):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

O arquivo `requirements.txt` inclui:

- numpy, matplotlib, torch, torchvision
- ipykernel, pandas, seaborn (para notebooks/relatórios)


## Como executar

Execução padrão (CPU):

```bash
python dlg-fb_tests.py
```

O script cria pastas de saída com carimbo de data/hora em `tests-outputs/dlg-fb-outputs/` e baixa dados em `datasets/dlg-fb-tests/` caso não existam.

Para um “smoke test” mais rápido, edite os seguintes parâmetros em `dlg-fb_tests.py` para diminuir o tempo:

- `TOTAL_EXP = 3` (em vez de 1200)
- `TOTAL_ITERATIONS = 60 + 1` (em vez de 300 + 1)

Mudando o dataset:

- Em `dlg-fb_tests.py`, ajuste a variável `dataset` para `"MNIST"` ou `"CIFAR100"`.

Uso de GPU (opcional):

- Por padrão o script força CPU (`use_cuda = False`). Caso tenha CUDA disponível, altere para `use_cuda = torch.cuda.is_available()` e ajuste `device` conforme necessário.


## Datasets suportados e estrutura

Datasets suportados nativamente:

- MNIST (10 classes, 28x28, 1 canal)
- CIFAR-100 (100 classes, 32x32, 3 canais)

Paths padrão criados/esperados:

- Dados: `datasets/dlg-fb-tests/`
- Saídas: `tests-outputs/dlg-fb-outputs/dlg-fb-output-[YYYY-MM-DD_HH-MM-SS]/`
	- Imagens: `visualizations-<DATASET>-<TIMESTAMP>/`
	- CSVs: `metrics-csv/`

O download automático via `torchvision.datasets` é feito diretamente para `datasets/dlg-fb-tests`. Se já tiver os dados, mantenha essa estrutura para reaproveitá-los.


## Parâmetros principais (em `dlg-fb_tests.py`)

- Geração e diretórios
	- `dataset`: "MNIST" | "CIFAR100" (padrão: CIFAR100)
	- `root_path`: raiz do projeto (padrão: ".")
	- `data_path`: `datasets/dlg-fb-tests`
	- `save_path` e `csv_save_path`: criados automaticamente com timestamp

- Reprodutibilidade
	- `random_seed = 42`

- Otimizador
	- LBFGS com `lr = 1.0`
	- `TOTAL_ITERATIONS = 300 + 1` (a iteração 0 é usada para visualizar o estado inicial)

- Critérios de parada
	- Convergência “humana” (privacidade vazada):
		- MNIST: loss <= 0.03
		- CIFAR-100: loss <= 0.09
	- Parada “precisa” por `CONVERGENCE_LOSS` (padrão `1e-6`) ou fim das iterações

- Número de execuções
	- `TOTAL_EXP = 1200` (experimentos independentes)
	- `num_dummy = 1` (uma imagem por experimento)


## Métodos, inicializadores e defesas

- Métodos: `methods = ["DLG", "iDLG"]`
	- DLG otimiza também rótulos dummy; iDLG estima o rótulo verdadeiro a partir do gradiente.

- Inicializadores: `initializers = ["random", "FB"]`
	- `random`: tensor aleatório como chute inicial.
	- `FB` (feedback): usa saídas convergidas do `random` como ponto de partida, com blending entre estados “old/new”.
	- Observação: `FB` só começa após pelo menos 3 convergências do `random` (para ter material de realimentação).

- Defesas (privadas do script; desativadas por padrão):
	- `defenses = ["None"]`
	- Magnitudes: `["None", 0.1, 0.01, 0.001, 0.0001]` (apenas usadas se `defense != "None"`).
	- Implementadas (com ruído nos gradientes): "Gaussian Noise" e "Laplacian Noise". Para ativar, edite as listas `defenses` e `magnitudes` conforme desejado.


## Saídas geradas

Todas as saídas de uma execução ficam em:

`tests-outputs/dlg-fb-outputs/dlg-fb-output-[YYYY-MM-DD_HH-MM-SS]/`

1) Visualizações (PNG)

- Pasta: `visualizations-<DATASET>-<TIMESTAMP>/`
- Arquivo por experimento/combinação: `exp_<EXP>_<METHOD>_on_img[<IMG_IDX>]_<_INITIALIZER_>_<_DEFENSE_>_<_MAGNITUDE_>.png`
- Cada figura mostra a imagem real (primeiro quadro) e a evolução do dummy em várias iterações.

2) Métricas (CSV)

- Pasta: `metrics-csv/`
- Hiperparâmetros: `metrics_hyperparameters_<DATASET>_at_<TIMESTAMP>.csv`
- Resultados: `metrics_<DATASET>_at_<TIMESTAMP>.csv`

Colunas do CSV principal:

- `img_idx`: índice da imagem no dataset
- `method`: DLG | iDLG
- `initializer`: random | FB
- `defense`: None | Gaussian Noise | Laplacian Noise
- `magnitude`: None | valor do ruído
- `exp`: id do experimento (0..TOTAL_EXP-1)
- `iters`: iterações usadas (última)
- `iter_privacy_leaked`: iteração em que ficou humanamente perceptível (ou -1)
- `gt_label`: rótulo verdadeiro
- `dummy_label`: rótulo estimado (DLG: argmax do último logit dummy; iDLG: rótulo previsto)
- `pred_label`: argmax da predição do modelo nas imagens dummy
- `converged`: True/False
- `loss`: loss final (diferença de gradientes)
- `mse`: MSE final entre dummy e ground truth


## Análise dos resultados (notebooks)

Os notebooks a seguir ajudam a consolidar métricas e produzir gráficos:

- `metrics.ipynb` (principal)
- `metrics-old.ipynb` (versão anterior)

Etapas típicas:

1. Garanta que as dependências para notebooks estão instaladas (`ipykernel`, `pandas`, `seaborn`).
2. Abra o notebook no VS Code/Jupyter.
3. Aponte para a pasta `tests-outputs/.../metrics-csv/` do experimento desejado.
4. Gere tabelas e gráficos agregados conforme as células do notebook.


## Dicas de uso

- Reduzindo tempo de execução: diminua `TOTAL_EXP` e `TOTAL_ITERATIONS` para testes rápidos.
- FB (feedback) só “anda” depois que o `random` obteve pelo menos 3 convergências; isso é intencional.
- Seeds: o script fixa seeds (`numpy`, `torch`, `torch.cuda`) para reprodutibilidade básica.
- CUDA: se usar GPU, habilite `use_cuda` e verifique se sua instalação de `torch` tem suporte CUDA.
- Defesas: para testar ruído, ative `defenses` e ajuste `magnitudes`; isso torna a recuperação mais difícil, como esperado.


## Estrutura do repositório (simplificada)

```
dlg-fb_tests.py              # Script principal dos experimentos
requirements.txt             # Dependências
metrics.ipynb                # Análises de métricas
metrics-old.ipynb            # Versão antiga do notebook
datasets/
	dlg-fb-tests/              # Baixa/guarda MNIST e CIFAR100 aqui
tests-outputs/
	dlg-fb-outputs/
		dlg-fb-output-[...]/     # Pasta de cada execução (timestamp)
			metrics-csv/           # CSVs de métricas e hiperparâmetros
			visualizations-*/      # PNGs com evolução das reconstruções
```


## Licença

Este projeto está licenciado sob a Licença MIT. Consulte o arquivo `LICENSE` na raiz do repositório para mais detalhes.


## Contato

Abra uma issue ou comente diretamente no repositório para dúvidas e sugestões.