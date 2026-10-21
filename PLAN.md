# Plano: segmentação de pulmões em raio-X de tórax (lung-seg-cxr)

## Objetivo

Projeto Python (scripts `.py`, sem notebooks) que:

1. baixa e prepara o dataset Montgomery + Shenzhen (Kaggle: `nikhilpandey360/chest-xray-masks-and-labels`);
2. treina uma U-Net (encoder ResNet34 pré-treinado no ImageNet) para segmentar os pulmões;
3. separa pulmão direito e esquerdo e calcula a razão entre as áreas;
4. avalia o modelo (Dice e erro da razão D/E);
5. faz uma análise exploratória (razão D/E em exames normais vs. tuberculose);
6. oferece uma demo local com Gradio e um comando de previsão para uma imagem.

Projeto **educacional**. Não é para uso clínico, e o README e a interface devem dizer isso.

## Ambiente alvo

- macOS em Apple Silicon (Mac M4), Python 3.11 ou 3.12, venv local.
- Deve rodar também em CUDA e CPU, sem mudar código.
- Dependências: `torch`, `torchvision`, `segmentation-models-pytorch`, `albumentations`, `opencv-python`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `tqdm`, `kagglehub`, `gradio`, `pyyaml`. Dev: `pytest`, `ruff`.
- Gerenciar com `pyproject.toml` (instalação via `pip install -e ".[dev]"`).

## Estrutura do repositório

```
lung-seg-cxr/
├── pyproject.toml
├── README.md
├── .gitignore              # data/, outputs/, .venv/, *.pt, __pycache__/
├── configs/
│   └── default.yaml
├── src/lungseg/
│   ├── __init__.py
│   ├── cli.py              # ponto de entrada: comando `lungseg <subcomando>`
│   ├── config.py           # carrega YAML em dataclass; permite override por CLI
│   ├── device.py           # escolha de dispositivo (cuda > mps > cpu)
│   ├── data.py             # download, pareamento, cache, split, Dataset, transforms
│   ├── model.py            # construção da U-Net, salvar/carregar checkpoint
│   ├── metrics.py          # dice, loss combinada
│   ├── postprocess.py      # lung_stats: duas maiores regiões, direito/esquerdo, razão
│   ├── train.py
│   ├── evaluate.py
│   ├── explore.py
│   ├── predict.py          # inferência em imagem de qualquer tamanho + overlay
│   └── app.py              # demo Gradio
├── tests/
│   ├── test_postprocess.py
│   ├── test_data.py
│   ├── test_metrics.py
│   └── test_device.py
├── data/                   # (gitignored) dataset bruto e cache
└── outputs/                # (gitignored) checkpoints, logs, figuras, CSVs
```

## Configuração (`configs/default.yaml`)

```yaml
seed: 42
data:
  kaggle_dataset: nikhilpandey360/chest-xray-masks-and-labels
  raw_dir: data/raw
  cache_path: data/cache_256.npz
  size: 256
  val_fraction: 0.15
model:
  arch: unet
  encoder: resnet34
  encoder_weights: imagenet
train:
  epochs: 25
  batch_size: 16
  lr: 3.0e-4
  weight_decay: 1.0e-4
  num_workers: 0
outputs_dir: outputs
threshold: 0.5
```

Qualquer chave pode ser sobrescrita pela CLI, por exemplo `lungseg train --epochs 5`.

## Comandos (CLI)

| Comando | O que faz | Saídas |
|---|---|---|
| `lungseg prepare` | baixa o dataset (se faltar), pareia imagens e máscaras, redimensiona e salva cache e split | `data/cache_256.npz`, `outputs/split.json` |
| `lungseg train` | treina e salva o melhor checkpoint por Dice de validação | `outputs/best.pt`, `outputs/train_log.csv`, `outputs/curves.png` |
| `lungseg evaluate` | Dice bruto e pós-processado, erro da razão D/E, figuras | `outputs/eval.csv`, `outputs/ratio_scatter.png`, `outputs/best_worst.png` |
| `lungseg explore` | razão D/E nas máscaras anotadas por fonte e rótulo de TB | `outputs/explore.csv`, `outputs/ratio_boxplot.png` |
| `lungseg predict IMG [--out PATH]` | segmenta uma imagem e imprime áreas e razão | overlay PNG |
| `lungseg app [--share]` | abre a demo Gradio em localhost:7860 | — |

Use `argparse` com subcomandos (sem dependência extra). Registrar o entrypoint `lungseg = "lungseg.cli:main"` no `pyproject.toml`.

## Detalhes por módulo

### `device.py`
- `get_device()` retorna `cuda` se disponível, senão `mps`, senão `cpu`, e registra no log qual foi escolhido.
- Antes de importar o torch no `cli.py`, definir `os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")`, para que operações sem suporte no MPS caiam para a CPU em vez de dar erro.

### `data.py`
- **Download:** `kagglehub.dataset_download(...)`. Se falhar por autenticação, mostrar mensagem clara explicando como colocar o token em `~/.kaggle/kaggle.json` ou nas variáveis `KAGGLE_USERNAME` e `KAGGLE_KEY`. Também aceitar `--raw-dir` apontando para uma pasta já extraída manualmente.
- **Pareamento:** imagens em `**/CXR_png/*.png`, máscaras em `**/masks/*.png` (busca recursiva). A chave é o nome do arquivo sem extensão. Nas máscaras, remover o sufixo `_mask` (as do Shenzhen têm esse sufixo; as do Montgomery não). Usar só a interseção. Imprimir quantas imagens, máscaras e pares foram encontrados. Se houver 0 pares, falhar listando as pastas encontradas.
- **Metadados pelo nome:** prefixo `MCU` → fonte Montgomery, `CHN` → Shenzhen. O último trecho após `_` é o rótulo de TB (0 = normal, 1 = TB). Se não for parseável, usar -1.
- **Cache:** ler em escala de cinza, redimensionar uma única vez para `size` (imagem com `INTER_AREA`, máscara com `INTER_NEAREST`), binarizar a máscara (`> 127`). Salvar em `.npz` comprimido: `images` (N,H,W uint8), `masks` (N,H,W uint8), `keys`. Guardar também os caminhos originais das imagens, para a demo usar como exemplos.
- **Split:** `train_test_split` estratificado pela fonte, com a seed do config. Salvar as listas de chaves em `outputs/split.json` e reutilizar nas próximas execuções.
- **Transforms de treino:** `Affine(scale=(0.9,1.1), translate_percent=(-0.05,0.05), rotate=(-10,10), p=0.7)`, `RandomBrightnessContrast(0.2,0.2,p=0.5)`, `CLAHE(p=0.3)`, `Normalize()`, `ToTensorV2()`.
- **⚠️ Proibido usar flip horizontal** (nem `HorizontalFlip` nem flip em rotações de 180°). Espelhar troca a anatomia de lado. Comentar isso no código.
- **Transforms de validação:** `Normalize()` + `ToTensorV2()`.
- **Dataset:** replica a imagem cinza em 3 canais (para o encoder ImageNet) e retorna `(image[3,H,W] float, mask[1,H,W] float)`.
- `num_workers` vem do config (padrão 0 no macOS).

### `model.py`
- `build_model(cfg)` usando `smp.Unet(encoder, encoder_weights, in_channels=3, classes=1)`.
- O checkpoint salva `state_dict`, o config usado, `size`, a época e o Dice de validação, para que `predict` e `app` reconstruam o modelo só a partir do arquivo.
- Ao carregar, sempre usar `torch.load(..., map_location=device)`.

### `metrics.py`
- Loss: `smp.losses.DiceLoss(mode="binary")` + `BCEWithLogitsLoss`.
- `dice(pred_bin, target)` com epsilon, calculado por imagem e depois em média.

### `postprocess.py` (núcleo do projeto)
- `lung_stats(mask) -> LungStats | None`: aplica `cv2.connectedComponentsWithStats` (conectividade 8), mantém as **duas maiores** regiões e ordena pelo centróide x.
- **Convenção radiológica:** a região mais à esquerda da imagem é o pulmão **direito** do paciente. Documentar isso na docstring.
- Retorna um dataclass com `area_direito`, `area_esquerdo`, `razao` (direito/esquerdo), centróides, mapa de rótulos e `clean_mask` (só as duas regiões).
- Retorna `None` se houver menos de duas regiões.

### `train.py`
- Fixar seeds (python, numpy, torch).
- Otimizador AdamW com CosineAnnealingLR.
- A cada época, logar loss de treino e Dice de validação no terminal (tqdm) e em CSV. Salvar o checkpoint quando o Dice de validação melhorar.
- Ao final, gerar `curves.png`.
- Flag `--epochs` para testes rápidos (ex.: `--epochs 1`).

### `evaluate.py`
- Para cada imagem de validação: Dice bruto, Dice após `clean_mask` e razão D/E prevista vs. razão nas máscaras anotadas.
- Imprimir as médias e o erro absoluto médio da razão.
- Figuras: dispersão razão real × prevista com linha y=x; painel com as 2 melhores e as 2 piores previsões (overlay azul = direito, verde = esquerdo).

### `explore.py`
- Razão D/E nas máscaras **anotadas** de todo o dataset, agrupada por fonte e rótulo de TB.
- Tabela resumo (count, mean, median, std) e boxplot com linha em 1,0.
- Imprimir um aviso: análise exploratória, com confundidores (hospital, aparelho, população); não serve para conclusões clínicas.

### `predict.py`
- `predict_image(model, img_rgb_or_gray, size, threshold, device)`: converte para cinza, redimensiona para `size`, roda o modelo, aplica `lung_stats` e redimensiona o mapa de rótulos de volta ao tamanho original com `INTER_NEAREST`.
- `render_overlay(...)`: pulmão direito em azul, esquerdo em verde (alpha ~0,45), letras "D" e "E" nos centróides, com tamanho de fonte proporcional à imagem.
- O comando da CLI salva o overlay e imprime a porcentagem da área pulmonar de cada lado e a razão.

### `app.py`
- Gradio `Interface`: entrada de imagem; saídas overlay + texto com as áreas.
- Exemplos: 4 imagens de validação (caminhos originais).
- Título e descrição com aviso de uso educacional.
- Se `lung_stats` retornar `None`, mostrar mensagem amigável.
- `launch()` local por padrão; `--share` opcional.

## Testes (pytest, sem precisar do dataset)

- `test_postprocess.py`: máscara sintética com dois retângulos de áreas diferentes. Verificar que o retângulo da esquerda da imagem vira o pulmão **direito**, que a razão está correta, que ruídos pequenos são descartados e que retorna `None` com só uma região.
- `test_data.py`: pareamento com nomes falsos (`CHNCXR_0001_0_mask.png` ↔ `CHNCXR_0001_0.png`, `MCUCXR_0001_0.png` ↔ `MCUCXR_0001_0.png`); parse de fonte e rótulo de TB; shapes e tipos do Dataset; o pipeline de treino não contém `HorizontalFlip`.
- `test_metrics.py`: Dice = 1 para máscaras iguais, ≈ 0 para disjuntas, sem NaN com máscaras vazias.
- `test_device.py`: `get_device()` retorna um dispositivo válido.

## Ordem de implementação (marcos)

1. **Esqueleto:** `pyproject.toml`, estrutura de pastas, `.gitignore`, `config.py`, `device.py`, CLI com subcomandos vazios. *Pronto quando:* `pip install -e ".[dev]"` funciona e `lungseg --help` lista os comandos.
2. **Pós-processamento e métricas + testes.** *Pronto quando:* `pytest` passa.
3. **Dados:** download, pareamento, cache, split, Dataset + testes. *Pronto quando:* `lungseg prepare` gera o cache e imprime cerca de 700 pares (Montgomery ~138, Shenzhen ~560; os números exatos podem variar).
4. **Treino.** *Pronto quando:* `lungseg train --epochs 1` roda de ponta a ponta no MPS e salva o checkpoint; com 25 épocas, Dice de validação ≥ 0,95.
5. **Avaliação e exploração.** *Pronto quando:* os CSVs e as figuras são gerados em `outputs/`.
6. **Predict e app.** *Pronto quando:* `lungseg predict` gera o overlay de uma imagem original em tamanho cheio e `lungseg app` abre no navegador.
7. **README:** objetivo, aviso educacional, setup no Mac, comandos, explicação da convenção radiológica, espaço para resultados (Dice, figuras) e créditos dos datasets (Montgomery County e Shenzhen, NLM/NIH).

## Convenções

- Type hints e docstrings curtas; `ruff` para lint e formatação.
- Usar `pathlib` para caminhos e `logging` em vez de `print` (exceto saídas finais da CLI).
- Nada de caminhos absolutos fixos: tudo vem do config.
- Commits pequenos, um por marco.
- Se algo no dataset não bater com o descrito aqui (nomes de pastas ou de arquivos), inspecionar e adaptar o pareamento, explicando o que mudou, em vez de forçar o formato.
