# lung-seg-cxr

Segmentação dos pulmões em raio-X de tórax com uma U-Net (encoder ResNet34 pré-treinado no ImageNet), separação em pulmão direito e esquerdo e cálculo da razão entre as áreas (D/E).

![Segmentação de pulmão direito (D, azul) e esquerdo (E, verde) em raio-X das bases Montgomery e Shenzhen](docs/showcase.png)

> **Projeto educacional. Não é um dispositivo médico e não deve ser usado para diagnóstico ou qualquer decisão clínica.**

## Setup (Mac com Apple Silicon)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Roda também em CUDA e CPU sem mudar nada: o dispositivo é escolhido automaticamente (cuda > mps > cpu).

O dataset vem do Kaggle (`nikhilpandey360/chest-xray-masks-and-labels`). Se o download pedir autenticação, salve o token de https://www.kaggle.com/settings em `~/.kaggle/kaggle.json` ou exporte `KAGGLE_USERNAME` e `KAGGLE_KEY`. Também dá para baixar manualmente e usar `lungseg prepare --raw-dir PASTA`.

## Comandos

| Comando | O que faz | Saídas |
|---|---|---|
| `lungseg prepare` | baixa o dataset, pareia imagens e máscaras, gera cache e split | `data/cache_256.npz`, `outputs/split.json` |
| `lungseg train` | treina e salva o melhor checkpoint (Dice de validação) | `outputs/best.pt`, `outputs/train_log.csv`, `outputs/curves.png` |
| `lungseg evaluate` | Dice bruto e pós-processado, erro da razão D/E | `outputs/eval.csv`, `outputs/ratio_scatter.png`, `outputs/best_worst.png` |
| `lungseg explore` | razão D/E nas máscaras anotadas por fonte e rótulo de TB | `outputs/explore.csv`, `outputs/ratio_boxplot.png` |
| `lungseg predict IMG [--out PATH]` | segmenta uma imagem e imprime áreas e razão | overlay PNG |
| `lungseg app [--share]` | demo Gradio em http://localhost:7860 | |

Qualquer chave de `configs/default.yaml` pode ser sobrescrita pela CLI, por exemplo `lungseg train --epochs 1 --batch-size 8`. Testes: `pytest`.

## Convenção radiológica

Num raio-X frontal (PA) o paciente aparece como se estivesse de frente para quem olha. Por isso **o pulmão do lado esquerdo da imagem é o pulmão direito do paciente**. No projeto, D = direito e E = esquerdo sempre se referem ao paciente; no overlay, azul = direito e verde = esquerdo.

Pelo mesmo motivo, o treino **não usa flip horizontal** como aumento de dados: espelhar a imagem trocaria a anatomia de lado.

O pós-processamento mantém as duas maiores regiões conexas da máscara prevista e as ordena pela posição horizontal do centróide.

## Resultados

| Métrica (validação) | Valor |
|---|---|
| Dice (bruto) | 0,961 |
| Dice (pós-processado) | 0,961 |
| Erro absoluto médio da razão D/E | 0,074 |

U-Net com encoder ResNet34, 25 épocas em um Mac Apple Silicon (MPS), cerca de 30 segundos por época (~13 minutos no total); melhor checkpoint na época 20. 106 imagens de validação.

![Loss de treino e Dice de validação por época](docs/curves.png)

### Comparação de arquiteturas

Mesmo encoder (ResNet34 ImageNet), mesmo split, mesma seed e 25 épocas para todas.

| Arquitetura | Parâmetros | Dice médio | Dice mediano | Pior Dice | Erro médio da razão D/E | Melhor época | Tempo de treino |
|---|---|---|---|---|---|---|---|
| U-Net | 24,4 M | 0,961 | 0,971 | 0,822 | 0,074 | 20 | ~13 min |
| U-Net++ | 26,1 M | 0,962 | 0,972 | 0,815 | 0,076 | 15 | ~21 min |

A U-Net++ ganha 0,001 de Dice e custa cerca de 60% mais tempo de treino. Com uma única seed e 106 imagens de validação, essa diferença está dentro do ruído: na prática as duas empatam, e a U-Net segue como padrão por ser mais rápida.

Para treinar outra arquitetura sem sobrescrever a atual:

```bash
lungseg train --arch unetplusplus --outputs-dir outputs/unetplusplus
lungseg evaluate --arch unetplusplus --outputs-dir outputs/unetplusplus
```

Copie `outputs/split.json` para a nova pasta antes, para avaliar no mesmo conjunto de validação.

Figuras em `outputs/`: `curves.png`, `ratio_scatter.png`, `best_worst.png`, `ratio_boxplot.png`.

A análise exploratória (`lungseg explore`) compara a razão D/E entre exames normais e com tuberculose. Fonte, hospital, aparelho e população se confundem com o rótulo, então as diferenças não servem para conclusões clínicas.

## Créditos dos dados

- **Montgomery County X-ray Set** e **Shenzhen Hospital X-ray Set**: National Library of Medicine (NLM/NIH). Jaeger S. et al., "Two public chest X-ray datasets for computer-aided screening of pulmonary diseases", *Quant Imaging Med Surg* 2014;4(6):475-477.
- Máscaras e organização no Kaggle: `nikhilpandey360/chest-xray-masks-and-labels`.
