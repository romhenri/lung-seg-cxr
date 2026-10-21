# lung-seg-cxr

Segmentação dos pulmões em raio-X de tórax com uma U-Net (encoder ResNet34 pré-treinado no ImageNet), separação em pulmão direito e esquerdo e cálculo da razão entre as áreas (D/E).

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

25 épocas em um Mac Apple Silicon (MPS), cerca de 1 minuto por época; melhor checkpoint na época 20. 106 imagens de validação.

Figuras em `outputs/`: `curves.png`, `ratio_scatter.png`, `best_worst.png`, `ratio_boxplot.png`.

A análise exploratória (`lungseg explore`) compara a razão D/E entre exames normais e com tuberculose. Fonte, hospital, aparelho e população se confundem com o rótulo, então as diferenças não servem para conclusões clínicas.

## Créditos dos dados

- **Montgomery County X-ray Set** e **Shenzhen Hospital X-ray Set**: National Library of Medicine (NLM/NIH). Jaeger S. et al., "Two public chest X-ray datasets for computer-aided screening of pulmonary diseases", *Quant Imaging Med Surg* 2014;4(6):475-477.
- Máscaras e organização no Kaggle: `nikhilpandey360/chest-xray-masks-and-labels`.
