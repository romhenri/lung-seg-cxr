"""Demo local com Gradio."""

import gradio as gr

from .config import Config
from .data import get_split, load_cache, subset
from .device import get_device
from .model import load_checkpoint
from .predict import describe, predict_image, render_overlay

DESCRIPTION = (
    "Segmentação dos pulmões em raio-X de tórax frontal com uma U-Net. "
    "Azul = pulmão direito do paciente (lado esquerdo da imagem), verde = pulmão esquerdo.\n\n"
    "**Projeto educacional. Não use para diagnóstico ou qualquer decisão clínica.**"
)


def run_app(cfg: Config, share: bool = False) -> None:
    device = get_device()
    model, ckpt = load_checkpoint(cfg.out / "best.pt", device)

    def segment(img):
        if img is None:
            return None, "Envie uma imagem."
        stats = predict_image(model, img, ckpt["size"], cfg.threshold, device)
        return render_overlay(img, stats), describe(stats)

    try:
        cache = load_cache(cfg)
        val = subset(cache, get_split(cfg, cache["keys"])["val"][:4])
        examples = [[str(p)] for p in cache["paths"][val]]
    except SystemExit:
        examples = None

    gr.Interface(
        fn=segment,
        inputs=gr.Image(type="numpy", label="Raio-X de tórax"),
        outputs=[gr.Image(label="Segmentação"), gr.Textbox(label="Áreas")],
        examples=examples,
        title="Segmentação de pulmões (uso educacional)",
        description=DESCRIPTION,
        flagging_mode="never",
    ).launch(share=share)
