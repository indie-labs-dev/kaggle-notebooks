# SKILL: sd-image-generator

Reference for `notebooks/sd-image-generator/notebook.ipynb`.

Read this file before executing any plan for this notebook.

---

## Notebook structure — 4 cells total

```
Cell 0 — Markdown header
Cell 1 — Setup          : pip install, env vars, imports
Cell 2 — Model config UI: STYLE_MODELS, SIZES, constants, load_pipeline, style radio + refiner checkbox
Cell 3 — All logic functions + widgets + handlers + gallery
```

Order within Cell 3 (strict):
1. image_to_b64, show_gallery
2. count_tokens, encode_prompt
3. make_generator, run_pipeline, load_face_adapter, get_face_bytes
4. All widgets (prompt_ta, token_label, neg_prompt_ta, ...)
5. update_token_count, on_random_seed, on_generate handlers
6. display() calls

Rule: Cell N must not call anything defined in Cell M where M > N.

---

## Cell 1 — Setup (canonical)

```python
import subprocess, sys, os

os.environ["HF_HOME"] = "/kaggle/working/hf_cache"
# NOTE: do NOT set TRANSFORMERS_CACHE — deprecated since transformers v5, causes FutureWarning

def install(pkg):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

PACKAGES = [
    "diffusers",
    "accelerate",
    "transformers",
    "huggingface_hub",
    "compel",
    "ipywidgets",
    "insightface",
    "onnxruntime-gpu",
    "opencv-python-headless",
]
for pkg in PACKAGES:
    install(pkg)

import torch
import ipywidgets as widgets
from IPython.display import display, clear_output
import random, io, base64
from PIL import Image, ImageOps

print(f"✅ Ready | PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
```

---

## Cell 2 — Model config (canonical)

### STYLE_MODELS — verified SDXL + diffusers format on HuggingFace

```python
STYLE_MODELS = {
    "Realism":       "SG161222/RealVisXL_V4.0",
    "Anime":         "cagliostrolab/animagine-xl-3.1",
    "Comics":        "Lykon/dreamshaper-xl-1-0",
    "Illustration":  "playgroundai/playground-v2.5-1024px-aesthetic",
    "Lineart":       "stabilityai/stable-diffusion-xl-base-1.0",
}
```

**All models work**: Stable, widely compatible pipelines on HuggingFace.

**Never use**: `stablediffusionapi/*`, `Linaqruf/lineart-anime-diffusion`, `ogkalu/Comic-Diffusion`,
`fluently/Fluently-XL-Final` — unavailable, SD 1.x, or single-file without diffusers structure.

### SIZES — SDXL-native, preserves YouTube aspect ratios

```python
SIZES = {
    "YouTube Video (16:9)":  (1152, 640),   # SDXL native training size, exact 16:9
    "YouTube Shorts (9:16)": (640, 1152),   # SDXL native training size, exact 9:16
}
```

### Balanced profile constants

```python
BASE_STEPS     = 40
DEFAULT_GUIDANCE = 6.5
```

### load_pipeline — canonical

```python
from diffusers import DiffusionPipeline
from huggingface_hub import file_exists

def load_pipeline(model_id: str):
    try:
        has_diffusers = file_exists(model_id, "model_index.json")
    except Exception:
        has_diffusers = False

    if not has_diffusers:
        raise ValueError(f"{model_id} has no model_index.json — only diffusers-format models supported")

    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16,
            use_safetensors=True, variant="fp16",
        )
    except Exception:
        pipe = DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, use_safetensors=True,
        )

    pipe.to("cuda")

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    return pipe
```

**Critical rules:**
- `pipe.to("cuda")` — NOT `enable_model_cpu_offload()` (too slow on T4)
- `file_exists()` for format detection — NOT `requests.head()` (HF returns 302)
- xformers in try/except — not available on Kaggle by default

---

## Cell 3 — Key functions (canonical)

### encode_prompt — compel for long prompts

```python
def encode_prompt(pipe, prompt: str, negative_prompt: str) -> dict:
    try:
        from compel import Compel, ReturnedEmbeddingsType
        compel = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
        )
        cond, pooled = compel(prompt)
        neg_cond, neg_pooled = compel(negative_prompt)
        [cond, neg_cond] = compel.pad_conditioning_tensors_to_same_length([cond, neg_cond])
        return {
            "prompt_embeds": cond, "pooled_prompt_embeds": pooled,
            "negative_prompt_embeds": neg_cond, "negative_pooled_prompt_embeds": neg_pooled,
        }
    except Exception:
        return {"prompt": prompt, "negative_prompt": negative_prompt}
```

When compel succeeds, returned dict has NO `"prompt"` or `"negative_prompt"` keys.
`run_pipeline` receives original strings separately via `plain_prompt`/`plain_negative` for refiner.

### load_face_adapter — canonical with VRAM flush

```python
def load_face_adapter(pipe, face_bytes: bytes, scale: float):
    face_img = ImageOps.exif_transpose(
        Image.open(io.BytesIO(face_bytes))
    ).convert("RGB")
    face_img.thumbnail((1024, 1024))
    if not getattr(pipe, "_ip_adapter_loaded", False):
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors",
        )
        pipe._ip_adapter_loaded = True
    pipe.set_ip_adapter_scale(scale)
    return face_img
```

**Critical**: always flush VRAM before loading IP-Adapter.
On T4 with pipe on CUDA (~6 GiB), IP-Adapter weights (~1.5 GiB) cause OOM without flush.

### run_pipeline — single image, no refiner

```python
def run_pipeline(pipe, prompt_kwargs: dict, width: int, height: int,
                  guidance: float, seed: int, face_img=None) -> Image.Image:
    """Single image generation — no refiner."""
    generator = make_generator(seed)
    common = {
        **prompt_kwargs,
        "width": width,
        "height": height,
        "generator": generator,
        "guidance_scale": guidance,
        "num_images_per_prompt": 1,
    }
    if face_img is not None:
        common["ip_adapter_image"] = face_img

    return pipe(
        num_inference_steps=BASE_STEPS,
        **common,
    ).images[0]
```

### on_generate — structure

```python
def on_generate(btn):
    # 1. Validate BEFORE disabling button
    prompt = prompt_ta.value.strip()
    if not prompt:
        status_label.value = "❌ Prompt is empty"
        return                              # button stays enabled — correct

    generate_btn.disabled = True           # disable AFTER validation
    ...
    try:
        prompt_kwargs = encode_prompt(pipe, prompt, negative_prompt)
        for i, seed in enumerate(seeds):
            img = run_pipeline(..., face_img=face_img)
    except Exception as e:
        status_label.value = f"❌ Error: {e}"
    finally:
        generate_btn.disabled = False
```

---

## VRAM budget on Kaggle T4 (14.56 GiB)

| Component | VRAM |
|---|---|
| SDXL base UNet fp16 | ~3.2 GiB |
| text_encoder + text_encoder_2 fp16 | ~1.9 GiB |
| VAE fp16 | ~0.5 GiB |
| CUDA context + overhead | ~0.8 GiB |
| Activations during generation | ~1.5 GiB |
| **Total without refiner** | **~7.9 GiB** |
| IP-Adapter weights | ~1.5 GiB |
| **Total with IP-Adapter** | **~9.4 GiB** |

Always flush VRAM before loading IP-Adapter: `gc.collect(); torch.cuda.empty_cache()`.

---

## Known issues

| Error | Cause | Fix |
|---|---|---|
| `FutureWarning: TRANSFORMERS_CACHE` | deprecated env var | Remove — only HF_HOME needed |
| cuFFT/cuDNN/cuBLAS factory warnings | Kaggle loads TF + PyTorch together | System noise, ignore |
| Flax deprecated warning | diffusers internal | Not our code, ignore |
| `variant=fp16 not found` | Model has no fp16 variant | Canonical falls back without variant |
| `xformers not installed` | Not on Kaggle by default | try/except in canonical |
| Generate btn stays disabled | empty prompt check after btn.disabled | Check prompt BEFORE disabling button |
| CUDA OOM on face mode | IP-Adapter weights (~1.5 GiB) on top of pipe | gc.collect() + empty_cache() before load_ip_adapter() |
