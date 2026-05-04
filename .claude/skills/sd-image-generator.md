# SKILL: sd-image-generator

Reference for `notebooks/sd-image-generator/notebook.ipynb`.

Read this file before executing any plan for this notebook.

---

## Notebook structure — 4 cells total

```
Cell 0 — Markdown header
Cell 1 — Setup          : pip install, env vars, imports
Cell 2 — Model config UI: STYLE_MODELS, SIZES, load_pipeline, style radio + refiner checkbox + load button
Cell 3 — Generation UI + all logic functions + gallery display
```

Order within Cell 3 (strict):
1. image_to_b64, show_gallery
2. generate_images, load_face_pipeline, generate_with_face
3. All widgets
4. get_seed, on_random_seed, on_generate handlers
5. display() calls

Rule: Cell N must not call anything defined in Cell M where M > N.

---

## Cell 1 — Setup (canonical)

```python
import subprocess, sys, os

os.environ["HF_HOME"] = "/kaggle/working/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/kaggle/working/hf_cache"

def install(pkg):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

PACKAGES = [
    "diffusers",
    "accelerate",
    "transformers",
    "huggingface_hub",
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
from PIL import Image

print(f"✅ Ready | PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
```

---

## Cell 2 — Model config (canonical)

### STYLE_MODELS — verified SDXL + diffusers format

```python
STYLE_MODELS = {
    "Realism":       "SG161222/RealVisXL_V4.0",
    "Anime":         "cagliostrolab/animagine-xl-3.1",
    "Comics":        "Lykon/dreamshaper-xl-1-0",
    "Illustration":  "playgroundai/playground-v2.5-1024px-aesthetic",
    "Lineart":       "stabilityai/stable-diffusion-xl-base-1.0",
}
```

**Never use**: `stablediffusionapi/*`, `Linaqruf/lineart-anime-diffusion`, `ogkalu/Comic-Diffusion`,
`fluently/Fluently-XL-Final` — unavailable, SD 1.x, or single-file without diffusers structure.

### Global state

```python
pipe = None
refiner = None
```

### load_pipeline — canonical with refiner support

```python
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
from huggingface_hub import file_exists, list_repo_files, hf_hub_url

def load_pipeline(model_id: str, load_refiner: bool = False):
    global refiner

    try:
        has_diffusers_format = file_exists(model_id, "model_index.json")
    except Exception:
        has_diffusers_format = False

    if has_diffusers_format:
        try:
            pipe = DiffusionPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16,
                use_safetensors=True, variant="fp16",
            )
        except Exception:
            pipe = DiffusionPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16, use_safetensors=True,
            )
    else:
        files = list(list_repo_files(model_id))
        sf = next((f for f in files if f.endswith(".safetensors") and "/" not in f), None)
        if sf is None:
            raise ValueError(f"No root-level .safetensors found in {model_id}")
        url = hf_hub_url(model_id, filename=sf)
        pipe = StableDiffusionXLPipeline.from_single_file(
            url, torch_dtype=torch.float16, use_safetensors=True,
        )

    pipe.enable_model_cpu_offload()
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    if load_refiner:
        # Reuse text_encoder_2 and vae from base — saves VRAM and load time
        refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=pipe.text_encoder_2,
            vae=pipe.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        refiner.enable_model_cpu_offload()
        try:
            refiner.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    else:
        refiner = None

    return pipe
```

**Critical rules:**
- Use `file_exists()` from huggingface_hub — NOT `requests.head()` (HF returns 302, requests doesn't follow)
- Single-file branch: filter `"/" not in f` to get root-level `.safetensors` only
- Use `DiffusionPipeline` for diffusers-format models
- NO `.to("cuda")` — conflicts with `enable_model_cpu_offload()`
- xformers in try/except — not on Kaggle by default
- Refiner reuses `text_encoder_2` and `vae` from base pipeline

### SIZES

```python
SIZES = {
    "YouTube Video (1920×1080)":  (1920, 1080),
    "YouTube Shorts (1080×1920)": (1080, 1920),
}
```

---

## Cell 3 — Generation functions (canonical)

### generate_images — with refiner two-pass

```python
def generate_images(pipe, prompt, negative_prompt, seed, count, width, height, guidance):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if refiner is not None:
        raw = pipe(
            prompt=prompt, negative_prompt=negative_prompt,
            num_images_per_prompt=count, width=width, height=height,
            generator=generator, num_inference_steps=40,
            denoising_end=0.8, guidance_scale=guidance,
            output_type="latent",
        ).images
        generator2 = torch.Generator(device="cuda").manual_seed(seed)
        return refiner(
            prompt=prompt, negative_prompt=negative_prompt,
            num_images_per_prompt=count, width=width, height=height,
            generator=generator2, num_inference_steps=40,
            denoising_start=0.8, guidance_scale=guidance,
            image=raw,
        ).images
    else:
        return pipe(
            prompt=prompt, negative_prompt=negative_prompt,
            num_images_per_prompt=count, width=width, height=height,
            generator=generator, num_inference_steps=30,
            guidance_scale=guidance,
        ).images
```

### generate_with_face — with refiner two-pass

Same pattern as `generate_images` but adds `ip_adapter_image=face_img` in the base pass only.
Refiner pass does NOT use `ip_adapter_image` — it works on latents, face already encoded.

### IP-Adapter

- Load only once: `getattr(pipe, "_ip_adapter_loaded", False)`
- Unload before non-face generation: `pipe.unload_ip_adapter(); pipe._ip_adapter_loaded = False`
- Weight: `ip-adapter-plus-face_sdxl_vit-h.safetensors` from `h94/IP-Adapter` / `sdxl_models`

---

## Required widgets (Cell 2 + Cell 3)

Cell 2:
- `style_radio` — RadioButtons (5 styles)
- `refiner_checkbox` — Checkbox (load refiner toggle)
- `load_btn` — Button

Cell 3:

| Variable | Widget type | Purpose |
|---|---|---|
| `prompt_ta` | `Textarea` | Main prompt |
| `neg_prompt_ta` | `Textarea` | Negative prompt |
| `count_slider` | `IntSlider` | Number of images (1–8) |
| `guidance_slider` | `FloatSlider` | Guidance scale (1.0–15.0) |
| `seed_input` | `IntText` | Seed value |
| `random_seed_btn` | `Button` | Generate random seed |
| `seed_mode_radio` | `RadioButtons` | "Fixed" / "Random each time" |
| `size_radio` | `RadioButtons` | Video / Shorts |
| `face_upload` | `FileUpload` | Optional face photo |
| `generate_btn` | `Button` | Trigger generation |
| `progress_bar` | `IntProgress` | Generation progress |
| `status_label` | `Label` | Current status text |
| `gallery_output` | `Output` | Gallery render target |

face_upload read pattern (ipywidgets >= 8): `face_upload.value[0]["content"]`

---

## Known issues

| Error | Cause | Fix |
|---|---|---|
| `has_diffusers_format always False` | `requests.head()` gets 302 from HF CDN | Use `file_exists()` — already in canonical |
| Single-file picks wrong file | `list_repo_files` returns all incl. subfolders | Filter `"/" not in f` — already in canonical |
| `variant=fp16 not found` | Model has no fp16 variant | Canonical falls back without variant |
| `xformers not installed` | Not on Kaggle by default | try/except in canonical |
| `CUDA out of memory` | Large size + many images | `enable_model_cpu_offload()` — never `.to("cuda")` |
| playground-v2.5 fails | Custom pipeline class | Use `DiffusionPipeline` — already in canonical |
| Refiner VRAM overflow | Loading full refiner separately | Reuse `text_encoder_2` and `vae` from base — already in canonical |
