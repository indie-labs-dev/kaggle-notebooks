# SKILL: sd-image-generator

Reference for `notebooks/sd-image-generator/notebook.ipynb`.

Read this file before executing any plan for this notebook.

---

## Notebook structure — 5 code cells

```
Cell 1 — Setup          : pip install, env vars, imports
Cell 2 — Model config   : style radio + "Load model" button
Cell 3 — Generation UI  : all ipywidgets (prompt, seed, count, size, face upload)
Cell 4 — Logic          : pure Python functions, no UI
Cell 5 — Gallery        : tile display + download buttons
```

Rule: Cell N must not import from Cell M where M > N.

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

Do NOT pin package versions (no `diffusers==0.28.0`) unless there is a confirmed conflict.

---

## Cell 2 — Model config (canonical)

```python
STYLE_MODELS = {
    "Realism":       "SG161222/RealVisXL_V4.0",
    "Anime":         "cagliostrolab/animagine-xl-3.1",
    "Comics":        "ogkalu/Comic-Diffusion",
    "Illustration":  "stablediffusionapi/illustration-diffusion",
    "Lineart":       "Linaqruf/lineart-anime-diffusion",
}

SIZES = {
    "YouTube Video (1920×1080)":  (1920, 1080),
    "YouTube Shorts (1080×1920)": (1080, 1920),
}

from diffusers import StableDiffusionXLPipeline

def load_pipeline(model_id: str):
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    return pipe

pipe = None

style_radio = widgets.RadioButtons(options=list(STYLE_MODELS.keys()), description="Style:")
load_btn    = widgets.Button(description="⚙️ Load model", button_style="primary")
model_status = widgets.Label(value="No model loaded")

def on_load(btn):
    global pipe
    model_status.value = f"⏳ Loading {style_radio.value}..."
    load_btn.disabled = True
    try:
        pipe = load_pipeline(STYLE_MODELS[style_radio.value])
        model_status.value = f"✅ {style_radio.value} ready"
    except Exception as e:
        model_status.value = f"❌ {e}"
    finally:
        load_btn.disabled = False

load_btn.on_click(on_load)
display(widgets.VBox([style_radio, widgets.HBox([load_btn, model_status])]))
```

---

## Cell 3 — Generation UI (required widgets)

| Variable | Widget type | Purpose |
|---|---|---|
| `prompt_ta` | `Textarea` | Main prompt |
| `neg_prompt_ta` | `Textarea` | Negative prompt |
| `count_slider` | `IntSlider` | Number of images (1–8) |
| `seed_input` | `IntText` | Seed value |
| `random_seed_btn` | `Button` | Generate random seed |
| `seed_mode_radio` | `RadioButtons` | "Fixed" / "Random each time" |
| `size_radio` | `RadioButtons` | Video / Shorts |
| `face_upload` | `FileUpload` | Optional face photo |
| `generate_btn` | `Button` | Trigger generation |
| `progress_bar` | `IntProgress` | Generation progress |
| `status_label` | `Label` | Current status text |
| `gallery_output` | `Output` | Gallery render target |

Seed logic:
```python
def get_seed():
    if seed_mode_radio.value == "Random each time":
        return random.randint(0, 2**32 - 1)
    return seed_input.value
```

---

## Cell 4 — Logic (canonical functions)

```python
def generate_images(pipe, prompt, negative_prompt, seed, count, width, height):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_images_per_prompt=count,
        width=width, height=height,
        generator=generator,
        num_inference_steps=30,
        guidance_scale=7.5,
    ).images

def load_face_pipeline(pipe, face_bytes):
    face_img = Image.open(io.BytesIO(face_bytes)).convert("RGB")
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors",
    )
    pipe.set_ip_adapter_scale(0.7)
    return pipe, face_img

def generate_with_face(pipe, face_img, prompt, negative_prompt, seed, count, width, height):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        ip_adapter_image=face_img,
        num_images_per_prompt=count,
        width=width, height=height,
        generator=generator,
        num_inference_steps=30,
        guidance_scale=7.5,
    ).images
```

IP-Adapter loads once on first face generation. Call `pipe.unload_ip_adapter()` before non-face generation if face was used previously.

---

## Cell 5 — Gallery (canonical)

```python
def image_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def show_gallery(images, seeds_used):
    cards = []
    for i, (img, seed) in enumerate(zip(images, seeds_used)):
        b64 = image_to_b64(img)
        dl  = widgets.HTML(
            f'<a download="img_{i+1}_seed{seed}.png" href="data:image/png;base64,{b64}">'
            f'<button>⬇ Download</button></a>'
        )
        thumb = widgets.Image(value=base64.b64decode(b64), format="png", width=320, height=180)
        cards.append(widgets.VBox([thumb, dl]))
    rows = [widgets.HBox(cards[i:i+4]) for i in range(0, len(cards), 4)]
    display(widgets.VBox(rows))
```

---

## Known issues

| Error | Cause | Fix |
|---|---|---|
| `CUDA out of memory` | Large size + many images | `enable_model_cpu_offload()` must be called |
| `xformers not installed` | xformers missing | Remove `enable_xformers_memory_efficient_attention()` |
| IP-Adapter fails to load | diffusers too old | Ensure diffusers >= 0.25.0 |
| Face not applied | scale is 0 | Confirm `set_ip_adapter_scale(0.7)` |
