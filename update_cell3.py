import json

# Read the notebook
with open("notebooks/sd-image-generator/notebook.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# New source code for Cell 3
new_source = """from diffusers import StableDiffusionXLPipeline

def image_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def show_gallery(images, seeds_used):
    cards = []
    for i, (img, seed) in enumerate(zip(images, seeds_used)):
        b64 = image_to_b64(img)
        dl = widgets.HTML(
            f'<a download="img_{i+1}_seed{seed}.png" href="data:image/png;base64,{b64}">'
            f'<button>⬇ Download</button></a>'
        )
        thumb = widgets.Image(value=base64.b64decode(b64), format="png", width=320, height=180)
        cards.append(widgets.VBox([thumb, dl]))
    rows = [widgets.HBox(cards[i:i+4]) for i in range(0, len(cards), 4)]
    display(widgets.VBox(rows))

def generate_images(pipe, prompt, negative_prompt, seed, count, width, height, guidance):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if refiner is not None and isinstance(pipe, StableDiffusionXLPipeline):
        raw = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=count,
            width=width, height=height,
            generator=generator,
            num_inference_steps=40,
            denoising_end=0.8,
            guidance_scale=guidance,
            output_type="latent",
        ).images
        generator2 = torch.Generator(device="cuda").manual_seed(seed)
        return refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
            width=width, height=height,
            generator=generator2,
            num_inference_steps=40,
            denoising_start=0.8,
            guidance_scale=guidance,
            image=raw,
        ).images
    else:
        return pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=count,
            width=width, height=height,
            generator=generator,
            num_inference_steps=30,
            guidance_scale=guidance,
        ).images

def load_face_pipeline(pipe, face_bytes):
    if not getattr(pipe, "_ip_adapter_loaded", False):
        face_img = Image.open(io.BytesIO(face_bytes)).convert("RGB")
        pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors",
        )
        pipe.set_ip_adapter_scale(0.7)
        pipe._ip_adapter_loaded = True
    else:
        face_img = Image.open(io.BytesIO(face_bytes)).convert("RGB")
    return pipe, face_img

def generate_with_face(pipe, face_img, prompt, negative_prompt, seed, count, width, height, guidance):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if refiner is not None and isinstance(pipe, StableDiffusionXLPipeline):
        raw = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            ip_adapter_image=face_img,
            num_images_per_prompt=count,
            width=width, height=height,
            generator=generator,
            num_inference_steps=40,
            denoising_end=0.8,
            guidance_scale=guidance,
            output_type="latent",
        ).images
        generator2 = torch.Generator(device="cuda").manual_seed(seed)
        return refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
            width=width, height=height,
            generator=generator2,
            num_inference_steps=40,
            denoising_start=0.8,
            guidance_scale=guidance,
            image=raw,
        ).images
    else:
        return pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            ip_adapter_image=face_img,
            num_images_per_prompt=count,
            width=width, height=height,
            generator=generator,
            num_inference_steps=30,
            guidance_scale=guidance,
        ).images

prompt_ta = widgets.Textarea(
    placeholder="Describe what you want to generate...",
    description="Prompt:",
    layout=widgets.Layout(width="600px", height="80px"),
)
neg_prompt_ta = widgets.Textarea(
    value="deformed, ugly, blurry, low quality, worst quality",
    description="Negative:",
    layout=widgets.Layout(width="600px", height="60px"),
)
count_slider = widgets.IntSlider(
    value=2, min=1, max=8, step=1,
    description="Count:",
    layout=widgets.Layout(width="400px"),
)
guidance_slider = widgets.FloatSlider(
    value=7.5, min=1.0, max=15.0, step=0.5,
    description="Guidance:",
    layout=widgets.Layout(width="400px"),
)
seed_input = widgets.IntText(
    value=42,
    description="Seed:",
    layout=widgets.Layout(width="200px"),
)
random_seed_btn = widgets.Button(
    description="🎲 Random seed",
    layout=widgets.Layout(width="160px"),
)
seed_mode_radio = widgets.RadioButtons(
    options=["Fixed", "Random each time"],
    value="Fixed",
    description="Seed mode:",
)
size_radio = widgets.RadioButtons(
    options=list(SIZES.keys()),
    description="Size:",
)
face_upload = widgets.FileUpload(
    accept="image/*",
    description="Face (opt.):",
    layout=widgets.Layout(width="300px"),
)
generate_btn = widgets.Button(
    description="🚀 Generate",
    button_style="success",
    layout=widgets.Layout(width="200px", height="40px"),
)
progress_bar = widgets.IntProgress(
    value=0, min=0, max=100,
    description="Progress:",
    layout=widgets.Layout(width="400px"),
)
status_label = widgets.Label(value="Ready")
gallery_output = widgets.Output()

def get_seed():
    if seed_mode_radio.value == "Random each time":
        return random.randint(0, 2**32 - 1)
    return seed_input.value

def on_random_seed(btn):
    seed_input.value = random.randint(0, 2**32 - 1)
random_seed_btn.on_click(on_random_seed)

def on_generate(btn):
    if pipe is None:
        status_label.value = "❌ Load a model first (run the cell above)"
        return
    generate_btn.disabled = True
    progress_bar.value = 0
    status_label.value = "⏳ Generating..."
    try:
        seed = get_seed()
        seed_input.value = seed
        width, height = SIZES[size_radio.value]
        guidance = guidance_slider.value
        # Warn if refiner loaded but current style doesn't support it
        if refiner is not None and not isinstance(pipe, StableDiffusionXLPipeline):
            status_label.value = "⚠️ Refiner not supported for this style, generating without it..."
        face_bytes = None
        if face_upload.value:
            face_bytes = face_upload.value[0]["content"]
        progress_bar.value = 30
        if face_bytes:
            updated_pipe, face_img = load_face_pipeline(pipe, face_bytes)
            images = generate_with_face(
                updated_pipe, face_img,
                prompt_ta.value, neg_prompt_ta.value,
                seed, count_slider.value, width, height, guidance,
            )
        else:
            if getattr(pipe, "_ip_adapter_loaded", False):
                pipe.unload_ip_adapter()
                pipe._ip_adapter_loaded = False
            images = generate_images(
                pipe,
                prompt_ta.value, neg_prompt_ta.value,
                seed, count_slider.value, width, height, guidance,
            )
        progress_bar.value = 90
        with gallery_output:
            clear_output(wait=True)
            show_gallery(images, [seed] * len(images))
        progress_bar.value = 100
        status_label.value = f"✅ Done — {len(images)} image(s) | seed: {seed}"
    except Exception as e:
        status_label.value = f"❌ Error: {e}"
    finally:
        generate_btn.disabled = False

generate_btn.on_click(on_generate)

display(widgets.VBox([
    prompt_ta,
    neg_prompt_ta,
    widgets.HBox([count_slider, guidance_slider]),
    widgets.HBox([seed_input, random_seed_btn]),
    seed_mode_radio,
    size_radio,
    face_upload,
    generate_btn,
    widgets.HBox([progress_bar, status_label]),
]))
display(gallery_output)"""

# Replace Cell 3 (index 3) - convert to list with newlines preserved
source_lines = new_source.split("\n")
nb["cells"][3]["source"] = [line + "\n" if i < len(source_lines) - 1 else line for i, line in enumerate(source_lines)]
nb["cells"][3]["outputs"] = []
nb["cells"][3]["execution_count"] = None

# Write back
with open("notebooks/sd-image-generator/notebook.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Cell 3 updated successfully")
