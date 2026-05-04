# SD Image Generator

Kaggle notebook for image generation via Stable Diffusion XL with an ipywidgets UI.

## Features

- Text-to-image generation with prompt
- Style selection: Realism, Anime, Comics, Illustration, Lineart
- YouTube-ready sizes: Video (1920×1080) and Shorts (1080×1920)
- Seed control: fixed or random per generation
- Face mode: generate with a reference face photo (IP-Adapter)
- Results gallery with download buttons

## Kaggle requirements

- GPU: T4 x2 or P100
- Internet: **On** (required — models are downloaded at runtime)
- RAM: 16 GB+

## How to run

1. Enable GPU and Internet in Kaggle notebook settings
2. Run All (Shift+F10)
3. In the Model Config section: select style and click "Load model" (~2–5 min)
4. In the Generation section: enter prompt and click "Generate"

## Stack

- `diffusers` — Stable Diffusion XL inference
- `ipywidgets` — in-notebook UI
- `torch` + `accelerate` — GPU
- `insightface` — face embedding for IP-Adapter
- `h94/IP-Adapter` — face-to-image adapter
