# CLAUDE.md

Always use Haiku model unless explicitly asked otherwise.

All files in this repo must be written in English only. Exception: `docs/plans/` — Russian is allowed there.

## Notebook rules

- **Stateless**: no datasets required on Kaggle, no external files at start
- **Self-contained**: notebook installs all packages and downloads all models itself on first run
- **UI-only config**: no hardcoded settings in code — all configuration through ipywidgets
- **No beta/experimental**: stable releases only (no rc, no dev, no experimental flags)
- **GPU**: optimized for Kaggle T4 x2 / P100; always use `torch_dtype=torch.float16`

## Kaggle environment

- `HF_HOME` must be set to `/kaggle/working/hf_cache` in Cell 1 (prevents `/root` overflow)
- Models are NOT cached between Kaggle sessions — downloaded fresh every run
- Internet must be enabled in Kaggle notebook settings
