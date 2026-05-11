# Speaker-Aware Transcription, Diarization, and ASR

This repository collects experiments around **who spoke when** (speaker diarization) and how that ties into **automatic speech recognition (ASR)**—so transcripts can be **speaker-aware** (utterances attributed to speakers, not just plain text).

## What is here

- **`diarization_finetune_lean (6).ipynb`** — A self-contained workflow to **fine-tune** [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0) on the **VoxConverse** diarization benchmark, then **evaluate** a full diarization stack (aligned with [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1)). The notebook pins a compatible **PyTorch / torchaudio / pyannote.audio** stack, validates imports, and supports a quick **smoke run** vs. full training.

## Before you run

1. **Hugging Face access** — Accept the model terms on Hugging Face for the segmentation and speaker-diarization models linked above (gated checkpoints).
2. **Authentication** — Log in as needed (`huggingface-cli login` or equivalent) so the notebook can download weights.
3. **Fast test vs. full run** — By default the notebook uses **`FAST_TEST=1`** for a small end-to-end check. For training on all benchmark files, set **`FAST_TEST=0`** in the environment before starting the kernel, or change the config in the notebook as documented in the first markdown cell.

## Requirements

Python environment with GPU recommended for training. Exact package pins and install logic are handled inside the notebook’s setup cell (Torch 2.4.1 stack, pyannote 3.x, Lightning, etc.).
