# Phase 6 - Publishing: Llama licence notes and benchmarks (2026-11-13)

The model card itself is `phase5/MODEL_CARD.md`; the release folder is built by `phase5/export_release.py` (name: Llama-WolfGPT-110M).

## What the Llama 3.1 licence asks (from the licence text)
- If Llama outputs are used to create/train/fine-tune an AI model that is distributed or made available, "Llama" must be at the beginning of that model's name.
- Prominently display "Built with Llama" (website, UI, blogpost, about page or documentation).
- Include a copy of the agreement with Llama Materials you distribute, and keep the attribution notice in a NOTICE text file: "Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright (c) Meta Platforms, Inc. All Rights Reserved."
- Use must comply with the Llama 3.1 Acceptable Use Policy.

## What of Llama was used
- Weights: no. Code/tokenizer/architecture files: no.
- Text outputs: yes. smol-magpie-ultra (made with Llama-3.1-405B-Instruct per the smoltalk dataset card) = 270,838 of 477,612 conversations (about 57%) in the SFT mix.
- everyday-conversations: the dataset card refers to a Llama-3.1-related source; the generator is not stated. Treated as possibly Llama.
- Other subsets: the generator is not named on the card (an arXiv paper says smol-constraints/summarize/rewrite were made with a Qwen2.5-72B-sized model; not fully verified). OpenHermes 2.5 and others come from other datasets with their own terms; not individually checked.

## Benchmarks
See the table in `phase5-sft-results.md`. Method: the model scores each answer text by log-likelihood; acc_norm per task (acc for Winogrande). The scoring code was tested against a one-by-one reference (difference 1e-5 in fp32).
- The base model is about level with GPT-2 124M on HellaSwag (GPT-2: 31.14 acc_norm in the Eleuther harness) and ahead on ARC-Easy (GPT-2: about 39.5); only roughly comparable because evaluation code differs.
- SFT costs about 8 points on ARC-Easy and 3 on PIQA; partly because the tests use plain text, not the chat format.

## Open points
- Not legal advice; read the licence before publishing.
- Dolly is CC BY-SA 3.0 (share-alike) and FineWeb-Edu is ODC-By: attribution required, check the exact terms.
- Name: the published name is "Llama-WolfGPT-110M" while the model calls itself WolfGPT (the identity data says WolfGPT).
- A Hugging Face access token was pasted into a chat during the project and was revoked. Never put tokens in files or commands that are committed.
