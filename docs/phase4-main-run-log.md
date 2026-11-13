# Phase 4 - Main run, 110M WolfGPT (2026-11-09 to 11-10) - DONE

Result: final val loss 3.129 (perplexity ~22.8) after 3.0B tokens. Best val = final val (no overfitting, no late problems).

Configuration (read from runs/main.json)
- Model: 110.1M params (12 layers, 12 heads, width 768, context 1024), tied embeddings, RoPE, RMSNorm, SwiGLU.
- Optimizer: Muon (lr 0.02, wd 0) on hidden matrices + AdamW (lr 2e-3) on embeddings/norms. Clip 1.0.
- 22,888 steps x 131,072 tokens = 3.0B tokens, one pass over the data (shuffled windows). Warmup 228 steps (1%), cosine decay to 10%.
- bf16 autocast, torch.compile, micro-batch 8, eval every 250 steps on a fixed 320-window validation set, checkpoint every 1000 steps.
- The run used --resume at least once (stopped and continued from a checkpoint): the checkpoint code worked in real use.

Speed
- 46.0k tokens/s, 33.0 TFLOP/s (with the corrected attention FLOP count), peak memory 6.5 GB. Training time ~18.1 h at that speed (better than the 21 h estimate, ~13% faster than the pilot suggested).

Loss curve (val): step 0 10.57 | 500 4.22 | 1000 3.90 | 2500 3.60 | 5000 3.43 | 10000 3.28 | 15000 3.19 | 20000 3.14 | 22888 3.129
- Max grad norm at any eval 2.28: no spikes. The last 1600 steps improved only 0.005 (expected as the cosine schedule reaches its floor).

Files (Linux disk, ~/wolfgpt-data/ckpt/): main_best.pt (421 MB, weights only), main_last.pt (937 MB, weights + optimizer). Also runs/main.json in phase3. generate.py loads main_best.pt to sample text.

## Base-model samples, record of what it said (2026-11-11, generate.py, main_best.pt)

### A. Temperature 0.8, top_k 50 (and 0.7 for the 1969 prompt)

Prompts: "The human heart is" (x3), "In 1969, the first", "hello".

What works
- Fluent, grammatical English in the style of educational web pages (FineWeb-Edu): proper sentences, paragraph breaks, topic stays on track for a few sentences, typographic apostrophes, plausible vocabulary (pulmonary artery, valves, oxygenated blood).
- Correct high-level ideas appear: the heart pumps blood, the pulmonary artery carries blood to the lungs, the heart is a muscle.

What does not
- Facts are invented: "about 3,800 heart cells", "pumps about 5,000 times per second", "85% of the blood volume", a "Russian archaeologist Vasiliy Lensky" who commanded the "U.S. Army's Central Command", Gypsies/Goths/Georgians as North American farmers. Treat every specific claim as unreliable.
- Local repetition and loops: "blood clotting, blood clotting, clotting of blood"; the pulmonary-artery sentence repeated; a made-up word ("capilli") and self-contradictions ("arteries are the tiny vessels", "veins" vs "arteries").
- Drifts off topic within one paragraph (the 1969 prompt turned into an expedition and army biography).
- "hello" is not a greeting for a base model: it continued as if "hello" were the start of an article (about Gypsy peoples). As expected, there is no chat behavior yet.

### B. Low temperature (0.5 for the heart, 0.3 for water), same model

"The human heart is" at temperature 0.5, 3 samples:
- Sample 1 fell into a loop: "It is the seat of the mind." repeated until the output ended, after saying the heart is "the largest organ of the body" and "the organ of the mind".
- Sample 2: says the heart "produces blood", "pumps blood in the same way as the kidneys", and "has two chambers, two main chambers, and two smaller chambers"; the sentence "The heart is an organ that produces blood to supply the body with oxygen" came back three times.
- Sample 3: a template sentence ("The heart is a complex organ with many functions, including ... the elimination of waste products") repeated four times with small changes; also claims the heart produces oxygen and carbon dioxide.

"Water boils at" at temperature 0.3:
- "Water boils at a temperature of about 15 degrees Celsius" (also "35"), then the same two sentences repeated over and over. The correct value is 100 C, so this is a wrong fact held with high confidence.

Reading: at low temperature the model picks its single most likely word each time, which feeds back into loops; it also shows the wrong facts are what the model actually believes, not random noise. Best range for this model: temperature about 0.7-0.9 with top_k 40-50 (and a repetition penalty for chat).

### C. Accident in interactive mode

A shell command (`python generate.py "The human heart is" --temperature 0.5 --n 3 ...`) was pasted into the "start of text>" prompt by mistake. The base model simply continued it as text: it repeated the command and wrote lines such as "The heart is a very hard place." followed by a drifting stream of questions about "computer models" of the heart. Nothing was executed (the model only produces text). It shows a base model has no concept of a command, a user or an answer: it only continues whatever text it is given. Fixing this is the purpose of Phase 5.

Overall reading: this matches a 110M-parameter model trained on 3B tokens: good surface language, little reliable knowledge, loops at low temperature.

Caveats
- Val loss is on 320 windows of FineWeb-Edu, so it is not comparable with Phase 1 character-level numbers or with other tokenizers.
- This is a base model: it continues text, it does not follow instructions yet.

Next: Phase 5 (post-training / SFT). It teaches format (answering, chatting), not new facts; factual reliability will stay limited at this size.
