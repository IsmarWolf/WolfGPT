# Phase 5 - SFT results (2026-11-12, overnight run + identity stage + chat fixes)

## Data (full run)
smol-smoltalk (all ~460k) + everyday-conversations (x3 in training) + Dolly. Dedup 477,612 -> 457,399 (20,213 repeated first prompts). 10,064 dropped (first turn over 1,024 tokens). Train 446,353 conversations / 267.5M tokens (77% answer tokens); validation 1,000 conversations. Only 9 everyday chats ended up in train (x3 = 27), so the greeting source is tiny.

## Learning-rate sweep (1 epoch on 150k conversations each, ~48 min each)
| lr | best val |
|---|---|
| 5e-5 | 1.761 |
| 1e-4 | 1.706 |
| 3e-4 | 1.630 (winner, highest value tested: a higher lr, e.g. 5e-4, might do better) |
Base model before SFT: 2.395.

## Final run (lr 3e-4, 2 epochs, 27,896 steps, ~4.6 h, 33.3k tok/s, 4.9 GB)
- Val 2.395 -> 1.484 (best checkpoint final_best.pt, step 27800). Epoch 1 ended at 1.541; epoch 2 added only 0.057; the last ~3,000 steps moved val by 0.003.
- Train loss in epoch 2 (~1.41) is below val (1.485): mild memorisation but val never rose, so no harmful overfitting.
- Per source val: self-oss-instruct 0.723, smollm-rewrite 0.926, smol-contraints 1.335, smol-magpie 1.538, openhermes 1.640, summarize-5k 2.067, explore-instruct-rewrite 2.572, dolly 2.774 (human-written text is the hardest to predict).

## Samples from final_best.pt (default sampling: temp 0.7, top-k 40, no repetition penalty)
- hello -> "Hello! How can I help you today?" (earlier sweep models failed this)
- Capital of France -> "The capital of France is Paris."
- Photosynthesis in two sentences -> fluent, mostly right, but 4 sentences and one false claim ("chlorophyll breaks down glucose")
- Three study tips -> 3 short, generic tips, stops correctly
- 12 + 15 -> "24." (wrong; the 3e-4 sweep model got 25, so arithmetic is luck at this size)
- Poem about the sea -> rhymeless, repeats itself, trails off at the length limit

## Free-form chat with final_best.pt (temp 0.7)
Identity failed as expected: no training data said who it is. "Do you know who you are?" got an off-topic reply, and "I'm your father" made it invent a father called David, a wife, etc. (synthetic data is full of persona/story writing). Replies drift into fiction over several turns.

## Identity stage (2026-11-13): teaching it its name
- Data: identity_data.py = 300 hand-written conversations (41 two-turn, drafted by a small model, reviewed). WolfGPT, small language model, ~110M parameters, trained from scratch, built by "Explorer" (the creator's game nickname). Nothing else personal. prep_identity.py mixes them with 10,000 random ordinary training chats; 20 identity chats held out for validation.
- Run 1 (wolf: lr 3e-5, id x3, 2 epochs, 676 steps): identity val 3.615 -> 2.294, general val unchanged (1.488 -> 1.490). Name still garbled ("I'm Wolf lend"): too gentle.
- Run 2 (wolf2: lr 1e-4, id x8 = 2,240 copies, 3 epochs, 1,146 steps, ~9 min): general val rose slightly (1.488 -> 1.516 at the end) while train loss fell to ~1.0: some memorisation of the repeated identity chats.
- GOTCHA: sft.py keeps the checkpoint with the best OVERALL validation loss, which is 98% ordinary chats. For wolf2 that was step 200, before the name was learned (wolf2_best.pt said "hello! Nice to meet you."). The end-of-training file wolf2_last.pt is the one that learned it. Lesson: for a small targeted stage, choose by the target metric (or use _last), not by overall val.
- RESULT, wolf2_last.pt (temp 0.3, rep penalty 0.3): "who are you?" -> "I'm WolfGPT, a small language model. I can chat, answer simple questions, and help with short writing." / "who made you?" -> "I was built by Explorer, a game nickname." / "are you chatgpt" -> "I'm not chatgpt. I'm WolfGPT, a small model built by Explorer." / capital of Japan -> "Tokyo." (general ability intact).
- SIDE EFFECT of the identity stage (id x8, 18% of that stage's data): in free chat, short or vague messages ("OK", "damm", "ALRIGHT") made it answer "I understand. I'm WolfGPT, a small language model." over and over. Reasons: (1) the chat history is in the prompt and small models copy their own earlier text; (2) the stage had almost no small-talk examples showing that identity answers belong only to identity questions; (3) low temperature (0.3) makes loops sticky; (4) --rep_penalty only looks at the current answer.
- DEMO MODEL: wolf2_last.pt.

## Chat-time fixes in chat.py / chatlib.py (no retraining)
- Drop invalid console bytes from the typed line (crash "TextInputSequence must be str").
- `--no_repeat N`: forbid re-generating an N-token phrase already used in this answer or earlier replies (tested offline: 112 repeated 4-grams -> 0). Side effect: when the model has nothing else to say it produces odd or broken words, so it is optional and off by default.
- Cut the reply at the first "###" (the model trying to open a new "### User:" block, then junk).
- `--retries K` (recommended 3): after each reply, score how much of it repeats earlier replies (share of repeated 4-token phrases, 100% if identical). Above `--repeat_thresh` (0.3) the reply is rejected and regenerated: attempt 2 warmer (+0.2 temperature), attempt 3 warmer with only the last exchange as memory, attempt 4 with no memory. The least repetitive attempt is shown. `--show_flags` prints a "[flag: ...]" line for each rejection (hidden by default).
- Test with `--retries 3` (temp 0.4, rep penalty 0.3): casual chat felt much better; remaining weaknesses are content (odd ideas, "damm" -> "the lifeblood of life", one-line poems), not loops. Sometimes the first words are glued ("Worldally"): model behaviour on unfamiliar topics, not fixed by the patch.

## Benchmarks (eval_bench.py, zero-shot, full sets, plain text)
| Task (n) | random | base (main_best) | SFT (final_best) | SFT + identity (wolf2_last) |
|---|---|---|---|---|
| HellaSwag (10,042) | 25 | 32.3 | 30.9 | 31.1 |
| ARC-Easy (2,376) | 25 | 48.2 | 40.5 | 40.3 |
| ARC-Challenge (1,172) | 25 | 26.0 | 24.9 | 24.6 |
| PIQA (1,838) | 50 | 62.2 | 59.2 | 58.4 |
| Winogrande (1,267) | 50 | 51.7 | 49.6 | 51.1 |
(acc_norm; acc for Winogrande. Approx. standard error 0.5 to 1.4 points.) A 100-question quick run gave HellaSwag 44%: small-sample noise. Chat tuning costs about 8 points on ARC-Easy and 3 on PIQA.

## Bugs / procedure notes
- Pasting a command while a chat is still open sends the command text to the model as a message. Exit the chat (empty line or Ctrl+C) and wait for the shell prompt first. Paste one command at a time.
- When copying files to a machine through a sync step, verify them byte for byte afterwards: an out-of-date copy can arrive silently.

## Verdict
SFT worked: the model greets, answers, follows the format, stops, and knows its name and builder. Weaknesses are what 110M parameters predict: shaky facts, no arithmetic, ignores "two sentences", short or repetitive poems, invents stories when role-played at, weak at casual one-word messages.

## Next ideas
Publish (see phase6 notes). Optional: polish round (about 150 varied small-talk chats, fewer identity copies, pick the checkpoint by an identity+small-talk metric), lr 5e-4 comparison, "scratchpad arithmetic" experiment with programmatically generated data, RAG demo over project notes, multi-machine low-communication training (DiLoCo) for a bigger v2.
