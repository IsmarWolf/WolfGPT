# What I learned

Português: [pt-BR/APRENDIZADOS.md](pt-BR/APRENDIZADOS.md)

Each lesson says where it came from. "Measured" means a number from our own runs; "advice" means something we read or were told and did not test.

## About training

1. **Judge by validation loss, never training loss.** Measured in Phase 1: validation bottomed at 1.469 and rose to 1.679 while training fell to 0.625. Keep the best-validation checkpoint and stop after a few evaluations without improvement.
2. **Losses are only comparable on the same data and tokenizer.** The character-level numbers (1.4) and the BPE numbers (3.1) are not comparable.
3. **The optimizer mattered a lot at small scale.** Measured: Muon beat AdamW by 0.86 in loss in the 52M-token pilot, for about 4% less speed. Caveat: short pilots overstate gaps, and AdamW was only tried at two learning rates.
4. **Fix the code before the long run.** The independent review found real problems (data seen with repeats, a Muon update that favoured one block, a validation set that depended on batch size, no resume). Resume was tested to give bit-identical weights, and the main run was stopped and resumed once or more without trouble.
5. **Tokens per parameter.** 3.0B tokens for 110M parameters is about 27 tokens per parameter, above the 20 rule of thumb, with one pass over the data: validation kept falling to the end (no overfitting).
6. **Speed is mostly about not wasting the GPU.** torch.compile gave +38% speed and -38% memory in the pilot. The loss is summed on the GPU with one sync per step. Training reached 33.0 TFLOP/s, about two thirds of the matmul peak.
7. **Honest FLOP counting.** Causal attention does half the work, so counting it fully overstated speed by 8 to 10%.

## About fine-tuning (SFT)

8. **SFT teaches format, not knowledge.** After SFT the model greets, answers and stops, but facts stay unreliable.
9. **Mask the prompt, keep the end token.** Loss only on answer tokens, and the end-of-text token is part of the answer so the model learns to stop.
10. **Deduplicate before splitting** so validation copies are not in training. 20,213 repeated first prompts were removed.
11. **Drop, do not cut, over-long examples.** Cutting would delete the end token and teach the model that answers stop at random points.
12. **More different examples beats more passes over the same ones.** Overfitting comes from repetition. The final run saw 446k distinct chats twice; the second epoch added only 0.057.
13. **Sweep the learning rate on a shorter run.** 3e-4 won 1.630 vs 1.706 vs 1.761, but it was the highest tried, so the optimum may be higher.
14. **Validation loss does not show drift.** Read samples. Early sweep models looped ("Read the text" x15); the final one did not.
15. **Small targeted stages need a targeted metric.** The training script kept the checkpoint with the best overall validation loss, which is 98% ordinary chats, so it picked step 200, before the name was learned. For a small stage use the last checkpoint or measure the target.
16. **Repeating a tiny dataset many times has a price.** Identity chats x8 taught the name but also memorised them (training loss ~1.0, validation up 1.488 to 1.516) and made the model recite its identity to unrelated short messages. The root fix, not done, is more varied small-talk data and fewer identity copies.
17. **A model copies itself.** The chat history is in the prompt, so a small model repeats its own earlier replies; low temperature makes loops sticky.

## About generation

18. **Low temperature loops.** Measured on the base model: at 0.3 to 0.5 it repeated the same sentences and held wrong facts with confidence (water boils at 15 C). The useful range was about 0.7 to 0.9 with top-k 40 to 50 for the base model.
19. **Fix repetition at chat time before retraining.** Score replies for repeated 4-token phrases, reject and retry warmer, cut at the first "###". Offline: 112 repeated phrases to 0. The anti-repeat n-gram ban alone can make words odd when the model has nothing else to say, so it is optional.
20. **Typing into the wrong prompt.** A shell command pasted into a chat is just text to the model; it continues it. Nothing executes.

## About measuring

21. **Small samples lie.** 100 HellaSwag questions gave 44%; the full 10,042 gave 32.3%. Report the sample size and the standard error.
22. **Test your evaluation code** against a slow one-by-one reference (difference 1e-5).
23. **The honest result is mixed.** The base model is above chance on HellaSwag, ARC-Easy and PIQA, at chance on ARC-Challenge and Winogrande. SFT lowers plain-text scores by a few points. The base model is about equal to GPT-2 124M on HellaSwag (31.1 reported for GPT-2, different evaluation code, only roughly comparable).

## About data and licences

24. **Synthetic data inherits its generator's terms.** 57% of the chat conversations came from Llama 3.1 405B, so the model carries the Llama name, a "Built with Llama" notice, the licence and a NOTICE file. Dolly is CC BY-SA 3.0 and FineWeb-Edu is ODC-By: attribution is required. Not legal advice.
25. **Do not re-upload web text.** Link to the dataset card instead.
26. **Synthetic persona data makes small models invent stories.** Role-playing at the model ("I'm your father") made it invent a father named David.

## About working

27. **Secrets.** A Hugging Face token was pasted into a terminal log; it was revoked immediately. Use environment variables, keep `.env` out of git, and never paste tokens into chats or logs.
28. **Check what you delivered.** Files sent to the PC sometimes arrived as the previous version. The fix: wait, compare byte for byte.
29. **Run one command at a time.** Pasting a command while a chat is still open sends it to the model as a message.
30. **Documenting as you go is cheap, reconstructing later is not.** Every phase has a dated log; this repository is built from them.

## Ideas, not tested

Only advice or plans; nothing was built or measured.

- Training together with other PCs: ordinary data-parallel training needs fast links between machines, and internet links are far too slow. Low-communication methods (DiLoCo, OpenDiLoCo, INTELLECT-1) exist for that; a two-machine test is an idea.
- A polish round (about 150 varied small-talk chats, fewer identity copies, pick the checkpoint by an identity plus small-talk metric).
- Learning rate 5e-4 for SFT; scratchpad arithmetic with programmatically generated data; a retrieval (RAG) demo over the project notes; more pretraining tokens (the 10B sample) or a larger model (about 325M with context 2,048).
- A "thinking" model was asked about but not attempted.

See also: [journey](JOURNEY.md), [decisions](DECISIONS.md), [glossary](GLOSSARY.md).
