# WolfGPT Glossary

Português: [pt-BR/GLOSSARIO.md](pt-BR/GLOSSARIO.md)

Plain-language definitions of the terms used while building WolfGPT, written as I learned them.

## Hardware and software

**GPU** - The graphics card. It does huge numbers of simple calculations at once, which is exactly what neural networks need. Ours: NVIDIA RTX 5060 Ti.

**VRAM** - The GPU's own memory (16 GB here). The model's weights, the data being processed and the training state must all fit in it. This is why a 1B-parameter model does not fit on this card.

**Tensor cores** - Special circuits on the GPU that multiply small blocks of numbers in a single step. The fast 16-bit speeds come from them.

**CUDA** - NVIDIA's software layer that lets programs run on the GPU.

**Compute capability / sm_120** - The GPU's architecture version. Ours is 12.0 (Blackwell). PyTorch must ship code compiled for it, which is why this project needs a CUDA 12.8+ build.

**WSL2** - Windows Subsystem for Linux 2: a real Linux running inside Windows. AI tools run on Linux, while still using the Windows GPU driver.

**venv (virtual environment)** - An isolated folder of Python packages for one project, so installs do not affect anything else. It must be activated in every new terminal (`source ~/wolfgpt-env/bin/activate`); the prompt then starts with (wolfgpt-env).

**PyTorch** - The Python library used to build and train neural networks on the GPU.

**torch.compile** - PyTorch's compiler for the model. In the pilot it gave +38% speed and -38% memory.

## Math and speed

**Matrix / matmul (matrix multiplication)** - A matrix is a grid of numbers. Multiplying two grids means: for each cell of the result, multiply a row of one by a column of the other and add it up. A GPT is mostly matmuls repeated billions of times, which is why we benchmark them.

**FLOP** - One arithmetic operation (one multiply or one add). Multiplying two N x N matrices costs about 2 x N^3 FLOPs.

**TFLOP/s** - Trillions of FLOPs per second. Our measured bf16 speed is about 50 TFLOP/s.

**Number formats**
- **fp32** - 32 bits per number. Precise but slow.
- **fp16** - 16 bits. Faster and half the memory, but a narrow range of sizes, so it can overflow.
- **bf16** - 16 bits with the same range as fp32 but less detail. Rarely overflows, so it is our training format.
- **TF32** - fp32 inputs rounded during the multiply. About half the speed of bf16 here.

**MFU (model FLOPs utilization)** - The share of the benchmarked peak that real training uses. Training also moves data and does other math, so 30-50% is typical. WolfGPT's main run used about two thirds (33.0 of about 50 TFLOP/s).

## The model

**Parameters (the "110M")** - The learned numbers inside the model. 110M means 110 million of them (110,119,680 exactly). More data never adds parameters; only a different model shape does.

**Weights** - Another word for the parameters. Training adjusts them.

**Token** - A chunk of text, roughly three-quarters of a word. "Tokens per second" is how fast training reads text.

**Tokenizer / BPE** - The program that cuts text into tokens. Byte-level BPE (byte-pair encoding) builds its vocabulary (32,768 tokens here) by repeatedly merging the most frequent pairs of bytes.

**Context window (context length)** - How many tokens the model can look at at once. WolfGPT: 1,024. Anything longer does not fit, and the model was only trained on positions up to 1,024.

**GPT** - Generative Pre-trained Transformer: a model trained to predict the next token.

**Attention, RoPE, RMSNorm, SwiGLU, tied embeddings** - Attention lets each token look at earlier tokens. RoPE tells attention where each token is. RMSNorm keeps the numbers in a stable range between layers. SwiGLU is a gated feed-forward layer (half of WolfGPT's parameters are in it). Tied embeddings means the word table at the input and the one at the output are the same numbers.

**Perplexity** - e raised to the loss: roughly, among how many words the model is torn at each step. WolfGPT's base model: about 22.8.

**Training** - Repeatedly showing the model text, measuring how wrong its next-token guesses are (the loss), and nudging the weights to be less wrong.

## How training is organised

**Epoch** - One full pass over the whole training dataset. Pretraining was less than one epoch (each of the 3B tokens seen about once). SFT uses 2 epochs. More passes help learning at first, but too many make the model memorize the examples (overfitting). Many different examples seen once or twice is safer than few examples seen many times.

**Step, batch, micro-batch** - A step is one update of the weights. A batch is the group of examples used for that update (32 conversations in SFT). A micro-batch is the slice that fits on the GPU at once; the results of several slices are added up before the update (gradient accumulation). In SFT the slices are sized by tokens (4,096), not by number of conversations.

**Padding** - Filler tokens added to short examples so every example in a batch has the same length. They are ignored in the loss.

**Optimizer (AdamW, Muon)** - The rule for how to update the weights from the gradient. AdamW is the usual choice. Muon is newer and is used for the hidden matrices; in the pilot it reached a loss of 4.418 against 5.280 for AdamW.

**Learning rate** - The size of each weight update. Too high: unstable training or forgetting what the model knew. Too low: slow. **Warm-up**: start small and ramp up over the first steps. **Cosine decay**: lower it smoothly, here to 10% of the peak by the end. A **learning-rate sweep** tries several values on a shorter run and keeps the best.

**Gradient clipping / gradient norm** - Limiting the size of a single update. A spike in the gradient norm warns that training is becoming unstable (the largest seen in the main run was 2.28: no spikes).

**Training loss vs validation loss** - Training loss is measured on data the model learns from; validation loss on held-out data it never trains on. Validation is the honest number.

**Overfitting** - Training loss keeps falling while validation loss stops improving or rises: the model is memorizing instead of learning (seen in Phase 1, within minutes). Keeping the best-validation checkpoint guards against it.

**Checkpoint** - A saved snapshot of the model (and optionally the optimizer and data position) so training can resume after a crash.

**Resume snapshot** - A checkpoint that also stores the optimizer state, the random-number state and the position in the data, so an interrupted run continues exactly where it stopped and ends with the same weights. Re-running run_overnight.sh uses it automatically.

**Smoke test** - A tiny run (about 20 steps) that checks everything works before starting the long run.

## Data

**Deduplication** - Removing repeated examples. Needed so a copy of a validation example is not also in the training data.

**Data leakage** - When validation examples (or copies of them) are seen during training, so the validation loss looks better than the model really is.

**Dropped conversations** - In SFT data prep, conversations whose first question and answer together exceed 1,024 tokens are removed (about 2%). Cutting them would delete the end of the answer, including the end-of-text token, and teach the model that answers stop at random points. Later turns of a multi-turn chat that do not fit are cut off whole.

**System message** - An instruction given to the model before the user's message (for example "summarize in one sentence"). We keep it only for the rewrite and summarize subsets, where the task lives in it.

**Synthetic data** - Training text written by another AI model instead of by humans. Cheap and plentiful, but it inherits that model's mistakes and style, and its terms of use. smol-smoltalk is synthetic; Dolly is human-written.

**Magpie** - A method for making synthetic instruction data. A chat model is given only the start of a user turn (the template that normally comes before a question), so it invents the question itself; then a model answers it. This yields huge, varied question sets without writing prompts by hand. magpie-ultra was made this way with Llama 3.1 405B-Instruct; **smol-magpie-ultra** is the shorter-conversation version prepared for models under 1B parameters, and it is the biggest part of smol-smoltalk (57% of the mix). Because the text comes from Llama, its licence terms apply to WolfGPT.

**Upsampling / repetition of a small source** - Showing a small but important source several times (here the everyday-conversation greetings, x3, in training only) so it is not drowned out by 450k other chats. Validation is never repeated.

## Making the model useful

**Base model** - A model that only continues text (what WolfGPT is after Phase 4). It does not answer questions or chat.

**Fine-tuning** - Continuing to train an already trained model on a smaller, targeted dataset, so its weights change. Best for changing behavior and format, not for adding many reliable facts. Like teaching a student a skill.

**SFT (supervised fine-tuning)** - Fine-tuning on example conversations (a question and the answer we want). This is Phase 5: it turns the base model into a chat model.

**Loss masking** - In SFT, only the answer tokens count toward the loss; the user's question is given to the model but not scored, so it learns to answer rather than to imitate questions.

**Chat template** - The fixed text layout of a conversation ("### User: ... ### Assistant: ..."), plus an end-of-text token that tells the model when its answer is finished.

**Identity stage** - A short extra fine-tuning on hand-written conversations that teach the model its name and who built it. Repeating a tiny dataset many times taught the name but also made the model memorize it and recite it at unrelated short messages.

**LoRA** - A cheaper fine-tuning method that trains only small add-on layers instead of all the weights. Saves memory; not needed here because the whole 110M model fits on the GPU.

**RAG (retrieval-augmented generation)** - At question time, search a document collection, paste the best passages into the prompt, and let the model answer from them. The model is not retrained, so facts can be changed by editing the documents. Like an open-book exam. Limited here by the 1,024-token context and the small model's weak reading. An idea, not built.

**Hallucination** - Fluent text that states invented facts. WolfGPT does this often at 110M parameters (the 3,800 heart cells, the boiling point of 15 degrees).

## Sampling (how text is generated)

**Temperature** - Controls randomness when picking the next word. Low (0.3): always the safest word, which causes loops. High (1.0+): more varied but more nonsense. 0.7-0.9 works best for the base model.

**Top-k** - Only choose among the k most likely next words (we use 40-50), which cuts out very unlikely words.

**Repetition penalty** - A sampling setting that lowers the score of words already used in the answer, which reduces loops. Ours is a fixed subtraction, off by default.

**No-repeat n-gram ban** - Forbid generating a phrase of N tokens that already appeared. It removed all repeated 4-token phrases in an offline test, but can produce odd words when the model has nothing else to say.

**Reject-and-retry** - After each reply, score how much of it repeats earlier replies; if it is above a threshold (0.3), throw it away and generate again, a bit warmer and with less memory. No retraining needed.

## Evaluation

**Zero-shot** - Tested with no examples in the prompt. **Benchmark** - A public test set with known answers.

**HellaSwag, ARC-Easy, ARC-Challenge, PIQA, Winogrande** - Multiple-choice common-sense and science tests. The model scores each answer text by log-likelihood and picks the best. Random guessing gives 25% on the first three and 50% on the last two.

**acc / acc_norm** - Accuracy, and accuracy with each answer's score divided by its length (so long answers are not punished).

**Standard error** - How much a score would wobble with a different sample of questions. About 0.5 points on HellaSwag and 1.4 on Winogrande.

## Licences

**Llama 3.1 Community License** - If outputs of Llama are used to train a model that is distributed, the model's name must begin with "Llama", show "Built with Llama", and include the licence and a NOTICE file. Not legal advice.

**ODC-By, CC BY-SA** - Licences that require attribution (and, for CC BY-SA, sharing under the same terms). FineWeb-Edu is ODC-By; Dolly is CC BY-SA 3.0.

**safetensors** - A safe file format for model weights (unlike pickle files, it cannot run code when loaded).
