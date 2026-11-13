"""Phase 6 step 2: build the publishable folder for Llama-WolfGPT-110M and check it.

  python export_release.py --ckpt ~/wolfgpt-data/ckpt/wolf2_last.pt
Writes ../release/Llama-WolfGPT-110M with: model.safetensors, config.json, tokenizer.json, model.py, chat.py,
README.md (the model card), NOTICE, benchmarks.json, and LICENSE if you put the Llama 3.1 licence text in
phase5/LICENSE (download it from Meta's Llama 3.1 licence page). Then it reloads the folder from scratch and checks
that the exported model gives exactly the same outputs as your checkpoint.
"""
import argparse, hashlib, importlib.util, json, os, shutil, sys
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from chatlib import load_model, DEFAULT_TOK
import model as model_module

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--ckpt", default=os.path.expanduser("~/wolfgpt-data/ckpt/wolf2_last.pt"))
p.add_argument("--out", default=os.path.join(HERE, "..", "release", "Llama-WolfGPT-110M"))
p.add_argument("--tokenizer", default=DEFAULT_TOK)
p.add_argument("--license", default=os.path.join(HERE, "LICENSE"))
p.add_argument("--card", default=os.path.join(HERE, "MODEL_CARD.md"))
p.add_argument("--notice", default=os.path.join(HERE, "NOTICE"))
p.add_argument("--benchmarks", default=os.path.join(HERE, "eval_results.json"))
args = p.parse_args()
out = os.path.abspath(args.out)
os.makedirs(out, exist_ok=True)

# ---- 1. weights: the output layer is tied to the embedding table, so store the table once
orig, ck = load_model(os.path.expanduser(args.ckpt), "cpu")
sd = ck["model"]
assert "head.weight" not in sd or torch.equal(sd["head.weight"], sd["tok_emb.weight"]), "head and embedding differ: tying assumption is wrong"
tensors = {k: v.detach().clone().contiguous().float() for k, v in sd.items() if k != "head.weight"}
save_file(tensors, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})

# ---- 2. config, tokenizer, code, documents
tok = Tokenizer.from_file(os.path.expanduser(args.tokenizer))
eot = tok.token_to_id("<|endoftext|>")
cfg = dict(ck["cfg"])
cfg.update(model_type="wolfgpt", architecture="decoder-only GPT: RoPE, RMSNorm, SwiGLU, tied embeddings, no dropout",
           tie_word_embeddings=True, eos_token_id=eot, parameters=orig.num_params(),
           chat_format="### User:\n{user}\n\n### Assistant:\n{assistant}<|endoftext|>")
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
shutil.copy(os.path.expanduser(args.tokenizer), os.path.join(out, "tokenizer.json"))
shutil.copy(model_module.__file__, os.path.join(out, "model.py"))
shutil.copy(os.path.join(HERE, "release_chat.py"), os.path.join(out, "chat.py"))
shutil.copy(args.card, os.path.join(out, "README.md"))
shutil.copy(args.notice, os.path.join(out, "NOTICE"))
if os.path.exists(args.benchmarks):
    r = json.load(open(args.benchmarks))
    json.dump({k: v for k, v in r.items() if k in ("main_best", "final_best", "wolf2_last")}, open(os.path.join(out, "benchmarks.json"), "w"), indent=1)
has_license = os.path.exists(args.license)
if has_license:
    shutil.copy(args.license, os.path.join(out, "LICENSE"))

# ---- 3. check: reload from the folder with the release code and compare with the original checkpoint
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("release_chat_check", os.path.join(out, "chat.py"))
rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
new, _ = rc.load(out, "cpu")
g = torch.Generator().manual_seed(0)
x = torch.randint(0, cfg["vocab_size"], (2, 64), generator=g)
with torch.no_grad():
    d = (orig(x)[0] - new(x)[0]).abs().max().item()
ids = rc.encode_prefix(tok, "who are you?")
a = rc.sample(orig, ids, eot, "cpu", 25, temperature=0)
b = rc.sample(new, ids, eot, "cpu", 25, temperature=0)
print(f"\nmax logit difference original vs exported: {d}")
print(f"greedy answers identical: {a == b}   ->  {tok.decode(b)!r}")
assert d == 0.0 and a == b, "EXPORT CHECK FAILED"
shutil.rmtree(os.path.join(out, "__pycache__"), ignore_errors=True)
sha = hashlib.sha256(open(os.path.join(out, "model.safetensors"), "rb").read()).hexdigest()
print("\nfolder:", out)
for f in sorted(os.listdir(out)):
    print(f"  {f:<20} {os.path.getsize(os.path.join(out, f))/1e6:9.2f} MB")
print("sha256 model.safetensors:", sha)
print("\nEXPORT CHECK PASSED")
if not has_license:
    print("\n!! LICENSE is missing: save the Llama 3.1 Community License text as", args.license, "and run this again.")
