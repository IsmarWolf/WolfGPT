"""Talk to WolfGPT (base model: it CONTINUES text, it does not answer questions yet).

  python generate.py "The capital of France is"
  python generate.py --prompt "Photosynthesis is" --temperature 0.7 --top_k 50 --max_new 200 --n 3
  python generate.py            # no prompt: interactive, type a start of text, empty line to quit
"""
import argparse, os, sys
import torch
from tokenizers import Tokenizer

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from model import GPT, Config

p = argparse.ArgumentParser()
p.add_argument("prompt", nargs="?", default=None)
p.add_argument("--prompt", dest="prompt_flag", default=None)
p.add_argument("--ckpt", default=os.path.expanduser("~/wolfgpt-data/ckpt/main_best.pt"))
p.add_argument("--tokenizer", default=os.path.expanduser("~/wolfgpt-data/data_fw/tokenizer.json"))
p.add_argument("--max_new", type=int, default=150)
p.add_argument("--temperature", type=float, default=0.8)   # lower = safer/repetitive, higher = more random
p.add_argument("--top_k", type=int, default=50)
p.add_argument("--n", type=int, default=1)                 # samples per prompt
p.add_argument("--seed", type=int, default=None)
p.add_argument("--cpu", action="store_true")
args = p.parse_args()

dev = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
tok = Tokenizer.from_file(args.tokenizer)
eot = tok.token_to_id("<|endoftext|>")
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
model = GPT(Config(**ck["cfg"]))
model.load_state_dict(ck["model"])
model.to(dev).eval()
T = model.c.block_size
if args.seed is not None:
    torch.manual_seed(args.seed)
print(f"loaded {args.ckpt} (step {ck.get('step', '?')}, {model.num_params()/1e6:.1f}M params) on {dev}\n")


@torch.no_grad()
def generate(text):
    ids = tok.encode(text).ids
    x = torch.tensor([ids], device=dev)
    for _ in range(args.max_new):
        with torch.autocast(dev, dtype=torch.bfloat16):
            logits = model(x[:, -T:])[0][:, -1, :].float()   # no KV cache: simple, fast enough for short text
        logits = logits / max(args.temperature, 1e-6)
        if args.top_k:
            kth = torch.topk(logits, args.top_k).values[:, -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        if nxt.item() == eot:
            break
        x = torch.cat([x, nxt], dim=1)
    return tok.decode(x[0].tolist())


prompt = args.prompt_flag or args.prompt
if prompt is not None:
    for i in range(args.n):
        print(f"--- sample {i+1} ---\n{generate(prompt)}\n")
else:
    while True:
        text = input("start of text> ")
        if not text.strip():
            break
        for i in range(args.n):
            print(f"\n{generate(text)}\n")
