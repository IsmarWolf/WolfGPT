"""WolfGPT - real matmul TFLOP/s on this GPU. Close other programs first."""
import json, statistics, torch

assert torch.cuda.is_available()
dev = torch.device("cuda")
name = torch.cuda.get_device_name(0)
torch.backends.cuda.matmul.allow_tf32 = True

def bench(dtype, n, iters=50, warmup=15):
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)
    for _ in range(warmup): a @ b
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); a @ b; e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e) / 1000)
    t = statistics.median(times)
    return 2 * n**3 / t / 1e12

results = {}
for label, dtype in [("bf16", torch.bfloat16), ("fp16", torch.float16), ("fp32(tf32)", torch.float32)]:
    for n in (4096, 8192):
        tf = bench(dtype, n)
        results[f"{label}_{n}"] = round(tf, 1)
        print(f"{label:11s} N={n:5d}  {tf:7.1f} TFLOP/s")

print(f"\nGPU: {name}")
print("Use the best bf16 number as your realistic compute ceiling; real training reaches ~30-50% of it (MFU).")
with open("benchmark_result.json", "w") as f:
    json.dump({"gpu": name, "torch": torch.__version__, "cuda": torch.version.cuda, "tflops": results}, f, indent=2)
