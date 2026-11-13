"""WolfGPT progress watcher: progress bar + ETA + loss curve, from runs/<tag>.json.

  python watch.py            keeps refreshing every 30 s (run it in a SECOND Ubuntu window)
  python watch.py --once     print once and exit
It also writes progress.html next to this file: open it in your Windows browser (it reloads itself
every 30 s). Data only changes when the trainer logs, i.e. about every 12 minutes.
"""
import html, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = next((a for a in sys.argv[1:] if not a.startswith("--")), "main")
PATH = os.path.join(HERE, "runs", f"{TAG}.json")

def dur(sec):
    h, r = divmod(int(sec), 3600)
    return f"{h} h {r // 60:02d} min" if h else f"{r // 60} min"

def stats():
    d = json.load(open(PATH))
    mtime = os.path.getmtime(PATH)
    a, hist = d["args"], d["history"]
    total, bt = a["steps"], a["batch_tokens"]
    last = hist[-1]
    step = last["step"]
    tps = last.get("tok_s") or d.get("tok_s") or 0
    eta = (total - step) * bt / tps if tps else None
    evals = [h for h in hist if h["step"] > 0 and "val" in h]
    return dict(total=total, bt=bt, step=step, frac=step / total, tps=tps, eta=eta, age=time.time() - mtime,
                interval=(250 * bt / tps) if tps else None, val=last.get("val"), train=last.get("train"),
                best=d.get("best_val"), gnorm=last.get("gnorm"), tflops=last.get("tflops"), evals=evals)

def note(s):
    if s["step"] >= s["total"]:
        return "FINISHED"
    if s["interval"] and s["age"] > 3 * s["interval"] + 60:
        return f"no update for {dur(s['age'])}: paused, crashed, or the PC slept"
    return "running"

def line(s):
    n = 30
    bar = "#" * int(n * s["frac"]) + "-" * (n - int(n * s["frac"]))
    fin = time.strftime("%a %H:%M", time.localtime(time.time() + s["eta"])) if s["eta"] else "?"
    return (f"[{bar}] {100*s['frac']:.1f}%  step {s['step']:,}/{s['total']:,}  val {s['val']:.3f}  "
            f"{s['tps']/1e3:.1f}k tok/s  left {dur(s['eta']) if s['eta'] else '?'} (done ~{fin})  [{note(s)}]")

def chart(evals, w=860, h=260, pad=46):
    pts = {"val": [(e["step"], e["val"]) for e in evals],
           "train": [(e["step"], e["train"]) for e in evals if "train" in e]}
    allp = pts["val"] + pts["train"]
    if len(pts["val"]) < 2:
        return "<p class='muted'>The loss curve appears after the second evaluation (about 25 minutes in).</p>"
    x0, x1 = min(p[0] for p in allp), max(p[0] for p in allp)
    y0, y1 = min(p[1] for p in allp), max(p[1] for p in allp)
    y0, y1 = y0 - 0.05 * (y1 - y0 + 1e-9), y1 + 0.05 * (y1 - y0 + 1e-9)
    X = lambda x: pad + (x - x0) / max(1e-9, x1 - x0) * (w - 2 * pad)
    Y = lambda y: (h - pad) - (y - y0) / max(1e-9, y1 - y0) * (h - 2 * pad)
    def poly(series, color):
        return f"<polyline fill='none' stroke='{color}' stroke-width='2' points='" + " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in series) + "'/>"
    grid = "".join(f"<line x1='{pad}' x2='{w-pad}' y1='{Y(v):.1f}' y2='{Y(v):.1f}' stroke='#2a3446'/>"
                   f"<text x='{pad-8}' y='{Y(v)+4:.1f}' fill='#8b98ad' font-size='11' text-anchor='end'>{v:.2f}</text>"
                   for v in [y0 + (y1 - y0) * i / 4 for i in range(5)])
    xl = f"<text x='{pad}' y='{h-18}' fill='#8b98ad' font-size='11'>step {int(x0):,}</text>" \
         f"<text x='{w-pad}' y='{h-18}' fill='#8b98ad' font-size='11' text-anchor='end'>step {int(x1):,}</text>"
    return (f"<svg viewBox='0 0 {w} {h}' width='100%'>{grid}{xl}{poly(pts['train'], '#5b8def')}{poly(pts['val'], '#34d399')}"
            f"<text x='{w-pad}' y='16' fill='#34d399' font-size='12' text-anchor='end'>validation loss</text>"
            f"<text x='{w-pad-120}' y='16' fill='#5b8def' font-size='12' text-anchor='end'>train loss</text></svg>")

def page(s):
    fin = time.strftime("%A %H:%M", time.localtime(time.time() + s["eta"])) if s["eta"] else "?"
    left = dur(s["eta"]) if s["eta"] else "?"
    tok = s["step"] * s["bt"]
    card = lambda k, v: f"<div class='card'><div class='k'>{k}</div><div class='v'>{v}</div></div>"
    st = note(s)
    cls = "ok" if st in ("running", "FINISHED") else "warn"
    return f"""<!doctype html><html><head><meta charset='utf-8'><meta http-equiv='refresh' content='30'>
<title>WolfGPT {TAG}: {100*s['frac']:.1f}%</title><style>
body{{background:#0e1420;color:#e6ecf5;font-family:system-ui,Segoe UI,sans-serif;max-width:920px;margin:28px auto;padding:0 16px}}
h1{{font-size:22px;margin:0 0 4px}} .muted{{color:#8b98ad;font-size:13px}}
.bar{{background:#1b2536;border-radius:10px;height:34px;overflow:hidden;margin:18px 0 6px}}
.fill{{background:linear-gradient(90deg,#34d399,#5b8def);height:100%;width:{100*s['frac']:.2f}%}}
.big{{font-size:34px;font-weight:700}} .row{{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0}}
.card{{background:#151e2e;border-radius:10px;padding:12px 16px;flex:1;min-width:150px}}
.k{{color:#8b98ad;font-size:12px}} .v{{font-size:20px;font-weight:600;margin-top:2px}}
.ok{{color:#34d399}} .warn{{color:#fbbf24;font-weight:600}}</style></head><body>
<h1>WolfGPT: {html.escape(TAG)} run</h1><div class='muted'>status: <span class='{cls}'>{html.escape(st)}</span> &middot; log updated {dur(s['age'])} ago &middot; page reloads every 30 s</div>
<div class='bar'><div class='fill'></div></div>
<div class='big'>{100*s['frac']:.1f}% <span class='muted'>step {s['step']:,} of {s['total']:,}</span></div>
<div class='row'>{card('Time left', left)}{card('Expected finish', fin)}{card('Tokens trained', f'{tok/1e9:.2f}B of {s["total"]*s["bt"]/1e9:.2f}B')}{card('Speed', f"{s['tps']/1e3:.1f}k tok/s")}</div>
<div class='row'>{card('Validation loss', f"{s['val']:.3f}")}{card('Best validation', f"{s['best']:.3f}" if s['best'] else '-')}{card('Gradient norm', f"{s['gnorm']:.2f}" if s['gnorm'] else '-')}{card('GPU compute', f"{s['tflops']:.1f} TFLOP/s" if s['tflops'] else '-')}</div>
{chart(s['evals'])}
<p class='muted'>Time left assumes the run is not paused. The estimate counts training time only, so expect it to run a few percent over because of evaluations and checkpoints.</p>
</body></html>"""

def once():
    if not os.path.exists(PATH):
        print(f"waiting for {PATH} (appears after the first evaluation)"); return False
    s = stats()
    open(os.path.join(HERE, "progress.html"), "w", encoding="utf-8").write(page(s))
    print(line(s)); return s["step"] < s["total"]

if __name__ == "__main__":
    if "--once" in sys.argv:
        once()
    else:
        print("Watching", PATH, "- Ctrl+C to stop. Open progress.html in your browser.")
        while True:
            try:
                if once() is False and os.path.exists(PATH):
                    break
            except Exception as e:
                print("waiting...", type(e).__name__)
            time.sleep(30)
