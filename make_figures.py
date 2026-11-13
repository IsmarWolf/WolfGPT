#!/usr/bin/env python3
"""Draw the WolfGPT infographics (hand-drawn look, dark background), in English and Portuguese.

Every number comes from a file in this repo (runs/*.json, eval_results.json, benchmark_result.json) or from a
measurement written in the docs (see docs/JOURNEY.md). Nothing is invented. Usage:

    python make_figures.py --data . --out docs/img

Needs matplotlib. Font: Patrick Hand (SIL OFL), pass --font path/to/PatrickHand-Regular.ttf (falls back to default).
"""
import argparse, json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, patheffects as pe
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=".")
ap.add_argument("--out", default="docs/img")
ap.add_argument("--font", default="")
args = ap.parse_args()
if args.font and os.path.exists(args.font):
    fm.fontManager.addfont(args.font)
    FONT = fm.FontProperties(fname=args.font).get_name()
else:
    FONT = "DejaVu Sans"

BG, FG, DIM = "#0d1117", "#f2f2f2", "#9aa4b2"
RED, TEAL, YEL, BLUE, GRN, PUR, ORG = "#ff6b6b", "#4ecdc4", "#ffd166", "#6c9eff", "#7bd88f", "#b794f4", "#ff9f43"
L = "en"


def t(en, pt):
    return en if L == "en" else pt


def num(x, nd=0):
    s = f"{x:,.{nd}f}"
    if L == "pt":
        s = s.replace(",", "§").replace(".", ",").replace("§", ".")
    return s


def J(p):
    return json.load(open(os.path.join(args.data, p)))


# ---------------------------------------------------------------- data (all from files)
main = J("phase3/runs/main.json")
pil = {k: J(f"phase3/runs/{k}.json") for k in ["muon_0.02", "muon_0.05", "adamw_6e-4", "adamw_2e-3"]}
sft = J("phase5/runs/final.json")
sweep = {k: J(f"phase5/runs/sweep_lr{k}.json") for k in ["5e-5", "1e-4", "3e-4"]}
wolf, wolf2 = J("phase5/runs/wolf.json"), J("phase5/runs/wolf2.json")
ev = J("phase5/eval_results.json")
gpu = J("benchmark_result.json")


def setup():
    plt.rcParams.update({"font.family": FONT, "axes.unicode_minus": False})


def style(ax):
    ax.set_facecolor(BG)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(FG)
        ax.spines[s].set_linewidth(2.2)
    ax.tick_params(colors=FG, labelsize=14, width=1.5)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)


def blank(ax):
    ax.set_facecolor(BG)
    ax.axis("off")


def frame(title, sub=None, size=(10, 6), **kw):
    fig = plt.figure(figsize=size, facecolor=BG)
    fig.text(0.04, 0.955, title, fontsize=27, color=FG, va="top", weight="normal")
    if sub:
        fig.text(0.04, 0.885, sub, fontsize=15, color=DIM, va="top")
    fig.text(0.975, 0.018, "WolfGPT", fontsize=13, color=DIM, ha="right")
    return fig


def save(fig, name):
    d = os.path.join(args.out, "en" if L == "en" else "pt-BR")
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, name + ".png"), dpi=130, facecolor=BG)
    plt.close(fig)


def sk():
    """enter the hand-drawn mode; return the context manager"""
    c = plt.xkcd(scale=0.9, length=90, randomness=2)
    return c


def after_xkcd():
    plt.rcParams["font.family"] = FONT
    plt.rcParams["path.effects"] = [pe.withStroke(linewidth=3.2, foreground=BG)]
    plt.rcParams["axes.unicode_minus"] = False


def bar_labels(ax, bars, fmt, dy=0.0, color=FG, size=14):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, fmt(b.get_height()), ha="center", va="bottom", color=color, fontsize=size)


def check(ax, x, y, s=1.0, c=GRN):
    ax.plot([x - 1.1 * s, x - 0.35 * s, x + 1.2 * s], [y + 0.1 * s, y - 0.7 * s, y + 1.2 * s], color=c, lw=4, solid_capstyle="round")


def cross(ax, x, y, s=1.0, c=RED):
    ax.plot([x - s, x + s], [y - s, y + s], color=c, lw=4, solid_capstyle="round")
    ax.plot([x - s, x + s], [y + s, y - s], color=c, lw=4, solid_capstyle="round")


def box(ax, x, y, w, h, color, lw=2.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6", fc="none", ec=color, lw=lw))


# ---------------------------------------------------------------- figures
def f01_journey():
    fig = frame(t("The whole journey", "A jornada completa"), t("seven phases, one RTX 5060 Ti, every number measured", "sete fases, uma RTX 5060 Ti, todos os números medidos"))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.80]); blank(ax); ax.set_xlim(0, 100); ax.set_ylim(0, 62)
    P = [
        ("0", t("Environment", "Ambiente"), t("WSL2 + PyTorch", "WSL2 + PyTorch"), t("bf16 up to 50 TFLOP/s", "bf16 até 50 TFLOP/s"), TEAL),
        ("1", t("Tiny GPT", "GPT minúsculo"), t("10.75M params, chars", "10,75M parâm., chars"), t("val loss 1.462", "val loss 1,462"), BLUE),
        ("2", t("Data + tokenizer", "Dados + tokenizer"), t("3.0B tokens", "3,0 bi de tokens"), t("vocab 32,768", "vocab 32.768"), YEL),
        ("3", t("Pilot", "Piloto"), t("Muon 4.418", "Muon 4,418"), t("AdamW 5.280", "AdamW 5,280"), ORG),
        ("4", t("Main run", "Treino principal"), t("110.1M params, 18.1 h", "110,1M parâm., 18,1 h"), t("val loss 3.129", "val loss 3,129"), RED),
        ("5", t("Chat (SFT)", "Chat (SFT)"), t("447k chats, 4.6 h", "447 mil chats, 4,6 h"), t("val 2.395 down to 1.484", "val 2,395 caindo a 1,484"), PUR),
        ("6", t("Release", "Publicação"), t("5 benchmarks", "5 benchmarks"), t("Llama licence, model card", "licença Llama, model card"), GRN),
    ]
    pos = [(2, 33), (27, 33), (52, 33), (77, 33), (6, 4), (31.5, 4), (57, 4)]
    for (i, ti, l1, l2, c), (x, y) in zip(P, pos):
        w, h = 21, 22
        box(ax, x, y, w, h, c)
        ax.text(x + 1.2, y + h - 2.3, t("Phase ", "Fase ") + i, color=c, fontsize=17, va="top")
        ax.text(x + 1.2, y + h - 7.4, ti, color=FG, fontsize=19, va="top")
        ax.text(x + 1.2, y + 7.2, l1, color=FG, fontsize=14, va="center")
        ax.text(x + 1.2, y + 2.6, l2, color=DIM, fontsize=13.5, va="center")
    for k in range(3):
        x = pos[k][0] + 21.8
        ax.annotate("", xy=(x + 3.0, 44), xytext=(x + 0.2, 44), arrowprops=dict(arrowstyle="-|>", color=FG, lw=2.4))
    ax.plot([87.5, 87.5, 16.5], [32.2, 29.6, 29.6], color=FG, lw=2.4)
    ax.annotate("", xy=(16.5, 26.6), xytext=(16.5, 29.6), arrowprops=dict(arrowstyle="-|>", color=FG, lw=2.4))
    for k in range(4, 6):
        x = pos[k][0] + 21.8
        ax.annotate("", xy=(x + 2.6, 16), xytext=(x + 0.2, 16), arrowprops=dict(arrowstyle="-|>", color=FG, lw=2.4))
    ax.text(99, 15, t("a model that\nchats and knows\nits own name", "um modelo que\nconversa e sabe\no próprio nome"), color=GRN, fontsize=14, ha="right", va="center")
    save(fig, "01_journey")


def f02_hardware():
    fig = frame(t("The machine", "A máquina"), t("what one gaming PC can do (matrix-multiply speed, measured)", "o que um PC gamer consegue (velocidade de multiplicação de matrizes, medida)"))
    ax = fig.add_axes([0.07, 0.19, 0.50, 0.57]); style(ax)
    names = ["bf16", "fp16", "tf32"]
    a = [gpu["tflops"]["bf16_4096"], gpu["tflops"]["fp16_4096"], gpu["tflops"]["fp32(tf32)_4096"]]
    b = [gpu["tflops"]["bf16_8192"], gpu["tflops"]["fp16_8192"], gpu["tflops"]["fp32(tf32)_8192"]]
    x = np.arange(3)
    b1 = ax.bar(x - 0.2, a, 0.38, color=TEAL, ec=FG, lw=2, label="N = 4096")
    b2 = ax.bar(x + 0.2, b, 0.38, color=BLUE, ec=FG, lw=2, label="N = 8192")
    bar_labels(ax, b1, lambda v: num(v, 1), size=12); bar_labels(ax, b2, lambda v: num(v, 1), size=12)
    tr = main["tflops"]
    ax.axhline(tr, color=RED, ls="--", lw=2.6)
    ax.text(2.55, tr + 1.2, t(f"real training: {num(tr,1)}", f"treino real: {num(tr,1)}"), color=RED, fontsize=14, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 62); ax.set_ylabel("TFLOP/s")
    ax.legend(frameon=False, labelcolor=FG, fontsize=13, loc="upper right", bbox_to_anchor=(1.0, 1.02))
    ax2 = fig.add_axes([0.62, 0.19, 0.36, 0.57]); blank(ax2); ax2.set_xlim(0, 10); ax2.set_ylim(-2, 10)
    rows = [("GPU", "RTX 5060 Ti, 16 GB"), ("CPU", "Ryzen 5 5600X"), ("RAM", "32 GB"), (t("Disk", "Disco"), "NVMe + SATA, ~1 TB"), ("OS", "Windows 11 + WSL2"), ("PyTorch", gpu["torch"])]
    for i, (k, v) in enumerate(rows):
        y = 9.0 - i * 1.6
        ax2.text(0, y, k, color=YEL, fontsize=16, va="center"); ax2.text(2.1, y, v, color=FG, fontsize=15, va="center")
    ax2.plot([0, 10], [9.9, 9.9], color=FG, lw=2)
    ax2.text(0, -1.2, t("training used about two thirds of the matmul peak", "o treino usou cerca de dois terços do pico de matmul"), color=DIM, fontsize=13, va="center")
    save(fig, "02_hardware")


def f03_model():
    emb = 32768 * 768
    att = 12 * (3 * 768 * 768 + 768 * 768)
    mlp = 12 * (768 * 4096 + 2048 * 768)
    nrm = 12 * 2 * 768 + 768
    tot = emb + att + mlp + nrm
    assert tot == main["params"], (tot, main["params"])
    fig = frame(t("What the 110 million numbers are", "O que são os 110 milhões de números"), t(f"{num(tot)} parameters, counted from the code", f"{num(tot)} parâmetros, contados a partir do código"))
    ax = fig.add_axes([0.04, 0.50, 0.92, 0.28]); blank(ax); ax.set_xlim(0, 100); ax.set_ylim(0, 10)
    parts = [(emb, t("word table (shared)", "tabela de palavras (compartilhada)"), YEL), (att, t("attention", "atenção"), BLUE), (mlp, t("feed-forward (SwiGLU)", "feed-forward (SwiGLU)"), RED)]
    x0 = 0
    for v, nme, c in parts:
        w = 100 * v / tot
        ax.add_patch(FancyBboxPatch((x0 + 0.3, 2.2), w - 0.6, 5, boxstyle="round,pad=0.1,rounding_size=0.6", fc=c, ec=FG, lw=2.4, alpha=0.9))
        ax.text(x0 + w / 2, 4.7, f"{num(100*v/tot,1)}%", color=BG, fontsize=22, ha="center", va="center", path_effects=[pe.Normal()])
        ax.text(x0 + w / 2, 0.8, f"{nme}\n{num(v/1e6,1)}M", color=c, fontsize=14, ha="center", va="top")
        x0 += w
    ax.text(100, 9.3, t(f"+ norms: {num(nrm)} numbers (0.02%)", f"+ normalizações: {num(nrm)} números (0,02%)"), color=DIM, fontsize=13, ha="right")
    ax2 = fig.add_axes([0.04, 0.04, 0.92, 0.36]); blank(ax2); ax2.set_xlim(0, 100); ax2.set_ylim(0, 10)
    specs = [("12", t("layers", "camadas")), ("12", t("attention heads", "cabeças de atenção")), ("768", t("width", "largura")), ("1,024" if L == "en" else "1.024", t("context tokens", "tokens de contexto")), ("32,768" if L == "en" else "32.768", t("vocabulary", "vocabulário"))]
    for i, (v, nme) in enumerate(specs):
        cx = 10 + i * 20
        box(ax2, cx - 8, 1.5, 16, 7, [TEAL, GRN, YEL, ORG, PUR][i])
        ax2.text(cx, 6.1, v, color=FG, fontsize=26, ha="center", va="center")
        ax2.text(cx, 3.1, nme, color=DIM, fontsize=13.5, ha="center", va="center")
    save(fig, "03_model")


def f04_tiny():
    fig = frame(t("Phase 1: overfitting in minutes", "Fase 1: overfitting em minutos"), t("tiny character-level GPT, 10.75M params, tiny Shakespeare", "GPT minúsculo de caracteres, 10,75M parâm., Shakespeare em miniatura"))
    ax = fig.add_axes([0.07, 0.19, 0.50, 0.57]); style(ax)
    vals = [0.625, 1.469, 1.679]; cols = [TEAL, YEL, RED]
    labs = [t("train loss\n(end)", "loss treino\n(fim)"), t("val loss\nbest", "val loss\nmelhor"), t("val loss\n(end)", "val loss\n(fim)")]
    bs = ax.bar(range(3), vals, 0.6, color=cols, ec=FG, lw=2)
    bar_labels(ax, bs, lambda v: num(v, 3))
    ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=13); ax.set_ylim(0, 2.1)
    ax.set_title(t("no early stopping", "sem parada antecipada"), color=FG, fontsize=16)
    ax2 = fig.add_axes([0.66, 0.19, 0.31, 0.57]); style(ax2)
    b2 = ax2.bar([0, 1], [1.462, 1.379], 0.55, color=[BLUE, GRN], ec=FG, lw=2)
    bar_labels(ax2, b2, lambda v: num(v, 3))
    ax2.set_xticks([0, 1]); ax2.set_xticklabels([t("English", "Inglês"), t("Portuguese", "Português")], fontsize=13); ax2.set_ylim(0, 2.1)
    ax2.set_title(t("best val loss", "melhor val loss"), color=FG, fontsize=16)
    fig.text(0.04, 0.02, t("the best-checkpoint trick: keep the best validation weights, stop after 4 evals without improvement", "o truque: guardar os melhores pesos de validação e parar após 4 avaliações sem melhora"), color=DIM, fontsize=13)
    save(fig, "04_tiny_gpt")


def f05_tokenizer():
    fig = frame(t("Phase 2: text into numbers", "Fase 2: texto em números"), t("own byte-level BPE tokenizer, trained in 57 seconds", "tokenizer BPE próprio, treinado em 57 segundos"))
    ax = fig.add_axes([0.07, 0.19, 0.38, 0.57]); style(ax)
    b = ax.bar([0, 1], [1.00, 4.61], 0.55, color=[BLUE, RED], ec=FG, lw=2)
    bar_labels(ax, b, lambda v: num(v, 2))
    ax.set_xticks([0, 1]); ax.set_xticklabels([t("Phase 1\n(characters)", "Fase 1\n(caracteres)"), t("WolfGPT\ntokenizer", "tokenizer\nWolfGPT")], fontsize=13)
    ax.set_ylim(0, 5.6); ax.set_title(t("characters per token", "caracteres por token"), color=FG, fontsize=16)
    ax2 = fig.add_axes([0.58, 0.19, 0.38, 0.57]); style(ax2)
    need = 20 * main["params"] / 1e9
    used = 3.0
    b2 = ax2.bar([0, 1], [need, used], 0.55, color=[YEL, TEAL], ec=FG, lw=2)
    bar_labels(ax2, b2, lambda v: num(v, 2))
    ax2.set_xticks([0, 1]); ax2.set_xticklabels([t("rule of thumb\n(20 tokens/param)", "regra prática\n(20 tokens/parâm.)"), t("tokens\nactually used", "tokens\nrealmente usados")], fontsize=13)
    ax2.set_ylim(0, 3.7); ax2.set_title(t("billions of tokens", "bilhões de tokens"), color=FG, fontsize=16)
    fig.text(0.04, 0.02, t("FineWeb-Edu sample-10BT: 3,002,016,685 training tokens, 10,032,689 validation tokens, split by document", "FineWeb-Edu sample-10BT: 3.002.016.685 tokens de treino, 10.032.689 de validação, separados por documento"), color=DIM, fontsize=13)
    save(fig, "05_tokenizer_data")


def f06_pilot():
    fig = frame(t("Phase 3: Muon vs AdamW", "Fase 3: Muon vs AdamW"), t("47.9M-param pilot, 400 steps, 52M tokens, same data order", "piloto de 47,9M parâm., 400 passos, 52M tokens, mesma ordem de dados"))
    ax = fig.add_axes([0.07, 0.19, 0.50, 0.57]); style(ax)
    meta = {"muon_0.02": ("Muon 0.02", TEAL), "muon_0.05": ("Muon 0.05", BLUE), "adamw_6e-4": ("AdamW 6e-4", RED), "adamw_2e-3": ("AdamW 2e-3", ORG)}
    for k, (nm, c) in meta.items():
        h = [(e["step"], e["val"]) for e in pil[k]["history"] if e["step"] >= 50]
        ax.plot(*zip(*h), color=c, lw=3, label=nm.replace(".", ",") if L == "pt" else nm)
    ax.set_xlabel(t("step", "passo")); ax.set_ylabel(t("validation loss", "loss de validação")); ax.legend(frameon=False, labelcolor=FG, fontsize=13)
    ax2 = fig.add_axes([0.66, 0.19, 0.31, 0.57]); style(ax2)
    order = ["muon_0.02", "muon_0.05", "adamw_6e-4", "adamw_2e-3"]
    fv = [pil[k]["final_val"] for k in order]
    b = ax2.bar(range(4), fv, 0.7, color=[meta[k][1] for k in order], ec=FG, lw=2)
    bar_labels(ax2, b, lambda v: num(v, 2), size=13)
    ax2.set_xticks(range(4)); ax2.set_xticklabels(["M\n.02", "M\n.05", "A\n6e-4", "A\n2e-3"], fontsize=12); ax2.set_ylim(0, 6.3)
    ax2.set_title(t("final loss", "loss final"), color=FG, fontsize=16)
    fig.text(0.04, 0.02, t("Muon: 0.86 lower loss (perplexity 2.4x lower) for about 4% less speed. Only 52M tokens: the gap shrinks in long runs.", "Muon: loss 0,86 menor (perplexidade 2,4x menor), ~4% mais lento. Só 52M tokens: a diferença encolhe em treinos longos."), color=DIM, fontsize=12.5)
    save(fig, "06_pilot_muon_adamw")


def f07_main():
    fig = frame(t("Phase 4: the main run", "Fase 4: o treino principal"), t("110.1M params, 3.0B tokens, one pass over the data, 18.1 hours", "110,1M parâm., 3,0 bi de tokens, uma passada pelos dados, 18,1 horas"))
    ax = fig.add_axes([0.08, 0.19, 0.88, 0.57]); style(ax)
    h = [e for e in main["history"] if e["step"] >= 250]
    xs = [e["step"] * 131072 / 1e9 for e in h]
    ax.plot(xs, [e["val"] for e in h], color=RED, lw=3.6, label=t("validation", "validação"))
    ax.plot(xs, [e["train"] for e in h], color=TEAL, lw=2.4, ls="--", label=t("training", "treino"))
    byst = {e["step"]: e["val"] for e in main["history"]}
    for st, dx, dy in [(500, 0.28, 0.35), (2500, 0.40, 0.34), (10000, 0.20, 0.38), (22888, -0.52, 0.42)]:
        if st not in byst:
            st = max(s for s in byst if s <= st)
        x, y = st * 131072 / 1e9, byst[st]
        ax.annotate(f"{num(y,3 if st==22888 else 2)}", xy=(x, y), xytext=(x + dx, y + dy), color=FG, fontsize=15, arrowprops=dict(arrowstyle="->", color=FG, lw=2))
    ax.set_xlim(0, 3.1); ax.set_ylim(3.0, 5.2)
    ax.set_xlabel(t("tokens seen (billions)", "tokens vistos (bilhões)")); ax.set_ylabel(t("loss", "loss"))
    ax.legend(frameon=False, labelcolor=FG, fontsize=14, loc="upper right")
    ax.text(0.45, 4.95, t("step 0: 10.57\n(off the chart)", "passo 0: 10,57\n(fora do gráfico)"), color=DIM, fontsize=14, va="top")
    fig.text(0.04, 0.02, t("perplexity 22.8 at the end; validation kept falling to the end: no overfitting", "perplexidade 22,8 no fim; a validação caiu até o fim: sem overfitting"), color=DIM, fontsize=13.5)
    save(fig, "07_main_run_loss")


def f08_hours():
    fig = frame(t("Where the GPU hours went", "Para onde foram as horas de GPU"), t("about 26 hours of GPU time in total", "cerca de 26 horas de GPU no total"))
    ax = fig.add_axes([0.30, 0.19, 0.64, 0.57]); style(ax)
    items = [(t("Tokenize the data", "Tokenizar os dados"), 0.75, YEL), (t("Pretraining (110M)", "Pré-treino (110M)"), 18.1, RED), (t("SFT lr sweep (3 runs)", "SFT: teste de lr (3 rodadas)"), 3 * 0.8, ORG), (t("SFT final run", "SFT: rodada final"), 4.6, PUR), (t("Identity stage", "Fase de identidade"), 0.15, GRN)]
    items = items[::-1]
    b = ax.barh(range(len(items)), [i[1] for i in items], 0.62, color=[i[2] for i in items], ec=FG, lw=2)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([i[0] for i in items], fontsize=15)
    for r, it in zip(b, items):
        ax.text(r.get_width() + 0.25, r.get_y() + r.get_height() / 2, (num(it[1], 1) if it[1] >= 1 else "~" + num(it[1] * 60) + " min") + (" h" if it[1] >= 1 else ""), color=FG, fontsize=15, va="center")
    ax.set_xlim(0, 21); ax.set_xlabel(t("hours", "horas"))
    fig.text(0.04, 0.02, t("sweep: about 48 min per run; identity stage (wolf2): about 9 min; the 110M main run: 46.0k tokens/s", "teste de lr: cerca de 48 min por rodada; identidade (wolf2): cerca de 9 min; treino principal: 46,0 mil tokens/s"), color=DIM, fontsize=13)
    save(fig, "08_gpu_hours")


def f09_data():
    fig = frame(t("Phase 5: the chat data", "Fase 5: os dados de chat"), t("from 477,612 conversations to what the model trained on", "de 477.612 conversas até o que o modelo treinou"))
    ax = fig.add_axes([0.04, 0.12, 0.58, 0.68]); blank(ax); ax.set_xlim(0, 100); ax.set_ylim(0, 10)
    steps = [(477612, t("collected", "coletadas"), BLUE), (457399, t("after removing repeats", "sem repetidas"), TEAL), (447335, t("after dropping long ones", "sem as muito longas"), YEL), (446353, t("train (with 3x greetings)", "treino (saudações 3x)"), RED)]
    for i, (v, nme, c) in enumerate(steps):
        y = 8 - i * 2.35
        w = 62 * v / 477612
        ax.add_patch(FancyBboxPatch((1, y), w, 1.6, boxstyle="round,pad=0.05,rounding_size=0.4", fc=c, ec=FG, lw=2.2, alpha=0.9))
        ax.text(2.2, y + 0.8, num(v), color=BG, fontsize=18, va="center", path_effects=[pe.Normal()])
        ax.text(1 + w + 1.5, y + 0.8, nme, color=FG, fontsize=14, va="center")
    ax.text(1, 0.15, t("+ 1,000 conversations held out for validation", "+ 1.000 conversas separadas para validação"), color=DIM, fontsize=13.5)
    ax2 = fig.add_axes([0.68, 0.19, 0.28, 0.57]); style(ax2)
    ll, tot = 270838, 477612
    b = ax2.bar([0], [ll], 0.55, color=PUR, ec=FG, lw=2); ax2.bar([0], [tot - ll], 0.55, bottom=[ll], color=BLUE, ec=FG, lw=2)
    ax2.text(0, ll / 2, f"{num(100*ll/tot)}%\n" + t("written by\nLlama 3.1 405B", "escritas pelo\nLlama 3.1 405B"), color=BG, fontsize=15, ha="center", va="center", path_effects=[pe.Normal()])
    ax2.text(0, ll + (tot - ll) / 2, f"{num(100*(tot-ll)/tot)}%\n" + t("other\nsources", "outras\nfontes"), color=BG, fontsize=15, ha="center", va="center", path_effects=[pe.Normal()])
    ax2.set_xticks([]); ax2.set_ylim(0, 520000)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(v/1000)}k"))
    ax2.set_title(t("who wrote it", "quem escreveu"), color=FG, fontsize=16)
    fig.text(0.04, 0.02, t("this is why the model carries the Llama name and a 'Built with Llama' notice", "por isso o modelo leva o nome Llama e o aviso 'Built with Llama'"), color=DIM, fontsize=13.5)
    save(fig, "09_sft_data")


def f10_sweep():
    fig = frame(t("Phase 5: picking the learning rate", "Fase 5: escolhendo a learning rate"), t("1 epoch on 150k conversations each, about 48 min per run", "1 época em 150 mil conversas cada, cerca de 48 min por rodada"))
    ax = fig.add_axes([0.08, 0.19, 0.55, 0.57]); style(ax)
    ks = ["5e-5", "1e-4", "3e-4"]; vals = [sweep[k]["best_val"] for k in ks]
    b = ax.bar(range(3), vals, 0.6, color=[BLUE, TEAL, RED], ec=FG, lw=2)
    bar_labels(ax, b, lambda v: num(v, 3))
    base = sft["base_val"]
    ax.axhline(base, color=YEL, ls="--", lw=2.4); ax.text(2.45, base + 0.04, t(f"before SFT: {num(base,3)}", f"antes do SFT: {num(base,3)}"), color=YEL, ha="right", fontsize=14)
    ax.set_xticks(range(3)); ax.set_xticklabels(["5e-5", "1e-4", "3e-4"], fontsize=15); ax.set_ylim(0, 2.8); ax.set_ylabel(t("best validation loss", "melhor loss de validação"))
    ax2 = fig.add_axes([0.68, 0.19, 0.29, 0.57]); blank(ax2); ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
    ax2.text(0, 8.2, t("winner: 3e-4", "vencedora: 3e-4"), color=RED, fontsize=20)
    ax2.text(0, 6.2, t("it is the highest\nvalue tested, so a\nhigher one (5e-4)\nmight do better.\nNot tried.", "é o maior valor\ntestado, então um\nmaior (5e-4) talvez\nfosse melhor.\nNão testado."), color=FG, fontsize=15, va="top")
    save(fig, "10_sft_lr_sweep")


def f11_sft():
    fig = frame(t("Phase 5: the final SFT run", "Fase 5: a rodada final de SFT"), t("lr 3e-4, 2 epochs, 27,896 steps, 4.6 hours", "lr 3e-4, 2 épocas, 27.896 passos, 4,6 horas"))
    ax = fig.add_axes([0.08, 0.19, 0.50, 0.57]); style(ax)
    h = sft["history"]
    ax.plot([e["step"] for e in h], [e["val"] for e in h], color=RED, lw=3.4, label=t("validation", "validação"))
    ht = [e for e in h if "train" in e]
    ax.plot([e["step"] for e in ht], [e["train"] for e in ht], color=TEAL, lw=1.8, ls="--", label=t("training", "treino"), alpha=0.9)
    ax.axvline(27896 / 2, color=YEL, ls=":", lw=2.4)
    ax.text(27896 / 2 + 500, 2.28, t("epoch 2\nstarts", "época 2\ncomeça"), color=YEL, fontsize=14, va="top")
    ax.set_ylim(1.3, 2.5); ax.set_xlim(0, 28500)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(v/1000)}k"))
    ax.set_xlabel(t("step", "passo")); ax.set_ylabel("loss"); ax.legend(frameon=False, labelcolor=FG, fontsize=13, loc="upper right")
    ax.text(900, 1.42, t(f"{num(sft['base_val'],3)} to {num(sft['best_val'],3)}", f"{num(sft['base_val'],3)} para {num(sft['best_val'],3)}"), color=FG, fontsize=18)
    ax2 = fig.add_axes([0.80, 0.19, 0.18, 0.57]); style(ax2)
    bs = wolf2["history"][0]["by_source"]
    names = {"self-oss-instruct": t("code", "código"), "smollm-rewrite-30k": t("rewrite", "reescrita"), "smol-contraints": t("constraints", "restrições"), "smol-magpie-ultra-short": "magpie", "openhermes-50k": "openhermes", "smol-summarize-5k": t("summary", "resumo"), "explore-instruct-rewrite": t("rewrite 2", "reescrita 2"), "dolly": "dolly (human)" if L == "en" else "dolly (humano)"}
    srt = sorted([(v, names[k]) for k, v in bs.items() if k in names])
    ax2.barh(range(len(srt)), [v for v, _ in srt], 0.65, color=PUR, ec=FG, lw=1.8)
    ax2.set_yticks(range(len(srt))); ax2.set_yticklabels([n for _, n in srt], fontsize=11)
    for i, (v, _) in enumerate(srt):
        ax2.text(v + 0.05, i, num(v, 2), color=FG, fontsize=11, va="center")
    ax2.set_xlim(0, 3.7); ax2.set_xticks([]); ax2.set_title(t("loss by source", "loss por fonte"), color=FG, fontsize=14)
    save(fig, "11_sft_final_run")


def f12_identity():
    fig = frame(t("Teaching it its name", "Ensinando o nome a ele"), t("300 hand-written conversations, built by 'Explorer'", "300 conversas escritas à mão, criado por 'Explorer'"))
    ax = fig.add_axes([0.07, 0.19, 0.28, 0.57]); style(ax)
    b = ax.bar([0, 1], [3.615, 2.294], 0.55, color=[RED, GRN], ec=FG, lw=2)
    bar_labels(ax, b, lambda v: num(v, 3))
    ax.set_xticks([0, 1]); ax.set_xticklabels([t("before", "antes"), t("after\nrun 1", "depois\nrodada 1")], fontsize=13); ax.set_ylim(0, 4.4)
    ax.set_title(t("identity questions", "perguntas de identidade"), color=FG, fontsize=15)
    ax2 = fig.add_axes([0.46, 0.19, 0.51, 0.57]); style(ax2)
    for d, c, nm in [(wolf, BLUE, t("run 1: lr 3e-5", "rodada 1: lr 3e-5")), (wolf2, RED, t("run 2: lr 1e-4", "rodada 2: lr 1e-4"))]:
        ax2.plot([e["step"] for e in d["history"]], [e["val"] for e in d["history"]], color=c, lw=3.2, marker="o", ms=5, label=nm + t(" (val)", " (val)"))
    h2 = [e for e in wolf2["history"] if "train" in e]
    ax2.plot([e["step"] for e in h2], [e["train"] for e in h2], color=TEAL, lw=2.4, ls="--", label=t("run 2 training loss", "rodada 2: loss de treino"))
    ax2.set_ylim(0.8, 1.7); ax2.set_xlabel(t("step", "passo")); ax2.set_ylabel(t("ordinary-chat loss", "loss em chats comuns"))
    ax2.legend(frameon=False, labelcolor=FG, fontsize=12, loc="lower left")
    ax2.set_title(t("did it forget the rest?", "esqueceu o resto?"), color=FG, fontsize=15)
    fig.text(0.04, 0.02, t("run 2 learned the name but memorised the repeated chats: training loss fell to ~1.0 while validation rose a little (1.488 to 1.516)", "a rodada 2 aprendeu o nome mas decorou os chats repetidos: loss de treino caiu a ~1,0 e a validação subiu um pouco (1,488 para 1,516)"), color=DIM, fontsize=12.5)
    save(fig, "12_identity_stage")


def f13_bench():
    fig = frame(t("Benchmarks: honest numbers", "Benchmarks: números honestos"), t("zero-shot, full test sets, accuracy %, dotted line = random guessing", "zero-shot, conjuntos completos, acurácia %, linha pontilhada = chute aleatório"))
    ax = fig.add_axes([0.07, 0.19, 0.90, 0.57]); style(ax)
    tasks = [("hellaswag", "HellaSwag", 25, "acc_norm"), ("arc_easy", "ARC-Easy", 25, "acc_norm"), ("arc_challenge", "ARC-Chall.", 25, "acc_norm"), ("piqa", "PIQA", 50, "acc_norm"), ("winogrande", "Winogrande", 50, "acc")]
    models = [("main_best", t("base model", "modelo base"), YEL), ("final_best", t("chat (SFT)", "chat (SFT)"), TEAL), ("wolf2_last", t("chat + identity", "chat + identidade"), RED)]
    w = 0.26
    for j, (mk, mn, c) in enumerate(models):
        vals = [ev[mk][tk][m] for tk, _, _, m in tasks]
        b = ax.bar(np.arange(5) + (j - 1) * w, vals, w * 0.92, color=c, ec=FG, lw=1.8, label=mn)
        for r, v in zip(b, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.8, num(v, 1), ha="center", color=FG, fontsize=11)
    for i, (_, _, rnd, _) in enumerate(tasks):
        ax.plot([i - 0.42, i + 0.42], [rnd, rnd], color=FG, ls=":", lw=2.6)
    ax.set_xticks(range(5)); ax.set_xticklabels([x[1] for x in tasks], fontsize=15); ax.set_ylim(0, 75); ax.set_ylabel("%")
    ax.legend(frameon=False, labelcolor=FG, fontsize=14, loc="upper right", ncol=3)
    fig.text(0.04, 0.02, t("standard error ~0.5 (HellaSwag) to ~1.4 (Winogrande) points. GPT-2 124M scores 31.1 on HellaSwag (different eval code).", "erro padrão de ~0,5 (HellaSwag) a ~1,4 (Winogrande) pontos. O GPT-2 124M faz 31,1 no HellaSwag (outro código de avaliação)."), color=DIM, fontsize=12.5)
    save(fig, "13_benchmarks")


def f14_flow():
    fig = frame(t("Stopping the repetition loop", "Parando o loop de repetição"), t("no retraining: the chat code flags a repeated reply and tries again", "sem retreinar: o código do chat marca a resposta repetida e tenta de novo"))
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.78]); blank(ax); ax.set_xlim(0, 100); ax.set_ylim(0, 60)
    def node(x, y, w, h, txt, c, fs=15):
        box(ax, x, y, w, h, c); ax.text(x + w / 2, y + h / 2, txt, color=FG, fontsize=fs, ha="center", va="center")
    node(2, 38, 17, 14, t("you type\na message", "você digita\numa mensagem"), BLUE)
    node(26, 38, 17, 14, t("model writes\na reply", "o modelo\nescreve"), TEAL)
    node(50, 38, 20, 14, t("repeat score\n< 0.3 ?", "pontuação de\nrepetição < 0,3 ?"), YEL)
    node(78, 38, 19, 14, t("show the\nreply", "mostra a\nresposta"), GRN)
    for x0, x1 in [(19.8, 25.2), (43.8, 49.2)]:
        ax.annotate("", xy=(x1, 45), xytext=(x0, 45), arrowprops=dict(arrowstyle="-|>", color=FG, lw=2.4))
    ax.annotate("", xy=(77.2, 45), xytext=(70.8, 45), arrowprops=dict(arrowstyle="-|>", color=GRN, lw=2.4)); ax.text(74, 47.2, t("yes", "sim"), color=GRN, fontsize=15, ha="center")
    ax.annotate("", xy=(60, 25.6), xytext=(60, 37.2), arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.4)); ax.text(62, 31.5, t("no: reject", "não: rejeita"), color=RED, fontsize=15)
    tries = [(t("try 2", "tentativa 2"), t("+0.2 temperature", "+0,2 de temperatura"), 2), (t("try 3", "tentativa 3"), t("+0.2, last turn only", "+0,2, só última troca"), 26), (t("try 4", "tentativa 4"), t("+0.3, no memory", "+0,3, sem memória"), 50)]
    node(2, 8, 22, 15, t("try 2\n+0.2 temp.", "tentativa 2\n+0,2 temp."), ORG, 14)
    node(30, 8, 22, 15, t("try 3\n+0.2, last turn only", "tentativa 3\n+0,2, só a última troca"), ORG, 13)
    node(58, 8, 22, 15, t("try 4\n+0.3, no memory", "tentativa 4\n+0,3, sem memória"), ORG, 14)
    for x0, x1 in [(24.8, 29.2), (52.8, 57.2)]:
        ax.annotate("", xy=(x1, 15.5), xytext=(x0, 15.5), arrowprops=dict(arrowstyle="-|>", color=FG, lw=2.4))
    ax.plot([60, 60, 13], [25.6, 25.2, 25.2], color=RED, lw=2.4); ax.annotate("", xy=(13, 23.8), xytext=(13, 25.2), arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.4))
    ax.text(84, 15.5, t("still repeating?\nshow the least\nrepetitive one", "ainda repete?\nmostra a menos\nrepetitiva"), color=DIM, fontsize=14, va="center")
    ax.text(2, 1.5, t("offline test of the 4-gram ban: 112 repeated phrases became 0.   User test with 3 retries: 'much better, perfect' for casual chat.", "teste offline da proibição de 4-gramas: 112 frases repetidas viraram 0.   Teste do usuário com 3 tentativas: 'muito melhor, perfeito' em conversa casual."), color=DIM, fontsize=12)
    save(fig, "14_repetition_fix")


def f15_decisions():
    fig = frame(t("What we decided, and the evidence", "O que decidimos, e a evidência"), None, size=(10, 7.4))
    ax = fig.add_axes([0.03, 0.02, 0.94, 0.86]); blank(ax); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    rows = [
        (t("English only, own 32,768-token BPE", "Só inglês, BPE próprio de 32.768 tokens"), t("4.61 characters per token", "4,61 caracteres por token"), BLUE),
        (t("Muon (lr 0.02) + AdamW for the rest", "Muon (lr 0,02) + AdamW no resto"), t("pilot loss 4.418 vs 5.280", "loss do piloto 4,418 vs 5,280"), TEAL),
        (t("3.0B tokens, a single pass", "3,0 bi de tokens, uma passada"), t("27 tokens per parameter", "27 tokens por parâmetro"), YEL),
        (t("SFT with lr 3e-4, 2 epochs", "SFT com lr 3e-4, 2 épocas"), t("1.630 vs 1.706 vs 1.761; epoch 2 gave only -0.057", "1,630 vs 1,706 vs 1,761; época 2 deu só -0,057"), ORG),
        (t("Identity: take the LAST checkpoint", "Identidade: usar o checkpoint FINAL"), t("the best-val one (step 200) had not learned it", "o de melhor val (passo 200) não tinha aprendido"), RED),
        (t("Anti-loop in the chat code, not retraining", "Anti-loop no código do chat, sem retreinar"), t("112 repeated phrases to 0", "112 frases repetidas para 0"), PUR),
        (t("Named Llama-WolfGPT-110M, 'Built with Llama'", "Nome Llama-WolfGPT-110M, 'Built with Llama'"), t("57% of chat data written by Llama 3.1", "57% dos dados de chat escritos pelo Llama 3.1"), GRN),
        (t("No more polishing, publish", "Sem mais polimento, publicar"), t("it already shows what 110M can do", "já mostra o que 110M consegue"), FG),
    ]
    ax.text(0, 98, t("DECISION", "DECISÃO"), color=DIM, fontsize=15, va="top"); ax.text(57, 98, t("EVIDENCE", "EVIDÊNCIA"), color=DIM, fontsize=15, va="top")
    ax.plot([0, 100], [92.5, 92.5], color=FG, lw=2.2)
    for i, (a, b, c) in enumerate(rows):
        y = 85 - i * 10.6
        box(ax, 0.5, y - 3.6, 55.5, 8.2, c, lw=2.2)
        ax.text(2.2, y + 0.5, a, color=FG, fontsize=14.5, va="center")
        ax.text(58, y + 0.5, b, color=c, fontsize=14.5, va="center")
    save(fig, "15_decisions")


def f16_report():
    fig = frame(t("Report card: what 110M parameters can do", "Boletim: o que 110M parâmetros conseguem"), t("from the real test chats (chat model)", "dos chats de teste reais (modelo de chat)"), size=(10, 7))
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.80]); blank(ax); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    good = [t("greets: 'Hello! How can I help you today?'", "cumprimenta: 'Hello! How can I help you today?'"), t("capital of France is Paris, Japan is Tokyo", "capital da França é Paris, do Japão é Tóquio"), t("stops when the answer is done", "para quando a resposta termina"), t("knows its name and who built it", "sabe o nome e quem o criou"), t("fluent, grammatical English", "inglês fluente e gramatical")]
    bad = [t("12 + 15 = 24 (arithmetic is luck)", "12 + 15 = 24 (conta é sorte)"), t("invents facts (base: water boils at 15 C)", "inventa fatos (base: água ferve a 15 C)"), t("ignores 'in two sentences'", "ignora 'em duas frases'"), t("poems: one line or repeating", "poemas: uma linha ou repetidos"), t("role-play: invents a father 'David'", "role-play: inventa um pai 'David'")]
    ax.text(0, 98, t("CAN", "CONSEGUE"), color=GRN, fontsize=22, va="top"); ax.text(52, 98, t("CAN'T", "NÃO CONSEGUE"), color=RED, fontsize=22, va="top")
    for i, s in enumerate(good):
        y = 82 - i * 15.5
        check(ax, 2.0, y, 1.6); ax.text(6, y, s, color=FG, fontsize=14, va="center", wrap=True)
    for i, s in enumerate(bad):
        y = 82 - i * 15.5
        cross(ax, 54.0, y, 1.4); ax.text(58, y, s, color=FG, fontsize=14, va="center")
    ax.plot([50, 50], [4, 92], color=DIM, lw=2, ls=":")
    save(fig, "16_report_card")


def f17_network():
    """Node-link picture of the real model: layer sizes come from the code, node counts are only for drawing."""
    fig = frame(t("The WolfGPT network, layer by layer", "A rede do WolfGPT, camada por camada"), t("every dot stands for many numbers; 12 identical blocks sit in the middle", "cada bolinha representa muitos números; 12 blocos idênticos no meio"), size=(12, 6.6))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.82]); blank(ax); ax.set_xlim(0, 100); ax.set_ylim(0, 60)
    rng = np.random.default_rng(7)
    cols = [  # x, n nodes, color, title, detail
        (8, 6, YEL, t("tokens", "tokens"), t("up to 1,024", "até 1.024")),
        (26, 9, ORG, t("embeddings", "embeddings"), "32,768 x 768" if L == "en" else "32.768 x 768"),
        (46, 9, BLUE, t("attention", "atenção"), t("Q, K, V + RoPE", "Q, K, V + RoPE")),
        (64, 9, RED, t("feed-forward", "feed-forward"), "SwiGLU"),
        (92, 6, GRN, t("next token", "próximo token"), "32,768" if L == "en" else "32.768"),
    ]
    pts = []
    for x, n, c, _, _ in cols:
        ys = np.linspace(14, 46, n)
        pts.append([(x, y) for y in ys])
    for a, b in zip(pts[:-1], pts[1:]):
        if a is pts[3]:
            continue
        for (x1, y1) in a:
            for (x2, y2) in b:
                ax.plot([x1, x2], [y1, y2], color=DIM, lw=0.7, alpha=0.45)
    for (x1, y1) in pts[3]:                       # feed-forward -> output
        for (x2, y2) in pts[4]:
            ax.plot([x1, x2], [y1, y2], color=DIM, lw=0.7, alpha=0.45)
    for (x, n, c, name, det), p in zip(cols, pts):
        for (px, py) in p:
            ax.add_patch(plt.Circle((px, py), 1.35, fc=c, ec=FG, lw=1.8, alpha=0.95, zorder=3))
        ax.text(x, 52, name, color=c, fontsize=17, ha="center", va="bottom")
        ax.text(x, 9.5, det, color=DIM, fontsize=13, ha="center", va="top")
    box(ax, 38.5, 10.5, 33.5, 38, PUR, lw=2.2)    # one block = attention + feed-forward
    ax.text(55.2, 4.0, t("block x 12: same shape, different weights", "bloco x 12: mesma forma, pesos diferentes"), color=PUR, fontsize=16, ha="center", va="center")
    save(fig, "17_network")


if __name__ == "__main__":
    only = os.environ.get("ONLY")
    for lang in ["en", "pt"]:
        L = lang
        with sk():
            after_xkcd()
            for f in [f01_journey, f02_hardware, f03_model, f04_tiny, f05_tokenizer, f06_pilot, f07_main, f08_hours, f09_data, f10_sweep, f11_sft, f12_identity, f13_bench, f14_flow, f15_decisions, f16_report, f17_network]:
                if only is None or f.__name__.startswith("f" + only):
                    f()
    print("ok")
